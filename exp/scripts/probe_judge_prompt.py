#!/usr/bin/env python3
"""探针：给 0.5B 找一种不退化的是/否判据写法。

背景：`simulate_spot_check.py` 的 J1 用「Is the information in the TURN
needed to answer the FINAL QUESTION? yes/no」，实测 58/60 全答 no —— 退化，
κ 无意义。这里试几种写法，看哪种的预测分布不是常数。

判定标准：正例率落在 0.2–0.8 之间，且与标签有可测的一致性。

用法：
    source exp/env.sh
    python exp/scripts/probe_judge_prompt.py --n 60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent.cache import ResponseCache                          # noqa: E402
from src.agent.llm import LocalQwen                               # noqa: E402
from src.common import dump_json, get_logger, load_yaml, path_of, set_seed  # noqa: E402
from src.memory.long_history import last_substantive              # noqa: E402
from src.memory.slice import group_by_dialogue, load_segments     # noqa: E402
from simulate_spot_check import (build_samples, cohen_kappa,       # noqa: E402
                                 judge_last_mention)
from verify_counterfactual import build_pairs, seg_text            # noqa: E402

log = get_logger("probe")

# ---- 写法 A：few-shot，一正一负各一个例子 --------------------------------
P_FEWSHOT = """Decide whether a turn of conversation is needed to answer a question.

Example 1
TURN:
User: i need a train to cambridge on tuesday
System: There are trains leaving at 9:15 and 10:30. Which do you prefer?
QUESTION: what time does the train leave
Answer: yes

Example 2
TURN:
User: hello
System: Hello, how can I help you today?
QUESTION: what time does the train leave
Answer: no

Now decide.
TURN:
{turn}
QUESTION: {query}
Answer:"""

# ---- 写法 B：抽取式，答不出就回 NONE ------------------------------------
P_EXTRACT = """Read the context and fill in the blank. Reply with only the missing words, or NONE.

Context:
{turn}

The final question is: {query}
The words in the context that answer it are: ___"""

# ---- 写法 C：显式选项 A/B ----------------------------------------------
P_AB = """TURN:
{turn}

QUESTION: {query}

Does the TURN contain information that answers the QUESTION?
A) yes, it contains the answer
B) no, it does not
Reply with only the letter. Answer:"""

VARIANTS = {"A_fewshot": P_FEWSHOT, "B_extract": P_EXTRACT, "C_ab": P_AB}


def parse_yesno(raw: str) -> int:
    low = raw.strip().lower()
    m = re.match(r"[^a-z]*(yes|no)", low)
    if m:
        return 1 if m.group(1) == "yes" else 0
    if "yes" in low and "no" not in low:
        return 1
    if "no" in low and "yes" not in low:
        return 0
    return 0


def parse_extract(raw: str) -> int:
    low = raw.strip().lower()
    if not low or low.startswith("none") or low in {"n/a", "na", "-"}:
        return 0
    return 1


def parse_ab(raw: str) -> int:
    low = raw.strip().lower()
    m = re.search(r"\b([ab])\b", low)
    if m:
        return 1 if m.group(1) == "a" else 0
    return parse_yesno(raw)


PARSERS = {"A_fewshot": parse_yesno, "B_extract": parse_extract, "C_ab": parse_ab}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--tag", default="judge_probe")
    args = ap.parse_args()

    set_seed(0)
    cfg = load_yaml(args.config)
    model_path = path_of(cfg["local"]["path"])

    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    pairs = build_pairs(by_d)
    samples = build_samples(pairs, args.n)
    log.info("探针 %d 条", len(samples))

    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(run_dir / "cache.jsonl", model_name=str(model_path))
    llm = LocalQwen(str(model_path), dtype=cfg["local"]["dtype"],
                    max_new_tokens=args.max_new_tokens, temperature=0.0)

    # 先把样本的 query / turn / label / J2 备好
    items = []
    for smp in samples:
        p, i, seg = smp["pair"], smp["i"], smp["seg"]
        items.append({
            "label": int(seg.label),
            "turn": seg_text(seg),
            "query": last_substantive(by_d[p["dialogue_id"]]).user,
            "j2": judge_last_mention(p["hist"], p["hist"][-1].state, i),
        })

    out = {}
    for name, tpl in VARIANTS.items():
        t0 = time.time()
        preds, raws = [], []
        for k, it in enumerate(items):
            prompt = tpl.format(turn=it["turn"], query=it["query"])
            hit = cache.get(prompt, args.max_new_tokens)
            if hit is not None:
                raw = hit["text"]
            else:
                g = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
                raw = g.text
                cache.put(prompt, args.max_new_tokens, raw, g.seconds, probe=name)
            preds.append(PARSERS[name](raw))
            raws.append(raw.strip().replace("\n", " ")[:60])
            if (k + 1) % 20 == 0:
                log.info("%s 进度 %d/%d", name, k + 1, len(items))

        labels = [it["label"] for it in items]
        j2s = [it["j2"] for it in items]
        km = cohen_kappa(labels, preds)
        out[name] = {
            "seconds": round(time.time() - t0, 1),
            "pred_pos_rate": round(sum(preds) / len(preds), 4),
            "label_vs_pred": km,
            "pred_vs_J2": cohen_kappa(preds, j2s),
            "n_distinct_raw": len(set(raws)),
            "sample_raws": raws[:8],
        }
        log.info("%s: 正例率 %.3f | κ(标签) %.4f 一致性 %.4f | κ(J2) %.4f",
                 name, out[name]["pred_pos_rate"], km["kappa"],
                 km["agreement"], out[name]["pred_vs_J2"]["kappa"])

    dump_json({"n": len(items), "variants": out}, run_dir / "probe.json")
    log.info("完成 → %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
