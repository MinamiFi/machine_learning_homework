#!/usr/bin/env python3
"""模拟人工抽检：用两个**独立于标签规则**的判据给 60 条样本打分，算 Cohen's κ。

## 为什么不能只做一次

标签定义是「本轮 delta 非空 且 至少一个 (slot,value) 在对话结束时仍有效」。
如果我用同一套 state/delta 数据再判一次，那叫同义反复，κ 必然接近 1，毫无意义。

所以这里用两个**决策过程不同**的判据：

**J1 — LLM 内容判据（完全看不到 state/delta）**
    给模型看「这一轮的 user+system 原文」和「对话最后的 query」，问：
    这一轮的信息对回答 query 是否必要？只答 yes/no。
    它只能读对话文本，不知道标签怎么算的 → 与标签管线独立。

**J2 — 规则判据「最后提及」（用 state，但规则不同）**
    标签问的是「引入」：这一轮新增了什么。
    J2 问的是「末次提及」：某个最终有效的槽位值，是否只在**这一轮及之前**
    出现过、之后再也不出现？若是，删掉这一轮就再也找不回来。
    这是 necessity 的近亲但口径不同（necessity 要求全对话唯一支撑；
    J2 只要求"此后不再出现"，允许之前出现过）。

两者都与标签**不同**，所以：
- κ(标签, J1) 和 κ(标签, J2) 衡量标签在两种独立视角下是否站得住；
- κ(J1, J2) 衡量两个判据自身是否一致——这是「模拟人类一致性」的代理。

## 这个结果不能当人工校验用

真正的 κ 需要人。这里给的是**模拟值**，文档里必须写明：
它证明的是「标签与两种独立的自动化判据一致」，不是「标签与人一致」。

用法：
    source exp/env.sh
    python exp/scripts/simulate_spot_check.py --n 60
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.cache import ResponseCache                       # noqa: E402
from src.agent.llm import LocalQwen                            # noqa: E402
from src.agent.prompt import render_context                    # noqa: E402
from src.common import dump_json, get_logger, load_yaml, path_of, set_seed  # noqa: E402
from src.memory.long_history import last_substantive           # noqa: E402
from src.memory.slice import group_by_dialogue, load_segments  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_counterfactual import (SYMBOLIC_VALUES, build_pairs,  # noqa: E402
                                   mentions, seg_text)

log = get_logger("spot_check")

# ---------------------------------------------------------------- LLM 判据

JUDGE_PROMPT = """You are judging whether one turn of a conversation is necessary.

A user and a system assistant talk over several turns to plan a trip.
At the very end the user asks something. Your job: decide whether the
information in the TURN below is needed to answer that FINAL QUESTION.

TURN:
{turn}

FINAL QUESTION:
{query}

Is the information in the TURN needed to answer the FINAL QUESTION?
Answer with exactly one word: yes or no.
Answer:"""


def judge_llm(llm, cache: ResponseCache, turn_text: str, query: str,
              max_new_tokens: int = 4) -> tuple[int, str]:
    prompt = JUDGE_PROMPT.format(turn=turn_text, query=query)
    hit = cache.get(prompt, max_new_tokens)
    if hit is not None:
        raw = hit["text"]
    else:
        g = llm.generate(prompt, max_new_tokens=max_new_tokens)
        raw = g.text
        cache.put(prompt, max_new_tokens, raw, g.seconds, judge="J1")
    low = raw.strip().lower()
    # 只认开头的 yes/no；模型偶尔会补一句解释
    m = re.match(r"[^a-z]*(yes|no)", low)
    if m:
        return (1 if m.group(1) == "yes" else 0), raw
    # 兜底：出现 yes 且没出现 no
    if "yes" in low and "no" not in low:
        return 1, raw
    if "no" in low and "yes" not in low:
        return 0, raw
    return 0, raw


# ---------------------------------------------------------------- 规则判据

def judge_last_mention(hist: list, gold: dict[str, str], i: int) -> int:
    """J2「末次提及」：存在某个最终有效槽位值，本轮提到它、之后不再提到。"""
    later_text = " ".join(seg_text(s) for s in hist[i + 1:]).lower()
    here = seg_text(hist[i]).lower()
    for slot, val in gold.items():
        if str(val).lower().strip() in SYMBOLIC_VALUES:
            continue
        if not mentions(val, here):
            continue
        if not mentions(val, later_text):
            return 1
    return 0


# ---------------------------------------------------------------- 一致性统计

def cohen_kappa(a: list[int], b: list[int]) -> dict:
    a = np.array(a)
    b = np.array(b)
    n = len(a)
    po = float((a == b).mean())
    # 两个评分者的边际分布
    pa1, pa0 = a.mean(), 1 - a.mean()
    pb1, pb0 = b.mean(), 1 - b.mean()
    pe = pa1 * pb1 + pa0 * pb0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    # 混淆矩阵
    tp = int(((a == 1) & (b == 1)).sum())
    tn = int(((a == 0) & (b == 0)).sum())
    fp = int(((a == 0) & (b == 1)).sum())
    fn = int(((a == 1) & (b == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "agreement": round(po, 4),
        "kappa": round(kappa, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pos_rate_a": round(float(a.mean()), 4),
        "pos_rate_b": round(float(b.mean()), 4),
    }


def kappa_ci(a: list[int], b: list[int], n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float]:
    """按样本重采样给 κ 一个置信区间。"""
    rng = np.random.default_rng(seed)
    a = np.array(a)
    b = np.array(b)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        k = cohen_kappa(a[idx].tolist(), b[idx].tolist())["kappa"]
        vals.append(k)
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


# ---------------------------------------------------------------- 主流程

def build_samples(pairs: list[dict], n: int, seed: int = 0) -> list[dict]:
    """与 write_spot_check 同一抽样逻辑，保证和 CSV 是同一批 60 条。"""
    rng = random.Random(seed)
    pos_rows, neg_rows = [], []
    for p in pairs:
        for i, s in enumerate(p["hist"]):
            row = {"pair": p, "i": i, "seg": s}
            (pos_rows if s.label == 1 else neg_rows).append(row)
    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)
    half = n // 2
    rows = pos_rows[:half] + neg_rows[:n - half]
    rng.shuffle(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--tag", default="spot_check")
    args = ap.parse_args()

    set_seed(0)
    t0 = time.time()
    cfg = load_yaml(args.config)
    model_path = path_of(cfg["local"]["path"])

    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    pairs = build_pairs(by_d)
    log.info("可用配对对话 %d", len(pairs))

    samples = build_samples(pairs, args.n)
    log.info("抽检 %d 条（正负例各半）", len(samples))

    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(run_dir / "cache.jsonl", model_name=str(model_path))

    llm = LocalQwen(str(model_path), dtype=cfg["local"]["dtype"],
                    max_new_tokens=args.max_new_tokens, temperature=0.0)
    log.info("backbone 就绪：%s", llm.name)

    rows = []
    for k, smp in enumerate(samples):
        p, i, seg = smp["pair"], smp["i"], smp["seg"]
        # query 取对话最后一个实质轮（build_pairs 里 hist 已截到 query 之前）
        query = last_substantive(by_d[p["dialogue_id"]]).user
        turn_text = seg_text(seg)

        j1, raw = judge_llm(llm, cache, turn_text, query, args.max_new_tokens)
        j2 = judge_last_mention(p["hist"], p["hist"][-1].state, i)

        rows.append({
            "dialogue_id": p["dialogue_id"],
            "turn_idx": seg.turn_idx,
            "label": int(seg.label),
            "J1_llm": j1,
            "J2_last_mention": j2,
            "query": query,
            "turn": turn_text[:300],
            "raw": raw.strip()[:40],
        })
        if (k + 1) % 10 == 0:
            log.info("进度 %d/%d", k + 1, len(samples))

    lab = [r["label"] for r in rows]
    j1s = [r["J1_llm"] for r in rows]
    j2s = [r["J2_last_mention"] for r in rows]

    res = {
        "split": args.split,
        "n": len(rows),
        "backbone": llm.name,
        "seconds": round(time.time() - t0, 1),
        "cache": {"hits": cache.hits, "misses": cache.misses},
        "label_vs_J1": cohen_kappa(lab, j1s),
        "label_vs_J2": cohen_kappa(lab, j2s),
        "J1_vs_J2": cohen_kappa(j1s, j2s),
        "rows": rows,
    }
    res["label_vs_J1"]["kappa_ci"] = kappa_ci(lab, j1s)
    res["label_vs_J2"]["kappa_ci"] = kappa_ci(lab, j2s)
    res["J1_vs_J2"]["kappa_ci"] = kappa_ci(j1s, j2s)

    dump_json(res, run_dir / "spot_check.json")

    # 同时把模拟判断写回 CSV 的「你的判断」列，标明来源
    csv_path = path_of("docs") / "标签抽检表.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig") as f:
            cr = list(csv.DictReader(f))
        key = {(r["dialogue_id"], int(r["turn_idx"])): r for r in cr}
        filled = 0
        for r in rows:
            c = key.get((r["dialogue_id"], r["turn_idx"]))
            if not c:
                continue
            # 两个判据一致时直接给结论，不一致时留空待人工
            if r["J1_llm"] == r["J2_last_mention"]:
                c["你的判断(1=该轮信息对回答query必要, 0=不必要)"] = str(r["J1_llm"])
                c["备注"] = "模拟：J1/J2 一致"
                filled += 1
            else:
                c["你的判断(1=该轮信息对回答query必要, 0=不必要)"] = ""
                c["备注"] = f"模拟：判据分歧 J1={r['J1_llm']} J2={r['J2_last_mention']}"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(cr[0].keys()))
            w.writeheader()
            w.writerows(cr)
        res["csv_filled"] = filled
        log.info("已把 %d/%d 条模拟结论写回 %s（分歧条留空）",
                 filled, len(rows), csv_path.name)

    log.info("κ(标签, J1_LLM) = %.4f  一致性 %.4f",
             res["label_vs_J1"]["kappa"], res["label_vs_J1"]["agreement"])
    log.info("κ(标签, J2_末次提及) = %.4f  一致性 %.4f",
             res["label_vs_J2"]["kappa"], res["label_vs_J2"]["agreement"])
    log.info("κ(J1, J2) = %.4f  一致性 %.4f",
             res["J1_vs_J2"]["kappa"], res["J1_vs_J2"]["agreement"])
    log.info("完成，用时 %.0fs → %s", res["seconds"], run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
