#!/usr/bin/env python3
"""L2 的 JGA 口径校验：删掉某组轮次后，LLM 还能答对多少。

## 为什么是这个口径

根 README §4.4 原方案是「删掉记忆后 JGA 是否下降」。直接算**绝对 JGA**
需要 backbone 会做对话状态追踪，而本机只有 Qwen2.5-0.5B——实测让它输出
整段对话的 slot/value JSON，它给出的是自造 schema（`hotel_options` 数组），
与 gold 完全不可比；定向抽取 12 个槽位只中 4 个。

所以改测**配对**口径：同一段对话、同一个槽位，只改「上下文里给了什么」，
比的是「删正例 vs 删负例，答对率各掉多少」。这样 0.5B 的绝对能力差被
两边同时扣掉，剩下的差值才是标签带来的。

前提是 0.5B 至少能从上下文里**抽出**答案。实测（`docs/L2_JGA校验.md` §2）：
给含 needle 的上下文答对 0.52，把上下文换成「无相关内容」只有 0.04，
13 倍差 —— 说明它确实在读上下文，不是瞎猜。

## 与反事实覆盖率校验的关系

覆盖率口径（`verify_counterfactual.py`）用字符串匹配判断「删掉后信息还在不在」，
零成本但只看字面。这里用 LLM 判断「删掉后还能不能答对」，有成本但看的是
下游效果。两个口径独立，结论一致才算站得住。

用法：
    source exp/env.sh
    python exp/scripts/jga_check.py --n-dialogues 50 --max-slots 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.cache import ResponseCache                          # noqa: E402
from src.agent.llm import build_llm                                # noqa: E402
from src.agent.prompt import (answer_matches, askable_slots,      # noqa: E402
                              build_prompt, render_context)
from src.common import dump_json, get_logger, load_yaml, path_of, set_seed  # noqa: E402
from src.memory.long_history import last_substantive              # noqa: E402
from src.memory.slice import group_by_dialogue, load_segments     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_counterfactual import measurable_slots                # noqa: E402

log = get_logger("jga_check")

# 四个条件。删轮量与反事实校验一致：k = min(n_pos, n_neg)。
CONDITIONS = ("full", "drop_pos", "drop_neg", "drop_rand")


def build_cases(segs_by_dialogue, n_dialogues: int, max_slots: int,
                seed: int) -> list[dict]:
    """挑出可做配对比较的对话，并为每个对话确定要问的槽位。"""
    rng = random.Random(seed)
    cases = []
    for d in sorted(segs_by_dialogue):
        v = segs_by_dialogue[d]
        q = last_substantive(v)
        hist = [s for s in v if s.turn_idx < q.turn_idx]
        if len(hist) < 3:
            continue
        gold = hist[-1].state
        ask = askable_slots(gold)
        if not ask:
            continue

        pos = [i for i, s in enumerate(hist) if s.label == 1]
        neg = [i for i, s in enumerate(hist) if s.label == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue

        # 只问「删正例时可能丢掉的」槽位，否则四条件答对率全 1，比不出东西。
        support = measurable_slots(hist, gold)
        drop_pos = set(pos[:k])
        affected = [s for s, idxs in support.items()
                    if s in ask and set(idxs) & drop_pos]
        if not affected:
            continue
        # 再要求「删负例时不该丢」——这类对话才是真正能区分的样本
        drop_neg = set(neg[:k])
        clean = [s for s in affected if not (set(support[s]) & drop_neg)]
        if not clean:
            continue

        slots = sorted(clean)[:max_slots]
        cases.append({
            "dialogue_id": d,
            "query": q.user,
            "n_hist": len(hist),
            "k": k,
            "slots": {s: gold[s] for s in slots},
            "hist": hist,
            "drop_pos": sorted(drop_pos),
            "drop_neg": sorted(drop_neg),
            "drop_rand": sorted(rng.sample(range(len(hist)), k)),
        })

    rng.shuffle(cases)
    return cases[:n_dialogues]


def context_for(case: dict, cond: str, max_turns: int | None) -> str:
    hist = case["hist"]
    drop = set(case[cond]) if cond != "full" else set()
    kept = [s for i, s in enumerate(hist) if i not in drop]
    if max_turns:
        kept = kept[-max_turns:]
    return render_context(kept)


class LazyLLM:
    """按需加载。全命中缓存时（换匹配规则重算）不加载模型，省 3.2 GB。"""

    def __init__(self, factory):
        self._factory = factory
        self._real = None
        self.name = "lazy"

    def _get(self):
        if self._real is None:
            self._real = self._factory()
            self.name = self._real.name
            log.info("backbone 就绪：%s", self.name)
        return self._real

    def generate(self, prompt: str, max_new_tokens: int | None = None):
        return self._get().generate(prompt, max_new_tokens=max_new_tokens)


def run(cases: list[dict], llm, cache: ResponseCache, max_turns: int | None,
        max_new_tokens: int, cache_only: bool = False,
        workers: int = 1) -> dict:
    """跑四条件。

    `cache_only=True`：只算「缓存齐全」的对话，缺的整段跳过。全量 5328 次
    生成在本地 0.5B 上要 2 小时以上，中途进程被杀（4 GB 内存上限，任何
    并行的模型加载都会 OOM）就什么都拿不到。有这个模式就能在任意时刻
    用已有缓存算出一份**部分结果**，并在产物里写明覆盖了多少段对话。

    `workers>1`：**先把所有缺缓存的 prompt 收集起来并发跑，再组装结果**，
    而不是在循环里逐条生成。串行 5328 次在 API 上约 62 分钟（0.7s/次），
    并发 4 实测 6.4 次/秒 → 约 14 分钟。并发 8 以上反而变慢（限流），
    默认 1 保持原有行为，脚本里按 backbone 选。
    """
    rows = []
    skipped: list[str] = []

    # 第一遍：规划。缓存命中的直接取，未命中的进待跑队列。
    plan: list[tuple[int, dict, str, str, str]] = []   # (case_i, case, slot, gold, cond)
    hit_map: dict[tuple[int, str, str], tuple[str, float]] = {}
    for ci, case in enumerate(cases):
        ctxs = {c: context_for(case, c, max_turns) for c in CONDITIONS}
        if cache_only:
            need = [build_prompt(ctxs[c], slot) for slot in case["slots"]
                    for c in CONDITIONS]
            if any(cache.get(p, max_new_tokens) is None for p in need):
                skipped.append(case["dialogue_id"])
                continue
        for slot, gold in case["slots"].items():
            for cond in CONDITIONS:
                prompt = build_prompt(ctxs[cond], slot)
                hit = cache.get(prompt, max_new_tokens)
                if hit is not None:
                    hit_map[(ci, slot, cond)] = (hit["text"], 0.0)
                else:
                    plan.append((ci, case, slot, gold, cond))

    # 第二遍：并发跑未命中的。
    log.info("待生成 %d 条（缓存命中 %d），workers=%d",
             len(plan), len(hit_map), workers)
    if plan:
        def _one(item):
            _ci, case, slot, _gold, cond = item
            prompt = build_prompt(context_for(case, cond, max_turns), slot)
            g = llm.generate(prompt, max_new_tokens=max_new_tokens)
            cache.put(prompt, max_new_tokens, g.text, g.seconds,
                      slot=slot, cond=cond, dialogue_id=case["dialogue_id"])
            return (item[0], slot, cond), (g.text, g.seconds)

        done = 0
        if workers <= 1:
            for item in plan:
                k, v = _one(item)
                hit_map[k] = v
                done += 1
                if done % 50 == 0:
                    log.info("生成 %d/%d，缓存命中 %d", done, len(plan), cache.hits)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_one, it) for it in plan]
                for f in as_completed(futs):
                    k, v = f.result()
                    hit_map[k] = v
                    done += 1
                    if done % 200 == 0:
                        log.info("生成 %d/%d", done, len(plan))

    # 第三遍：组装。
    for ci, case in enumerate(cases):
        if cache_only and case["dialogue_id"] in skipped:
            continue
        for slot, gold in case["slots"].items():
            per_cond = {}
            for cond in CONDITIONS:
                pred, _sec = hit_map[(ci, slot, cond)]
                per_cond[cond] = {"pred": pred, "ok": answer_matches(gold, pred)}
            rows.append({"dialogue_id": case["dialogue_id"], "slot": slot,
                         "gold": gold, "by_cond": per_cond})

    if skipped:
        log.warning("cache_only：跳过 %d 段缓存不全的对话（已算 %d 段）",
                    len(skipped), len({r["dialogue_id"] for r in rows}))
    return {"rows": rows, "n_skipped_incomplete": len(skipped)}


def summarize(rows: list[dict], cases: list[dict]) -> dict:
    """槽位级准确率 + 对话级 JGA（该对话抽到的槽位全对）。"""
    acc = {c: [] for c in CONDITIONS}
    for r in rows:
        for c in CONDITIONS:
            acc[c].append(float(r["by_cond"][c]["ok"]))

    # 对话级：按 dialogue_id 聚合，全对记 1
    by_d: dict[str, dict[str, list[bool]]] = {}
    for r in rows:
        d = r["dialogue_id"]
        by_d.setdefault(d, {c: [] for c in CONDITIONS})
        for c in CONDITIONS:
            by_d[d][c].append(bool(r["by_cond"][c]["ok"]))

    jga = {c: [] for c in CONDITIONS}
    for d, m in by_d.items():
        for c in CONDITIONS:
            jga[c].append(1.0 if m[c] and all(m[c]) else 0.0)

    out = {
        "n_dialogues": len(by_d),
        "n_slot_rows": len(rows),
        "slot_acc": {c: round(float(np.mean(acc[c])), 4) for c in CONDITIONS},
        "jga": {c: round(float(np.mean(jga[c])), 4) for c in CONDITIONS},
    }

    # 「答得出来」的子集：full 条件下就答对的行。
    # 必须单独看——0.5B 对某些槽位即使给全上下文也答不对，把这些算进
    # 分母会同时稀释两边，差值被压向 0（噪声也会被放大）。
    answerable = [r for r in rows if r["by_cond"]["full"]["ok"]]
    out["slot_acc_answerable"] = {
        c: (round(float(np.mean([r["by_cond"][c]["ok"] for r in answerable])), 4)
            if answerable else 0.0)
        for c in CONDITIONS
    }
    out["n_answerable"] = len(answerable)
    out["answerable_rate"] = round(len(answerable) / max(len(rows), 1), 4)

    # 配对 bootstrap：正例组与负例组的差值（按对话重采样）
    rng = np.random.default_rng(0)
    keys = sorted(by_d)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        a = [1.0 if all(by_d[keys[i]]["drop_pos"]) else 0.0 for i in pick]
        b = [1.0 if all(by_d[keys[i]]["drop_neg"]) else 0.0 for i in pick]
        diffs.append(float(np.mean(a) - np.mean(b)))
    diffs = np.array(diffs)
    out["paired"] = {
        "diff": round(float(out["jga"]["drop_pos"] - out["jga"]["drop_neg"]), 4),
        "ci_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci_high": round(float(np.percentile(diffs, 97.5)), 4),
        "p_two_sided": round(float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())), 4),
    }

    # 全部两两对比。只报一组会让报告里的其它说法（"删负例反而比全量略好"）
    # 在产物里查不到出处。
    def _cmp(a: str, b: str, n_boot: int = 5000) -> dict:
        rng2 = np.random.default_rng(0)
        d = []
        for _ in range(n_boot):
            pick = rng2.choice(len(keys), size=len(keys), replace=True)
            av = [1.0 if all(by_d[keys[i]][a]) else 0.0 for i in pick]
            bv = [1.0 if all(by_d[keys[i]][b]) else 0.0 for i in pick]
            d.append(float(np.mean(av) - np.mean(bv)))
        d = np.array(d)
        pt = (float(np.mean([1.0 if all(by_d[k][a]) else 0.0 for k in keys]))
              - float(np.mean([1.0 if all(by_d[k][b]) else 0.0 for k in keys])))
        return {
            "diff": round(pt, 4),
            "ci_low": round(float(np.percentile(d, 2.5)), 4),
            "ci_high": round(float(np.percentile(d, 97.5)), 4),
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
        }

    pairs = [("drop_pos", "drop_neg"), ("drop_pos", "drop_rand"),
             ("drop_neg", "drop_rand"), ("full", "drop_neg"),
             ("drop_rand", "full")]
    out["paired_all"] = {f"{a}-{b}": _cmp(a, b) for a, b in pairs}

    # ---- 分槽位细分
    # 主表只给全局平均，会把「所有槽位一视同仁」这个错误印象固化下来。
    # 实测 train-day / restaurant-day / train-leaveat 在 full 条件下就是 0，
    # 而 hotel-day 是 1.0——差距大到必须分开报，否则平均值没有意义。
    # n 小时不给数字（样本太少，CI 会宽到没用），只报计数。
    by_slot: dict[str, dict] = {}
    for r in rows:
        sl = r["slot"]
        d = by_slot.setdefault(sl, {
            "n": 0,
            **{c: 0 for c in CONDITIONS},
            "full_ok": 0,
        })
        d["n"] += 1
        for c in CONDITIONS:
            d[c] += int(bool(r["by_cond"][c]["ok"]))
        d["full_ok"] += int(bool(r["by_cond"]["full"]["ok"]))
    for sl, d in by_slot.items():
        n = d["n"]
        d["slot_acc"] = {c: round(d[c] / n, 4) for c in CONDITIONS}
        d["answerable_rate"] = round(d["full_ok"] / n, 4)
    out["by_slot"] = dict(sorted(by_slot.items(),
                                 key=lambda kv: (-kv[1]["n"], kv[0])))
    return out


def to_markdown(res: dict) -> str:
    s = res["summary"]
    lines = [
        "# L2 JGA 口径校验结果", "",
        f"- 对话数 **{s['n_dialogues']}**，槽位提问 **{s['n_slot_rows']}** 条",
        f"- backbone `{res['backbone']}`，max_new_tokens {res['max_new_tokens']}",
        f"- 耗时 {res['seconds']:.0f} 秒，缓存命中 {res['cache']['hits']} / "
        f"未命中 {res['cache']['misses']}",
        "", "## 槽位级准确率", "",
        "| 条件 | 准确率 |", "|---|---|",
    ]
    for c in CONDITIONS:
        lines.append(f"| {c} | {s['slot_acc'][c]:.4f} |")
    lines += ["", "## 对话级 JGA（该对话抽到的槽位全对）", "",
              "| 条件 | JGA |", "|---|---|"]
    for c in CONDITIONS:
        lines.append(f"| {c} | {s['jga'][c]:.4f} |")
    lines += [
        "", "## 只在「给全上下文就答对」的子集上比", "",
        f"0.5B 对部分槽位即使给全上下文也答不出（可答率 "
        f"{s['answerable_rate']:.3f}，{s['n_answerable']} / {s['n_slot_rows']} 条）。"
        "只在这个子集上看删除的影响，才是干净的对比。", "",
        "| 条件 | 准确率 |", "|---|---|",
    ]
    for c in CONDITIONS:
        lines.append(f"| {c} | {s['slot_acc_answerable'][c]:.4f} |")
    p = s["paired"]
    lines += [
        "", "## 配对差值（drop_pos − drop_neg）", "",
        f"- 差值 **{p['diff']:+.4f}**，95% CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]，"
        f"p = {p['p_two_sided']:.4f}",
    ]
    if "paired_all" in s:
        lines += ["", "### 全部两两对比（5000 次 bootstrap，按对话重采样）", "",
                  "| 对比 | 差值 | 95% CI | p |", "|---|---|---|---|"]
        for k, v in s["paired_all"].items():
            a, b = k.split("-", 1)
            lines.append(
                f"| {a} − {b} | {v['diff']:+.4f} | "
                f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {v['p_two_sided']:.4f} |")

    # ---- 分槽位细分
    # 只报 n>=5 的（n 太小 CI 宽到没用，数字反而误导）；完整表在 json 里。
    bs = s.get("by_slot") or {}
    big = {k: v for k, v in bs.items() if v["n"] >= 5}
    if big:
        lines += ["", "## 分槽位细分", "",
                  f"共 {len(bs)} 个槽位类型，其中 {len(big)} 个 n≥5（下表）。"
                  "完整表（含 n<5）见 `jga.json` 的 `summary.by_slot`。", "",
                  "| 槽位 | n | full | drop_pos | drop_neg | drop_rand | 可答率 |",
                  "|---|---|---|---|---|---|---|"]
        for sl, d in big.items():
            a = d["slot_acc"]
            lines.append(
                f"| {sl} | {d['n']} | {a['full']:.3f} | {a['drop_pos']:.3f} | "
                f"{a['drop_neg']:.3f} | {a['drop_rand']:.3f} | "
                f"{d['answerable_rate']:.3f} |")
        # 只挑出真正值得注意的：全量上下文都答不出、或删正例无损的
        zero_full = [k for k, v in big.items() if v["slot_acc"]["full"] == 0.0]
        pos_harmless = [k for k, v in big.items()
                        if v["slot_acc"]["full"] > 0
                        and v["slot_acc"]["drop_pos"] >= v["slot_acc"]["full"]]
        if zero_full:
            lines += ["", f"- **全量上下文也答不出**（0.5B 能力所限，"
                          f"不是记忆选择的问题）：{', '.join(zero_full)}"]
        if pos_harmless:
            lines += ["", f"- **删正例不掉分**（说明这些槽位的信息在别处也有）："
                          f"{', '.join(pos_harmless)}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-dialogues", type=int, default=50)
    ap.add_argument("--max-slots", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="只保留最近 N 轮（默认全部）")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--tag", default="jga")
    ap.add_argument("--resume", default=None,
                    help="复用某个已有 run 目录的 cache.jsonl（换匹配规则重算时不重跑生成）")
    ap.add_argument("--workers", type=int, default=0,
                    help="并发数。0=自动（api 用 4，local 用 1）。"
                         "API 实测并发 4 约 6.4 次/秒，8 以上反而被限流")
    ap.add_argument("--cache-only", action="store_true",
                    help="只用缓存里已有的生成结果，缺的对话整段跳过。"
                         "用于全量跑不完时先拿一份部分结果（产物会记录覆盖对话数）")
    args = ap.parse_args()

    set_seed(0)
    t0 = time.time()
    cfg = load_yaml(args.config)

    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    cases = build_cases(by_d, args.n_dialogues, args.max_slots, seed=0)
    log.info("挑出 %d 段可配对对话（删正例会丢、删负例不丢的槽位）", len(cases))
    if not cases:
        log.error("没有可用对话")
        return 1
    n_slots = sum(len(c["slots"]) for c in cases)
    log.info("共 %d 条槽位提问 × 4 条件 = %d 次生成", n_slots, n_slots * 4)

    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # backbone 从 configs/agent.yaml 的 `backbone` 决定（local / api / dummy）。
    # 早先这里硬编码 LocalQwen，接上 API 后仍会加载 0.5B——配置改了却没生效，
    # 而且 4 GB 内存下同时跑本地模型和 API 请求是没必要的浪费。
    backbone = (cfg.get("backbone") or "local").lower()
    if backbone == "api":
        cache_model = f"api:{cfg['api']['model']}"
    elif backbone == "dummy":
        cache_model = "dummy"
    else:
        cache_model = str(path_of(cfg["local"]["path"]))
    cache_path = (path_of("runs", args.resume) / "cache.jsonl"
                  if args.resume else run_dir / "cache.jsonl")
    # 缓存 key 含 model 名，所以 local 与 api 的缓存天然隔离，不会互相污染
    cache = ResponseCache(cache_path, model_name=cache_model)
    if args.resume:
        log.info("复用缓存 %s（%d 条，model=%s）", cache_path, len(cache), cache_model)

    llm = LazyLLM(lambda: build_llm(cfg, max_new_tokens=args.max_new_tokens))

    workers = args.workers
    if workers <= 0:
        # 本地模型单进程跑满 2 核，并发只会互相抢；API 是网络等待，并发有效。
        workers = 3 if backbone == "api" else 1
    out = run(cases, llm, cache, args.max_turns, args.max_new_tokens,
              cache_only=args.cache_only, workers=workers)
    summary = summarize(out["rows"], cases)

    res = {
        "split": args.split,
        "backbone": llm.name if llm.name != "lazy" else "lazy(全命中缓存)",
        "max_new_tokens": args.max_new_tokens,
        "max_turns": args.max_turns,
        "seconds": time.time() - t0,
        "cache": {"hits": cache.hits, "misses": cache.misses},
        "cache_only": bool(args.cache_only),
        "n_skipped_incomplete": out.get("n_skipped_incomplete", 0),
        "n_cases_requested": len(cases),
        "summary": summary,
        "rows": out["rows"],
    }
    dump_json(res, run_dir / "jga.json")
    (run_dir / "report.md").write_text(to_markdown(res), encoding="utf-8")

    log.info("槽位准确率：%s", summary["slot_acc"])
    log.info("可答子集（%d/%d，%.3f）：%s", summary["n_answerable"],
             summary["n_slot_rows"], summary["answerable_rate"],
             summary["slot_acc_answerable"])
    log.info("对话级 JGA：%s", summary["jga"])
    log.info("配对差值 %+.4f CI[%+.4f, %+.4f] p=%.4f",
             summary["paired"]["diff"], summary["paired"]["ci_low"],
             summary["paired"]["ci_high"], summary["paired"]["p_two_sided"])
    bs = summary.get("by_slot") or {}
    big = [(k, v) for k, v in bs.items() if v["n"] >= 5]
    if big:
        log.info("分槽位（n>=5，共 %d 个）：", len(big))
        for k, v in big:
            a = v["slot_acc"]
            log.info("  %-24s n=%3d  full=%.3f  drop_pos=%.3f  drop_neg=%.3f",
                     k, v["n"], a["full"], a["drop_pos"], a["drop_neg"])
    log.info("完成，用时 %.0fs → %s", res["seconds"], run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
