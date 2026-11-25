#!/usr/bin/env python3
"""L2 评测：任务完成率 / AR / token 消耗（需要 LLM）。

与 L1 的分工（`src/README.md` §4）：
    L1 只算「needle 有没有被选进上下文」，秒级、零 LLM 成本，回答方法成不成立。
    L2 真正让 LLM 读上下文答题，小时级，把 CR 换算成下游效果。

## 为什么用「填空答对率」而不是 JGA

题目要的是任务完成率（JGA）。但 JGA 要求模型输出完整的 slot/value 状态，
本机 0.5B 做不到（实测输出自造 schema，见 `docs/L2_JGA校验.md` §2）。
所以这里用**逐槽位填空答对率**作为代理指标，并明确标注：

    * 槽位准确率（slot accuracy）：抽到的每个槽位单独判对错
    * 场景成功率（scenario success）：该场景抽到的所有槽位全对 → 1
    * AR（answerable rate）：只在「给全上下文就答对」的子集上重算，
      排除 0.5B 本来就答不出的槽位

「场景成功率」是最接近 JGA 的口径，但它只覆盖被抽到的 ≤3 个槽位，
不是全部状态，所以**不能直接当 JGA 报**。报告里必须写清这一点。

## 三种 token 口径

    context_tokens  选中的记忆 + query 拼成的上下文长度（选择器的直接代价）
    prompt_tokens   完整 prompt（含填空模板与指令）
    total_tokens    输入 + 输出（真实花钱的部分）

B0（预算内最近优先）与 B0FULL（不设预算的全量上下文）必须分开：
前者是「不做记忆管理 + 硬预算」，在预算下等价于 R1；后者是题目字面意义的
「原始全量上下文记忆」。混在一起会把「截断的代价」和「不截断的代价」搞混。

**AR 的基准是 NEEDLE，不是 B0FULL。** AR 要回答的是「这个槽位从证据里能不能
抽出来」，不需要几百轮无关背景；而 B0FULL 在 L=512 时 bank 约 18,655 token，
prefill 173 tok/s → 单次 112 秒，60 次就 67 分钟，成本与收益完全不成比例。
NEEDLE 只在 needle 那一轮上问同一个槽位——它是该槽位信息的唯一来源，
在它上面答对就等于「可答」。B0FULL 仍可用 `--with-b0full` 打开（仅建议小 L）。

用法：
    source exp/env.sh
    python3 exp/scripts/eval_l2.py --n-per-length 10 --max-slots 2 --tag l2_smoke
    python3 exp/scripts/eval_l2.py --n-per-length 30 --max-slots 3 --tag l2_main
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.cache import ResponseCache                        # noqa: E402
from src.agent.llm import DummyLLM, build_llm                    # noqa: E402
from src.agent.loop import MemoryAgent                           # noqa: E402
from src.agent.prompt import askable_slots                       # noqa: E402
from src.common import (dump_json, get_logger, load_yaml,        # noqa: E402
                        path_of, set_seed)
from src.eval.report import to_table                             # noqa: E402
from src.memory.embed import embed_segments                      # noqa: E402
from src.memory.long_history import (build_scenarios,            # noqa: E402
                                     scenario_token_stats)
from src.memory.slice import group_by_dialogue, load_segments    # noqa: E402
from src.memory.types import LongScenario                        # noqa: E402

log = get_logger("eval_l2")

# 默认跑的选择器。B0FULL 需 --with-b0full 显式打开（走全量上下文，不走预算）。
DEFAULT_SELECTORS = ("B0", "B1", "B2", "R0", "R1", "R2")


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """均值的 bootstrap 95% CI。场景数少，必须给区间。"""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return {
        "mean": round(float(v.mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
    }


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 5000,
                     seed: int = 0) -> dict:
    """配对差值 a-b 的 bootstrap。按场景配对（同一场景同一槽位）。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) != len(y) or len(x) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan")}
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # 双侧 p：bootstrap 分布中越过 0 的比例 ×2
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "diff": round(float(d.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "p": round(float(min(p, 1.0)), 4),
    }


def answerable_keys(all_rows: list[dict],
                    baseline: str = "B0FULL") -> set[tuple[str, str]]:
    """「给全上下文就答对」的 (scenario_id, slot) 集合，AR 的分母。

    两个必须注意的点：

    1. 必须在**全部行**上算，不能在按 selector 过滤后的子集上算——
       子集里没有基准行，AR 会恒为 NaN（第一版就是这样踩的，冒烟里
       12 行全 NaN 才暴露）。
    2. `baseline` 必须与调用处 `--ar-baseline` 一致。这里曾经**硬编码
       `selector == "NEEDLE"`**，而 CLI 默认已改成 b0full，于是传进来的
       全量行里一条 NEEDLE 都不匹配 → 空集 → AR 依然全 NaN。硬编码一个
       默认值、再用另一个默认值调用，是最难查的那类 bug：两次都"看起来对"。
    """
    return {
        (r["scenario_id"], r["slot"])
        for r in all_rows
        if r.get("selector") == baseline and r["correct"]
    }


def aggregate_rows(rows: list[dict],
                   answerable: set[tuple[str, str]] | None = None,
                   baseline: str = "B0FULL") -> dict:
    """把逐槽位的行聚合成一组指标。

    rows 每行：{scenario_id, selector, slot, gold, pred, correct,
                context_tokens, prompt_tokens, n_new_tokens, cr, mode}
    answerable : 由 `answerable_keys(all_rows)` 算出的可答槽位集合。
        None 时退化为「在本组行里找 NEEDLE」——只在 rows 含 NEEDLE 时正确。
    """
    if not rows:
        return {}

    acc = [float(r["correct"]) for r in rows]
    # 场景成功率：同一 (scenario_id) 下所有抽到的槽位全对
    by_sc: dict[str, list[bool]] = {}
    for r in rows:
        by_sc.setdefault(r["scenario_id"], []).append(bool(r["correct"]))
    succ = [1.0 if all(v) else 0.0 for v in by_sc.values()]

    if answerable is None:
        answerable = answerable_keys(rows, baseline=baseline)
    ans_rows = [r for r in rows
                if (r["scenario_id"], r["slot"]) in answerable]
    ans_acc = [float(r["correct"]) for r in ans_rows]

    return {
        "n_slots": len(rows),
        "n_scenarios": len(by_sc),
        "slot_acc": bootstrap_ci(acc),
        "scenario_success": bootstrap_ci(succ),
        "n_answerable_slots": len(ans_rows),
        "ar": bootstrap_ci(ans_acc) if ans_rows else {
            "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
        "context_tokens": round(float(np.mean([r["context_tokens"] for r in rows])), 1),
        "prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rows])), 1),
        "total_tokens": round(float(np.mean([r["total_tokens"] for r in rows])), 1),
        "cr": round(float(np.mean([r["cr"] for r in rows])), 4),
        # 用 .get：手工构造的行（单测、复用旧 rows）不一定带 seconds，
        # 缺了不该让整个汇总崩掉。
        "seconds": round(float(np.mean([r.get("seconds", 0.0) for r in rows])), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--n-per-length", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max-slots", type=int, default=2,
                    help="每个场景最多问几个槽位")
    ap.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--tag", default="l2")
    ap.add_argument("--resume", default=None, help="复用该 run 目录的 cache.jsonl")
    ap.add_argument("--dummy", action="store_true", help="用 DummyLLM 打通管线，不加载模型")
    ap.add_argument("--ar-baseline", choices=["oracle", "b0full", "needle"],
                    default="oracle",
                    help="AR（可答子集）的基准。b0full=给全量上下文能否答对（方法论上正确），"
                         "needle=只给 needle 那一轮（便宜但会低估可答性）。"
                         "本地 0.5B 时被迫用 needle（B0FULL 单次 112 秒），"
                         "API 上 19k token 只要 2 秒，应该用 b0full")
    ap.add_argument("--workers", type=int, default=0,
                    help="并发数。0=自动（api 用 3，local 用 1）。该端点套餐上限 3 并发")
    args = ap.parse_args()

    set_seed(0)
    t0 = time.time()

    cfg = load_yaml("configs/agent.yaml")
    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    log.info("%s: %d 段 / %d 对话", args.split, len(segs), len(by_d))

    scenarios = build_scenarios(
        segs,
        lengths=tuple(args.lengths),
        n_per_length=args.n_per_length,
        seeds=tuple(args.seeds),
    )
    # 只留有可问槽位的场景，否则会白跑
    scenarios = [s for s in scenarios if askable_slots(s.gold_state or {})]
    log.info("可用场景 %d 个，token 统计 %s", len(scenarios),
             scenario_token_stats(scenarios))

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_eval_l2_{args.tag}"
    run_dir = path_of("runs", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = path_of("runs", args.resume) if args.resume else run_dir
    cache = ResponseCache(cache_dir / "cache.jsonl")

    if args.dummy:
        llm = DummyLLM(reply="unknown")
        log.info("使用 DummyLLM（只验证管线，结果无意义）")
    else:
        llm = build_llm(cfg, max_new_tokens=args.max_new_tokens)
        log.info("backbone 就绪：%s", llm.name)

    # B2 需要打分网络。训练 11 秒，直接现场训。
    # DummyLLM 模式下不训练：管线验证不需要，省 12 秒。
    model = None
    if not args.dummy and any(s.upper().startswith("B2") for s in args.selectors):
        from src.scripts_helpers import load_scorer

        model, hist = load_scorer()
        log.info("打分网络：val_auc %.4f（位置基线 %.4f）",
                 hist.best_val_auc, hist.position_auc)

    agent = MemoryAgent(
        llm=llm,
        scorer_model=model,
        budget=args.budget,
        reserve_recent=cfg["budget"].get("reserve_recent_turns", 2),
        cache=cache,
        max_new_tokens=args.max_new_tokens,
    )

    # 没有打分网络时剔除 B2/B2R，否则会在第一个场景就抛错
    selectors = [s for s in args.selectors
                 if model is not None or not s.upper().startswith("B2")]
    if len(selectors) != len(args.selectors):
        log.warning("无打分网络，跳过 %s",
                    sorted(set(args.selectors) - set(selectors)))

    # ---- 主循环
    all_rows: list[dict] = []
    for si, sc in enumerate(scenarios, 1):
        vecs = embed_segments(sc.bank)     # 每场景编码一次，所有选择器复用
        # 所有选择器问**同一批槽位**。这不是省事，是 AR 口径的前提：
        # AR = 只在「给全上下文就答对」的槽位子集上重算准确率，
        # 若各选择器问的槽位不同，子集就落不到同一批 key 上，
        # AR 会恒为 NaN（第一版就是这样踩的）。
        ask = sorted(askable_slots(sc.gold_state or {}))[: args.max_slots]

        # AR 基准：只在 needle 那一轮上问。
        # 原方案用 B0FULL（全量上下文）当基准，实测代价不可接受：
        # L=512 时 bank 约 18,655 token，prefill 173 tok/s → 单次 112 秒，
        # 60 次就 67 分钟，且它回答的问题也不对——AR 要测的是
        # 「这个槽位从证据里能不能抽出来」，不需要 512 轮无关背景。
        # needle 轮是信息的**唯一来源**，在它上面答对就等于「可答」。
        # **必须问与其它选择器完全相同的那批槽位**（`ask` 全部，不是 `ask[0]`）。
        # 第一版只问 ask[0]，而 B0/B1/B2 问 ask 的全部槽位，导致：
        #   ① AR 的 answerable_keys 只覆盖 1 个槽位，其余槽位匹配不到基准，
        #      AR 恒为 NaN（表里 5 个选择器的 ar 列全是 nan，就是这个 bug）
        #   ② NEEDLE 的 slot_acc 也失真——它被强制问 ask[0]，而 ask[0] 未必
        #      是 needle 那一轮里提到的槽位，答错不是模型的问题
        # AR 基准 2：ORACLE —— 只给**本对话的真实历史**（去掉人为干扰轮）。
        #
        # 为什么不用 B0FULL：B0FULL 把 3 个其他对话的干扰轮也放进去，
        # 模型会被带偏。实测 18 行里 10 行答错，错的样子很说明问题——
        # `hotel-area` gold='centre' 答 'north'、`hotel-stars` gold='0' 答 '4'，
        # 都是被别的对话的值盖掉了。拿它当 AR 分母，会把「选择本该赢」的
        # 槽位判成「不可答」，AR 因此虚低（0.625 vs B1 的 0.875）。
        #
        # 干扰轮是**人为加进去的**，不是原始数据的一部分；AR 要问的是
        # 「这个槽位在本对话历史里能不能答出来」，所以基准必须排除干扰轮。
        if args.ar_baseline in ("oracle", "b0full"):
            oracle_segs = [x for x in sc.bank if x.dialogue_id == sc.dialogue_id]
            if oracle_segs and ask:
                ov = embed_segments(oracle_segs)
                for slot in ask:
                    t = agent.run_turn(
                        LongScenario(
                            scenario_id=sc.scenario_id + "_ORACLE",
                            dialogue_id=sc.dialogue_id, bank=oracle_segs,
                            query=sc.query, gold_state=sc.gold_state,
                            needle_id=sc.needle_id, meta={}, target_len=1),
                        "B0", full_context=True, slot=slot, vecs=ov)
                    if t.meta.get("skipped"):
                        continue
                    all_rows.append({
                        "scenario_id": t.scenario_id.replace("_ORACLE", ""),
                        "length": sc.target_len,
                        "selector": "ORACLE",
                        "slot": t.meta.get("slot"),
                        "gold": t.meta.get("gold"),
                        "pred": t.output,
                        "correct": bool(t.meta.get("correct")),
                        "context_tokens": t.context_tokens,
                        "prompt_tokens": t.prompt_tokens,
                        "n_new_tokens": t.n_new_tokens,
                        "total_tokens": t.total_tokens,
                        "cr": t.cr,
                        "seconds": t.seconds,
                        "cache_hit": t.cache_hit,
                    })

        needle_segs = [x for x in sc.bank if x.seg_id == sc.needle_id]
        if needle_segs and ask:
            needle_vecs = embed_segments(needle_segs)
            for slot in ask:
                t = agent.run_turn(
                    LongScenario(
                        scenario_id=sc.scenario_id + "_NEEDLE",
                        dialogue_id=sc.dialogue_id, bank=needle_segs,
                        query=sc.query, gold_state=sc.gold_state,
                        needle_id=sc.needle_id, meta={}, target_len=1),
                    "B0", full_context=True, slot=slot, vecs=needle_vecs)
                if t.meta.get("skipped"):
                    continue
                all_rows.append({
                    "scenario_id": t.scenario_id.replace("_NEEDLE", ""),
                    "length": sc.target_len,
                    "selector": "NEEDLE",
                    "slot": t.meta.get("slot"),
                    "gold": t.meta.get("gold"),
                    "pred": t.output,
                    "correct": bool(t.meta.get("correct")),
                    "context_tokens": t.context_tokens,
                    "prompt_tokens": t.prompt_tokens,
                    "n_new_tokens": t.n_new_tokens,
                    "total_tokens": t.total_tokens,
                    "cr": t.cr,
                    "seconds": t.seconds,
                    "cache_hit": t.cache_hit,
                })

        # B0FULL：题目字面义的「原始全量上下文」。
        # 这既是**必做要求里的基线①**，也是 AR 口径的正确基准——
        # 「给全上下文就答对」才叫可答，只给 needle 一轮会低估可答性
        # （一个槽位可能由多轮共同确定，needle 单轮未必含答案）。
        # 本地 0.5B 时这一步要 112 秒/次（prefill 173 tok/s），所以当初
        # 退化成了 NEEDLE 基准；接上 API 后 19k token 只要 2 秒，代价不再是问题。
        if ask:
            for slot in ask:
                t = agent.run_turn(sc, "B0", full_context=True, slot=slot,
                                   vecs=vecs)
                if t.meta.get("skipped"):
                    continue
                all_rows.append({
                    "scenario_id": t.scenario_id,
                    "length": sc.target_len,
                    "selector": "B0FULL",
                    "slot": t.meta.get("slot"),
                    "gold": t.meta.get("gold"),
                    "pred": t.output,
                    "correct": bool(t.meta.get("correct")),
                    "context_tokens": t.context_tokens,
                    "prompt_tokens": t.prompt_tokens,
                    "n_new_tokens": t.n_new_tokens,
                    "total_tokens": t.total_tokens,
                    "cr": t.cr,
                    "seconds": t.seconds,
                    "cache_hit": t.cache_hit,
                })

        for sel_name in selectors:
            turn_list = [
                agent.run_turn(sc, sel_name, full_context=False,
                               slot=slot, vecs=vecs)
                for slot in ask
            ]
            for t in turn_list:
                if t.meta.get("skipped"):
                    continue
                all_rows.append({
                    "scenario_id": t.scenario_id,
                    "length": sc.target_len,
                    "selector": t.selector,
                    "slot": t.meta.get("slot"),
                    "gold": t.meta.get("gold"),
                    "pred": t.output,
                    "correct": bool(t.meta.get("correct")),
                    "context_tokens": t.context_tokens,
                    "prompt_tokens": t.prompt_tokens,
                    "n_new_tokens": t.n_new_tokens,
                    "total_tokens": t.total_tokens,
                    "cr": t.cr,
                    "seconds": t.seconds,
                    "cache_hit": t.cache_hit,
                })
        if si % 5 == 0:
            log.info("进度 %d/%d 场景，%d 行，缓存命中 %d / 未命中 %d，已用 %.0fs",
                     si, len(scenarios), len(all_rows), cache.hits, cache.misses,
                     time.time() - t0)

    # ---- 汇总
    out: dict = {
        "split": args.split,
        "lengths": args.lengths,
        "n_per_length": args.n_per_length,
        "max_slots": args.max_slots,
        "budget": args.budget,
        "backbone": llm.name,
        "n_scenarios": len(scenarios),
        "ar_baseline": args.ar_baseline,
        "scenario_tokens": {str(k): v for k, v in scenario_token_stats(scenarios).items()},
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "overall": {},
        "by_length": {},
    }

    # AR 基准：默认用 B0FULL（给全上下文能否答对）。
    # 退回 needle 时会低估可答性——needle 单轮未必包含槽位的答案。
    base_sel = {"oracle": "ORACLE", "b0full": "B0FULL",
                "needle": "NEEDLE"}[args.ar_baseline]
    ans = answerable_keys(all_rows, baseline=base_sel)
    out["n_answerable_slots"] = len(ans)
    report_sels = tuple(selectors) + ("ORACLE", "B0FULL", "NEEDLE")
    for sel_name in report_sels:
        rows = [r for r in all_rows if r["selector"] == sel_name]
        if rows:
            out["overall"][sel_name] = aggregate_rows(rows, ans, baseline=base_sel)

    for L in args.lengths:
        sub = [r for r in all_rows if r["length"] == L]
        if not sub:
            continue
        sub_ans = answerable_keys(sub, baseline=base_sel)
        out["by_length"][str(L)] = {
            sel: aggregate_rows([r for r in sub if r["selector"] == sel], sub_ans,
                                baseline=base_sel)
            for sel in report_sels
            if any(r["selector"] == sel for r in sub)
        }

    # ---- 配对比较：B2 对每个对照
    if "B2" in out["overall"]:
        def _vec(sel: str, key: str = "correct") -> list[float]:
            # 按 (scenario_id, slot) 对齐，保证配对
            d = {(r["scenario_id"], r["slot"]): float(r[key])
                 for r in all_rows if r["selector"] == sel}
            return d

        b2 = _vec("B2")
        out["paired"] = {}
        for other in ("B1", "B0", "R0", "R1", "R2"):
            o = _vec(other)
            keys = sorted(set(b2) & set(o))
            if not keys:
                continue
            out["paired"][f"B2-{other}"] = paired_bootstrap(
                [b2[k] for k in keys], [o[k] for k in keys])
            out["paired"][f"B2-{other}"]["n_pairs"] = len(keys)

    out["total_seconds"] = round(time.time() - t0, 1)
    dump_json(out, run_dir / "L2_generation.json")
    with (run_dir / "L2_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 打印
    print("\n" + "=" * 88)
    print(f"L2 评测 | backbone={llm.name} | 场景 {len(scenarios)} | 预算 {args.budget}")
    print("=" * 88)
    cols = ["selector", "n_slots", "slot_acc", "scenario_success", "ar",
            "context_tokens", "total_tokens", "cr"]
    table_rows = []
    for sel, m in out["overall"].items():
        table_rows.append({
            "selector": sel,
            "n_slots": m["n_slots"],
            "slot_acc": m["slot_acc"]["mean"],
            "scenario_success": m["scenario_success"]["mean"],
            "ar": m["ar"]["mean"],
            "context_tokens": m["context_tokens"],
            "total_tokens": m["total_tokens"],
            "cr": m["cr"],
        })
    print(to_table(table_rows, cols))

    if out.get("paired"):
        print("\n## 配对比较（B2 减对照，按场景×槽位配对）")
        prow = [{"cmp": k, **v} for k, v in out["paired"].items()]
        print(to_table(prow, ["cmp", "diff", "ci_low", "ci_high", "p", "n_pairs"]))

    print("=" * 88)
    print(f"\n产物：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
