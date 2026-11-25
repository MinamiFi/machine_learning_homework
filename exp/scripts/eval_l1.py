#!/usr/bin/env python3
"""L1 评测：选择质量，不需要 LLM。

秒级跑完 7 组选择器 × 3 个长度，回答"打分网络在真实的选择任务上
是否优于零成本基线"。

用法：
    source exp/env.sh
    python exp/scripts/eval_l1.py
    python exp/scripts/eval_l1.py --lengths 128 256 512 --n-per-length 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import dump_json, get_logger, path_of, set_seed
from src.eval.report import to_markdown, to_table
from src.eval.runners import run_baseline_l1
from src.memory.embed import embed_split
from src.memory.long_history import build_scenarios, scenario_token_stats
from src.memory.slice import load_segments, group_by_dialogue
from src.scorer.dataset import build_dataset
from src.scorer.model import build_model
from src.scorer.train import train

log = get_logger("eval_l1")


def load_or_train_scorer(cfg_path: str = "configs/scorer.yaml"):
    """没有已保存的模型就现场训一个（13 秒，不值得做缓存）。"""
    from src.common import load_yaml

    cfg = load_yaml(cfg_path)
    tr_segs = load_segments("train")
    tr_emb = embed_split("train", tr_segs)
    tr_full = build_dataset(tr_segs, tr_emb)
    from src.scorer.dataset import split_by_dialogue

    tr_ds, va_ds = split_by_dialogue(tr_full, 0.1, cfg["train"].get("seed", 0))
    model, hist = train(cfg, tr_ds, va_ds, verbose=False)
    log.info("打分网络已训练：val_auc %.4f（位置基线 %.4f，超出 %+.4f）",
             hist.best_val_auc, hist.position_auc, hist.margin_over_position)
    return model, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--n-per-length", type=int, default=30)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--tag", default="l1")
    args = ap.parse_args()

    set_seed(0)
    t0 = time.time()

    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    log.info("%s: %d 段 / %d 对话", args.split, len(segs), len(by_d))

    scenarios = build_scenarios(
        segs,
        lengths=tuple(args.lengths),
        n_per_length=args.n_per_length,
        seeds=tuple(args.seeds),
    )
    log.info("场景 token 统计：%s", scenario_token_stats(scenarios))

    model, hist = load_or_train_scorer()

    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_eval_l1_{args.tag}")
    out: dict = {
        "split": args.split,
        "lengths": args.lengths,
        "n_scenarios_per_length": args.n_per_length * len(args.seeds),
        "budget": args.budget,
        "scorer": {
            "val_auc": round(hist.best_val_auc, 4),
            "position_auc": round(hist.position_auc, 4),
            "margin": round(hist.margin_over_position, 4),
        },
        "scenario_tokens": {str(k): v for k, v in scenario_token_stats(scenarios).items()},
        "by_length": {},
    }

    # 按长度分组跑：不同 L 的可检索难度不同，混在一起看会掩盖趋势
    for L in args.lengths:
        # scenario_id 形如 MUL0001.json_L256_p0.42_s0，用 _L{L}_ 精确筛
        sub = [s for s in scenarios if f"_L{L}_" in s.scenario_id]
        if not sub:
            continue
        log.info("--- L=%d，%d 个场景 ---", L, len(sub))
        results = run_baseline_l1(
            sub, model=model, budget=args.budget, top_m=50, seed=0,
        )
        out["by_length"][str(L)] = {k: v.summary() for k, v in results.items()}

    # 汇总：跨长度平均
    agg: dict[str, dict] = {}
    for L, d in out["by_length"].items():
        for name, r in d.items():
            a = agg.setdefault(name, {"CR": [], "Recall@budget": [], "NDCG@k": [], "fill_rate": []})
            for k in a:
                a[k].append(r[k])
    out["overall"] = {
        name: {
            "CR": round(float(np.mean(v["CR"])), 4),
            "Recall@budget": round(float(np.mean(v["Recall@budget"])), 4),
            "NDCG@k": round(float(np.mean(v["NDCG@k"])), 4),
            "fill_rate": round(float(np.mean(v["fill_rate"])), 4),
        }
        for name, v in agg.items()
    }

    out["total_seconds"] = round(time.time() - t0, 1)
    dump_json(out, run_dir / "L1_selection.json")

    # 打印
    print("\n" + "=" * 78)
    for L, d in out["by_length"].items():
        print(f"\n## L={L}")
        print(to_table([{**v, "name": k} for k, v in d.items()],
                       ["name", "CR", "Recall@budget", "NDCG@k", "n_chosen", "fill_rate"]))
    print("\n## 跨长度平均")
    print(to_table([{**v, "name": k} for k, v in out["overall"].items()],
                   ["name", "CR", "Recall@budget", "NDCG@k", "fill_rate"]))
    print("=" * 78)
    print(f"\n产物：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
