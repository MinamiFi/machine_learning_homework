#!/usr/bin/env python3
"""预算扫描：B 从 256 到 2048，看压缩率对选择质量的影响。

动机：L=128 的 bank 只有约 4,700 token，而默认 B=2048，
相当于保留了 44% 的历史——"压缩"实际上没发生，各方法的 CR 会被稀释。
只有把 B 压到真正的低压缩率，"记忆过滤"才有意义。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import dump_json, get_logger, path_of, set_seed
from src.eval.report import to_table
from src.eval.runners import make_selector, precompute_vectors, run_l1
from src.memory.long_history import build_scenarios
from src.memory.slice import load_segments
from src.scripts_helpers import load_scorer

log = get_logger("budget_sweep")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--n-per-length", type=int, default=20)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--names", nargs="+",
                    default=["B1", "B2", "B2R", "R0", "R0A", "R1", "R2", "LEN"])
    ap.add_argument("--tag", default="budget")
    args = ap.parse_args()

    set_seed(0)
    t0 = time.time()

    segs = load_segments(args.split)
    scenarios = build_scenarios(segs, lengths=tuple(args.lengths),
                               n_per_length=args.n_per_length, seeds=tuple(args.seeds))
    log.info("%d 个场景", len(scenarios))

    model, hist = load_scorer()
    vec_cache = precompute_vectors(scenarios)

    out: dict = {
        "split": args.split, "budgets": args.budgets, "lengths": args.lengths,
        "n_per_length": args.n_per_length, "seeds": args.seeds,
        "scorer": {"val_auc": round(hist.best_val_auc, 4),
                   "position_auc": round(hist.position_auc, 4)},
        "grid": {},
    }

    for L in args.lengths:
        sub = [s for s in scenarios if f"_L{L}_" in s.scenario_id]
        if not sub:
            continue
        bank_tok = float(np.mean([sum(s.n_tokens for s in sc.bank) for sc in sub]))
        log.info("=== L=%d  bank≈%.0f token ===", L, bank_tok)
        for B in args.budgets:
            row = {}
            for name in args.names:
                sel = make_selector(name, model=model)
                r = run_l1(sub, sel, name, budget=B, vec_cache=vec_cache)
                row[name] = r.summary()
            comp = B / bank_tok
            out["grid"][f"L{L}_B{B}"] = {
                "L": L, "B": B, "bank_tokens": round(bank_tok),
                "compression_ratio": round(comp, 4), "results": row,
            }
            print(f"\n### L={L}  B={B}  (保留 {100*comp:.1f}% 历史)")
            print(to_table([{**v, "name": k} for k, v in row.items()],
                           ["name", "CR", "Recall@budget", "NDCG@k", "n_chosen"]))

    out["seconds"] = round(time.time() - t0, 1)
    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    dump_json(out, run_dir / "budget_sweep.json")
    print(f"\n产物：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
