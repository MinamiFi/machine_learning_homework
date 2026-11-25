#!/usr/bin/env python3
"""消融实验：网络层数 / 学习率 / L2 正则强度（题目必做要求 4）。

单因素扫描，其余固定。训练只要 13 秒，整个网格几分钟跑完。

用法：
    source exp/env.sh
    python exp/scripts/run_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import dump_json, get_logger, load_yaml, path_of, set_seed
from src.memory.embed import embed_split
from src.memory.slice import load_segments
from src.scorer.dataset import build_dataset, position_baseline_auc, split_by_dialogue
from src.scorer.train import evaluate_auc, train

log = get_logger("ablation")

# 因素名 → (配置段, 字段名)。写错段会静默失效（踩过：
# num_layers 写进 train 段，build_model 读 model 段，结果 4 个取值
# 参数全一样、AUC 全一样，看起来像"层数无影响"，其实是没生效）。
GRIDS = {
    "num_layers": ("model", [1, 2, 3, 4]),
    "lr": ("train", [1e-4, 3e-4, 1e-3, 3e-3]),
    "weight_decay": ("train", [0.0, 1e-5, 1e-4, 1e-3]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="ablation")
    args = ap.parse_args()

    base = load_yaml("configs/scorer.yaml")
    tr = load_segments("train")
    tr_emb = embed_split("train", tr)
    full = build_dataset(tr, tr_emb)

    te = load_segments("test")
    te_ds = build_dataset(te, embed_split("test", te))

    pos_auc = max(position_baseline_auc(te_ds).get("R0_rel", np.nan),
                  position_baseline_auc(te_ds).get("R0_abs", np.nan))
    log.info("test 位置基线 AUC = %.4f", pos_auc)

    rows = []
    prev_params: dict[str, int] = {}
    t0 = time.time()

    for factor, (section, values) in GRIDS.items():
        for v in values:
            aucs = []
            n_params = 0
            for seed in args.seeds:
                cfg = json.loads(json.dumps(base))          # deep copy
                cfg["train"]["seed"] = seed
                cfg[section][factor] = v                    # 写对段！

                tr_ds, va_ds = split_by_dialogue(full, 0.1, seed)
                model, hist = train(cfg, tr_ds, va_ds, verbose=False)
                aucs.append(evaluate_auc(model, te_ds))
                n_params = hist.n_params

            # 自检：不同取值必须产生不同参数量或不同 AUC，否则配置没生效
            if factor == "num_layers" and n_params == prev_params.get(factor):
                raise RuntimeError(
                    f"num_layers={v} 参数量与上一个取值相同（{n_params}），配置未生效")

            row = {
                "factor": factor,
                "value": v,
                "test_auc_mean": round(float(np.mean(aucs)), 4),
                "test_auc_std": round(float(np.std(aucs)), 4),
                "n_params": n_params,
                "margin_over_position": round(float(np.mean(aucs)) - pos_auc, 4),
                "aucs": [round(a, 4) for a in aucs],
            }
            prev_params[factor] = n_params
            rows.append(row)
            log.info("%-13s = %-8s | test AUC %.4f ± %.4f | 参数 %s",
                     factor, v, row["test_auc_mean"], row["test_auc_std"], f"{n_params:,}")

    out = {
        "seeds": args.seeds,
        "position_auc": round(pos_auc, 4),
        "n_train": len(full),
        "n_test": len(te_ds),
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }
    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    dump_json(out, run_dir / "ablation.json")

    # markdown 表
    print("\n## 消融实验结果（test AUC，%d seeds）\n" % len(args.seeds))
    print("| 因素 | 取值 | test AUC | 标准差 | 参数 | 超出位置基线 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['factor']} | {r['value']} | {r['test_auc_mean']:.4f} | "
              f"{r['test_auc_std']:.4f} | {r['n_params']:,} | {r['margin_over_position']:+.4f} |")
    print(f"\n位置基线 AUC = {pos_auc:.4f}")
    print(f"产物：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
