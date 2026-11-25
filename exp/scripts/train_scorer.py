#!/usr/bin/env python3
"""第 2 步（关键决策点）：训练打分网络，回答"方法成不成立"。

这一步的输出决定后面要不要继续写代码：
    若 val_auc - position_auc < 0.03 → 打分网络没读到内容，只是复现了
    "越靠前越重要"的启发式。此时应回头检查标签（根 README §5.6 预案），
    而不是继续往下写 agent 和 L2 评测。

用法：
    source exp/env.sh
    python exp/scripts/train_scorer.py                    # 默认配置
    python exp/scripts/train_scorer.py --loss bce+rank
    python exp/scripts/train_scorer.py --num-layers 1 --lr 1e-3
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

from src.common import dump_json, get_logger, load_yaml, path_of
from src.memory.slice import load_segments
from src.memory.embed import embed_split
from src.scorer.dataset import build_dataset, split_by_dialogue, position_baseline_auc
from src.scorer.train import train, evaluate_auc
from src.scorer.baselines import baseline_table

log = get_logger("train_scorer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scorer.yaml")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="test", help="最终报这个 split 的 AUC")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--loss", default=None, choices=["bce", "bce+rank"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="base", help="实验名，用于 run 目录")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.num_layers is not None:
        cfg["model"]["num_layers"] = args.num_layers
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.loss is not None:
        cfg["train"]["loss"] = args.loss
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    t0 = time.time()

    # ---- 数据
    tr_segs = load_segments(args.train_split)
    tr_emb = embed_split(args.train_split, tr_segs)
    tr_full = build_dataset(tr_segs, tr_emb, drop_unlabeled=True)
    tr_ds, va_ds = split_by_dialogue(tr_full, val_ratio=0.1, seed=cfg["train"].get("seed", 0))

    log.info("训练集 %d 段 / 验证集 %d 段（按 dialogue_id 划分，无泄漏）",
             len(tr_ds), len(va_ds))

    # ---- 训练
    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_scorer_{args.tag}_s{cfg['train'].get('seed',0)}")
    model, hist = train(cfg, tr_ds, va_ds, log_path=run_dir / "train_history.json")

    # ---- 在独立的 test split 上评估
    te_segs = load_segments(args.eval_split)
    te_emb = embed_split(args.eval_split, te_segs)
    te_ds = build_dataset(te_segs, te_emb, drop_unlabeled=True)
    te_auc = evaluate_auc(model, te_ds)
    te_base = position_baseline_auc(te_ds)

    # ---- 零成本基线全表（用原始 segs，需要 turn_idx/n_tokens）
    te_segs_labeled = [s for s in te_segs if s.dialogue_has_state]
    bt = baseline_table(te_segs_labeled)

    result = {
        "tag": args.tag,
        "config": cfg,
        "n_params": hist.n_params,
        "train_seconds": round(hist.train_seconds, 2),
        "total_seconds": round(time.time() - t0, 1),
        "val_auc": round(hist.best_val_auc, 6),
        "val_position_auc": round(hist.position_auc, 6),
        "val_margin": round(hist.margin_over_position, 6),
        "test_auc": round(te_auc, 6),
        "test_baselines": {k: round(v, 6) for k, v in bt.items()},
        "test_position_auc": {k: round(v, 6) for k, v in te_base.items()},
        "test_margin_vs_best_baseline": round(
            te_auc - max(bt.values()) if bt else float("nan"), 6),
        "best_epoch": hist.best_epoch,
        "verdict": "",
    }

    margin = result["test_margin_vs_best_baseline"]
    if margin >= 0.03:
        result["verdict"] = "PASS：显著超过所有零成本基线，方法成立，可以继续写下游"
    elif margin > 0:
        result["verdict"] = f"MARGINAL：仅超出 {margin:+.4f}，不足 0.03，需检查是否靠位置信息"
    else:
        result["verdict"] = f"FAIL：未超过最强零成本基线（{margin:+.4f}），应回头改标签"

    dump_json(result, run_dir / "result.json")

    print("\n" + "=" * 72)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 72)
    print(f"\n{result['verdict']}\n")
    print(f"产物：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
