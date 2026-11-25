#!/usr/bin/env python3
"""对照实验：把标签从「引入」换成「必要」，L1 的 CR 会更高吗？

背景（`exp/docs/反事实删除校验.md` §4）：
现行标签 label = 1 表示「本轮引入了对话结束时仍有效的新信息」。反事实
校验证明这个方向是对的（删正例丢失率是删负例的 11.2 倍），但也发现它
**偏保守**：逐轮看 47.9% 的正例不是任何槽位的唯一支撑；集合覆盖下
51.4% 的轮次其实可以删。

于是有个直接的假设：**如果按「必要」重新定义标签，打分器应该更会挑**。
这个脚本就检验这个假设，而不是把它当成理所当然的推论。

两种标签：
  intro  （现行）label = 1  iff 本轮 delta 非空且至少一个值在最终状态里仍有效
  nec    （新）  label = 1  iff 本轮是某个最终状态槽位值的**唯一**文本支撑

关键：**两者的 gold 完全相同**（都用对话最后一个非空 state），
区别只在「怎么算重要」。这样比较才是干净的。

用法：
    source exp/env.sh
    python exp/scripts/train_necessity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import dump_json, get_logger, path_of, set_seed      # noqa: E402
from src.memory.embed import embed_split                              # noqa: E402
from src.memory.slice import group_by_dialogue, load_segments         # noqa: E402
from src.scorer.dataset import ScorerDataset, split_by_dialogue       # noqa: E402
from src.scorer.train import train                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_counterfactual import measurable_slots, mentions, seg_text  # noqa: E402

log = get_logger("train_necessity")


def final_state(turns) -> dict[str, str]:
    """对话最后一个非空 state —— 与原标签的 `final` 口径一致。"""
    for s in reversed(turns):
        if s.state:
            return dict(s.state)
    return {}


def necessity_labels(turns, gold: dict[str, str]) -> np.ndarray:
    """每轮的必要性：它是多少个 gold 槽位值的**唯一**文本支撑。"""
    support = measurable_slots(turns, gold)
    nec = np.zeros(len(turns), dtype=np.int32)
    for _slot, sup in support.items():
        if len(sup) == 1:
            nec[sup[0]] += 1
    return nec


def build_labels(segs, name: str) -> np.ndarray:
    """按 dialogue 逐段算标签，返回与 segs 等长的数组。

    name = "intro" → 用数据里现成的 label（现行定义）
    name = "nec"   → necessity > 0
    """
    by_d = group_by_dialogue(segs)
    out = np.zeros(len(segs), dtype=np.float32)
    pos_of = {s.seg_id: i for i, s in enumerate(segs)}

    n_slots = n_nec_pos = 0
    for d in sorted(by_d):
        turns = by_d[d]
        gold = final_state(turns)
        if name == "intro":
            for s in turns:
                out[pos_of[s.seg_id]] = float(s.label)
            continue

        nec = necessity_labels(turns, gold)
        n_slots += len(measurable_slots(turns, gold))
        for i, s in enumerate(turns):
            out[pos_of[s.seg_id]] = float(nec[i] > 0)
            n_nec_pos += int(nec[i] > 0)

    if name == "nec":
        log.info("necessity 标签：正例 %d / %d = %.3f（可测槽位 %d 个）",
                 n_nec_pos, len(segs), n_nec_pos / max(len(segs), 1), n_slots)
    return out


def make_ds(segs, emb, labels, keep_unlabeled: bool) -> ScorerDataset:
    """用自定义标签构造数据集（复用 build_dataset 的过滤与字段）。"""
    from src.scorer.dataset import build_dataset

    ds = build_dataset(segs, emb, drop_unlabeled=not keep_unlabeled)
    # build_dataset 已按 dialogue_has_state 过滤，这里把标签对齐到它的 seg_ids
    lab_of = {s.seg_id: labels[i] for i, s in enumerate(segs)}
    ds.label = np.array([lab_of[sid] for sid in ds.seg_ids], dtype=np.float32)
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scorer.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 epochs")
    ap.add_argument("--tag", default="necessity")
    args = ap.parse_args()

    from src.common import load_yaml

    set_seed(0)
    t0 = time.time()
    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    segs = load_segments("train")
    emb = embed_split("train", segs)
    log.info("train %d 段，编码 %s", len(segs), emb.shape)

    results: dict[str, dict] = {}
    models: dict[str, object] = {}

    for name in ("intro", "nec"):
        log.info("=== 标签口径 %s ===", name)
        labels = build_labels(segs, name)
        ds = make_ds(segs, emb, labels, keep_unlabeled=False)
        tr, va = split_by_dialogue(ds, 0.1, cfg["train"].get("seed", 0))
        log.info("%s：train %d / val %d，正例率 %.3f",
                 name, len(tr), len(va), tr.pos_rate)

        model, hist = train(cfg, tr, va, verbose=False)
        results[name] = {
            "pos_rate": round(tr.pos_rate, 4),
            "n_params": hist.n_params,
            "best_val_auc": round(hist.best_val_auc, 4),
            "position_auc": round(hist.position_auc, 4),
            "margin": round(hist.margin_over_position, 4),
            "best_epoch": hist.best_epoch,
            "seconds": round(hist.train_seconds, 2),
        }
        models[name] = model
        log.info("%s：val_auc %.4f（位置基线 %.4f，超出 %+.4f）| %.1fs",
                 name, hist.best_val_auc, hist.position_auc,
                 hist.margin_over_position, hist.train_seconds)

    # 两个模型在**同一套** val 上比（用 intro 标签作评价标准），
    # 否则"nec 在自己标签上 AUC 更高"是循环论证。
    from src.scorer.train import evaluate_auc

    labels_intro = build_labels(segs, "intro")
    ds_intro = make_ds(segs, emb, labels_intro, keep_unlabeled=False)
    _tr, va_intro = split_by_dialogue(ds_intro, 0.1, cfg["train"].get("seed", 0))
    common = {
        name: round(evaluate_auc(models[name], va_intro), 4)
        for name in models
    }
    log.info("统一评价（val，intro 标签）：%s", common)

    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json({"config": cfg, "by_label": results,
               "val_auc_on_intro_labels": common}, run_dir / "necessity.json")
    log.info("完成，用时 %.1fs → %s", time.time() - t0, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
