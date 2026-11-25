"""脚本共用的加载/训练助手。放这里避免每个脚本重复实现。"""

from __future__ import annotations

from src.common import get_logger, load_yaml
from src.memory.embed import embed_split
from src.memory.slice import load_segments
from src.scorer.dataset import build_dataset, split_by_dialogue
from src.scorer.train import train

log = get_logger(__name__)


def load_scorer(cfg_path: str = "configs/scorer.yaml"):
    """训练打分网络（13 秒，不值得做持久化缓存）。"""
    cfg = load_yaml(cfg_path)
    tr = load_segments("train")
    full = build_dataset(tr, embed_split("train", tr))
    tr_ds, va_ds = split_by_dialogue(full, 0.1, cfg["train"].get("seed", 0))
    model, hist = train(cfg, tr_ds, va_ds, verbose=False)
    log.info("打分网络：val_auc %.4f | 位置基线 %.4f | 超出 %+.4f",
             hist.best_val_auc, hist.position_auc, hist.margin_over_position)
    return model, hist
