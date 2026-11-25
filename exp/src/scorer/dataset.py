"""训练数据构造。

两个必须处理的问题：

1. **过滤 dialogue_has_state=False 的轮次**。
   这类轮次 label 全 0，但这是"标注缺失"不是"不重要"。
   混进训练集等于教模型"没有标注的对话全不重要"，会污染决策边界。
   train 里有 520 段这样的对话（2,987 轮）。

2. **必须按 dialogue_id 划分 train/val，不能按轮随机划分**。
   同一段对话的轮次高度相关（话题、实体、位置都连着），
   按轮随机划分会造成信息泄漏，val AUC 虚高。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..common import get_logger
from ..memory.types import MemorySegment

log = get_logger(__name__)


@dataclass
class ScorerDataset(Dataset):
    """(embedding, label) 对。"""

    emb: np.ndarray            # (N, 384)
    label: np.ndarray          # (N,)
    seg_ids: list[str]
    dialogue_ids: list[str]
    turn_idx: np.ndarray       # (N,) 绝对轮次，R0/R1 基线要用
    n_tokens: np.ndarray       # (N,) 长度基线要用

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.emb[i]), torch.tensor(self.label[i], dtype=torch.float32)

    @property
    def pos_rate(self) -> float:
        return float(self.label.mean()) if len(self.label) else 0.0


def build_dataset(
    segs: list[MemorySegment],
    emb: np.ndarray,
    drop_unlabeled: bool = True,
) -> ScorerDataset:
    """从片段 + 向量构造数据集。

    segs 与 emb 必须一一对应（embed_split 保证了这一点）。
    """
    if len(segs) != len(emb):
        raise ValueError(f"segs({len(segs)}) 与 emb({len(emb)}) 行数不符")

    keep = np.ones(len(segs), dtype=bool)
    if drop_unlabeled:
        keep = np.array([s.dialogue_has_state for s in segs], dtype=bool)
        n_drop = int((~keep).sum())
        if n_drop:
            log.info("剔除无状态标注的轮次 %d 条（占 %.1f%%）",
                     n_drop, 100 * n_drop / len(segs))

    idx = np.where(keep)[0]
    return ScorerDataset(
        emb=emb[idx].astype(np.float32),
        label=np.array([segs[i].label for i in idx], dtype=np.float32),
        seg_ids=[segs[i].seg_id for i in idx],
        dialogue_ids=[segs[i].dialogue_id for i in idx],
        turn_idx=np.array([segs[i].turn_idx for i in idx], dtype=np.int32),
        n_tokens=np.array([segs[i].n_tokens for i in idx], dtype=np.int32),
    )


def split_by_dialogue(
    ds: ScorerDataset,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[ScorerDataset, ScorerDataset]:
    """按 dialogue_id 划分，同一对话的轮次只出现在一侧。"""
    rng = np.random.RandomState(seed)
    dids = np.array(sorted(set(ds.dialogue_ids)))
    rng.shuffle(dids)
    n_val = max(1, int(len(dids) * val_ratio))
    val_dids = set(dids[:n_val].tolist())

    is_val = np.array([d in val_dids for d in ds.dialogue_ids])
    return _subset(ds, ~is_val), _subset(ds, is_val)


def _subset(ds: ScorerDataset, mask: np.ndarray) -> ScorerDataset:
    idx = np.where(mask)[0]
    return ScorerDataset(
        emb=ds.emb[idx],
        label=ds.label[idx],
        seg_ids=[ds.seg_ids[i] for i in idx],
        dialogue_ids=[ds.dialogue_ids[i] for i in idx],
        turn_idx=ds.turn_idx[idx],
        n_tokens=ds.n_tokens[idx],
    )


def build_loaders(
    train_ds: ScorerDataset,
    val_ds: ScorerDataset,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False),
    )


def resolve_pos_weight(cfg_value, label: np.ndarray) -> float:
    """把 config 里的 pos_weight 解析成数值。

    注意方向：BCEWithLogitsLoss 的 pos_weight 是"正例的权重"，
    数学上等于 负例数/正例数 才平衡。本项目正例率 55.6%，
    该比值约 0.80，即应当**下调**正例权重。
    如果误写成 正例数/负例数（1.25），会把模型进一步推向全判 1。

    "auto" → 按上述定义算；数值 → 直接用。
    """
    if cfg_value is None:
        return 1.0
    if isinstance(cfg_value, str) and cfg_value.lower() == "auto":
        pos = float((label == 1).sum())
        neg = float((label == 0).sum())
        if pos == 0:
            return 1.0
        w = neg / pos
        log.info("pos_weight=auto → %.4f（负 %d / 正 %d）", w, int(neg), int(pos))
        return w
    return float(cfg_value)


def position_baseline_auc(ds: ScorerDataset) -> dict[str, float]:
    """在数据集上算零成本基线 AUC。train() 每轮调用，用于监控退化。

    rel_pos 用对话内相对位置（按 dialogue_id 分组），实测最强，AUC 0.82。
    """
    from sklearn.metrics import roc_auc_score

    y = ds.label
    if len(np.unique(y)) < 2:
        return {}

    # 相对位置
    from collections import defaultdict

    by_d: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(ds.dialogue_ids):
        by_d[d].append(i)
    rel = np.zeros(len(y), dtype=np.float64)
    for idxs in by_d.values():
        n = len(idxs)
        for rank, i in enumerate(sorted(idxs, key=lambda j: ds.turn_idx[j])):
            rel[i] = rank / (n - 1) if n > 1 else 0.0

    out = {}
    for name, sc in {
        "R0_rel": -rel,
        "R0_abs": -ds.turn_idx.astype(np.float64),
        "R1_rel": rel,
        "LEN": ds.n_tokens.astype(np.float64),
    }.items():
        try:
            out[name] = float(roc_auc_score(y, sc))
        except ValueError:
            out[name] = float("nan")
    return out
