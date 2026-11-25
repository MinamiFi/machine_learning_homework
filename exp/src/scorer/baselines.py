"""基线打分器 —— 零成本的对照组。

这些不是"凑数的对照"，而是判断打分网络到底有没有学到内容的标尺。

实测（test split）：
    -rel_pos  (对话内相对位置)  AUC 0.8172
    -turn_idx (绝对轮次)        AUC 0.7636
    规则"前 3 轮"               AUC 0.6585

含义：一个完全不懂内容、只会说"越靠前越重要"的规则就能拿到 0.82。
所以打分网络必须显著超过 0.80 才算真的读到了内容。
这个数字写进 train() 的每轮日志，防止自欺。
"""

from __future__ import annotations

import numpy as np

from ..memory.types import MemorySegment


def _rel_pos(segs: list[MemorySegment]) -> np.ndarray:
    """对话内相对位置 ∈ [0, 1]。需要按 dialogue_id 分组归一化。"""
    from collections import defaultdict

    by_d: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(segs):
        by_d[s.dialogue_id].append(i)

    out = np.zeros(len(segs), dtype=np.float64)
    for idxs in by_d.values():
        n = len(idxs)
        for rank, i in enumerate(sorted(idxs, key=lambda j: segs[j].turn_idx)):
            out[i] = rank / (n - 1) if n > 1 else 0.0
    return out


def score_position(segs: list[MemorySegment], mode: str = "rel") -> np.ndarray:
    """R0 位置基线（首因：越靠前分越高）。

    mode="rel"  相对位置，实测 AUC 0.8172（最强零成本基线）
    mode="abs"  绝对轮次，实测 AUC 0.7636
    """
    if mode == "rel":
        pos = _rel_pos(segs)
    elif mode == "abs":
        pos = np.array([s.turn_idx for s in segs], dtype=np.float64)
    else:
        raise ValueError(f"未知 mode: {mode}")
    return -pos


def score_recency(segs: list[MemorySegment], mode: str = "rel") -> np.ndarray:
    """R1 recency 基线（近因：越靠后分越高）。与 R0 方向相反。"""
    return -score_position(segs, mode)


def score_random(segs: list[MemorySegment], seed: int = 0) -> np.ndarray:
    """R2 随机基线。下界参考。"""
    rng = np.random.RandomState(seed)
    return rng.rand(len(segs))


def score_first_k(segs: list[MemorySegment], k: int = 3) -> np.ndarray:
    """规则基线：前 k 轮重要。README 里提到的 0.6585。"""
    return np.array([1.0 if s.turn_idx < k else 0.0 for s in segs], dtype=np.float64)


def score_length(segs: list[MemorySegment]) -> np.ndarray:
    """长度基线：越长的轮次越重要。

    加这条是为了排除"打分器其实只学会了长度"这种可能。
    属于必须报告的可疑基线之一。
    """
    return np.array([s.n_tokens for s in segs], dtype=np.float64)


BASELINES = {
    "R0": lambda segs, **kw: score_position(segs, kw.get("mode", "rel")),
    "R0_abs": lambda segs, **kw: score_position(segs, "abs"),
    "R1": lambda segs, **kw: score_recency(segs, kw.get("mode", "rel")),
    "R2": lambda segs, **kw: score_random(segs, kw.get("seed", 0)),
    "R3": lambda segs, **kw: score_first_k(segs, kw.get("k", 3)),
    "LEN": lambda segs, **kw: score_length(segs),
}


def baseline_table(segs: list[MemorySegment]) -> dict[str, float]:
    """把所有零成本基线在给定片段上跑一遍 AUC。

    这是 train() 每轮都要打印的表——它回答"模型有没有超过启发式"。
    """
    from sklearn.metrics import roc_auc_score

    y = np.array([s.label for s in segs], dtype=np.int64)
    if len(np.unique(y)) < 2:
        return {}

    out: dict[str, float] = {}
    for name, fn in BASELINES.items():
        s = fn(segs)
        try:
            out[name] = float(roc_auc_score(y, s))
        except ValueError:
            out[name] = float("nan")
    return out
