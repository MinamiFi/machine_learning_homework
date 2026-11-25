"""向量检索：cosine top-M。

因为 embed 阶段已 L2 归一化，内积就等于 cosine，不用再除模长。
top_m=50 是"重排候选池"的默认值：足够覆盖真正相关的片段，
又小到让后续打分/去冗余的代价可忽略。
"""

from __future__ import annotations

import numpy as np


def cosine_sim(query_vec: np.ndarray, bank_vecs: np.ndarray) -> np.ndarray:
    """(D,) × (N,D) → (N,)。输入已归一化时等价于内积。"""
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    b = np.asarray(bank_vecs, dtype=np.float32)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    qn = np.linalg.norm(q)
    bn = np.linalg.norm(b, axis=1)
    bn[bn == 0] = 1e-8
    return (b @ q) / (bn * (qn if qn > 0 else 1e-8))


def retrieve(
    query_vec: np.ndarray,
    bank_vecs: np.ndarray,
    top_m: int = 50,
) -> np.ndarray:
    """返回 top-M 的下标，按相似度降序。M 大于库大小时返回全部。"""
    sims = cosine_sim(query_vec, bank_vecs)
    n = len(sims)
    m = min(top_m, n)
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    # argpartition 是 O(N)，全排序是 O(N log N)；N 可能上万
    idx = np.argpartition(-sims, m - 1)[:m]
    return idx[np.argsort(-sims[idx])].astype(np.int64)


def retrieve_batch(
    query_vecs: np.ndarray,
    bank_vecs: np.ndarray,
    top_m: int = 50,
) -> list[np.ndarray]:
    """批量版本。逐个调用即可，N 小时不值得上矩阵分块。"""
    return [retrieve(q, bank_vecs, top_m) for q in np.atleast_2d(query_vecs)]
