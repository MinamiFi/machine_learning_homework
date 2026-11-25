"""L1 指标：全部不需要 LLM。

CR (context recall) 是最关键的一个：它只问"埋进去的 needle 有没有被选进
上下文"，纯查 id 集合，成本为零。有了它，打分器的好坏可以在秒级判断，
不必等 8 小时的 LLM 评测。
"""

from __future__ import annotations

import numpy as np


def auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, y_score))


def context_recall(selected_ids: list[str], needle_id: str | None) -> float:
    """needle 是否被选进上下文。0 或 1。

    这是"关键信息不丢失"最直接的度量，也是题目验收指标之一。
    """
    if needle_id is None:
        return float("nan")
    return 1.0 if needle_id in set(selected_ids) else 0.0


def recall_at_budget(
    selected_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """预算内召回了多少比例的相关片段。

    比 CR 更细：CR 只看 needle 一个，这个看所有 label=1 的片段。
    """
    if not relevant_ids:
        return float("nan")
    rel = set(relevant_ids)
    return len(rel & set(selected_ids)) / len(rel)


def ndcg_at_k(scores, labels, k: int) -> float:
    """按分数排序取前 k，算 NDCG。二值相关性。"""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(s)
    if n == 0 or k <= 0:
        return float("nan")
    k = min(k, n)

    order = np.argsort(-s)[:k]
    gains = y[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())

    ideal = np.sort(y)[::-1][:k]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def token_cost(result: dict) -> dict:
    """一次评测的 token 开销。"""
    return {
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
        "total_tokens": int(result.get("prompt_tokens", 0)) + int(result.get("completion_tokens", 0)),
        "n_memories": int(result.get("n_memories", 0)),
    }


def task_success(pred_state: dict, gold_state: dict) -> float:
    """对话状态是否完全匹配（JGA, joint goal accuracy）。

    任务完成率的严格版本：所有 slot 值都对才算成功。
    """
    if not gold_state:
        return float("nan")
    return 1.0 if pred_state == gold_state else 0.0


def slot_f1(pred: dict, gold: dict) -> dict:
    """槽位级 P/R/F1。比 JGA 宽松，能看出"差一点"的程度。"""
    if not gold:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    tp = sum(1 for k, v in pred.items() if gold.get(k) == v)
    n_pred, n_gold = len(pred), len(gold)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def aggregate(values: list[float]) -> dict:
    """跳过 NaN 的均值/中位/计数。"""
    v = np.array([x for x in values if not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
    }
