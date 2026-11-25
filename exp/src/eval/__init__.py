"""eval — 指标与实验调度。

核心设计：两层评测。

    L1 选择质量（不需要 LLM，秒级）
       AUC / CR(context recall) / NDCG@k / Recall@budget
       可跑 6.5 万轮全量

    L2 任务完成（需要 LLM，17 秒/样本）
       任务完成率 / AR / token 消耗
       只在 300 条子集上跑

先 L1 全量扫，若打分网络赢不过位置基线就停，不浪费 8 小时跑 L2。
这能把"发现方法不行"的时间从 8 小时压到 1 分钟。
"""

from .metrics import (
    auc,
    context_recall,
    ndcg_at_k,
    recall_at_budget,
    token_cost,
    task_success,
    slot_f1,
)
from .runners import run_l1, run_baseline_l1, L1Result
from .report import to_markdown, to_table

__all__ = [
    "auc",
    "context_recall",
    "ndcg_at_k",
    "recall_at_budget",
    "token_cost",
    "task_success",
    "slot_f1",
    "run_l1",
    "run_baseline_l1",
    "L1Result",
    "to_markdown",
    "to_table",
]
