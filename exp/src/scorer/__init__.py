"""scorer — 浅层全连接打分网络 + 基线打分器 + 训练。

题目要求的核心模块：输入记忆片段的向量表征，输出重要性分数。
"""

from .model import ImportanceScorer, build_model, count_params
from .dataset import ScorerDataset, build_dataset, build_loaders
from .baselines import score_position, score_recency, score_random, BASELINES
from .train import train, evaluate_auc, TrainHistory

__all__ = [
    "ImportanceScorer",
    "build_model",
    "count_params",
    "ScorerDataset",
    "build_dataset",
    "build_loaders",
    "score_position",
    "score_recency",
    "score_random",
    "BASELINES",
    "train",
    "evaluate_auc",
    "TrainHistory",
]
