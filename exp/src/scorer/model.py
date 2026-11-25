"""打分网络：浅层全连接 MLP。

结构（对应根 README §3.4）：

    e ∈ R^384
      → LayerNorm(384)
      → [Linear → GELU → Dropout] × num_layers
      → Linear(·, 1)
      → (训练时输出 logit，推理时 sigmoid)

num_layers=2 时是 384→256→256→1，参数量 165,377
（含 LayerNorm 的 768）。这里踩过一个坑：如果写成
`dims = [input] + [hidden]*(num_layers-1) + [1]`，
num_layers=2 只会得到 1 个隐藏层（99,585 参数），
与题目"浅层"的消融设计不符。正确写法见 build_model。

为什么输出 logit 而不是概率：BCEWithLogitsLoss 内部做 log-sum-exp，
数值更稳；sigmoid 只在推理和评测时套。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImportanceScorer(nn.Module):
    """记忆片段重要性打分器。"""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation.lower()]

        self.norm = nn.LayerNorm(input_dim)

        # 关键：num_layers 个隐藏层，不是 num_layers-1 个
        dims = [input_dim] + [hidden_dim] * num_layers + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # 最后一层是输出，不加激活/丢弃
                layers.append(act())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 384) → (B,) logit。"""
        h = self.mlp(self.norm(x))
        return h.squeeze(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理：返回 sigmoid 后的概率 (B,)。"""
        self.eval()
        return torch.sigmoid(self.forward(x))


def build_model(cfg: dict) -> ImportanceScorer:
    """从 configs/scorer.yaml 的 model 段建模型。"""
    m = cfg.get("model", cfg)
    return ImportanceScorer(
        input_dim=m.get("input_dim", 384),
        hidden_dim=m.get("hidden_dim", 256),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("dropout", 0.1),
        activation=m.get("activation", "gelu"),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
