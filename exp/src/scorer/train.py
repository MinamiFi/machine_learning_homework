"""训练循环。

三个设计要点，都是为了"不自欺"：

1. **每轮同时报位置基线 AUC**。如果 `val_auc - position_auc < 0.03`，
   说明模型没读到内容，只是复现了"越靠前越重要"。训练脚本直接打警告，
   而不是等到报告里才发现。

2. **早停看 val AUC，不看 loss**。下游只关心排序，不关心概率校准。

3. **日志逐 step 落 JSONL**。交付物要求"训练日志"，且消融要能复盘曲线。

实测：65k 样本 × 5 epoch ≈ 6 秒（2 核 CPU），所以训练本身不是瓶颈，
可以把消融网格跑满。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..common import ensure_dir, get_logger, path_of, set_seed
from .dataset import ScorerDataset, build_loaders, position_baseline_auc, resolve_pos_weight
from .model import ImportanceScorer, build_model, count_params

log = get_logger(__name__)


@dataclass
class TrainHistory:
    """训练过程记录。可直接 json.dump 成交付物里的训练日志。"""

    steps: list[dict] = field(default_factory=list)
    epochs: list[dict] = field(default_factory=list)
    best_val_auc: float = 0.0
    best_epoch: int = -1
    position_auc: float = 0.0
    n_params: int = 0
    train_seconds: float = 0.0

    @property
    def margin_over_position(self) -> float:
        """比位置基线高多少。这是全项目最关键的一个数。"""
        return self.best_val_auc - self.position_auc

    def to_dict(self) -> dict:
        return {
            "best_val_auc": self.best_val_auc,
            "best_epoch": self.best_epoch,
            "position_auc": self.position_auc,
            "margin_over_position": self.margin_over_position,
            "n_params": self.n_params,
            "train_seconds": self.train_seconds,
            "epochs": self.epochs,
            "n_steps": len(self.steps),
        }

    def save(self, p: Path) -> None:
        ensure_dir(p.parent)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- 损失

class BCERankLoss(nn.Module):
    """BCE + pairwise margin ranking。

    下游只关心 top-k 的排序。加一个 batch 内的 pairwise 项可以让梯度
    直接优化相对顺序，而不只是逐点概率。作为消融开关（cfg.loss）。
    """

    def __init__(self, pos_weight: float = 1.0, margin: float = 0.5, rank_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.margin = margin
        self.rank_weight = rank_weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.bce(logits, y)
        pos = logits[y == 1]
        neg = logits[y == 0]
        if len(pos) and len(neg):
            # 随机配对，避免 O(n²)
            k = min(len(pos), len(neg))
            pi = torch.randperm(len(pos))[:k]
            ni = torch.randperm(len(neg))[:k]
            rank = torch.clamp(self.margin - (pos[pi] - neg[ni]), min=0).mean()
            loss = loss + self.rank_weight * rank
        return loss


# ---------------------------------------------------------------- 评估

@torch.no_grad()
def evaluate_auc(
    model: nn.Module,
    ds: ScorerDataset,
    batch_size: int = 1024,
) -> float:
    """在数据集上算 AUC。"""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(ds.label)) < 2:
        return float("nan")

    model.eval()
    scores = []
    for i in range(0, len(ds), batch_size):
        x = torch.from_numpy(ds.emb[i:i + batch_size])
        scores.append(model(x).numpy())
    s = np.concatenate(scores)
    return float(roc_auc_score(ds.label, s))


# ---------------------------------------------------------------- 主循环

def train(
    cfg: dict,
    train_ds: ScorerDataset,
    val_ds: ScorerDataset,
    log_path: Path | None = None,
    verbose: bool = True,
) -> tuple[ImportanceScorer, TrainHistory]:
    """训练打分网络。

    cfg 是 configs/scorer.yaml 的内容（或消融后的副本）。
    """
    tcfg = cfg.get("train", cfg)
    seed = int(tcfg.get("seed", 0))
    set_seed(seed)

    bs = int(tcfg.get("batch_size", 256))
    epochs = int(tcfg.get("epochs", 20))
    patience = int(tcfg.get("patience", 3))
    lr = float(tcfg.get("lr", 3e-4))
    wd = float(tcfg.get("weight_decay", 1e-4))
    loss_kind = str(tcfg.get("loss", "bce")).lower()

    pw = resolve_pos_weight(tcfg.get("pos_weight", 1.0), train_ds.label)

    model = build_model(cfg)
    hist = TrainHistory(n_params=count_params(model))

    # 位置基线：在 val 上算，与 val_auc 同口径比较
    base = position_baseline_auc(val_ds)
    hist.position_auc = max(
        base.get("R0_rel", float("nan")),
        base.get("R0_abs", float("nan")),
    )

    if verbose:
        log.info("模型参数 %s | 训练 %d / 验证 %d | pos_rate %.3f | pos_weight %.3f",
                 f"{hist.n_params:,}", len(train_ds), len(val_ds),
                 train_ds.pos_rate, pw)
        log.info("位置基线 AUC（val）：%s", {k: round(v, 4) for k, v in base.items()})

    if loss_kind == "bce+rank":
        crit: nn.Module = BCERankLoss(pos_weight=pw)
    else:
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader, val_loader = build_loaders(train_ds, val_ds, bs, seed)

    best_state = None
    bad_epochs = 0
    t0 = time.time()
    global_step = 0

    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            ep_loss += float(loss.item())
            nb += 1
            global_step += 1
            if global_step % 50 == 0:
                hist.steps.append({
                    "step": global_step, "epoch": ep,
                    "loss": float(loss.item()), "lr": opt.param_groups[0]["lr"],
                })

        val_auc = evaluate_auc(model, val_ds)
        train_loss = ep_loss / max(nb, 1)
        rec = {
            "epoch": ep,
            "train_loss": round(train_loss, 6),
            "val_auc": round(val_auc, 6),
            "position_auc": round(hist.position_auc, 6),
            "margin": round(val_auc - hist.position_auc, 6),
        }
        hist.epochs.append(rec)

        if verbose:
            flag = ""
            if val_auc - hist.position_auc < 0.03:
                flag = "  ⚠ 未显著超过位置基线"
            log.info("ep %2d | loss %.4f | val_auc %.4f | 位置基线 %.4f | 超出 %+.4f%s",
                     ep, train_loss, val_auc, hist.position_auc,
                     val_auc - hist.position_auc, flag)

        if val_auc > hist.best_val_auc:
            hist.best_val_auc = val_auc
            hist.best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    log.info("早停：val_auc 连续 %d 轮未提升", patience)
                break

    hist.train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        log.info("训练完成 %.1fs | best val_auc %.4f @ep%d | 位置基线 %.4f | 超出 %+.4f",
                 hist.train_seconds, hist.best_val_auc, hist.best_epoch,
                 hist.position_auc, hist.margin_over_position)
        if hist.margin_over_position < 0.03:
            log.warning("⚠ 打分网络未显著超过位置基线（margin=%.4f < 0.03）。"
                        "按根 README §5.6 预案，应回头检查标签而非继续写下游代码。",
                        hist.margin_over_position)

    if log_path is not None:
        hist.save(log_path)
        with open(str(log_path).replace(".json", ".steps.jsonl"), "w", encoding="utf-8") as f:
            for r in hist.steps:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return model, hist
