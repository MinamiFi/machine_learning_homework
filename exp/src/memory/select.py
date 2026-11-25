"""预算内选择 —— 全项目实验公平性的代码级保证。

B1 / B2 / R0 / R1 / R2 / R3 全部调用这一个函数，
唯一变量是 scores 从哪来：

    B1 纯向量检索     scores = cosine(query, seg)
    B2 检索+打分网络   scores = scorer(seg_emb)
    R0 位置基线        scores = -turn_idx          （首因：越早越重要）
    R1 recency         scores = +turn_idx          （近因：越晚越重要）
    R2 随机            scores = rand()
    R3 打乱时间顺序     scores = B2 的分数，但拼接顺序随机（不影响选择，影响 prompt）

所以"预算怎么截断""最近几轮保底"对所有组完全一致，
不会出现"某组因为实现细节多塞了几轮"这种不可比的情况。

token 数用真实 tokenizer 计数（segment.n_tokens），不按空格估。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import MemorySegment


@dataclass
class SelectResult:
    """选择结果。带明细，便于错误分析和报告。"""

    chosen: list[int]                      # 选中的下标，按分数降序（不是时间序）
    scores: np.ndarray                     # 全量分数，(N,)
    total_tokens: int = 0
    budget: int = 0
    dropped: int = 0                       # 因预算被丢掉的候选数
    reserved: list[int] = field(default_factory=list)   # 保底选中的下标

    @property
    def n_chosen(self) -> int:
        return len(self.chosen)

    @property
    def fill_rate(self) -> float:
        """预算填充率。远小于 1 说明候选池太小，不是选择器的问题。"""
        return self.total_tokens / self.budget if self.budget else 0.0


def select(
    scores: np.ndarray,
    tokens: np.ndarray,
    budget: int,
    reserve_recent: int = 2,
    mmr_lambda: float = 0.0,
    bank_vecs: np.ndarray | None = None,
    tie_break: str = "index",
) -> SelectResult:
    """按分数降序累加 token 直到 budget，返回选中下标。

    参数
    ----
    scores : (N,) 越高越重要。可以是任意来源，函数不关心。
    tokens : (N,) 每段的真实 token 数。
    budget : token 预算 B。
    reserve_recent : 保底保留时间上最近的 k 段（下标最大的 k 个）。
        即使它们分数很低也要留下——否则"最后一个用户问题"可能被丢掉。
        所有基线共用这个规则，故不破坏公平性。
    mmr_lambda : >0 时做最大边际相关去冗余，惩罚与已选片段过近的候选。
        需要 bank_vecs。默认 0（纯按分数）。
    tie_break : 分数相同时的次序。"index" = 靠后的优先（更近因）。

    返回
    ----
    SelectResult
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    n = len(scores)
    if n == 0:
        return SelectResult(chosen=[], scores=scores, total_tokens=0, budget=budget)

    if len(tokens) != n:
        raise ValueError(f"scores 与 tokens 长度不符：{n} vs {len(tokens)}")

    # ---- 1. 保底：时间上最近的 k 段
    k = max(0, min(reserve_recent, n))
    reserved = list(range(n - k, n)) if k else []
    used = int(tokens[reserved].sum()) if reserved else 0

    # 保底本身就可能超预算（极端情况：单段很长）。此时只保留最后一段。
    if used > budget and reserved:
        reserved = [n - 1]
        used = int(tokens[n - 1])

    # ---- 2. 其余候选按分数降序
    candidates = [i for i in range(n) if i not in set(reserved)]
    if tie_break == "index":
        # 分数降序，同分时下标大的（更近）优先
        order = sorted(candidates, key=lambda i: (-scores[i], -i))
    else:
        order = sorted(candidates, key=lambda i: (-scores[i], i))

    chosen_extra: list[int] = []
    chosen_vecs: list[np.ndarray] = []

    if mmr_lambda > 0 and bank_vecs is not None:
        # MMR：每步重算一次，N 小（≤ 预算能装下的量级）时可接受
        pool = list(order)
        while pool:
            best_i, best_val = None, -np.inf
            for i in pool:
                if used + tokens[i] > budget:
                    continue
                rel = scores[i]
                if chosen_vecs:
                    sim = max(float(bank_vecs[i] @ v) for v in chosen_vecs)
                else:
                    sim = 0.0
                val = (1 - mmr_lambda) * rel - mmr_lambda * sim
                if val > best_val:
                    best_val, best_i = val, i
            if best_i is None:
                break
            chosen_extra.append(best_i)
            chosen_vecs.append(bank_vecs[best_i])
            used += int(tokens[best_i])
            pool.remove(best_i)
    else:
        for i in order:
            if used + tokens[i] > budget:
                continue          # 跳过装不下的，继续试更小的（贪心装填）
            chosen_extra.append(i)
            used += int(tokens[i])

    # ---- 3. 合并。返回顺序：先按分数降序的主体，再接保底段
    chosen = chosen_extra + reserved
    return SelectResult(
        chosen=chosen,
        scores=scores,
        total_tokens=int(used),
        budget=int(budget),
        dropped=n - len(chosen),
        reserved=reserved,
    )


def order_by_time(chosen: list[int], turn_idx: np.ndarray | None = None) -> list[int]:
    """把选中下标重排成时间序，用于拼 prompt。

    R3（打乱时间顺序）不走这里，它显式打乱。
    时间序是默认：LLM 读按时间排列的记忆更自然。
    """
    if turn_idx is None:
        return sorted(chosen)
    t = np.asarray(turn_idx)
    return sorted(chosen, key=lambda i: t[i])
