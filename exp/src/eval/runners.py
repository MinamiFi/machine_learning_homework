"""L1 评测：只算选择质量，不加载 LLM。

七组选择器共用同一个 select()，唯一变量是 scores 从哪来。
这是实验公平性的代码级保证。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..common import get_logger, stable_seed
from ..memory.embed import embed_queries, embed_segments
from ..memory.retrieve import cosine_sim, retrieve
from ..memory.select import SelectResult, order_by_time, select
from ..memory.types import LongScenario, MemorySegment
from .metrics import aggregate, context_recall, ndcg_at_k, recall_at_budget

log = get_logger(__name__)

# selector: (scenario, bank_vecs) -> scores (N,)
Selector = Callable[[LongScenario, np.ndarray], np.ndarray]


@dataclass
class L1Result:
    """一组的 L1 结果。"""

    name: str
    cr: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    ndcg: list[float] = field(default_factory=list)
    n_chosen: list[float] = field(default_factory=list)
    fill_rate: list[float] = field(default_factory=list)
    tokens: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_scenarios": len(self.cr),
            "CR": round(aggregate(self.cr)["mean"], 4),
            "Recall@budget": round(aggregate(self.recall)["mean"], 4),
            "NDCG@k": round(aggregate(self.ndcg)["mean"], 4),
            "n_chosen": round(aggregate(self.n_chosen)["mean"], 2),
            "fill_rate": round(aggregate(self.fill_rate)["mean"], 4),
            "tokens": round(aggregate(self.tokens)["mean"], 1),
            "seconds": round(self.seconds, 2),
        }


# ---------------------------------------------------------------- 选择器工厂

def make_selector(
    name: str,
    model=None,
    top_m: int = 50,
    seed: int = 0,
) -> Selector:
    """按名字造选择器。

    B0  全量：不做选择，全部片段进上下文（基线 1，可能超预算）
    B1  纯向量检索：cosine(query, seg)
    B2  检索 + 打分网络：scorer(seg_emb)
    R0  位置基线（首因）：-rel_pos
    R0a 绝对轮次：-turn_idx
    R1  recency：+rel_pos
    R2  随机
    R3  前 3 轮规则
    LEN 长度基线
    """
    name = name.upper()

    def b1(sc: LongScenario, vecs: np.ndarray) -> np.ndarray:
        q = embed_queries([sc.query])[0]
        return cosine_sim(q, vecs)

    def _score_all(vecs: np.ndarray) -> np.ndarray:
        import torch

        if model is None:
            raise ValueError("需要传入已训练的 model")
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(vecs.astype(np.float32))).numpy().astype(np.float64)

    def b2(sc: LongScenario, vecs: np.ndarray) -> np.ndarray:
        """B2：打分网络对**全库**稠密打分，不做检索过滤。

        这样 B2 测的是打分器本身的能力，与检索质量解耦。
        两阶段版本见 B2R。
        """
        return _score_all(vecs)

    def b2r(sc: LongScenario, vecs: np.ndarray) -> np.ndarray:
        """B2R：两阶段（检索 top-M → 打分器重排），即题目设计的完整系统。

        池外给 -inf（严格两阶段）。不用 cosine 兜底：
        混两种分数量纲后，"打分网络"组会部分退化成 B1，
        消融就测不出打分器的独立贡献了。

        注意：若 needle 不在 top-M 池内，B2R 必然漏掉它。
        这正是"检索召回是打分器上限"的体现，报告里要单独讨论。
        """
        q = embed_queries([sc.query])[0]
        pool = retrieve(q, vecs, top_m=min(top_m, len(vecs)))
        scores = np.full(len(vecs), -np.inf, dtype=np.float64)
        scores[pool] = _score_all(vecs[pool])
        return scores

    def rel_pos(sc: LongScenario) -> np.ndarray:
        n = len(sc.bank)
        return np.arange(n, dtype=np.float64) / max(n - 1, 1)

    table: dict[str, Selector] = {
        # B0 无记忆模块：不排序，按时间序（最近优先）填满预算。
        # 在硬预算下这等价于 R1（recency 截断）——这本身是个结论：
        # 「不做记忆管理」的最佳策略就是「留最近的」。
        "B0": lambda sc, v: np.arange(len(v), dtype=np.float64),
        "B1": b1,
        "B2": b2,
        "B2R": b2r,
        "R0": lambda sc, v: -rel_pos(sc),
        "R0A": lambda sc, v: -np.array([s.turn_idx for s in sc.bank], dtype=np.float64),
        "R1": lambda sc, v: rel_pos(sc),
        "R2": lambda sc, v: np.random.RandomState(seed + stable_seed(sc.scenario_id, mod=10000)).rand(len(v)),
        "R3": lambda sc, v: np.array(
            [1.0 if s.turn_idx < 3 else 0.0 for s in sc.bank], dtype=np.float64),
        # 长度基线：检验"越长越重要"这个可疑假设。
        # 若它很强，说明打分器可能只学到了长度，报告里必须交代。
        "LEN": lambda sc, v: np.array([s.n_tokens for s in sc.bank], dtype=np.float64),
    }
    if name not in table:
        raise ValueError(f"未知选择器 {name}，可选 {sorted(table)}")
    return table[name]


# ---------------------------------------------------------------- L1 主循环

def precompute_vectors(scenarios: list[LongScenario]) -> dict[str, np.ndarray]:
    """预先编码所有场景的 bank。

    必须预计算：7 个选择器共用同一批向量，否则同一段文本要被编码 7 次。
    L=512 × 90 场景约 4.6 万段，编码一次约 4 分钟，编码 7 次就是半小时。
    """
    cache: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, sc in enumerate(scenarios, 1):
        cache[sc.scenario_id] = embed_segments(sc.bank)
        if i % 20 == 0:
            log.info("预编码 %d/%d 场景… %.0fs", i, len(scenarios), time.time() - t0)
    log.info("预编码完成 %d 场景，%.1fs", len(scenarios), time.time() - t0)
    return cache


def run_l1(
    scenarios: list[LongScenario],
    selector: Selector,
    name: str,
    budget: int = 2048,
    reserve_recent: int = 2,
    mmr_lambda: float = 0.0,
    ndcg_k: int = 20,
    vec_cache: dict[str, np.ndarray] | None = None,
) -> L1Result:
    """在一个选择器上跑完所有场景的 L1 指标。"""
    t0 = time.time()
    res = L1Result(name=name)

    for sc in scenarios:
        vecs = (vec_cache or {}).get(sc.scenario_id)
        if vecs is None:
            vecs = embed_segments(sc.bank)
        scores = selector(sc, vecs)
        tokens = np.array([s.n_tokens for s in sc.bank], dtype=np.int64)

        sel: SelectResult = select(
            scores, tokens, budget,
            reserve_recent=reserve_recent,
            mmr_lambda=mmr_lambda,
            bank_vecs=vecs if mmr_lambda > 0 else None,
        )
        chosen_ids = [sc.bank[i].seg_id for i in sel.chosen]
        # 相关性只算"对本 query 相关"的主对话轮次（构造场景时已标好），
        # 不能用全库 label==1——干扰轮在它自己的对话里 label=1，
        # 但对当前 query 无关，算进去会让 Recall 完全失去意义。
        relevant = sc.meta.get("relevant_ids") or [s.seg_id for s in sc.bank if s.label == 1]

        res.cr.append(context_recall(chosen_ids, sc.needle_id))
        res.recall.append(recall_at_budget(chosen_ids, relevant))
        rel_set = set(sc.meta.get("relevant_ids") or [])
        rel_labels = [1.0 if s.seg_id in rel_set else 0.0 for s in sc.bank]
        res.ndcg.append(ndcg_at_k(scores, rel_labels, ndcg_k))
        res.n_chosen.append(sel.n_chosen)
        res.fill_rate.append(sel.fill_rate)
        res.tokens.append(sel.total_tokens)

    res.seconds = time.time() - t0
    log.info("%-4s | CR %.3f | Recall %.3f | NDCG %.3f | 选中 %.1f 段 | 填充 %.1f%% | %.1fs",
             name, aggregate(res.cr)["mean"], aggregate(res.recall)["mean"],
             aggregate(res.ndcg)["mean"], aggregate(res.n_chosen)["mean"],
             100 * aggregate(res.fill_rate)["mean"], res.seconds)
    return res


def run_baseline_l1(
    scenarios: list[LongScenario],
    names: tuple[str, ...] = ("B0", "B1", "B2", "B2R", "R0", "R0A", "R1", "R2", "R3", "LEN"),
    model=None,
    budget: int = 2048,
    vec_cache: dict[str, np.ndarray] | None = None,
    **kw,
) -> dict[str, L1Result]:
    """一次跑完所有组。返回 {name: L1Result}。"""
    if vec_cache is None:
        vec_cache = precompute_vectors(scenarios)

    out: dict[str, L1Result] = {}
    for n in names:
        try:
            sel = make_selector(n, model=model, **kw)
        except Exception as e:                     # noqa: BLE001
            log.warning("跳过 %s：%s", n, e)
            continue
        out[n] = run_l1(scenarios, sel, n, budget=budget, vec_cache=vec_cache)
    return out
