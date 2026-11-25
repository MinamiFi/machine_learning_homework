"""L2 主循环：检索 → 打分 → 选择 → 拼上下文 → 生成 → 记录开销。

这个模块把 `memory/`（选择）、`scorer/`（打分）、`agent/`（LLM）串起来，
是 `eval/` 做 L2 评测时唯一需要调用的入口。

设计上的三个约束（都是被实测逼出来的）：

1. **选择逻辑必须复用 `memory.select.select()`**，不能在这里另写一套截断。
   B0/B1/B2/R0/R1/R2/R3 共用同一个预算逻辑，唯一变量是 scores 来源——
   这是实验公平性的代码级保证（见 `memory/select.py` 模块 docstring）。

2. **token 数必须用 backbone 自己的 tokenizer 精确计数**，不能按空格估。
   L2 的核心指标之一就是 token 消耗，估出来的数字没法写进报告。
   `LocalQwen.count_tokens()` 与 `ApiClient` 的实现可能不同，所以计数
   一律走 `llm.count_tokens()`。

3. **缓存不是优化而是必需**。一次全量 L2 是小时级，中途崩了没有缓存
   等于白跑。`ResponseCache` 的 key 含 model 名，换 backbone 不会误命中。

关于 B0（全量上下文）的一个区分
--------------------------------
`memory.select.select()` 总是受 budget 约束，所以 `make_selector("B0")`
得到的是「按时间序（最近优先）填满预算」，在硬预算下等价于 R1。
但题目里的「原始全量上下文记忆」指的是**不设预算、全部塞进去**。
两者不是一回事：前者测「截断的代价」，后者测「不截断的代价（token 爆炸）」。

因此这里提供 `full_context=True` 开关，走 `_select_all()`，
并在 `TurnResult.meta["mode"]` 里标明 `"full"` 还是 `"budget"`，
避免报告里把两种 B0 混为一谈。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..common import get_logger, stable_seed
from ..memory.embed import embed_queries, embed_segments
from ..memory.select import select
from ..memory.types import LongScenario, MemorySegment
from .cache import ResponseCache
from .prompt import askable_slots, answer_matches, build_prompt, render_context

log = get_logger(__name__)

__all__ = [
    "TurnResult",
    "MemoryAgent",
    "run_turn",
    "run_scenario",
    "select_all",
]


@dataclass
class TurnResult:
    """一次 turn 的完整记录。带明细是为了错误分析。"""

    scenario_id: str
    selector: str
    # 选择结果
    chosen_ids: list[str] = field(default_factory=list)
    n_chosen: int = 0
    n_bank: int = 0
    context_tokens: int = 0        # 选中的记忆 + query 拼成的上下文，真实 token 数
    fill_rate: float = 0.0
    # 生成结果
    output: str = ""
    prompt_tokens: int = 0         # 完整 prompt（含填空模板）的 token 数
    n_new_tokens: int = 0
    seconds: float = 0.0
    cache_hit: bool = False
    # 上下文召回（零成本，顺便算）
    cr: float = float("nan")
    # 元信息
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """一次 turn 的总 token 开销 = 输入 + 输出。"""
        return self.prompt_tokens + self.n_new_tokens

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["total_tokens"] = self.total_tokens
        return d


def select_all(bank: list[MemorySegment]) -> list[int]:
    """全量上下文：不排序、不截断，按时间序全选。

    这是题目里「原始全量上下文记忆」的字面含义。
    L=512 场景下会到 2 万 token 量级，成本是 B2 的十几倍——
    这个倍数本身就是 L2 要报的数字之一。
    """
    return list(range(len(bank)))


class MemoryAgent:
    """带记忆模块的对话智能体。

    只持有 LLM 与打分模型，不持有数据。这样同一个 agent 可以跑所有场景。

    参数
    ----
    llm : 实现 `LLMClient` 协议的对象（`LocalQwen` / `ApiClient` / `DummyLLM`）
    scorer_model : 已训练的 `ImportanceScorer`，B2 用。None 时 B2 不可用。
    budget : token 预算 B
    reserve_recent : 保底保留最近 k 段
    cache : `ResponseCache`，None 时不缓存（只用于调试）
    """

    def __init__(
        self,
        llm,
        scorer_model=None,
        budget: int = 2048,
        reserve_recent: int = 2,
        cache: ResponseCache | None = None,
        max_new_tokens: int = 24,
    ):
        self.llm = llm
        self.scorer_model = scorer_model
        self.budget = budget
        self.reserve_recent = reserve_recent
        self.cache = cache
        self.max_new_tokens = max_new_tokens

    # ---------------------------------------------------------------- 打分

    def _score_with_model(self, vecs: np.ndarray) -> np.ndarray:
        import torch

        if self.scorer_model is None:
            raise ValueError("B2 需要已训练的 scorer_model")
        self.scorer_model.eval()
        with torch.no_grad():
            return (
                self.scorer_model(torch.from_numpy(vecs.astype(np.float32)))
                .numpy()
                .astype(np.float64)
            )

    def score_bank(self, scenario: LongScenario, vecs: np.ndarray, selector: str) -> np.ndarray:
        """按选择器名给候选打分的唯一入口。

        与 `eval/runners.make_selector` 保持同一套语义，但这里不依赖 eval，
        避免 agent → eval 的反向依赖。
        """
        name = selector.upper()
        n = len(scenario.bank)

        if name in ("B2", "B2R"):
            scores = self._score_with_model(vecs)
            if name == "B2R":
                # 两阶段：检索 top-M 后重排，池外 -inf
                from ..memory.retrieve import cosine_sim, retrieve

                q = embed_queries([scenario.query])[0]
                pool = retrieve(q, vecs, top_m=min(50, n))
                masked = np.full(n, -np.inf, dtype=np.float64)
                masked[pool] = scores[pool]
                return masked
            return scores

        if name == "B1":
            from ..memory.retrieve import cosine_sim

            q = embed_queries([scenario.query])[0]
            return cosine_sim(q, vecs)

        if name == "B0":
            # 预算内的「不做记忆管理」：最近优先
            return np.arange(n, dtype=np.float64)

        rel = np.arange(n, dtype=np.float64) / max(n - 1, 1)
        if name == "R0":
            return -rel
        if name == "R0A":
            return -np.array([s.turn_idx for s in scenario.bank], dtype=np.float64)
        if name == "R1":
            return rel
        if name == "R2":
            return np.random.RandomState(
                stable_seed(scenario.scenario_id, mod=10000)
            ).rand(n)
        if name == "R3":
            return np.array(
                [1.0 if s.turn_idx < 3 else 0.0 for s in scenario.bank],
                dtype=np.float64,
            )
        if name == "LEN":
            return np.array([s.n_tokens for s in scenario.bank], dtype=np.float64)
        raise ValueError(f"未知选择器 {selector}")

    # ---------------------------------------------------------------- 一次 turn

    def run_turn(
        self,
        scenario: LongScenario,
        selector: str = "B2",
        full_context: bool = False,
        slot: str | None = None,
        vecs: np.ndarray | None = None,
    ) -> TurnResult:
        """跑一个 turn：选择 → 拼上下文 → 生成 → 记录。

        参数
        ----
        scenario : `LongScenario`
        selector : B0/B1/B2/B2R/R0/R0A/R1/R2/R3/LEN
        full_context : True 时忽略预算与选择器，全量按时间序塞进去（题目的 B0 字面义）
        slot : 要问的槽位。None 时从 `scenario.gold_state` 里自动挑一个可问的。
            挑不到可问槽位时直接返回空结果（该场景对填空口径不可测）。
        vecs : 预先编码好的 bank 向量，避免重复编码
        """
        t0 = time.time()
        bank = scenario.bank
        if vecs is None:
            vecs = embed_segments(bank)
        tokens = np.array([s.n_tokens for s in bank], dtype=np.int64)

        # ---- 1. 选择
        if full_context:
            chosen = select_all(bank)
            mode = "full"
            fill = 1.0
            scores = np.zeros(len(bank), dtype=np.float64)
        else:
            scores = self.score_bank(scenario, vecs, selector)
            sel = select(
                scores, tokens, self.budget, reserve_recent=self.reserve_recent
            )
            chosen = sel.chosen
            mode = "budget"
            fill = sel.fill_rate

        chosen_sorted = sorted(chosen)  # 按时间序拼进 prompt
        chosen_segs = [bank[i] for i in chosen_sorted]
        chosen_ids = [s.seg_id for s in chosen_segs]

        # ---- 2. 拼上下文与 prompt
        ctx = render_context(chosen_segs)
        ctx_tokens = self.llm.count_tokens(ctx)

        res = TurnResult(
            scenario_id=scenario.scenario_id,
            selector=selector if not full_context else "B0FULL",
            chosen_ids=chosen_ids,
            n_chosen=len(chosen_ids),
            n_bank=len(bank),
            context_tokens=ctx_tokens,
            fill_rate=fill,
            cr=(
                1.0 if scenario.needle_id and scenario.needle_id in set(chosen_ids) else 0.0
            ),
            meta={"mode": mode, "slot": slot},
        )

        # ---- 3. 挑槽位（没有可问的就到此为止，不算一次生成）
        ask = askable_slots(scenario.gold_state or {})
        if slot is None:
            if not ask:
                res.meta["skipped"] = "no_askable_slot"
                res.seconds = time.time() - t0
                return res
            slot = sorted(ask)[0]
        elif slot not in ask:
            res.meta["skipped"] = f"slot_not_askable:{slot}"
            res.seconds = time.time() - t0
            return res

        gold = ask[slot]
        res.meta["slot"] = slot
        res.meta["gold"] = gold

        prompt = build_prompt(ctx, slot)

        # ---- 4. 生成（先查缓存）
        mx = self.max_new_tokens
        cached = self.cache.get(prompt, mx) if self.cache else None
        if cached is not None:
            text = cached["text"]
            res.cache_hit = True
            res.output = text
            res.prompt_tokens = self.llm.count_tokens(prompt)
            res.n_new_tokens = cached.get("n_new_tokens") or self.llm.count_tokens(text)
            res.seconds = cached.get("seconds", 0.0)
        else:
            gen = self.llm.generate(prompt, max_new_tokens=mx)
            text = gen.text
            res.output = text
            res.prompt_tokens = gen.n_prompt_tokens or self.llm.count_tokens(prompt)
            res.n_new_tokens = gen.n_new_tokens or self.llm.count_tokens(text)
            res.seconds = gen.seconds
            if self.cache:
                self.cache.put(
                    prompt,
                    mx,
                    text,
                    seconds=gen.seconds,
                    n_new_tokens=res.n_new_tokens,
                    scenario_id=scenario.scenario_id,
                    selector=res.selector,
                    slot=slot,
                )

        # ---- 5. 判对
        res.meta["correct"] = bool(answer_matches(gold, text))
        res.seconds = max(res.seconds, 0.0)
        if not res.cache_hit:
            res.meta["wall_seconds"] = round(time.time() - t0, 3)
        return res


# ---------------------------------------------------------------- 便捷函数


def run_turn(agent: MemoryAgent, scenario: LongScenario, selector: str = "B2", **kw) -> TurnResult:
    """模块级入口，与设计文档 `run_turn(agent, scenario, selector)` 对齐。"""
    return agent.run_turn(scenario, selector=selector, **kw)


def run_scenario(
    agent: MemoryAgent,
    scenario: LongScenario,
    selectors: tuple[str, ...] = ("B0", "B1", "B2", "R0"),
    full_context: bool = False,
    max_slots: int = 3,
    vecs: np.ndarray | None = None,
) -> list[TurnResult]:
    """在一个场景上跑多个选择器，每个选择器问最多 `max_slots` 个槽位。

    同一场景内 bank 向量只编码一次（`vecs` 复用），
    这是把 4 组 × 3 槽位从「12 次编码」压到「1 次编码」的关键。
    """
    if vecs is None:
        vecs = embed_segments(scenario.bank)

    ask = askable_slots(scenario.gold_state or {})
    slots = sorted(ask)[:max_slots]
    out: list[TurnResult] = []

    for sel_name in selectors:
        if not slots:
            out.append(agent.run_turn(scenario, sel_name, full_context, None, vecs))
            continue
        for slot in slots:
            out.append(
                agent.run_turn(scenario, sel_name, full_context, slot, vecs)
            )
    return out
