"""memory 的数据类型。

用 dataclass 而不是裸 dict：字段拼错要报错，不能静默变成 None。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class MemorySegment:
    """一段记忆 = 一轮对话（user + system）。

    这是全项目统一的记忆粒度（根 README §3.2）。
    """

    seg_id: str                 # f"{dialogue_id}:{turn_idx}"，全局唯一
    dialogue_id: str
    turn_idx: int
    user: str
    system: str
    state: dict[str, str] = field(default_factory=dict)
    delta: dict[str, str] = field(default_factory=dict)
    n_valid: int = 0
    label: int = 0
    dialogue_has_state: bool = True
    services: list[str] = field(default_factory=list)
    n_tokens: int = 0           # 真实 tokenizer 计数，slice 时填

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemorySegment":
        """JSONL 里没有 seg_id（由 dialogue_id + turn_idx 推出），这里补上。"""
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        kw.setdefault("seg_id", f"{d['dialogue_id']}:{d['turn_idx']}")
        return cls(**kw)


@dataclass
class Scenario:
    """一次评测场景：一条待回答的 query + 一个可检索的记忆库。

    bank 是候选片段（可以是整个对话，也可以是膨胀后的长历史）。
    query 是当前轮用户的话。
    """

    scenario_id: str
    dialogue_id: str
    bank: list[MemorySegment]          # 候选记忆，按时间序
    query: str
    gold_state: dict[str, str] = field(default_factory=dict)
    needle_id: str | None = None       # 埋入的关键信息片段 id（L1 的 CR 用）
    decoy_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LongScenario(Scenario):
    """膨胀后的长历史场景。

    在 Scenario 基础上记录合成参数，便于报告里交代"历史是怎么造出来的"。
    """

    target_len: int = 0
    needle_pos: float = 0.0
    seed: int = 0
    n_distractors: int = 0
