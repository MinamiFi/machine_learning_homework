"""memory — 记忆库：切片、编码、检索、预算内选择、长历史构造。

这个包只依赖 numpy，不依赖 torch / transformers。
编码器通过参数传入，方便单测时用假编码器。
"""

from .types import MemorySegment, LongScenario, Scenario
from .slice import slice_turns, load_segments, segment_text, count_tokens
from .embed import embed_segments, embed_queries, Embedder
from .retrieve import retrieve, retrieve_batch, cosine_sim
from .select import select, SelectResult
from .long_history import build_long_history, build_scenarios

__all__ = [
    "MemorySegment",
    "LongScenario",
    "Scenario",
    "slice_turns",
    "load_segments",
    "segment_text",
    "count_tokens",
    "embed_segments",
    "embed_queries",
    "Embedder",
    "retrieve",
    "retrieve_batch",
    "cosine_sim",
    "select",
    "SelectResult",
    "build_long_history",
    "build_scenarios",
]
