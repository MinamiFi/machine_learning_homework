"""切片：turn 级 JSONL → MemorySegment，并用真实 tokenizer 计数。

为什么不用 len(text.split())：实测单轮中位 40 token，按空格估会低估约 20%，
B=2048 时误差能到 10 轮以上，直接影响"预算内装多少"的结论。
"""

from __future__ import annotations

import functools
from pathlib import Path

from ..common import path_of, read_jsonl, get_logger
from .types import MemorySegment

log = get_logger(__name__)

DATA_DIR = "data/processed/multiwoz22"


# ---------------------------------------------------------------- tokenizer

@functools.lru_cache(maxsize=2)
def _get_tokenizer(name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """延迟加载 Qwen tokenizer。只用来计数，不参与生成。

    LLM 侧的 token 预算必须用同一个 tokenizer，否则预算就不是预算了。
    """
    from transformers import AutoTokenizer

    local = path_of("models", "Qwen2.5-0.5B-Instruct")
    src = str(local) if local.exists() else name
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    return tok


def count_tokens(text: str) -> int:
    """真实 tokenizer 计数（不含特殊 token）。"""
    if not text:
        return 0
    return len(_get_tokenizer()(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------- 文本拼装

def segment_text(seg: MemorySegment, include_system: bool = True) -> str:
    """把一段记忆渲染成进 prompt 的文本。

    格式固定，所有基线一致；R3（打乱时间顺序）只改拼接顺序，不改这个格式。
    """
    parts = [f"User: {seg.user}"]
    if include_system and seg.system and seg.system != "none":
        parts.append(f"System: {seg.system}")
    return "\n".join(parts)


# ---------------------------------------------------------------- 主入口

def slice_turns(split: str, data_dir: str = DATA_DIR) -> list[MemorySegment]:
    """读一个 split 的 JSONL，转成 MemorySegment 列表（按 dialogue_id, turn_idx 排序）。

    保留 label=0 的轮次：过滤是训练侧的事（scorer.build_dataset 负责），
    不是切片侧的事——评测时负例片段同样是候选记忆。
    """
    p = path_of(data_dir, f"{split}.jsonl")
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，先跑 exp/scripts/prepare_multiwoz.py")

    segs: list[MemorySegment] = []
    for r in read_jsonl(p):
        seg = MemorySegment.from_dict(r)
        seg.seg_id = f"{seg.dialogue_id}:{seg.turn_idx}"
        segs.append(seg)

    segs.sort(key=lambda s: (s.dialogue_id, s.turn_idx))
    log.info("slice_turns(%s): %d 段 / %d 对话",
             split, len(segs), len({s.dialogue_id for s in segs}))
    return segs


def attach_token_counts(segs: list[MemorySegment], include_system: bool = True) -> None:
    """就地填 n_tokens。编码一次约 185 段/秒，6.5 万段约 6 分钟。"""
    for s in segs:
        if not s.n_tokens:
            s.n_tokens = count_tokens(segment_text(s, include_system))


def load_segments(
    split: str,
    data_dir: str = DATA_DIR,
    with_tokens: bool = True,
) -> list[MemorySegment]:
    """切片 + 填 token 数。带 npz 缓存，第二次调用是秒级。"""
    import numpy as np

    cache = path_of(data_dir, f"{split}.tokens.npz")
    segs = slice_turns(split, data_dir)

    if with_tokens:
        if cache.exists():
            counts = np.load(cache)["n_tokens"]
            if len(counts) == len(segs):
                for s, c in zip(segs, counts):
                    s.n_tokens = int(c)
                log.info("token 计数命中缓存 %s", cache.name)
                return segs
        attach_token_counts(segs)
        np.savez_compressed(cache, n_tokens=np.array([s.n_tokens for s in segs], dtype=np.int32))
        log.info("token 计数已缓存 → %s", cache.name)

    return segs


def group_by_dialogue(segs: list[MemorySegment]) -> dict[str, list[MemorySegment]]:
    out: dict[str, list[MemorySegment]] = {}
    for s in segs:
        out.setdefault(s.dialogue_id, []).append(s)
    for v in out.values():
        v.sort(key=lambda x: x.turn_idx)
    return out
