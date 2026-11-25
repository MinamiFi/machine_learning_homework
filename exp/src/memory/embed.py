"""句向量编码：all-MiniLM-L6-v2，384 维。

实测 185 段/秒（2 核），6.5 万段约 6 分钟。结果落盘 npz，评测时不重算。

编码对象是 segment_text()（user + system），不是单独 user：
记忆片段是"这一轮发生了什么"，system 回复里含检索到的实体名，是关键信息载体。
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from ..common import path_of, get_logger
from .slice import segment_text
from .types import MemorySegment

log = get_logger(__name__)

EMBED_DIM = 384
EMBED_DIR = "data/processed/memory"


class Embedder:
    """句向量模型包装。延迟加载，避免 import 时就吃内存。"""

    def __init__(self, model_dir: str = "models/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_dir = str(path_of(model_dir))
        self.device = device
        self._m = None

    @property
    def model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer

            log.info("加载句向量模型 %s", self.model_dir)
            self._m = SentenceTransformer(self.model_dir, device=self.device)
        return self._m

    def encode(self, texts: list[str], batch_size: int = 32, show: bool = False) -> np.ndarray:
        v = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # 归一化后内积即 cosine
            show_progress_bar=show,
        )
        return v.astype(np.float32)


@functools.lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    return Embedder()


# ---------------------------------------------------------------- 编码

def embed_segments(
    segs: list[MemorySegment],
    embedder: Embedder | None = None,
    batch_size: int = 32,
    include_system: bool = True,
    show: bool = False,
) -> np.ndarray:
    """编码所有片段 → (N, 384) float32，已 L2 归一化。"""
    embedder = embedder or default_embedder()
    texts = [segment_text(s, include_system) for s in segs]
    return embedder.encode(texts, batch_size=batch_size, show=show)


def embed_queries(
    queries: list[str],
    embedder: Embedder | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    embedder = embedder or default_embedder()
    return embedder.encode(queries, batch_size=batch_size)


# ---------------------------------------------------------------- 缓存

def embed_split(
    split: str,
    segs: list[MemorySegment],
    embedder: Embedder | None = None,
    batch_size: int = 32,
    force: bool = False,
) -> np.ndarray:
    """编码一个 split 并落盘缓存。已有缓存且行数一致则直接读。

    缓存键只含 split 名：换编码器要手动删 npz，否则会读到旧向量。
    在文件名里带上模型名可以避免这个坑。
    """
    out = path_of(EMBED_DIR, f"{split}.minilm.npz")
    if out.exists() and not force:
        z = np.load(out)
        if z["emb"].shape[0] == len(segs):
            log.info("句向量命中缓存 %s  shape=%s", out.name, z["emb"].shape)
            return z["emb"].astype(np.float32)
        log.warning("缓存行数不符（%d vs %d），重算", z["emb"].shape[0], len(segs))

    log.info("编码 %s：%d 段…", split, len(segs))
    emb = embed_segments(segs, embedder, batch_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=emb, seg_ids=np.array([s.seg_id for s in segs]))
    log.info("已缓存 → %s  (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return emb
