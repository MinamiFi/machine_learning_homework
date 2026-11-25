#!/usr/bin/env python3
"""第 1 步：切片 + 编码，产出可训练的向量库。

用法：
    source exp/env.sh
    python exp/scripts/build_memory.py              # dev + test
    python exp/scripts/build_memory.py --split train # 加 train（约 6 分钟）

产物：
    data/processed/memory/{split}.minilm.npz   (N, 384) float32 已归一化
    data/processed/multiwoz22/{split}.tokens.npz  token 计数
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.common import get_logger, path_of, dump_json
from src.memory.slice import load_segments
from src.memory.embed import embed_split

log = get_logger("build_memory")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["dev", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="忽略缓存重算")
    args = ap.parse_args()

    summary = {}
    for split in args.split:
        t0 = time.time()
        segs = load_segments(split)
        emb = embed_split(split, segs, batch_size=args.batch_size, force=args.force)
        dt = time.time() - t0

        toks = np.array([s.n_tokens for s in segs])
        summary[split] = {
            "segments": len(segs),
            "dialogues": len({s.dialogue_id for s in segs}),
            "emb_shape": list(emb.shape),
            "emb_norm_mean": round(float(np.linalg.norm(emb, axis=1).mean()), 6),
            "tokens_mean": round(float(toks.mean()), 2),
            "tokens_median": int(np.median(toks)),
            "tokens_p90": int(np.percentile(toks, 90)),
            "tokens_max": int(toks.max()),
            "pos_rate": round(float(np.mean([s.label for s in segs])), 4),
            "seconds": round(dt, 1),
        }
        log.info("%s 完成 %.1fs | %s | 归一化模长 %.4f",
                 split, dt, emb.shape, summary[split]["emb_norm_mean"])

    dump_json(summary, path_of("data/processed/memory/summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
