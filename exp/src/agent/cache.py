"""生成结果缓存。

L2 一次跑几十分钟，崩了不能从头来。key = sha256(model + prompt + params)，
落盘 JSONL，命中就直接读。这不是优化，是必需（`src/README.md` §3）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["ResponseCache"]


class ResponseCache:
    def __init__(self, path: str | Path, model_name: str = ""):
        self.path = Path(path)
        self.model_name = model_name
        self._mem: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        # 并发跑实验时多个线程会同时 put。dict 赋值受 GIL 保护，
        # 但「追加写文件」不是原子的——必须加锁，否则 JSONL 会串行错乱。
        import threading
        self._lock = threading.Lock()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._mem[row["key"]] = row

    def _key(self, prompt: str, max_new_tokens: int) -> str:
        raw = f"{self.model_name}\x00{prompt}\x00{max_new_tokens}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, prompt: str, max_new_tokens: int) -> dict | None:
        k = self._key(prompt, max_new_tokens)
        row = self._mem.get(k)
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return row

    def put(self, prompt: str, max_new_tokens: int, text: str,
            seconds: float = 0.0, **meta) -> None:
        k = self._key(prompt, max_new_tokens)
        row = {"key": k, "text": text, "seconds": seconds,
               "model": self.model_name, **meta}
        with self._lock:
            self._mem[k] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._mem)
