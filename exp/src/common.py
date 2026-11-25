"""公共工具：路径解析、配置读取、随机种子、日志。

不放任何模型逻辑，只放"所有模块都要用"的东西。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------- 路径

def exp_root() -> Path:
    """exp/ 的绝对路径。src/ 的上一级。"""
    return Path(__file__).resolve().parent.parent


def path_of(*parts: str) -> Path:
    """相对 exp/ 拼路径。所有路径都从这里出，不硬编码。"""
    return exp_root().joinpath(*parts)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- 配置

def load_yaml(p: str | Path) -> dict[str, Any]:
    import yaml

    with open(path_of(str(p)) if not os.path.isabs(str(p)) else p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, patch: dict) -> dict:
    """递归合并，patch 优先。用于消融时只改几个字段。"""
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- 随机

def set_seed(seed: int) -> None:
    """一次设齐三个随机源。torch 延迟导入，避免 memory 模块被迫依赖 torch。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU 上部分算子无确定实现
    except ImportError:
        pass


# ---------------------------------------------------------------- 日志

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def dump_json(obj: Any, p: Path, indent: int = 2) -> None:
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, p: Path) -> None:
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stable_seed(*parts: object, mod: int | None = None) -> int:
    """跨进程稳定的种子。**绝不能用内置 hash()**。

    Python 3 默认对 str/bytes/tuple 的 hash 加随机盐（PYTHONHASHSEED），
    所以 `hash(("MUL0003.json", 128, 0))` 在两次进程里得到不同的数。
    用它当 random.Random 的种子，会让「同一份数据 + 同一个 seed」
    每次都生成不同的场景——实验完全不可复现，而且缓存全部失效。

    实测：同一段代码在两个进程里分别得到 needle_pos 0.86 与 0.69，
    场景 id 里带位置后缀，所以连"哪些场景"都对不上。

    用 sha256 前 8 字节转整数，稳定且分布均匀。
    """
    h = hashlib.sha256()
    for x in parts:
        h.update(repr(x).encode("utf-8"))
        h.update(b"\x00")
    v = int.from_bytes(h.digest()[:8], "big")
    return v % mod if mod else v
