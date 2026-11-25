#!/usr/bin/env python3
"""把 `交付/` 打成单个 zip。

先跑 `make_delivery.py` 生成 `交付/`，再跑本脚本。打包时逐文件做密钥扫描，
命中即跳过并报告——所以本脚本自身也会被扫到，**不能把密钥字面量写在这里**，
只能用正则模式匹配。

用法：
    python3 exp/scripts/make_zip.py            # 打包
    python3 exp/scripts/make_zip.py --check    # 只校验已有 zip 的完整性与密钥
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "交付"
OUT = ROOT / "高级机器学习_题目3_记忆打分_交付包.zip"

# 只匹配密钥的形状，不匹配任何真实值——否则扫描器会扫到自己。
SECRET_PATTERNS = [
    re.compile(rb"sk-[0-9a-f]{32}"),
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
]


def scan(data: bytes) -> str | None:
    for pat in SECRET_PATTERNS:
        m = pat.search(data)
        if m:
            return m.group(0)[:8].decode("ascii", "replace") + "…"
    return None


def build() -> int:
    if not SRC.is_dir():
        print(f"!! 找不到 {SRC}，先跑 make_delivery.py", file=sys.stderr)
        return 1

    n = 0
    hits: list[str] = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(SRC.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            data = p.read_bytes()
            if (sig := scan(data)) is not None:
                hits.append(f"{rel}  ({sig})")
                continue
            z.write(p, rel.as_posix())
            n += 1

    size = OUT.stat().st_size / 1024 / 1024
    print(f"写入 {n} 个文件，{size:.2f} MB → {OUT.name}")
    if hits:
        print(f"!! 跳过 {len(hits)} 个疑似含密钥的文件：")
        for h in hits:
            print("   ", h)
        return 2
    print("密钥扫描：0 命中")
    return 0


def check() -> int:
    if not OUT.exists():
        print(f"!! 找不到 {OUT}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(OUT) as z:
        bad = z.testzip()
        names = [x for x in z.namelist() if not x.endswith("/")]
        hits = []
        for name in names:
            if (sig := scan(z.read(name))) is not None:
                hits.append(f"{name}  ({sig})")
    size = OUT.stat().st_size / 1024 / 1024
    print(f"{OUT.name}: {len(names)} 文件，{size:.2f} MB")
    print("完整性：", "OK" if bad is None else f"损坏于 {bad}")
    print("密钥扫描：", "0 命中" if not hits else f"{len(hits)} 命中")
    for h in hits:
        print("   ", h)
    return 0 if bad is None and not hits else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="只校验已有 zip")
    a = ap.parse_args()
    return check() if a.check else build()


if __name__ == "__main__":
    raise SystemExit(main())
