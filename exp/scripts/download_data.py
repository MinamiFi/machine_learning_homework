#!/usr/bin/env python3
"""下载 MultiWOZ 2.2 的逐轮版本（parquet）。

    source exp/env.sh
    python exp/scripts/download_data.py            # 下载并落到 data/raw/multiwoz22/
    python exp/scripts/download_data.py --check    # 只校验

数据源：HF `pietrolesci/multiwoz_all_versions`
  - 已把 MultiWOZ 2.0/2.1/2.2/2.3 的逐轮状态对齐到同一张表，每行 = 一轮（user + system）
  - 我们只用 `states_22` 这一列（MultiWOZ 2.2 的 belief state）
  - 原始官方仓库 github.com/budzianowski/multiwoz 直连约 80 KB/s，不适合拉全量

落盘后结构：
    data/raw/multiwoz22/{train,dev,test}.parquet
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(EXP_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

RAW = EXP_ROOT / "data" / "raw" / "multiwoz22"
REPO = "pietrolesci/multiwoz_all_versions"

# split -> HF 仓库里的文件名
REMOTE = {
    "train": "data/train-00000-of-00001-cac863914a66066c.parquet",
    "dev": "data/validation-00000-of-00001-e59981d70cc52f88.parquet",
    "test": "data/test-00000-of-00001-ef025efe32e8c18e.parquet",
}


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def download(split: str, force: bool) -> None:
    from huggingface_hub import hf_hub_download

    dst = RAW / f"{split}.parquet"
    if dst.exists() and not force:
        print(f"  [skip] {split:<5} {human(dst.stat().st_size)}")
        return

    print(f"  [get ] {split:<5} {REMOTE[split]}")
    tmp = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=REMOTE[split],
        local_dir=str(RAW / "_tmp"),
    )
    shutil.move(tmp, dst)
    print(f"  [ok  ] {split:<5} {human(dst.stat().st_size)}")


def check() -> int:
    import pyarrow.parquet as pq

    bad = 0
    print(f"{'split':<8}{'size':>10}  {'rows':>8}  {'dialogues':>10}  status")
    for split in REMOTE:
        p = RAW / f"{split}.parquet"
        if not p.exists():
            print(f"{split:<8}{'-':>10}  {'-':>8}  {'-':>10}  MISSING")
            bad += 1
            continue
        t = pq.read_table(p, columns=["dialogue_id"])
        print(f"{split:<8}{human(p.stat().st_size):>10}  {t.num_rows:>8}  "
              f"{len(set(t.column('dialogue_id').to_pylist())):>10}  OK")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"repo   = {REPO}")
    print(f"target = {RAW}\n")

    if args.check:
        return 1 if check() else 0

    for split in REMOTE:
        download(split, args.force)
    tmp_dir = RAW / "_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print()
    return 1 if check() else 0


if __name__ == "__main__":
    sys.exit(main())
