#!/usr/bin/env python3
"""把实验要用的模型下到 exp/models/，已存在则跳过。

    source exp/env.sh
    python exp/scripts/download_models.py              # 全部
    python exp/scripts/download_models.py --only embedder
    python exp/scripts/download_models.py --check      # 只校验，不下载

说明：本环境访问不了 huggingface.co，默认走 hf-mirror.com。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(EXP_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODELS = EXP_ROOT / "models"

# 每项：key -> (HF repo id, 说明, 大致体积, 只取这些文件)
# allow 必须写全，否则 all-MiniLM 会把 openvino/onnx/tf 三套权重全拖下来（900MB+）。
SPECS: dict[str, tuple[str, str, str, list[str]]] = {
    "embedder": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "句向量模型，384 维，记忆检索与打分网络的输入",
        "~90 MB",
        ["*.json", "*.txt", "*.safetensors", "1_Pooling/*", "2_Normalize/*"],
    ),
    "llm": (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "对话智能体 backbone 的开发期占位模型（CPU 可跑）",
        "~1.0 GB",
        ["*.json", "*.safetensors", "*.txt", "*.model"],
    ),
}


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def is_ready(local: Path) -> bool:
    """有 config.json 且目录非空，就认为下好了。"""
    return local.is_dir() and (local / "config.json").is_file()


def download(key: str, force: bool) -> None:
    from huggingface_hub import snapshot_download

    repo, desc, approx, allow = SPECS[key]
    local = MODELS / Path(repo).name

    if is_ready(local) and not force:
        print(f"  [skip] {key:<9} {local.name} 已存在（{human(dir_size(local))}）")
        return

    print(f"  [get ] {key:<9} {repo}  {approx}")
    print(f"         → {local}")
    snapshot_download(
        repo_id=repo,
        local_dir=str(local),
        allow_patterns=allow,
        max_workers=4,
    )
    print(f"  [ok  ] {key:<9} {human(dir_size(local))}")


def check() -> int:
    bad = 0
    print(f"{'key':<10}{'dir':<34}{'size':>10}  status")
    for key, (repo, _, _, _) in SPECS.items():
        local = MODELS / Path(repo).name
        ok = is_ready(local)
        size = human(dir_size(local)) if local.exists() else "-"
        print(f"{key:<10}{local.name:<34}{size:>10}  {'OK' if ok else 'MISSING'}")
        bad += 0 if ok else 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SPECS), help="只处理某一个")
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    ap.add_argument("--check", action="store_true", help="只校验不下载")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
    print(f"HF_HOME     = {os.environ['HF_HOME']}")
    print(f"target      = {MODELS}\n")

    if args.check:
        return 1 if check() else 0

    keys = [args.only] if args.only else list(SPECS)
    for k in keys:
        download(k, args.force)
    print()
    return 1 if check() else 0


if __name__ == "__main__":
    sys.exit(main())
