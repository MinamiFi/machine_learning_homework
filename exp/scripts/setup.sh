#!/usr/bin/env bash
# 一键安装实验依赖（纯 CPU）。可重复执行，已装的会跳过。
#
#   bash exp/scripts/setup.sh
#
# 环境约束：2 核 / 4GB / 无 GPU。torch 必须走 CPU 索引，否则会拖进 CUDA 依赖。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

echo "[1/3] python: $($PY -V 2>&1)  ($(command -v "$PY"))"

echo "[2/3] torch (CPU-only)  ← $TORCH_INDEX"
$PY -m pip install --index-url "$TORCH_INDEX" -r "$HERE/requirements-torch.txt"

echo "[3/3] 其余依赖"
$PY -m pip install -r "$HERE/requirements.txt"

echo
echo "---- 安装结果 ----"
$PY - <<'PY'
import torch, transformers, sentence_transformers, sklearn, huggingface_hub
print(f"torch                {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"transformers         {transformers.__version__}")
print(f"sentence-transformers{sentence_transformers.__version__:>9}")
print(f"huggingface-hub      {huggingface_hub.__version__}")
print(f"scikit-learn         {sklearn.__version__}")
PY
