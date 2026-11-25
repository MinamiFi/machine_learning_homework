# 实验环境变量。用法：source exp/env.sh
# 目的：模型缓存、镜像、线程数全部收敛到 exp/ 内部，换机器不用改代码。

export EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HuggingFace 走国内镜像（官方站在本环境不可达）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$EXP_ROOT/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# 2 核机器：把 BLAS 线程钉死，避免线程超订反而变慢
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false

export MODELS_DIR="$EXP_ROOT/models"
export PYTHONPATH="$EXP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
