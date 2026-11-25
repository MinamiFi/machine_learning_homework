#!/usr/bin/env bash
# 一条命令复现全部实验（根 README §6 交付物验收要求）。
#
# 用法：
#   bash exp/scripts/run_all.sh              # 全流程（首次约 25 分钟 + L2 小时级）
#   bash exp/scripts/run_all.sh --skip-l2    # 跳过 L2（不需要 LLM，约 25 分钟）
#   bash exp/scripts/run_all.sh --check-only # 只跑文档数字自检（20 秒）
#
# 前置：已经 source 过 env.sh 的话脚本会重新 source，重复无害。
# 幂等：每步产物落 runs/{timestamp}_*，重跑不会覆盖旧结果。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source exp/env.sh

SKIP_L2=0
CHECK_ONLY=0
for a in "$@"; do
  case "$a" in
    --skip-l2)    SKIP_L2=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数 $a"; exit 2 ;;
  esac
done

step() { echo; echo "═══ $* ═══"; echo; }
t_start=$(date +%s)

if [ "$CHECK_ONLY" -eq 1 ]; then
  step "文档数字自检"
  python3 exp/scripts/verify_docs.py
  echo "完成，$(( $(date +%s) - t_start )) 秒"
  exit 0
fi

# ---- 0. 单元自检（1 秒，不加载模型）
step "0/8 四模块单元自检"
python3 exp/scripts/test_src.py

# ---- 1. 环境与模型（幂等：已装/已下会秒过）
step "1/8 依赖与模型"
bash exp/scripts/setup.sh
python3 exp/scripts/download_models.py

# ---- 2. 数据
step "2/8 下载并解析 MultiWOZ 2.2"
python3 exp/scripts/download_data.py
python3 exp/scripts/prepare_multiwoz.py

# ---- 3. 标签校验（C1–C5）
step "3/8 标签校验"
python3 exp/scripts/verify_labels.py --split train

# ---- 4. 切片与编码（约 7 分钟，train 最慢）
step "4/8 切片 + 句向量编码"
python3 exp/scripts/build_memory.py --split train dev test

# ---- 5. 打分网络
step "5/8 训练打分网络"
python3 exp/scripts/train_scorer.py

# ---- 6. L1 评测（不需要 LLM）
step "6/8 L1 选择质量评测"
python3 exp/scripts/eval_l1.py --tag eval_l1_full2

# ---- 7. 消融 + 预算扫描 + 标签三重校验
step "7/8 消融 / 预算扫描 / 标签校验"
python3 exp/scripts/run_ablation.py
python3 exp/scripts/budget_sweep.py
python3 exp/scripts/verify_counterfactual.py --split test --spot-check 60
python3 exp/scripts/verify_counterfactual.py --split dev
python3 exp/scripts/verify_counterfactual.py --split train
python3 exp/scripts/simulate_spot_check2.py --n 200

# ---- 8. L2（需要 LLM，小时级；0.5B 只能出配对结论）
if [ "$SKIP_L2" -eq 0 ]; then
  step "8/8 L2 任务完成率评测（小时级）"
  python3 exp/scripts/jga_check.py --n-dialogues 120 --max-slots 3 --tag jga_main_fix2
  python3 exp/scripts/eval_l2.py --n-per-length 10 --max-slots 2 --tag l2_main
else
  step "8/8 跳过 L2（--skip-l2）"
fi

# ---- 收尾：图 + 文档数字自检
step "生成图（报告/PPT 用）"
python3 exp/scripts/make_figures.py

step "文档数字自检（不一致则非 0 退出）"
python3 exp/scripts/verify_docs.py

echo
echo "全部完成，$(( $(date +%s) - t_start )) 秒"
echo "产物在 exp/runs/ 下，报告在 exp/docs/ 下"
