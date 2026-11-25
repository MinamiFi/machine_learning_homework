# exp/ — 题目三实验环境

对应根目录 [README.md](../README.md) 的方案。包含环境、模型、数据、四模块代码（`src/`）与全部实验产物（`runs/`）。

自检状态：**13/13 通过，峰值内存 3.24 GB / 上限 4.29 GB**。

---

## 0. 现在有什么

| 组件 | 状态 | 位置 | 说明 |
|---|---|---|---|
| Python | ✅ 3.11.2 | 系统 | |
| torch | ✅ 2.5.1+cpu | 用户 site-packages | CPU 版，无 CUDA |
| transformers / sentence-transformers | ✅ 4.46.3 / 3.3.1 | 同上 | 版本互相验证过 |
| 句向量模型 | ✅ 87.4 MB | `models/all-MiniLM-L6-v2` | 384 维，英文 |
| 对话 backbone | ✅ 953.3 MB | `models/Qwen2.5-0.5B-Instruct` | **开发占位**，见 §3 |
| 环境自检 | ✅ 13/13 | `scripts/smoke_test.py` | 见 §6 留档 |
| MultiWOZ 2.2 数据 | ✅ 52 MB | `data/` | 8,437 / 1,000 / 1,000 对话，见 [`data/README.md`](data/README.md) |
| 弱监督标签 | ✅ 已生成并校验 | `data/processed/multiwoz22/` | 正例 55.6%，3 split × 5 项校验全通过；有位置偏置。原理见 [`data/标签说明.md`](data/标签说明.md) |
| API backbone | ✅ 已接入 | `configs/agent.yaml` | 同济 `https://agi.tongji.edu.cn/v1` / `DeepSeek-V4.1-Flash`；key 只从环境变量 `DSH_EXP_API_KEY` 读（`env.api.sh`，已 gitignore） |
| 实验代码 | ✅ 四模块均可用 | `src/` | 设计见 [`src/README.md`](src/README.md)；`agent/loop.py` 多轮主循环已实现，含 `full_context` 开关与三种 backbone |
| **L1 实验结果** | ✅ 已跑完 | [`docs/L1实验结果.md`](docs/L1实验结果.md) | B2 跨长度 CR **0.7667**，纯检索 0.4778，位置基线 0.2778 |
| **L2 任务完成率** | ✅ 已跑完（API） | [`docs/L2任务完成率.md`](docs/L2任务完成率.md) | B2 slot_acc **0.3778** / 场景成功率 0.1667 / ctx 1,489 token；ORACLE 0.8444 只用 216 token |
| 消融实验 | ✅ 已跑完 | `runs/*_ablation/` | 层数/学习率/L2 三因素均无显著影响（见 L1 文档 §2.3） |
| 标签可信度 | ✅ 三个口径 | [`docs/反事实删除校验.md`](docs/反事实删除校验.md) · [`docs/L2_JGA校验.md`](docs/L2_JGA校验.md) · [`docs/标签抽检.md`](docs/标签抽检.md) | 覆盖率 10.56×、JGA 差 30.8 个百分点、模拟 κ 0.43–0.50 |

> ⚠️ 2026-11-24 修正了两处错误，本页数字均为修正后值：
> 1. **数据层 system 错位**：MultiWOZ 原始数据一轮内先 system 后 user，原先取 `(usr[t], sys[t])` 是错位的。见根 README 附录 E。
> 2. **`hash()` 随机盐**：`build_scenarios` 用内置 `hash()` 生成 `needle_pos`，Python 3 每次进程加随机盐 → **同一份数据每次跑出的场景完全不同**。已改为 `stable_seed()`（sha256），L1/L2/budget/necessity 全部重跑。见根 README 附录 F。

总计占用 1.2 GB（模型 1.04 GB + 数据 52 MB + 依赖）。

---

## 1. 快速开始

```bash
cd ~/高级机器学习
source exp/env.sh                      # 必须，否则模型路径/镜像/线程数都不对

python3 exp/scripts/download_models.py --check   # 校验模型是否齐
python3 exp/scripts/download_data.py --check     # 校验数据是否齐
python3 exp/scripts/verify_labels.py             # 校验标签（5 项）
python3 exp/scripts/smoke_test.py                # 全链路自检
python3 exp/scripts/smoke_test.py --skip-llm     # 只测前两段，省 3 GB 内存

# L1 流水线（首次建库 7 分钟，之后都是秒级）
python3 exp/scripts/build_memory.py --split train dev test
python3 exp/scripts/train_scorer.py              # 20 秒
python3 exp/scripts/eval_l1.py --lengths 128 256 512 --n-per-length 30 --seeds 0 1 2

# 标签可信度：反事实删除校验（1.4 秒）+ necessity 标签对照（6 分钟）
python3 exp/scripts/verify_counterfactual.py --split test --spot-check 60
python3 exp/scripts/train_necessity.py                          # AUC 对比，31 秒
python3 exp/scripts/compare_necessity_l1.py --n-per-length 60   # CR 对比，5 分钟

# L2 JGA 口径校验（配对，0.5B，21 分钟；复用缓存重算 0 秒）
python3 exp/scripts/jga_check.py --n-dialogues 120 --max-slots 3 --tag jga_main
python3 exp/scripts/jga_check.py --n-dialogues 607 --max-slots 3 --tag jga_wide   # 全量扩样，2 小时+
python3 exp/scripts/jga_check.py --n-dialogues 607 --max-slots 3 --resume <dir> --cache-only  # 只用已有缓存出部分结果

python3 exp/scripts/verify_docs.py               # 20 秒，校验文档 181 项数字
```

换机器重装：

```bash
bash exp/scripts/setup.sh              # 装依赖（幂等）
python3 exp/scripts/download_models.py  # 下模型（已存在则跳过）
```

---

## 2. 目录

```
exp/
├── README.md                # 本文件
├── env.sh                   # 环境变量（source 它）
├── requirements.txt         # 直接依赖（torch 除外）
├── requirements-torch.txt   # torch，必须走 CPU 索引
├── requirements.lock.txt    # 实测版本快照
├── configs/
│   ├── env.yaml             # 路径、模型名、线程数
│   ├── scorer.yaml          # 打分网超参（消融就改这里）
│   └── agent.yaml           # backbone 与 token 预算
├── scripts/
│   ├── setup.sh                  # 一键装依赖
│   ├── download_models.py        # 下模型 + 校验
│   ├── download_data.py          # 下 MultiWOZ 2.2
│   ├── prepare_multiwoz.py       # 转 JSONL + 生成差分标签（含 system 错位修正）
│   ├── verify_labels.py          # 标签逻辑校验（C1–C5）
│   ├── build_memory.py           # 切片 + 编码 + token 缓存
│   ├── train_scorer.py           # 训练打分网
│   ├── eval_l1.py                # L1 选择质量评测
│   ├── run_ablation.py           # 消融网格
│   ├── budget_sweep.py           # token 预算扫描
│   ├── eval_l2.py                # L2 任务完成率 / AR / token 消耗
│   ├── jga_check.py              # JGA 配对口径校验（含 --cache-only 部分结果模式）
│   ├── verify_counterfactual.py  # 反事实删除校验
│   ├── train_necessity.py        # necessity 标签对照（AUC）
│   ├── compare_necessity_l1.py   # necessity 标签对照（CR，配对 bootstrap）
│   ├── simulate_spot_check.py    # 模拟抽检 v1（LLM 判据，已证伪）
│   ├── simulate_spot_check2.py   # 模拟抽检 v2（J2/J3 规则判据）
│   ├── probe_judge_prompt.py     # 探针：0.5B 判必要性四种写法全退化
│   ├── verify_docs.py            # 文档数字自检（181 项，20 秒）
│   ├── run_all.sh                # 一条命令复现全部实验
│   ├── make_figures.py           # 出图（7 张，报告/PPT 用）
│   ├── make_ppt.js               # 生成 35 页汇报 PPT（pptxgenjs + 主题后处理）
│   ├── md2docx.py                # 报告 md → docx（无 pandoc 环境的轻量转换）
│   ├── test_src.py               # 四模块单元自检（48 项，1 秒，不加载模型）
│   └── smoke_test.py             # 三段式自检（加载 Qwen，90 秒）
├── src/
│   ├── README.md            # 四模块设计与接口
│   ├── common.py            # 路径/YAML/日志/读写工具
│   ├── memory/              # 切片、编码、长历史构造、检索、选择
│   ├── scorer/              # 打分网、数据集、基线、训练
│   ├── eval/                # 指标、L1/L2 调度、报告
│   └── agent/               # ✅ prompt / llm / cache / loop 四件齐全
├── docs/
│   ├── L1实验结果.md        # 主表 + 消融 + 四个设计坑
│   ├── 反事实删除校验.md    # 标签可信度（10.56 倍）+ necessity 对照（负结果，§8）
│   ├── L2_JGA校验.md        # JGA 口径复核（API：full 0.9174 / drop_pos 0.2477）
│   ├── L2任务完成率.md      # 任务完成率 / AR / token 消耗主表（ORACLE 216 token 84.4%）
│   ├── 标签抽检.md          # 模拟 Cohen's κ（0.43–0.50），明确非人工校验
│   └── 标签抽检表.csv       # 60 行抽检表（模拟值已填 50 条，10 条分歧留空）
├── figures/                 # 报告/PPT 用图（7 张 300dpi PNG，make_figures.py 生成）
├── slides/                  # 汇报 PPT（题目三_记忆打分_汇报.pptx + pdf 预览）
├── models/                  # 模型权重（gitignore）
├── .cache/huggingface/      # HF 缓存（gitignore）
├── data/                    # MultiWOZ 2.2 + 标签（见 data/README.md）
├── logs/                    # 日志
└── runs/                    # 实验产物（每次一个时间戳目录）
```

`runs/` 里已有（**2026-11-24 修正 system 错位后的有效结果**）：

| 目录 | 内容 |
|---|---|
| `20261124_021731_scorer_scorer_base_s0_s0/` | 打分网（test AUC **0.9332**，位置基线 0.8172，margin +0.1160） |
| `20261124_092829_eval_l1_eval_l1_full2/` | L1 完整评测（3 长度 × 90 场景，B2 CR **0.7667**） |
| `20261124_022842_ablation/` | 消融（3 seeds，1 层 0.9352 最优） |
| `20261124_023207_budget/` | 预算扫描（12 组） |
| `20261124_032401/022143/022148_counterfactual_*/` | 反事实校验三 split（10.56× / 13.23× / 10.71×） |
| `20261124_030223_necessity/` + `_030727_necessity_l1_n60/` | necessity 标签对照（CR 无显著差异 p=0.4996） |
| `20261124_023556_jga_main_fix2/` | L2 JGA 配对校验（258 条，1371 秒） |
| `20261124_061833_jga_wide3/` | L2 JGA **扩样**（**607 段 / 1332 条**，4879 秒；分槽位细分用这一版） |
| `20261124_081620_jga_api_full/` | L2 JGA **真 backbone**（607 段 / 1332 条，1234 秒，DeepSeek-V4.1-Flash；**绝对 JGA 用这一版**） |
| `20261124_033153_spot_check2_n200/` | 模拟抽检 κ（n=200） |
| `20261124_032705_judge_probe/` | 探针：LLM 判必要性四种写法全退化 |

`20261119_*` 系列是**修正前的旧结果**，已被上面的目录取代，仅留档对照。`20261119_085551_ablation` 另有一处配置写错段，作废。

---

## 3. 模型选择：为什么是这两个

**句向量用 `all-MiniLM-L6-v2`**：384 维、87 MB、CPU 上 6 句 0.05 秒。MultiWOZ 是英文语料，**不能用 `bge-small-zh`** 这类中文模型（v1 README 的错误，已在根 README 附录 A 更正）。

**对话 backbone 用 `Qwen2.5-0.5B-Instruct`**，原因是内存：本机 cgroup 上限 4.29 GB，0.5B 的 fp32 权重约 2 GB，实测峰值 3.24 GB；1.5B 的 fp32 要 6 GB，装不下。

> ⚠️ **0.5B 只能用来打通管线和调 prompt，不能用来出主表。**
> 它的对话状态追踪能力很弱，任务完成率大概率接近 0，三组基线之间分不出差异。
> 正式实验两条路：
> 1. **API**（推荐）：改 `configs/agent.yaml` 的 `backbone: api`，key 走环境变量；
> 2. 换一台有 GPU 的机器，跑 7B 级别。

这一点必须在组会上说清楚，否则会拿一堆全 0 的结果白跑两周。

---

## 4. 内存账（4.29 GB 上限）

| 阶段 | 峰值 RSS | 备注 |
|---|---|---|
| 只加载 embedder | ~0.6 GB | |
| 加载 LLM（fp32） | **3.24 GB** | 余量不到 1 GB |
| 打分网训练 | 可忽略 | 16.5 万参数 |

推论：

- **不要并行跑两个 LLM 进程**，评测必须串行。
- 真实评测时进程里还要放记忆库向量（几百 MB 量级），余量会更紧。如果 OOM，把 `configs/agent.yaml` 的 `dtype` 改成 `bfloat16`（权重降到约 1 GB），但 CPU 上 bf16 算子可能更慢，需要实测。
- 内存不是主要瓶颈，**速度才是**：0.5B 在 2 核上约 4.3 tok/s，一轮完整对话要好几秒。三组基线 × 几百条样本 × 多个种子，跑之前先估算总时长。

---

## 5. 环境变量（`source exp/env.sh`）

| 变量 | 值 | 作用 |
|---|---|---|
| `HF_ENDPOINT` | `https://hf-mirror.com` | 本机访问不了 huggingface.co |
| `HF_HOME` | `exp/.cache/huggingface` | 缓存收在项目内，便于打包 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `2` | 钉死线程数，避免超订变慢 |
| `MODELS_DIR` | `exp/models` | 脚本读模型用 |
| `PYTHONPATH` | `exp/` | 之后 `import src.xxx` 能直接用 |

---

## 6. 自检留档（2026-11-18）

```
[1/3] 句向量模型
  [PASS] embedder 模型存在        all-MiniLM-L6-v2
  [PASS] 输出维度 == 384          dim=384
  [PASS] 向量已归一化
  [PASS] 同义句相似度 > 跨主题     intra=0.685 inter=0.261
  [PASS] 编码速度                 load 0.2s, 6 句 0.05s

[2/3] 打分网络训练链路（合成数据）
  [PASS] MLP 参数量               165,377
  [PASS] loss 下降                0.6772 → 0.0799
  [PASS] 合成任务 AUC > 0.95      AUC=0.9807
  [PASS] 训练速度                 96 step / 0.4s

[3/3] 对话 backbone
  [PASS] llm 模型存在             Qwen2.5-0.5B-Instruct
  [PASS] llm 加载                 1.4s, 峰值 3242 MB
  [PASS] 能生成非空回复            5 tok / 1.2s = 4.3 tok/s
  [PASS] 内存 < 4GB               峰值 3242 MB
```

第 3 段里模型拿到「记忆片段 + 当前问题」后正确答出 `acorn guest house`，说明 prompt 模板的形状是对的。

第 2 段用的是**合成数据**（固定线性 teacher），只证明 BCE + AdamW + MLP 这条训练链路能收敛，**不代表真实标签可用**。真实标签要靠 MultiWOZ 的 belief state 差分，见根 README §4.3。

---

## 7. 踩过的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `all-MiniLM` 下出 932 MB | 仓库含 openvino / onnx / tf / flax 四套权重 | `download_models.py` 加 `allow_patterns`，已清理到 88 MB |
| huggingface.co 超时 | 官方站不可达 | 默认 `HF_ENDPOINT=https://hf-mirror.com` |
| 默认源装 torch 拉 CUDA 依赖 | Linux 默认轮子是 CUDA 版 | 固定 `--index-url .../whl/cpu` |
| `num_layers=2` 只有 1 个隐藏层 | 维度列表写法差一 | 已改为 `[in] + [hidden]*num_layers + [1]`，参数量对齐 16.5 万 |
| Qwen 生成告警 top_p/top_k | 自带 generation_config 与贪心解码冲突 | 加载后置 `top_p=1.0, top_k=0` |
| 输出以 `user\n` 开头 | 没传 `add_generation_prompt=True` | `llm.py` 的 `apply_chat_template` 补上 |
| 撇号导致假阴性 | `Queen's` 去标点后变 `queen s`，与 gold `queens` 对不上 | 撇号**直接删除**而非换空格 |
| 模型路径被当成 HF repo id | `configs/agent.yaml` 写的是相对路径 | 用 `path_of()` 解析到 exp 根目录 |
| **记忆片段全是「本轮提问 + 上一轮提问的回复」** | MultiWOZ 原始数据一轮内**先 system 后 user**，`sys[t]` 回复的是 `usr[t-1]` | `prepare_multiwoz.py` 改用 `sys[t+1]`；三条探针确认（postcode 落 `sys[t+1]` 266/309、turn 0 的 `sys` 1000/1000 为 none） |
| 文档写「47.9% 正例可删」 | necessity 是逐轮判据，同一值被两轮提到时两轮都 necessity=0，但同时删就丢 | 改用集合覆盖上界（49.2% 轮次、33.3% 正例）；删光 necessity=0 实测仍丢 12.93% |
| 消融四档参数量完全相同 | `num_layers` 写进 `cfg["train"]`，而 `build_model` 读 `cfg["model"]` | 网格改三元组 `(段, 取值)` + 自检「参数量重复即 raise」 |
| 位置基线凭空赢了 | 主对话轮次全挤在开头，干扰轮均匀铺开 | 主对话轮等距铺开；干扰轮只从 `turn_idx ∈ hist_idx` 抽 |
| query 取最后一轮全是空话 | 1000/1000 段对话以 `none`/客套话结尾 | `is_substantive()` 判 `delta` 非空 + `last_substantive()` 从后往前找 |
| 0.5B 判必要性全退化 | 四种 yes/no 写法正例率 0.033 / 0.100 / 1.000 / 0.983 | 改用两个规则判据 J2/J3 做模拟 κ；JGA 改用填空式抽取 |
| 数字问答 0.5B 只有 0.20 | 问句自带数字，模型回显问句 | 改填空式模板（30 个模板一律不含数字），命中率 0.52 |
| `python` 命令不存在 | 容器只有 `python3` | 所有文档与脚本命令统一 `python3` |

---

## 8. 下一步

1. ~~MultiWOZ 2.2 下载与解析~~ **已完成**，见 [`data/README.md`](data/README.md)。
2. ~~`memory` / `scorer` / `eval`（L1）~~ **已完成**，见 [`docs/L1实验结果.md`](docs/L1实验结果.md)。
3. ~~长历史构造 + needle 注入，L 扫描（128/256/512）~~ **已完成**。
4. ~~三基线 + 对照基线、消融、预算扫描~~ **已完成**。
5. ~~L2 任务完成率主表（AR、token 消耗、任务完成率）~~ **已完成**（API backbone），
   见 [`docs/L2任务完成率.md`](docs/L2任务完成率.md)。90 场景 / 180 槽位 / 1685 秒。
   B2 slot_acc **0.3778**、场景成功率 0.1667、ctx 1,489 token；**B2−B1 只 +0.0222（p=0.6964，不显著）**，
   唯一显著赢的是位置基线 R0（+0.1667，p<0.0001）。ORACLE 只用 **216 token** 就到 **0.8444**——
   中间 46.6 个百分点是「区分对话归属」这件事的价值。
6. ~~反事实删除校验~~ **已完成**，见 [`docs/反事实删除校验.md`](docs/反事实删除校验.md)。
   结论：标签方向正确（同分母配对下删正例丢失率是删负例的 **10.56 倍**），但标签标的是「引入」
   而非「必要」：集合覆盖显示 **49.2%** 的轮次可删（正例中 **33.3%**）。
7. **真人抽检 60 条 + Cohen's κ**：表在 [`docs/标签抽检表.csv`](docs/标签抽检表.csv)。模拟值已填 50 条、
   10 条判据分歧留空，见 [`docs/标签抽检.md`](docs/标签抽检.md)。**模拟 κ ≠ 人工 κ**。
8. ~~用 necessity 作辅助标签试一版~~ → **已做，CR 无显著提升**（0.7778 vs 0.7667，配对 p=0.896，90 场景里 75 个持平），结论是不改标签。详见 [`docs/反事实删除校验.md`](docs/反事实删除校验.md) §8。
9. ~~L2 JGA 口径校验~~ **已完成（配对口径）**，见 [`docs/L2_JGA校验.md`](docs/L2_JGA校验.md)。
   **0.5B 配对口径**：可答子集 152 条上，删正例答对率 100%→**26.3%**，删负例 100%→**92.8%**；
   对话级差值 **−0.3083**，CI[−0.3917,−0.2250]，p<0.001。与覆盖率口径（10.56 倍）方向一致。
   **换真 backbone 后（`exp/runs/20261124_081620_jga_api_full/`）**：`full` **0.9174** / `drop_pos` **0.2477** /
   `drop_neg` 0.9204，差值扩大到 **−0.6755**（p<0.001）；`full − drop_neg` 从 −0.075（显著）变成
   **−0.0082（p=0.49 不显著）**——「删噪声有益」只在弱模型上成立。

### L1 已给出的结论（决定后面怎么写）

- **打分网络成立**：跨长度 CR **0.7667**，纯检索 0.4778，位置基线 0.2778。但「历史越长优势越大」**不成立**：
  B2/B1 比值 1.43× → 1.92× → 1.60× 非单调，绝对差 0.300 → 0.367 → 0.200 也非单调。**L=512 的 CR 0.5333 是最明显短板**。
- **两阶段（检索 top-50 → 重排）被检索召回卡死**：needle 进 top-50 的比例在 L=128/256/512 分别是
  0.678 / 0.456 / 0.344，B2R 的 CR 与之一一吻合（0.633 / 0.456 / 0.311），上界锁死。
  **B2 走全库稠密打分**（8,372 段 batch 推理 **0.014 秒**，成本可忽略）。这条要写进报告的"局限"。
- **浅层确实够**：1 层（99,585 参数）AUC 0.9352，4 层 0.9339。题目写"浅层全连接"不是妥协。
- **超参都不敏感**：lr 与 weight_decay 的 AUC 跨度（0.0013）与种子标准差（0.0007）同量级；
  weight_decay 四个取值小数点后四位完全相同——早停在 best ep5 先触发，L2 没机会起作用。
- **低预算下打分器反而不如余弦**：B=256 时 L=256（0.100 vs 0.125）、L=512（0.050 vs 0.075）两格 B2 输给 B1。
  预算只够放 8–9 个片段时，打分器学的「整体重要性」不如余弦的「直接对齐 query」。
- **L2 只需跑 300 条**：L1 已经把"方法行不行"回答完了，L2 只用来把 CR 换算成任务完成率。

### L2 配对校验已给出的结论

- **标签方向再次被独立验证**：覆盖率口径（字符串）说删正例丢 **26.8%**、删负例丢 **3.5%**；
  JGA 口径（LLM 答题）说删正例答对率掉到 **26.3%**、删负例只掉到 **92.8%**。两个口径测的层面不同，结论一致。
- **0.5B 只能做配对比较**：`full` 条件只有 **0.589**，绝对 JGA 不可信。
  **已换真 backbone（DeepSeek-V4.1-Flash）重跑**：`full` **0.9174**、可答率 0.9174、
  对话级 JGA 0.8270，`drop_pos − drop_neg` 从 −0.3083 扩大到 **−0.6755**（p<0.001）。
  1234 秒跑完 5328 次生成（并发 3）。产物 `exp/runs/20261124_081620_jga_api_full/`。
- **prompt 影响 2.6 倍**：填空式 0.52 vs 问答式 0.20。问答式的问句自带数字，0.5B 会回显问句本身。
  `agent/prompt.py` 的 30 个模板一律不含数字。
- **「删噪声有益」在真 backbone 上不成立**（已验证）：0.5B 时 `full − drop_neg` = −0.075
  （p=0.0336 显著），API 上 **−0.0082（p=0.49 不显著）**。强模型不容易被噪声干扰，
  删冗余对它既无益也无害。**不能拿 0.5B 的数字当普适结论。**
- **标签可靠性随槽位类型变化（扩样 607 段 / 1332 条，规模 5.2 倍）**：一次性信息（名字/类型/星级，
  n=295）删掉重要轮后答对率只剩 **0.112**（`drop_pos/full`=**0.16**）；可复述信息（区域/食物/价格/日期/人数/时间，
  n=1001）还剩 **0.223**（0.37）。**2.3 倍差**。机制：`anatolia` 只出现一次，`centre` 反复出现。
  → 标签在一次性信息上可靠，在可复述信息上会高估必要性。产物 `exp/runs/20261124_061833_jga_wide3/`。
  换成 API 后差距收窄到 **1.45 倍**（0.20 vs 0.29），方向不变。

### L2 任务完成率已给出的结论

- **CR 高不等于答对率高**：B2 的 CR 0.7667 是 B0 的 4.6 倍，但 slot_acc 只从 0.3167 到 0.3778。
  needle 在上下文里、和模型能答对，中间隔着一道鸿沟。
- **鸿沟来自「跨对话的冲突干扰」**：30 个 L=128 场景里 33 个（场景,槽位）组合的选中上下文含
  「同槽位、不同值」的干扰轮（gold=`centre` 答 `north`、gold=`0` 答 `4`）。
  **纯文本打分器看不到对话边界**，这是它的固有盲区。
- **B0FULL（7,277 token）只有 0.3222，比 B2（1,489 token）还低**，比 ORACLE（216 token，0.8444）
  低 52 个百分点。**不滤上下文不是「信息更全」，是「更容易答错」**——这是记忆过滤最直接的论据。
- **B2 相对 B1 的优势不显著**（+0.0222，p=0.6964）：180 条样本功效不够。要判定需要 500+ 场景。
