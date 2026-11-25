# src/ 四个模块：设计与接口

> 这是设计文档，代码还没写。所有数字都来自本机实测（2026-11-18/19）。

## 0. 先说为什么这样切

实测瓶颈决定架构：

| 环节 | 实测速度 | 含义 |
|---|---|---|
| 句向量编码 | **185 轮/秒** | 6.5 万轮全量编码 5.9 分钟，可接受 |
| MLP 训练 | 65k 样本 × 5 epoch ≈ **6 秒** | 几乎免费，消融可以随便跑 |
| LLM prefill | **173 tok/s** | B=2048 约 11.8 秒 |
| LLM decode | **5.9 tok/s** | 生成 30 token 约 5.1 秒 |
| **一次评测样本** | **≈ 17 秒** | 300 样本 × 6 组 = **8.5 小时** |

内存上限 4.29 GB，LLM 加载后占 3.24 GB，**无法并行**，只能串行。

所以：**LLM 是唯一昂贵的环节，其余全部免费。** 架构必须围绕这一点组织——凡是能不用 LLM 算的指标，绝不用 LLM。

推论：**把「选择」和「生成」彻底分开**。选择器（scorer）的优劣用 AUC / CR / NDCG 衡量，全部零 LLM 成本，可以跑满 6.5 万轮；只有「任务完成率」需要 LLM，只在小子集上跑。

---

## 1. `memory/` — 记忆库与检索

**职责**：把 JSONL 的轮次变成「可检索、可打分、可装进预算」的记忆片段。

### 输入输出

```
data/processed/multiwoz22/*.jsonl   (turn 级)
        │  slice()          切片 + 编码
        ▼
data/processed/memory/*.npz        (片段 + 384 维向量)
        │  build_long_history()   构造长历史 + 注入 needle
        ▼
data/processed/long_history/*.jsonl (L=128/256/512 三种长度)
        │  retrieve()        向量召回 top-M
        ▼
        │  select()          预算内选择
        ▼
   选中的片段 id 列表 + 实际 token 数
```

### 关键接口

```python
# memory/slice.py
def slice_turns(split: str) -> list[MemorySegment]:
    """一轮 = 一段。字段见 data/README.md。"""

def embed_segments(segs, model, batch_size=32) -> np.ndarray:
    """离线算好落盘，评测时不再编码。"""

# memory/long_history.py
def build_long_history(
    dialogue_id: str,
    target_len: int,          # 128 / 256 / 512
    needle_pos: float,        # 0.0–0.9，随机！见根 README §4.2
    seed: int,
) -> LongScenario:
    """单对话膨胀：保留原轮次顺序，插入同领域干扰轮，埋 needle 与诱饵。"""

# memory/retrieve.py
def retrieve(query_vec, bank_vecs, top_m=50) -> np.ndarray:
    """cosine top-M。M=50 是重排的候选池。"""

# memory/select.py
def select(
    scores: np.ndarray,
    tokens: np.ndarray,
    budget: int,
    reserve_recent: int = 2,   # 最近 2 轮保底
    mmr_lambda: float = 0.0,   # >0 时做去冗余
) -> list[int]:
    """按分数降序累加 token 直到 budget。返回选中下标。"""
```

### 为什么 `select()` 单独成一个函数

三组基线（B1/B2）和四条对照（R0/R1/R2/R3）**共用同一个选择逻辑**，区别只在 `scores` 从哪来：

| 组 | scores 来源 |
|---|---|
| B1 纯检索 | `retrieve()` 的 cosine 相似度 |
| B2 检索+打分 | `scorer.predict()` 的输出 |
| R0 位置基线 | `-turn_idx` |
| R1 recency | `+turn_idx` |
| R2 随机 | `np.random.rand()` |

这样「预算截断」「保底规则」对所有组完全一致，**唯一变量是打分**。这是实验公平性的代码级保证，也是根 README §5.1 那条「同一 prompt 模板、同一 B」的落地方式。

### 关键设计：token 预算用真实 tokenizer 算

不能用 `len(text.split())` 估算。实测单轮 39.8 token（中位 40，p90 62），用 Qwen tokenizer 精确计数。B=2048 约装 51 轮。

---

## 2. `scorer/` — 打分网络

**职责**：题目要求的浅层全连接网络，以及训练、推理、消融。

### 模型

```
输入 e_i ∈ R^384
  → LayerNorm
  → Linear(384, 256) → GELU → Dropout(0.1)
  → Linear(256, 256) → GELU → Dropout(0.1)
  → Linear(256, 1) → Sigmoid
参数量 165,377
```

已实测：65k 样本 5 epoch 训练 **6 秒**。

### 关键接口

```python
# scorer/model.py
class ImportanceScorer(nn.Module):
    def forward(self, e: Tensor) -> Tensor:   # (B,384) -> (B,1)

# scorer/dataset.py
def build_dataset(split, bank) -> Dataset:     # (embedding, label)
# 过滤 dialogue_has_state=False 的轮（标注缺失，全 0 不是"不重要"）

# scorer/train.py
def train(cfg, train_ds, val_ds) -> (model, history):
    """BCEWithLogitsLoss + AdamW + early stop on val AUC。
    日志写 logs/{run_id}.jsonl，含 step/loss/lr/val_auc/seed（交付物）。"""

# scorer/baselines.py
def score_position(segs) -> np.ndarray     # R0: -turn_idx
def score_recency(segs) -> np.ndarray      # R1: +turn_idx
def score_random(segs, seed) -> np.ndarray # R2
```

### 必须处理的三件事

**① 正例率 55.6%，`pos_weight` 不能用「负/正」。**
实测正例偏多，若按 `neg/pos ≈ 0.8` 加权会进一步推向全判 1。默认 `pos_weight=1.0`，需要时下调。

**② 位置偏置必须显式测量。**
标签正例率从 turn 0 的 85% 降到 turn 10 的 16%，`-rel_pos` 单独就有 **AUC 0.803**。所以：

- `train()` 每轮除了报 `val_auc`，**必须同时报位置基线的 AUC**；
- 若 `val_auc - position_auc < 0.03`，说明模型没读到内容，训练脚本直接打警告。

这是把「防止退化成启发式」写进代码，而不是只写在报告里。

**③ 损失可能不止 BCE。**
下游只关心 top-k 的**排序**，不关心绝对概率。保留 pairwise margin ranking loss 作为可选开关（`cfg.loss: bce | bce+rank`），消融时对比。

---

## 3. `agent/` — 对话智能体

**职责**：拼 prompt、调 LLM、缓存结果。**不做选择决策**（那是 memory 的事）。

### 关键接口

```python
# agent/prompt.py
def build_prompt(memories: list[str], recent: list[str], query: str) -> list[dict]:
    """固定三段：system(角色+记忆使用说明) → memory block(按时间序) → 最近轮+query。
    所有基线共用这一个模板，只换 memory block 的内容。"""

# agent/llm.py
class LLMClient(Protocol):
    def generate(self, messages, max_new_tokens) -> Response: ...

class LocalQwen(LLMClient):   # Qwen2.5-0.5B，CPU
class ApiClient(LLMClient):   # 任意 OpenAI 兼容端点

# agent/cache.py
class ResponseCache:
    """key = sha256(model + prompt + params)，落盘 JSONL。
    重跑不重复花钱，也保证可复现。"""

# agent/loop.py
def run_turn(agent, scenario, selector) -> TurnResult:
    """检索 → 打分 → 选择 → 拼 prompt → 生成 → 记录 token 与输出。"""
```

### 实际实现与设计的差异（L2 JGA 校验时补的）

设计里 `build_prompt(memories, recent, query)` 返回 `list[dict]`，实际写成
`build_prompt(ctx: str, slot: str) -> str`，原因：JGA 口径校验只需要
「给定上下文 + 填空模板」，不需要多轮 messages 结构。真正的多轮 agent
prompt 等 API backbone 到位后再加，届时保留这个函数作为单轮抽取的底层。

`agent/prompt.py` 里 `SLOT_BLANK` 的 30 个模板是实测调出来的：
问答式（"for how many people?"）在 0.5B 上只有 20%，**填空式 52%**，
因为问句里自带数字会让模型回显问句。模板本身必须不含数字。

### `agent/loop.py` 实际实现（L2 主循环补齐时写）

设计写 `run_turn(agent, scenario, selector) -> TurnResult`，实现保持这个签名，
但把逻辑放进 `MemoryAgent` 类，模块级 `run_turn()` 只做转发。理由：agent 要
持有 LLM、打分模型、缓存、预算，用类才不用传 6 个参数。

三个实现上的决定：

1. **`score_bank()` 复刻了 `eval/runners.make_selector` 的语义**，而不是 import 它。
   因为依赖方向是 `memory → scorer → agent → eval`，agent 不能反向依赖 eval。
   两处语义必须保持一致，改一处要同步改另一处——这是唯一的重复代码，
   已在两个 docstring 里互相点名。

2. **`full_context=True` 开关**。题目里的「原始全量上下文记忆」指不设预算全塞进去，
   而 `select()` 总是受 budget 约束，`make_selector("B0")` 得到的其实是
   「预算内最近优先」（等价 R1）。两者不是一回事：前者测不截断的代价（token 爆炸），
   后者测截断的代价。`TurnResult.meta["mode"]` 记 `"full"` / `"budget"`，
   L2 表里 B0FULL 与 B0 分行报。

3. **`n_new_tokens` 与 `prompt_tokens` 走 backbone 自己的计数**。
   `LocalQwen` 用 tokenizer 精确计数，`ApiClient` 用服务端返回的 `usage`。
   换 backbone 时 token 口径不会漂移，这是 token 消耗能进主表的前提。

### AR 基准为什么是 NEEDLE 而不是 B0FULL（`scripts/eval_l2.py`）

第一版拿 B0FULL（全量上下文）当「可答」基准，跑通后才发现两个问题：

1. **代价随 L 爆炸**。L=512 时 bank 约 18,655 token，prefill 173 tok/s →
   单次 112 秒（L=128 31 秒、L=256 58 秒），60 次就 67 分钟，而预算模式只要 12.4 秒。
2. **它回答的问题不对**。AR 要测「这个槽位从证据里能不能抽出来」，
   不需要 512 轮无关背景；全量上下文反而给模型塞了一堆干扰。

改成 **NEEDLE 基准**：只在 `needle_id` 那一轮上问同一个槽位。needle 是该槽位
信息的唯一来源，在它上面答对就等于「可答」。B0FULL 仍保留为
`--with-b0full` 显式开关（只在 L≤128 时建议打开），用于单独报「不截断的代价」。

配套约束：**所有选择器必须问同一批槽位**（`ask = sorted(askable_slots(gold))[:max_slots]`
提到循环外）。否则 AR 的 `(scenario_id, slot)` key 对不上，子集落空、AR 恒为 NaN。

### `agent/llm.py` 的 `build_llm()`

按 `configs/agent.yaml` 的 `backbone` 字段三选一：`local` / `api` / `dummy`。
**API key 只从环境变量读**（`api.api_key_env` 指定变量名，默认
`DEEPSEEK_API_KEY`），不写进配置文件、不写进代码、不接受命令行参数。
缺 key 时 `ApiClient.__init__` 直接抛 `ValueError` 并打印设置方法，
避免跑到一半才发现没配。

### 为什么 LLM 要抽象成 Protocol

三个原因：

1. **0.5B 只能打通管线，不能出主表**（`exp/README.md` §3）。正式实验必须能一键切到 API。
2. 换 backbone 时，`memory/` 和 `scorer/` 一行都不用改。
3. 测试时可以用 `DummyLLM`（固定返回），让 `eval/` 的单元测试不依赖 3.2 GB 内存。

### 缓存不是优化，是必需

一次全量评测 8.5 小时。如果中途崩了要重跑，没有缓存等于白干。而且 `temperature=0` 下结果确定，缓存命中率会很高（尤其 R2 随机过滤跑多个种子时，prompt 大量重复）。

---

## 4. `eval/` — 指标与实验调度

**职责**：算指标、跑基线、做消融、出表。

### 关键设计：两层评测

这是整个架构最重要的一个决定。

| 层 | 指标 | 要 LLM 吗 | 成本 | 样本量 |
|---|---|---|---|---|
| **L1 选择质量** | AUC、CR（needle 是否被选入）、NDCG@k、Recall@budget | ❌ 不要 | 秒级 | 6.5 万轮全量 |
| **L2 任务完成** | 任务完成率、AR、token 消耗 | ✅ 要 | 17 秒/样本 | 300 条子集 |

**CR（context recall）是这里的关键杠杆**：它只问「needle 那段有没有被选进上下文」，纯查 id 集合，**完全不需要 LLM**。

所以流程是：

```
1. L1 全量扫（秒级）
   → 6 组 × 3 个种子 × 512 轮场景，全部跑完
   → 立刻能回答：打分网络有没有赢过位置基线 0.803？
   → 如果没赢，说明标签或模型有问题，此时不要浪费 8 小时跑 L2

2. L2 小样本（小时级）
   → 只在 L1 筛出的有效配置上跑
   → 300 条 × 6 组 ≈ 8.5 小时，可过夜
```

**这个顺序能把「发现方法不行」的时间从 8 小时压到 1 分钟。** 对 2 核机器是决定性的。

### 关键接口

```python
# eval/metrics.py
def context_recall(selected_ids, needle_id) -> float    # L1，不要 LLM
def ndcg_at_k(scores, labels, k) -> float               # L1
def task_success(pred_state, gold_state) -> float       # L2
def slot_f1(pred, gold) -> float                        # L2
def token_cost(result) -> dict                          # L2

# eval/runners.py
def run_baseline(name: str, scenarios, llm=None, tier="L1") -> dict:
    """name ∈ {B0, B1, B2, R0, R1, R2, R3}"""

# eval/ablation.py
def grid(base_cfg) -> list[cfg]:
    """3 因素 × 4 取值，单因素扫描（根 README §5.4）"""

# eval/report.py
def to_markdown(results) -> str
```

### 输出约定

```
runs/{run_id}/
├── config.yaml       # 完整配置，可复现
├── L1_selection.json # 选择质量（秒级产物）
├── L2_generation.jsonl
├── metrics.json      # 汇总表
└── report.md         # 给人看的
```

`run_id` = `{timestamp}_{exp_name}_{seed}`，日志里每一行都带 `run_id`，方便对齐。

---

## 5. 模块依赖关系

```
        data/processed/multiwoz22/*.jsonl
                    │
              ┌─────▼─────┐
              │  memory   │  切片 / 长历史 / 检索 / 预算选择
              └─────┬─────┘
                    │ segments + 向量
              ┌─────▼─────┐
              │  scorer   │  MLP + 基线打分器
              └─────┬─────┘
                    │ scores
              ┌─────▼─────┐
              │  agent    │  prompt + LLM + 缓存
              └─────┬─────┘
                    │ 输出
              ┌─────▼─────┐
              │   eval    │  L1(无LLM) / L2(有LLM)
              └───────────┘
```

依赖是**单向**的：`memory → scorer → agent → eval`。没有反向依赖，所以：

- `eval/` 可以只导入前三个模块跑 L1，不加载 LLM（省 3.2 GB）；
- `memory/` 和 `scorer/` 可以单独单元测试，不需要模型。

---

## 6. 开发顺序（按「最早发现问题」排）

| 步 | 模块 | 产出 | 为什么这个顺序 |
|---|---|---|---|
| 1 | `memory/slice` + `embed` | 片段 + 向量 | 一切的基础，且实测只要 6 分钟 |
| 2 | `scorer` + `baselines` | AUC vs 0.803 | **最早回答「方法成不成立」** |
| 3 | `memory/long_history` | L=128/256/512 场景 | 位置随机化在这里落地 |
| 4 | `eval` L1 | CR / NDCG 表 | 仍不需要 LLM，秒级迭代 |
| 5 | `agent` + `eval` L2 | 任务完成率 | 最贵，放最后，且此时已有 L1 筛选 |
| 6 | `eval/ablation` | 消融网格 | 训练只要 6 秒，随时可跑 |

第 2 步是关键决策点：如果 MLP 打不过位置基线 0.803，就得回去改标签（根 README §5.6 预案），而不是继续往下写。**先花半天验证方法，比写完全部代码再发现不成立要好。**

---

## 7. 已知风险

| 风险 | 影响 | 应对 |
|---|---|---|
| LLM 评测 8.5 小时 | 迭代慢 | L1 优先；缓存；夜间跑 |
| 0.5B 能力不足 | L2 全 0，分不出差异 | L2 只做 sanity check，主结论靠 L1 + API |
| 位置偏置 | 模型退化成启发式 | 训练时监控 `val_auc - position_auc` |
| 4.29 GB 内存 | 不能并行 | 串行；L1 不加载 LLM |
| 长历史是合成的 | 被质疑不真实 | 报告声明 + 真实短对话 sanity check |
