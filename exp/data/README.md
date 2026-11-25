# 数据说明：MultiWOZ 2.2

## 来源

- 原始数据集：**MultiWOZ 2.2**（Zang et al., 2020），官方仓库 `budzianowski/multiwoz`。
- 实际使用的镜像：HuggingFace `pietrolesci/multiwoz_all_versions`（parquet，逐轮对齐，含 2.0/2.1/2.2/2.3 四套状态）。
- 为什么不用官方仓库：本机访问 `github.com` 约 80 KB/s，拉 10,438 个 JSON 文件不现实；HF 镜像 10 MB 秒级下完。

> 该镜像比官方少 1 段 train 对话（8437 vs 8438），应为镜像侧过滤，不影响实验；正式报告里注明数据版本与来源即可。

## 文件

```
data/
├── raw/multiwoz22/          # 原始 parquet，不要改
│   ├── train.parquet        # 7.5 MB
│   ├── dev.parquet          # 1.0 MB
│   └── test.parquet         # 1.0 MB
└── processed/multiwoz22/    # 解析产物
    ├── train.jsonl          # 34 MB
    ├── dev.jsonl            # 4.5 MB
    ├── test.jsonl           # 4.5 MB
    ├── sample.jsonl         # 3 条样例，给人看
    └── stats.json           # 统计
```

重建与校验：

```bash
source exp/env.sh
python exp/scripts/download_data.py     # 下 parquet（已存在则跳过）
python exp/scripts/prepare_multiwoz.py  # 转 JSONL + 打标签
python exp/scripts/verify_labels.py     # 标签校验，5 项（三个 split 均通过）
```

## 一行一条：turn 级 schema

每行是**一轮**（user 一句 + 它引出的 system 回复），这是全项目统一的记忆粒度（根 README §3.2）。

> ⚠️ **system 已做错位修正。** 原始 parquet 沿用官方 MultiWOZ 的约定，一轮内**先 system 后 user**，
> 所以 `sys_utt[t]` 回复的是 `usr_utt[t-1]`。三条独立探针确认（test split）：
> 用户问 postcode 时邮编落在 `sys[t+1]` 占 86.1%、落在 `sys[t]` 仅 2.3%；用户提 booking 时确认落在
> `sys[t+1]` 381 次 vs `sys[t]` 35 次；`sys` 以问号结尾时短答落在 `usr[t+1]` 1015 次 vs `usr[t]` 514 次；
> 而 turn 0 的 `sys_utt` 在 1000/1000 段对话里都是 `none`（首轮没有"上一轮的回复"）。
> 状态标注是对齐 user 的（delta 值出现在 `usr[t]` 占 71.8%，`usr[t-1]` 仅 2.5%），所以正确轮次是
> `turn t = (usr_utt[t], sys_utt[t+1], states_22[t])`。`system_raw` 保留原始同轮 `sys_utt[t]` 备查。

| 字段 | 类型 | 含义 |
|---|---|---|
| `dialogue_id` | str | 对话 ID，如 `MUL0001.json` |
| `turn_idx` | int | 对话内轮次，从 0 开始 |
| `services` | list[str] | 涉及领域（restaurant / hotel / attraction / taxi / train / hospital / police） |
| `active_intents` | list[str] | 当前意图 |
| `user` | str | 本轮用户话语 |
| `system` | str | **下一轮**的 system 回复（即对本轮 user 的回答，见上方说明） |
| `system_raw` | str | 原始同轮 `sys_utt[t]`，仅供核对，不要用于训练/评测 |
| `state` | dict | **累积**对话状态 `{slot: value}`（MultiWOZ 2.2 belief state） |
| `delta` | dict | 相对上一轮新增/更新的 slot |
| `n_valid` | int | `delta` 中在对话结束时**仍然有效**的 slot 个数 |
| `label` | 0/1 | `n_valid > 0` |
| `dialogue_has_state` | bool | 整段对话是否有状态标注；False 时 label 全 0 属**标注缺失**，训练应过滤 |

关键点：`state` 是**逐轮累积**的，不是每轮独立的。t3 把 `hotel-stay` 从 3 改成 2，`delta` 就是 `{"hotel-stay": "2"}`，直接对应「新增或更新」。

> 标签的完整讲解、实例与校验结果见 [`标签说明.md`](标签说明.md)。

## 标签定义（根 README §4.3）

> 第 t 轮为 important ⟺ 相对 t-1 轮新增/更新了某个 slot-value，**且该值在对话结束时仍然有效**。

第二个条件是本组加的，v1 只看「有变化」，会把后来被覆盖的废弃值也标成重要。实测被覆盖的轮：train 2,104 轮（占 3.2%）。

## 实测统计

| split | 对话 | 轮数 | 正例率 | 有状态变化 | 被覆盖 |
|---|---|---|---|---|---|
| train | 8,437 | 65,213 | 53.8% | 57.0% | 2,104 |
| dev | 1,000 | 8,374 | 56.8% | 59.2% | 200 |
| test | 1,000 | 8,372 | 56.6% | 59.2% | 219 |

- 平均 7.7–8.4 轮/对话（这是**用户轮**数，总轮次约翻倍）。
- train 有 520 段对话完全没有状态标注（2,987 轮），`dialogue_has_state=false`；剔除后正例率 55.6%。
- 正例率 **55.6%**，明显高于根 README 原先预估的 25–40%。含义：**不能用 `pos_weight=负/正` 去补偿，反而要压正例**，否则模型会倾向于全判 1。

## ⚠️ 一个必须处理的问题：标签与位置强相关

正例率随轮次单调下降：

```
turn 0: 85.0%   turn 3: 61.2%   turn 6: 39.2%   turn 9: 18.6%
turn 1: 77.2%   turn 4: 55.8%   turn 7: 28.8%   turn 10: 16.0%
turn 2: 67.0%   turn 5: 50.0%   turn 8: 21.5%
```

后果：一个**零成本的「越靠前越重要」启发式**就能拿到：

| 基线 | AUC |
|---|---|
| `-turn_idx`（绝对位置） | **0.739** |
| `-rel_pos`（对话内相对位置） | **0.803** |
| 规则「前 3 轮」 | 0.659 |

也就是说，如果打分网络只学会「早=重要」，AUC 能到 0.80，但完全没读内容。根 README §5.2 原来只防了「越新越重要」（recency），实测方向**恰好相反**——是 primacy（首因）偏置。

处理办法（需组会定，按代价从低到高）：

1. **加位置基线 R0**：把「按轮次从早到晚取」作为一条对照，所有内容模型必须显著超过 0.803。
2. **训练时按位置重加权**：让每个位置的正负比接近 1:1，逼模型读内容。
3. **换标签**：改为「该 slot 在后续轮被用户**引用/请求**过」，比「最终状态仍有效」更贴近「记忆有没有用」。
4. **在长历史里随机化 needle 位置**（根 README §4.2）：评测阶段位置不提供信息，模型只能靠内容。

第 4 条本来就在计划里，现在看是**必须做**，不是可选项。

## 引用

```
@inproceedings{zang-etal-2020-multiwoz,
  title     = {MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines},
  author    = {Zang, Xiaoxue and Rastogi, Abhinav and Sunkara, Srinivas and Gupta, Raghav and Zhang, Kai and Chen, Jindong},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI},
  year      = {2020}
}
```
