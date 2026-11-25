#!/usr/bin/env python3
"""把 MultiWOZ 2.2 parquet 转成逐轮 JSONL，并生成重要性标签。

    source exp/env.sh
    python exp/scripts/prepare_multiwoz.py

标签定义（根 README §4.3）：
  第 t 轮为 important ⟺ 它相对 t-1 轮新增/更新了某个 slot-value，
                        且该值在对话结束时仍然有效（未被后续轮覆盖）。

## system 语句错位一轮（必须修正）

`pietrolesci/multiwoz_all_versions` 沿用了官方 MultiWOZ 的存储约定：一轮内
**先 system 后 user**，所以 `sys_utt[t]` 回复的是 `usr_utt[t-1]`，不是同一轮的
user。三条独立探针确认（test split）：

| 探针 | 落在 sys[t] | 落在 sys[t+1] |
|---|---|---|
| 用户问 postcode，回复落在哪 | 1 | **59** |
| 用户提 booking，确认落在哪 | 35 | **381** |
| sys 以问号结尾，短答落在哪 | 514 | **1015** |

决定性证据：**turn 0 的 sys_utt 在 1000/1000 段对话里都是 `none`**——首轮没有
"上一轮的回复"。

而状态标注是对齐 user 的（delta 值出现在 `usr_utt[t]` 占 71.8%，出现在
`usr_utt[t-1]` 仅 2.5%）。所以正确的轮次定义是：

    turn t = (usr_utt[t], sys_utt[t+1], states_22[t])

这里把 `system` 取成 `sys_utt[t+1]`，使每一段记忆自身是「用户提问 + 系统回复」
的完整一对。原始同轮 `sys_utt[t]` 保留在 `system_raw` 字段备查。

输出：
    data/processed/multiwoz22/{train,dev,test}.jsonl
    data/processed/multiwoz22/stats.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

EXP_ROOT = Path(__file__).resolve().parents[1]
RAW = EXP_ROOT / "data" / "raw" / "multiwoz22"
OUT = EXP_ROOT / "data" / "processed" / "multiwoz22"
SPLITS = ("train", "dev", "test")


def norm_state(raw) -> dict[str, str]:
    """{slot: [value, ...]} → {slot: value}，丢掉空值；None 视为无状态。"""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        try:
            if len(v) == 0:
                continue
            out[k] = str(v[0])
        except TypeError:
            out[k] = str(v)
    return out


def process_split(split: str) -> tuple[list[dict], dict]:
    df = pq.read_table(RAW / f"{split}.parquet").to_pandas()
    df = df.sort_values(["dialogue_id", "turn_id"], kind="mergesort")

    rows: list[dict] = []
    n_overwritten = 0
    n_empty_dialogue = 0

    for did, g in df.groupby("dialogue_id", sort=False):
        # 先按 turn_id 收集，再统一做 system 的错位修正（见模块 docstring）。
        raw_turns = []
        prev: dict[str, str] = {}
        for r in g.itertuples():
            cur = norm_state(r.states_22)
            delta = {k: v for k, v in cur.items() if prev.get(k) != v}
            raw_turns.append({
                "dialogue_id": did,
                "turn_idx": int(r.turn_id),
                "services": list(r.services) if r.services is not None else [],
                "active_intents": list(r.active_intents) if r.active_intents is not None else [],
                "user": r.usr_utt or "",
                "system_raw": r.sys_utt or "",
                "state": cur,
                "delta": delta,
            })
            prev = cur

        # system 错位修正：turn t 的 system 是 sys_utt[t+1]（回复本轮 user）。
        # 最后一轮没有下一条 system，用空串，同时该轮 user 常为 "none"（对话已结束）。
        for i, t in enumerate(raw_turns):
            nxt = raw_turns[i + 1]["system_raw"] if i + 1 < len(raw_turns) else ""
            t["system"] = nxt

        turns = raw_turns

        # 对话结束时的状态：取最后一个非空 state（末尾可能有未标注轮）
        final: dict[str, str] = {}
        for t in turns:
            if t["state"]:
                final = t["state"]
        if not final:
            n_empty_dialogue += 1

        for t in turns:
            t["n_valid"] = sum(1 for k, v in t["delta"].items() if final.get(k) == v)
            t["label"] = int(t["n_valid"] > 0)
            # 整段对话没有任何状态标注时，全 0 标签是"标注缺失"而非"全不重要"
            t["dialogue_has_state"] = bool(final)
            if t["delta"] and t["label"] == 0:
                n_overwritten += 1
        rows.extend(turns)

    labels = [r["label"] for r in rows]
    stats = {
        "split": split,
        "dialogues": int(df.dialogue_id.nunique()),
        "turns": len(rows),
        "avg_turns_per_dialogue": round(len(rows) / max(df.dialogue_id.nunique(), 1), 2),
        "positive_rate": round(sum(labels) / max(len(labels), 1), 4),
        "turns_with_delta": int(sum(1 for r in rows if r["delta"])),
        "overwritten_turns": n_overwritten,
        "dialogues_without_state": n_empty_dialogue,
        "domains": dict(Counter(d for r in rows for d in r["services"]).most_common()),
        "n_valid_dist": dict(sorted(Counter(r["n_valid"] for r in rows).items())),
    }
    return rows, stats


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for split in SPLITS:
        rows, stats = process_split(split)
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_stats[split] = stats
        print(f"{split:<5} {stats['dialogues']:>5} 对话 / {stats['turns']:>6} 轮  "
              f"正例 {stats['positive_rate']:.1%}  "
              f"有变化 {stats['turns_with_delta']/max(stats['turns'],1):.1%}  "
              f"→ {path.name}")

    (OUT / "stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 抽 3 条给人看
    with (OUT / "train.jsonl").open(encoding="utf-8") as f:
        sample = [json.loads(next(f)) for _ in range(3)]
    (OUT / "sample.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in sample), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
