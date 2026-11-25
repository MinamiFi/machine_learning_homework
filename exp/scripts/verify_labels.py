#!/usr/bin/env python3
"""校验 belief state 差分标签的正确性。

    source exp/env.sh
    python exp/scripts/verify_labels.py            # 默认查 train
    python exp/scripts/verify_labels.py --split test

跑五组检查，任何一组不通过就说明标签逻辑有问题：

  C1 一致性   label 必须严格等于 (n_valid > 0)
  C2 差分正确 delta 必须等于 state 相对上一轮的差集（逐槽位核对）
  C3 回退统计 槽位值 A→B→A 的次数（数据噪声，非错误）
  C4 空值处理 空状态轮不得夹在中间（否则会把 prev 清空）
  C5 覆盖统计 报告「真覆盖」与「槽位消失」的占比

退出码：0 全通过，1 有失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
PROC = EXP_ROOT / "data" / "processed" / "multiwoz22"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")


def load(split: str) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    with (PROC / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by[r["dialogue_id"]].append(r)
    for g in by.values():
        g.sort(key=lambda r: r["turn_idx"])
    return by


def c1_consistency(by) -> None:
    bad = 0
    for g in by.values():
        for r in g:
            if r["label"] != int(r["n_valid"] > 0):
                bad += 1
            if r["n_valid"] > len(r["delta"]):
                bad += 1
    total = sum(len(g) for g in by.values())
    check("C1 label 与 n_valid 一致", bad == 0, f"{total} 轮，异常 {bad}")


def c2_delta(by) -> None:
    """delta 必须 == {k: v in state if prev.get(k) != v}，空状态轮跳过。"""
    bad = 0
    for g in by.values():
        prev: dict[str, str] = {}
        for r in g:
            if not r["state"] and not r["delta"]:
                continue          # 标注缺失轮，不参与比对
            expect = {k: v for k, v in r["state"].items() if prev.get(k) != v}
            if expect != r["delta"]:
                bad += 1
            prev = r["state"]
    total = sum(1 for g in by.values() for r in g if r["state"] or r["delta"])
    check("C2 delta 等于状态差集", bad == 0, f"{total} 轮，异常 {bad}")


def c3_reverts(by) -> None:
    """同一槽位出现 A→B→A 的回退，说明标注不稳定（用户确实改主意也可能）。

    这不是错误，是数据本身的噪声，只做统计不做断言。
    """
    reverts = 0
    examples: list[str] = []
    for did, g in by.items():
        hist: dict[str, list[str]] = defaultdict(list)
        for r in g:
            for k, v in r["state"].items():
                hist[k].append(v)
        for k, vs in hist.items():
            seq: list[str] = []
            for v in vs:
                if not seq or seq[-1] != v:
                    seq.append(v)
            if len(seq) >= 3 and seq[0] == seq[-1]:
                reverts += 1
                if len(examples) < 3:
                    examples.append(f"{did}:{k}={seq}")
    check("C3 槽位值回退统计", True,
          f"A→B→A 共 {reverts} 次（噪声，非错误）")
    for e in examples:
        print(f"        例：{e}")


def c4_empty_states(by) -> None:
    """空状态轮在对话中的位置：开头/结尾无害，夹在中间会污染 prev。"""
    lead = all_empty = mid = 0
    for g in by.values():
        states = [bool(r["state"]) for r in g]
        has_any = any(states)
        for i, has in enumerate(states):
            if has:
                continue
            if not has_any:
                all_empty += 1          # 整段对话无标注，不参与训练
            elif any(states[:i]):
                mid += 1                # 前后都有状态：会污染 prev
            else:
                lead += 1               # 开头空，等价于从零开始，无害
    check("C4 空状态不夹在中间", mid == 0,
          f"开头 {lead} / 全空对话 {all_empty} / 中间 {mid}")


def c5_overwrite(by) -> None:
    """按 slot 统计：结束时仍有效 / 被改值 / 槽位消失。"""
    still = replaced = gone = 0
    for g in by.values():
        final = next((r["state"] for r in reversed(g) if r["state"]), None)
        if final is None:
            continue
        for r in g:
            for k, v in r["delta"].items():
                if k not in final:
                    gone += 1
                elif final[k] == v:
                    still += 1
                else:
                    replaced += 1
    tot = still + replaced + gone
    check("C5 slot 级覆盖统计", tot > 0,
          f"有效 {still} ({still/tot:.1%}) / 改值 {replaced} ({replaced/tot:.1%}) "
          f"/ 消失 {gone} ({gone/tot:.2%})")
    print(f"        → 真·覆盖（同槽位换值）只占 {replaced/tot:.1%}，"
          f"说明「仍有效」这条条件主要剔除的是被改写的旧值")


def position_bias(by) -> None:
    """标签与轮次的相关性——这是最强的零成本启发式，必须报出来。"""
    from sklearn.metrics import roc_auc_score

    rows = [r for g in by.values() for r in g if r.get("dialogue_has_state", True)]
    if not rows:
        return
    y = [r["label"] for r in rows]
    if len(set(y)) < 2:
        print("        （标签全为同一值，跳过位置偏置）")
        return
    t = [r["turn_idx"] for r in rows]
    maxlen = {r["dialogue_id"]: max(x["turn_idx"] for x in g) for r in rows
              for g in [by[r["dialogue_id"]]]}
    rel = [r["turn_idx"] / max(maxlen[r["dialogue_id"]], 1) for r in rows]
    print(f"        位置基线 AUC：-turn_idx={roc_auc_score(y, [-x for x in t]):.4f}  "
          f"-rel_pos={roc_auc_score(y, [-x for x in rel]):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = ap.parse_args()

    print("=" * 72)
    print(f"belief state 差分标签校验  split={args.split}")
    print("=" * 72)

    by = load(args.split)
    n_turn = sum(len(g) for g in by.values())
    print(f"  {len(by)} 段对话 / {n_turn} 轮\n")

    c1_consistency(by)
    c2_delta(by)
    c3_reverts(by)
    c4_empty_states(by)
    c5_overwrite(by)
    print()
    position_bias(by)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 72)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过")
    if failed:
        print("失败：" + "；".join(failed))
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
