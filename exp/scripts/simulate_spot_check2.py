#!/usr/bin/env python3
"""模拟人工抽检（第二版）：用两个能工作的独立判据算 Cohen's κ。

## 第一版为什么废了

`simulate_spot_check.py` 的 J1 想让 0.5B 直接读对话答 yes/no「这一轮对回答
query 是否必要」。实测**四种写法全部退化**（`runs/20261124_032705_judge_probe/`）：

| 写法 | 正例率 | κ(标签) |
|---|---|---|
| 直接问 yes/no | 0.033（58/60 全答 no） | 0.067 |
| few-shot 一正一负 | 0.100 | −0.067 |
| 抽取式「填不出就 NONE」 | 1.000（全填） | 0.000 |
| A/B 选项 | 0.983 | 0.033 |

模型没有判断力，只是被句法形式牵着走。**0.5B 不能做语义必要性判断**，
这也是为什么前面的 JGA 校验不用 yes/no，而用「填空能不能答对」这种
任务型探针（with-needle 0.52 vs no-needle 0.04）。

## 这一版用什么

两个都基于 state 数据，但**规则与标签不同**，且都实测有区分度：

**J2 — 末次提及**：存在某个最终有效槽位值，本轮提到它、之后不再提到。
    标签问「引入」（delta 非空 + 结束仍有效），J2 问「此后不可替代」。
    允许之前出现过，所以口径更宽。

**J3 — 唯一支撑（necessity）**：存在某个最终有效槽位值，全对话只有本轮提到它。
    比 J2 严格（J2 允许多轮提到、只要本轮是最后一次）。

两者夹住标签：标签 ⊂ 广义必要（J2），J3 ⊂ 标签。
κ(J2, J3) 是「两个判据之间的一致性」，用它模拟「两个标注者之间的一致性」。

## 这仍然不是人工校验

真 κ 要人。这里给的是模拟值：证明「标签落在两种独立自动化判据之间且与之
一致」，不是「标签与人一致」。CSV 的备注列会写明来源。

用法：
    source exp/env.sh
    python exp/scripts/simulate_spot_check2.py --n 60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import dump_json, get_logger, path_of, set_seed   # noqa: E402
from src.memory.long_history import last_substantive              # noqa: E402
from src.memory.slice import group_by_dialogue, load_segments     # noqa: E402
from simulate_spot_check import (build_samples, cohen_kappa,       # noqa: E402
                                 judge_last_mention, kappa_ci)
from verify_counterfactual import build_pairs                      # noqa: E402

log = get_logger("spot2")


def judge_unique_support(support: dict[str, list[int]], i: int) -> int:
    """J3：本轮是某个最终有效槽位值的唯一支撑轮。"""
    return int(any(sup == [i] for sup in support.values()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tag", default="spot_check2")
    args = ap.parse_args()

    set_seed(0)
    t0 = time.time()

    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    pairs = build_pairs(by_d)
    samples = build_samples(pairs, args.n)
    log.info("抽检 %d 条（正负例各半，与 CSV 同一批）", len(samples))

    rows = []
    for smp in samples:
        p, i, seg = smp["pair"], smp["i"], smp["seg"]
        gold = p["hist"][-1].state
        rows.append({
            "dialogue_id": p["dialogue_id"],
            "turn_idx": seg.turn_idx,
            "label": int(seg.label),
            "J2_last_mention": judge_last_mention(p["hist"], gold, i),
            "J3_unique_support": judge_unique_support(p["support"], i),
            "n_support_slots": sum(1 for sup in p["support"].values() if sup == [i]),
            "query": last_substantive(by_d[p["dialogue_id"]]).user,
            "user": seg.user,
            "system": seg.system,
        })

    lab = [r["label"] for r in rows]
    j2s = [r["J2_last_mention"] for r in rows]
    j3s = [r["J3_unique_support"] for r in rows]

    res = {
        "split": args.split,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "note": "模拟值，非人工校验；J1(LLM yes/no) 已实测退化，不纳入",
        "label_vs_J2": cohen_kappa(lab, j2s),
        "label_vs_J3": cohen_kappa(lab, j3s),
        "J2_vs_J3": cohen_kappa(j2s, j3s),
        "rows": rows,
    }
    for k, (a, b) in {
        "label_vs_J2": (lab, j2s),
        "label_vs_J3": (lab, j3s),
        "J2_vs_J3": (j2s, j3s),
    }.items():
        res[k]["kappa_ci"] = kappa_ci(a, b)

    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(res, run_dir / "spot_check2.json")

    # 回写 CSV：J2 与 J3 一致时给结论；分歧留空
    csv_path = path_of("docs") / "标签抽检表.csv"
    filled = disagree = 0
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig") as f:
            cr = list(csv.DictReader(f))
        key = {(r["dialogue_id"], int(r["turn_idx"])): r for r in cr}
        col = "你的判断(1=该轮信息对回答query必要, 0=不必要)"
        for r in rows:
            c = key.get((r["dialogue_id"], r["turn_idx"]))
            if not c:
                continue
            if r["J2_last_mention"] == r["J3_unique_support"]:
                c[col] = str(r["J2_last_mention"])
                c["备注"] = f"模拟：J2/J3 一致（J2=末次提及 J3=唯一支撑）"
                filled += 1
            else:
                c[col] = ""
                c["备注"] = (f"模拟：判据分歧 J2={r['J2_last_mention']} "
                            f"J3={r['J3_unique_support']}（留空待人工）")
                disagree += 1
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(cr[0].keys()))
            w.writeheader()
            w.writerows(cr)
        res["csv_filled"] = filled
        res["csv_disagree"] = disagree
        dump_json(res, run_dir / "spot_check2.json")
        log.info("CSV：%d 条填结论、%d 条分歧留空", filled, disagree)

    for k in ("label_vs_J2", "label_vs_J3", "J2_vs_J3"):
        m = res[k]
        log.info("κ(%s) = %.4f  一致性 %.4f  CI %s  (A正例率 %.2f / B正例率 %.2f)",
                 k, m["kappa"], m["agreement"], m["kappa_ci"],
                 m["pos_rate_a"], m["pos_rate_b"])
    log.info("完成，用时 %.1fs → %s", res["seconds"], run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
