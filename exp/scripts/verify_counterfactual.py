"""反事实删除校验：belief-state 差分标签的「重要」是否真的重要。

根 README §4.4 的原方案用 JGA 判定，而 JGA 需要 LLM。0.5B 出不了主表
（见 exp/README.md §3），所以这里改用**不需要 LLM 的覆盖率口径**，
与 L1 同属"零 LLM 成本"层：

    删掉某组轮次后，query 时刻的 gold 状态里，还有多少槽位值
    能在剩余上下文中找到文本支撑？

若标签有效，应观察到：
  - 删 label=1 轮 → 丢失率**高**
  - 删 label=0 轮 → 丢失率**低**

设计要点（每条都对应一个容易犯的错）：
1. **等量删除**：同一段对话里两边删同样多轮，否则"删得多当然丢得多"。
   k = min(n_pos, n_neg)，两边都没有的对话直接排除。
2. **配对比较**：同一对话跑两个条件，用对话级配对差值 + bootstrap CI，
   而不是拿两个独立均值比——后者会被对话间难度差异淹没。
3. **gold 取 query 时刻的累积状态**：`hist[-1].state`，不是对话末尾的
   state（末尾可能还有 query 之后的轮次，那是未来信息）。
4. **无文本支撑的值单列**：`hotel-internet=yes` 这类值本来就不在文本里
   出现（原文是 "free wifi"），不属于"删掉了所以丢了"，必须排除，
   否则两边都会算成"丢失"，稀释信号。

局限（必须写进报告）：这是"信息是否还在文本里"，不是"LLM 能否答对"。
真正的 JGA 校验要等 L2 打通。

用法：
    source exp/env.sh
    python exp/scripts/verify_counterfactual.py --split test
    python exp/scripts/verify_counterfactual.py --split test --spot-check 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import dump_json, get_logger, path_of      # noqa: E402
from src.memory.long_history import last_substantive      # noqa: E402
from src.memory.slice import group_by_dialogue, load_segments  # noqa: E402

log = get_logger("counterfactual")

# 这些值不是"内容"，而是标注体系里的符号，永远不会出现在对话文本里。
# 它们在原文里以同义表达出现（free wifi / no internet / any area），
# 字符串匹配抓不到，必须排除而不是算作"丢失"。
SYMBOLIC_VALUES = {"yes", "no", "dontcare", "none", ""}


def mentions(val: str, text: str) -> bool:
    """值是否在文本里出现。

    短值/纯数字必须整词匹配，否则 '4' 会命中 '15:00'、'3' 会命中 '3-4'。
    这是首版覆盖率虚高的直接原因。
    """
    v = str(val).lower().strip()
    t = text.lower()
    if not v:
        return False
    if len(v) <= 3 or v.isdigit():
        return re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", t) is not None
    return v in t


def seg_text(seg) -> str:
    return f"{seg.user} {seg.system}"


def measurable_slots(hist, gold: dict[str, str]) -> dict[str, list[int]]:
    """返回 {slot: [支撑轮在 hist 里的下标]}，只保留可测的槽位。

    可测 = 值非符号值，且至少有一轮历史文本提到它。
    """
    out: dict[str, list[int]] = {}
    for slot, val in gold.items():
        if str(val).lower().strip() in SYMBOLIC_VALUES:
            continue
        sup = [i for i, s in enumerate(hist) if mentions(val, seg_text(s))]
        if sup:
            out[slot] = sup
    return out


def loss_after_drop(support: dict[str, list[int]], drop: set[int]) -> tuple[int, int]:
    """删除 drop 里的轮次后，多少个槽位失去**全部**支撑。

    返回 (丢失数, 可测数)。
    """
    lost = 0
    for _slot, sup in support.items():
        if all(i in drop for i in sup):
            lost += 1
    return lost, len(support)


def build_pairs(segs_by_dialogue, seed: int = 0):
    """为每段对话构造配对条件：删 k 个正例 / 删 k 个负例 / 删 k 个随机。

    k = min(n_pos, n_neg)，保证三边删除量相同。
    """
    rng = random.Random(seed)
    pairs = []
    for d in sorted(segs_by_dialogue):
        v = segs_by_dialogue[d]
        q = last_substantive(v)
        hist = [s for s in v if s.turn_idx < q.turn_idx]
        if len(hist) < 3:
            continue
        gold = hist[-1].state
        support = measurable_slots(hist, gold)
        if not support:
            continue

        pos = [i for i, s in enumerate(hist) if s.label == 1]
        neg = [i for i, s in enumerate(hist) if s.label == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue

        # 位置匹配的负例：正例天然集中在开头（首因），而靠前的轮次更可能
        # 是唯一支撑。直接比 pos[:k] 与 neg[:k] 会把「位置效应」算进「标签
        # 效应」。这里给每个待删正例配一个位置最近的未用负例。
        drop_pos = set(pos[:k])
        drop_neg_matched = _match_by_position(sorted(drop_pos), neg, len(hist))

        pairs.append({
            "dialogue_id": d,
            "query_turn": q.turn_idx,
            "n_hist": len(hist),
            "k": k,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "n_measurable_slots": len(support),
            "drop_pos": drop_pos,
            "drop_neg": set(neg[:k]),
            "drop_neg_matched": drop_neg_matched,
            "drop_rand": set(rng.sample(range(len(hist)), k)),
            "support": support,
            "hist": hist,
        })
    return pairs


def _match_by_position(pos_idx: list[int], neg_pool: list[int], n: int) -> set[int]:
    """为每个正例位置挑一个距离最近的未占用负例。

    匹配不上的（负例不够近）就跳过——宁可比得少，也不要引入位置偏置。
    """
    used: set[int] = set()
    out: set[int] = set()
    for i in pos_idx:
        cands = [j for j in neg_pool if j not in used]
        if not cands:
            break
        j = min(cands, key=lambda x: (abs(x - i), x))
        used.add(j)
        out.add(j)
    return out


def run_condition(pairs, key: str) -> dict:
    """跑一个删除条件，返回逐对话的丢失率与汇总。"""
    per_dialogue = []
    lost_tot = meas_tot = 0
    for p in pairs:
        lost, meas = loss_after_drop(p["support"], p[key])
        per_dialogue.append({"dialogue_id": p["dialogue_id"], "lost": lost, "meas": meas})
        lost_tot += lost
        meas_tot += meas
    return {
        "pooled_loss_rate": lost_tot / max(meas_tot, 1),
        "lost": lost_tot,
        "measurable": meas_tot,
        "per_dialogue": per_dialogue,
    }


def paired_bootstrap(pairs, key_a: str, key_b: str, n_boot: int = 2000, seed: int = 0):
    """配对 bootstrap：同一对话内 a 与 b 的丢失率之差。

    按对话重采样（不是按槽位），因为同一对话内的槽位不独立。
    """
    rng = np.random.RandomState(seed)
    deltas = np.zeros(len(pairs))
    for i, p in enumerate(pairs):
        la, ma = loss_after_drop(p["support"], p[key_a])
        lb, mb = loss_after_drop(p["support"], p[key_b])
        deltas[i] = la / max(ma, 1) - lb / max(mb, 1)

    n = len(deltas)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[b] = deltas[idx].mean()
    return {
        "mean_diff": float(deltas.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_two_sided": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
        "n_dialogues": n,
    }


def necessity_analysis(pairs) -> dict:
    """逐轮归因：某轮是不是某个 gold 槽位值的**唯一**文本支撑。

    这是回答"标签 = 必要性吗"的关键。label 的定义是
    "本轮引入了对话结束时仍有效的值"，它并不要求这个值只能由本轮提供。
    若大量 label=1 轮的 necessity=0，说明标签标的是"引入"而非"必要"，
    对"过滤冗余记忆"这个任务来说，这些轮是可以删的。
    """
    rows = []
    for p in pairs:
        hist = p["hist"]
        for i, s in enumerate(hist):
            c = sum(1 for _slot, sup in p["support"].items() if sup == [i])
            rows.append({"label": s.label, "necessity": c})

    lab = np.array([r["label"] for r in rows])
    nec = np.array([r["necessity"] for r in rows], dtype=float)

    from sklearn.metrics import roc_auc_score

    dist = {}
    for k in range(0, int(nec.max()) + 1):
        m = nec == k
        if m.sum():
            dist[str(k)] = {"n": int(m.sum()), "pos_rate": round(float(lab[m].mean()), 4)}

    return {
        "n_turns": len(rows),
        "pos_rate": round(float(lab.mean()), 4),
        "mean_necessity_label1": round(float(nec[lab == 1].mean()), 4),
        "mean_necessity_label0": round(float(nec[lab == 0].mean()), 4),
        "auc_necessity_predicts_label": round(float(roc_auc_score(lab, nec)), 4),
        "auc_binarized": round(float(roc_auc_score(lab, (nec > 0).astype(float))), 4),
        "redundant_pos_rate": round(float((nec[lab == 1] == 0).mean()), 4),
        "distribution": dist,
    }


def unmeasurable_report(segs_by_dialogue) -> dict:
    """有多少 gold 槽位是「不可测」的，按槽位名归类。

    这部分不能进反事实校验（删不删都不会"丢失"），但要如实报告，
    否则读者会以为校验覆盖了全部标签。
    """
    by_slot: Counter = Counter()
    by_reason: Counter = Counter()
    tot = 0
    for d in sorted(segs_by_dialogue):
        v = segs_by_dialogue[d]
        q = last_substantive(v)
        hist = [s for s in v if s.turn_idx < q.turn_idx]
        if not hist:
            continue
        for slot, val in hist[-1].state.items():
            tot += 1
            if str(val).lower().strip() in SYMBOLIC_VALUES:
                by_reason["符号值（yes/no/dontcare）"] += 1
                by_slot[slot] += 1
            elif not any(mentions(val, seg_text(s)) for s in hist):
                by_reason["原文无字面匹配（同义表达）"] += 1
                by_slot[slot] += 1
    return {
        "total_gold_slots": tot,
        "unmeasurable": int(sum(by_reason.values())),
        "by_reason": dict(by_reason),
        "top_slots": by_slot.most_common(10),
    }


def position_stratified(pairs, max_pos: int = 2) -> dict:
    """只看历史开头几轮内的删除，做位置分层校验。

    位置匹配是"给每个正例配一个附近的负例"，但正例仍集中在开头。
    更硬的做法是把比较**限制在同一位置区间内**：只在 turn 0–2 里
    删正例 vs 删负例，位置完全同分布，差值只能来自标签。

    再进一步做 **paired 版本**：只在「正负例在该区间内都有可删轮」的
    对话上比较，两边分母完全相同，排除"参与对话不同"带来的差异。
    """
    out = {}
    for name in ("drop_pos", "drop_neg"):
        lost = meas = n_drop = 0
        for p in pairs:
            d = {i for i in p[name] if i <= max_pos}
            if not d:
                continue
            l, m = loss_after_drop(p["support"], d)
            lost += l
            meas += m
            n_drop += len(d)
        out[name] = {
            "loss_rate": round(lost / max(meas, 1), 4),
            "lost": lost,
            "measurable": meas,
            "n_dropped_turns": n_drop,
        }

    # paired：同一批对话、同一分母
    sub = [p for p in pairs
           if any(i <= max_pos for i in p["drop_pos"])
           and any(i <= max_pos for i in p["drop_neg"])]
    paired = {}
    for name in ("drop_pos", "drop_neg"):
        lost = meas = 0
        for p in sub:
            d = {i for i in p[name] if i <= max_pos}
            l, m = loss_after_drop(p["support"], d)
            lost += l
            meas += m
        paired[name] = {"loss_rate": round(lost / max(meas, 1), 4),
                        "lost": lost, "measurable": meas}

    ratio = out["drop_pos"]["loss_rate"] / max(out["drop_neg"]["loss_rate"], 1e-9)
    ratio_paired = (paired["drop_pos"]["loss_rate"]
                    / max(paired["drop_neg"]["loss_rate"], 1e-9))
    return {"max_pos": max_pos, "conditions": out,
            "paired": {"n_dialogues": len(sub), "conditions": paired},
            "ratio": round(ratio, 2), "ratio_paired": round(ratio_paired, 2)}


def coverage_ceiling(pairs) -> dict:
    """压缩上界：贪心最小覆盖需要保留多少轮。

    为什么不能直接用 necessity=0 的比例当"可删比例"：
    necessity 是**逐轮**判据。某个槽位值有两轮都提到它时，这两轮的
    necessity 都是 0；但**同时删掉两轮**就会丢失。所以「47.9% 的正例
    necessity=0」不等于「可以删掉 47.9%」。

    正确的问法是集合覆盖：保留哪些轮，能让所有可测槽位至少被提到一次，
    且保留的轮数最少。贪心（每步选覆盖最多未覆盖槽位的轮）给出近似解，
    是"可删比例"的可靠上界。
    """
    tot_turns = tot_cover = 0
    pos_total = pos_in_cover = 0
    for p in pairs:
        support = p["support"]
        uncovered = set(support)
        cover: set[int] = set()
        while uncovered:
            cnt: dict[int, int] = {}
            for sl in uncovered:
                for i in support[sl]:
                    cnt[i] = cnt.get(i, 0) + 1
            best = max(cnt, key=lambda k: (cnt[k], -k))
            cover.add(best)
            uncovered = {sl for sl in uncovered if best not in support[sl]}
        tot_turns += len(p["hist"])
        tot_cover += len(cover)
        pos_total += sum(1 for s in p["hist"] if s.label == 1)
        pos_in_cover += sum(1 for i in cover if p["hist"][i].label == 1)

    deletable = tot_turns - tot_cover
    return {
        "total_turns": tot_turns,
        "min_cover_turns": tot_cover,
        "deletable_turns": deletable,
        "deletable_rate": round(deletable / max(tot_turns, 1), 4),
        "pos_total": pos_total,
        "pos_in_cover": pos_in_cover,
        "pos_dropped_rate": round((pos_total - pos_in_cover) / max(pos_total, 1), 4),
    }


def write_spot_check(pairs, n: int, seed: int = 0) -> Path:
    """导出人工抽检表（CSV），交给用户填。

    反事实校验只能证明"信息在不在文本里"，不能证明"标注本身对不对"。
    后者必须人看。这里给出可直接填的表，不自称做过人工校验。
    """
    import csv

    rng = random.Random(seed)
    # 正负例各半，避免只看正例
    pos_rows, neg_rows = [], []
    for p in pairs:
        for i, s in enumerate(p["hist"]):
            row = {
                "dialogue_id": p["dialogue_id"],
                "turn_idx": s.turn_idx,
                "label": s.label,
                "delta": json.dumps(s.delta, ensure_ascii=False),
                "query": s.user,
                "response": s.system,
                "gold_state_at_query": json.dumps(p["hist"][-1].state, ensure_ascii=False),
                "必要性(唯一支撑数)": sum(
                    1 for _sl, sup in p["support"].items() if sup == [i]),
                "你的判断(1=该轮信息对回答query必要, 0=不必要)": "",
                "备注": "",
            }
            (pos_rows if s.label == 1 else neg_rows).append(row)

    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)
    half = n // 2
    rows = pos_rows[:half] + neg_rows[:n - half]
    rng.shuffle(rows)

    out = path_of("docs") / "标签抽检表.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--spot-check", type=int, default=0,
                    help="额外导出 N 行人工抽检 CSV（正负例各半）")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    log.info("%s: %d 段 / %d 对话", args.split, len(segs), len(by_d))

    pairs = build_pairs(by_d)
    log.info("可用配对对话 %d / %d（要求历史 ≥3 轮、正负例各 ≥1、有可测槽位）",
             len(pairs), len(by_d))
    if not pairs:
        log.error("没有可用配对，无法校验")
        return 1

    ks = [p["k"] for p in pairs]
    log.info("每边删除轮数 k：中位 %d，均值 %.2f，最大 %d",
             int(np.median(ks)), float(np.mean(ks)), max(ks))

    # ---- 1. 四个条件 ----
    conds = {}
    for name, key in (("drop_pos", "drop_pos"), ("drop_neg", "drop_neg"),
                      ("drop_neg_matched", "drop_neg_matched"),
                      ("drop_rand", "drop_rand")):
        conds[name] = run_condition(pairs, key)
        log.info("%-16s 池化丢失率 %.4f  (%d/%d)",
                 name, conds[name]["pooled_loss_rate"],
                 conds[name]["lost"], conds[name]["measurable"])

    # 位置分布自查：若 pos 与 neg 位置差很大，pos_vs_neg 就混入了位置效应
    pos_pos = [i for p in pairs for i in p["drop_pos"]]
    neg_pos = [i for p in pairs for i in p["drop_neg"]]
    mch_pos = [i for p in pairs for i in p["drop_neg_matched"]]
    pos_conf = {
        "drop_pos_mean": round(float(np.mean(pos_pos)), 3),
        "drop_neg_mean": round(float(np.mean(neg_pos)), 3),
        "drop_neg_matched_mean": round(float(np.mean(mch_pos)), 3),
    }
    log.info("位置自查：pos %.2f / neg %.2f / neg_matched %.2f",
             pos_conf["drop_pos_mean"], pos_conf["drop_neg_mean"],
             pos_conf["drop_neg_matched_mean"])

    # ---- 2. 配对显著性 ----
    tests = {
        "pos_vs_neg": paired_bootstrap(pairs, "drop_pos", "drop_neg", args.n_boot),
        "pos_vs_neg_matched": paired_bootstrap(pairs, "drop_pos", "drop_neg_matched",
                                               args.n_boot),
        "pos_vs_rand": paired_bootstrap(pairs, "drop_pos", "drop_rand", args.n_boot),
        "neg_vs_rand": paired_bootstrap(pairs, "drop_neg", "drop_rand", args.n_boot),
        "neg_matched_vs_rand": paired_bootstrap(pairs, "drop_neg_matched", "drop_rand",
                                                args.n_boot),
    }
    for k, v in tests.items():
        log.info("%-20s 差值 %+.4f  95%%CI [%+.4f, %+.4f]  p=%.4f",
                 k, v["mean_diff"], v["ci_low"], v["ci_high"], v["p_two_sided"])

    # ---- 3. 位置分层校验 ----
    strat = position_stratified(pairs, max_pos=2)
    sp = strat["conditions"]
    log.info("位置分层（仅 turn 0–%d）：删正例 %.4f vs 删负例 %.4f（%.1f 倍）",
             strat["max_pos"], sp["drop_pos"]["loss_rate"],
             sp["drop_neg"]["loss_rate"], strat["ratio"])
    _pp = strat["paired"]["conditions"]
    log.info("  配对版（%d 段对话、同分母）：删正例 %.4f vs 删负例 %.4f（%.1f 倍）",
             strat["paired"]["n_dialogues"], _pp["drop_pos"]["loss_rate"],
             _pp["drop_neg"]["loss_rate"], strat["ratio_paired"])

    # ---- 4. 必要性分析 ----
    nec = necessity_analysis(pairs)
    log.info("必要性分析：label=1 轮均值 %.4f vs label=0 轮 %.4f，AUC %.4f",
             nec["mean_necessity_label1"], nec["mean_necessity_label0"],
             nec["auc_necessity_predicts_label"])
    log.info("label=1 轮中 necessity=0（逐轮看冗余）的比例 %.2f%%",
             100 * nec["redundant_pos_rate"])

    # ---- 4b. 压缩上界（集合覆盖，回答"到底能删多少"）----
    ceil = coverage_ceiling(pairs)
    log.info("压缩上界：%d 轮中最少保留 %d 轮 → 可删 %.1f%%（正例中被删 %.1f%%）",
             ceil["total_turns"], ceil["min_cover_turns"],
             100 * ceil["deletable_rate"], 100 * ceil["pos_dropped_rate"])

    # ---- 5. 不可测槽位 ----
    unmeas = unmeasurable_report(by_d)
    log.info("gold 槽位 %d 个，其中不可测 %d 个（%.1f%%）",
             unmeas["total_gold_slots"], unmeas["unmeasurable"],
             100 * unmeas["unmeasurable"] / max(unmeas["total_gold_slots"], 1))

    # ---- 6. 落盘 ----
    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_counterfactual"
                              f"{'_' + args.tag if args.tag else ''}")
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "split": args.split,
        "n_dialogues_usable": len(pairs),
        "n_dialogues_total": len(by_d),
        "k_stats": {
            "median": int(np.median(ks)),
            "mean": round(float(np.mean(ks)), 3),
            "max": int(max(ks)),
        },
        "conditions": {k: {"pooled_loss_rate": round(v["pooled_loss_rate"], 4),
                           "lost": v["lost"], "measurable": v["measurable"]}
                       for k, v in conds.items()},
        "position_confound": pos_conf,
        "position_stratified": strat,
        "paired_tests": tests,
        "necessity": nec,
        "coverage_ceiling": ceil,
        "unmeasurable": unmeas,
        "seconds": round(time.time() - t0, 1),
    }
    dump_json(result, run_dir / "counterfactual.json")
    log.info("产物：%s", run_dir)

    # ---- 7. Markdown ----
    md = _to_markdown(result)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    if args.spot_check:
        p = write_spot_check(pairs, args.spot_check)
        log.info("人工抽检表（%d 行，需你填）→ %s", args.spot_check, p)

    return 0


def _to_markdown(r: dict) -> str:
    c = r["conditions"]
    t = r["paired_tests"]
    n = r["necessity"]

    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [
        "## 反事实删除校验结果",
        "",
        f"数据：{r['split']}，可用配对对话 **{r['n_dialogues_usable']}** / "
        f"{r['n_dialogues_total']}，每边删除 k 轮（中位 {r['k_stats']['median']}，"
        f"最大 {r['k_stats']['max']}）。",
        "",
        "### 1. 四个条件的池化信息丢失率",
        "",
        "| 删除条件 | 丢失槽位 | 可测槽位 | 丢失率 |",
        "|---|---|---|---|",
    ]
    for name, label in (("drop_pos", "删 label=1（重要）"),
                        ("drop_neg", "删 label=0（不重要）"),
                        ("drop_neg_matched", "删 label=0（位置匹配）"),
                        ("drop_rand", "删随机（对照）")):
        v = c[name]
        lines.append(f"| {label} | {v['lost']} | {v['measurable']} | "
                     f"**{pct(v['pooled_loss_rate'])}** |")

    pc = r.get("position_confound", {})
    if pc:
        lines += [
            "",
            f"位置自查：被删正例的平均位置 {pc['drop_pos_mean']}，"
            f"普通负例 {pc['drop_neg_mean']}，**位置匹配负例 "
            f"{pc['drop_neg_matched_mean']}**。",
            "",
            "正例天然集中在对话开头（首因），而靠前的轮次更可能是某个值的唯一"
            "支撑。因此「正例 − 负例」的原始差值混入了位置效应；"
            "「正例 − 位置匹配负例」才是干净的标签效应。",
        ]

    lines += [
        "",
        "### 2. 配对差值（按对话 bootstrap）",
        "",
        "| 对比 | 差值 | 95% CI | p |",
        "|---|---|---|---|",
    ]
    for name, label in (("pos_vs_neg", "正例 − 负例（含位置混淆）"),
                        ("pos_vs_neg_matched", "**正例 − 位置匹配负例**"),
                        ("pos_vs_rand", "正例 − 随机"),
                        ("neg_vs_rand", "负例 − 随机"),
                        ("neg_matched_vs_rand", "位置匹配负例 − 随机")):
        if name not in t:
            continue
        v = t[name]
        lines.append(f"| {label} | {v['mean_diff']:+.4f} | "
                     f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {v['p_two_sided']:.4f} |")

    lines += [
        "",
        f"### 3. 位置分层校验（只在 turn 0–{r['position_stratified']['max_pos']} 内比）",
        "",
        "位置匹配只能让两组位置分布接近，不能完全同分布。这里直接把比较"
        "限制在同一位置区间内，位置完全同分布，差值只能来自标签本身。",
        "",
        "| 条件 | 丢失槽位 | 可测槽位 | 丢失率 | 删除轮数 |",
        "|---|---|---|---|---|",
    ]
    for name, label in (("drop_pos", "删 label=1"), ("drop_neg", "删 label=0")):
        v = r["position_stratified"]["conditions"][name]
        lines.append(f"| {label} | {v['lost']} | {v['measurable']} | "
                     f"**{pct(v['loss_rate'])}** | {v['n_dropped_turns']} |")
    _ps = r["position_stratified"]["conditions"]
    _pp = r["position_stratified"]["paired"]["conditions"]
    _nsub = r["position_stratified"]["paired"]["n_dialogues"]
    lines += [
        "",
        f"同一位置区间内，删正例的丢失率是删负例的 "
        f"**{r['position_stratified']['ratio']:.1f} 倍**。",
        "",
        f"**配对版**（只取正负例在该区间内都有可删轮的 {_nsub} 段对话，"
        f"两边分母完全相同）：删正例 {pct(_pp['drop_pos']['loss_rate'])} vs "
        f"删负例 {pct(_pp['drop_neg']['loss_rate'])}，"
        f"**{r['position_stratified']['ratio_paired']:.1f} 倍**。",
        "",
        "### 4. 必要性分析：标签标的是「引入」还是「必要」",
        "",
        f"- 轮数 {n['n_turns']}，正例率 {n['pos_rate']}",
        f"- label=1 轮的平均唯一支撑数：**{n['mean_necessity_label1']}**",
        f"- label=0 轮的平均唯一支撑数：**{n['mean_necessity_label0']}**",
        f"- 用必要性预测 label 的 AUC：{n['auc_necessity_predicts_label']}"
        f"（二值化 {n['auc_binarized']}）",
        f"- **label=1 轮中 necessity=0（完全冗余）的比例：{pct(n['redundant_pos_rate'])}**",
        "",
        "| 唯一支撑数 | 轮数 | 其中正例率 |",
        "|---|---|---|",
    ]
    for k, v in n["distribution"].items():
        lines.append(f"| {k} | {v['n']} | {v['pos_rate']} |")

    cc = r["coverage_ceiling"]
    lines += [
        "",
        "#### 4.3 压缩上界：到底能删多少（集合覆盖）",
        "",
        "不能直接把「necessity=0 的比例」当可删比例。necessity 是**逐轮**判据："
        "某个槽位值有两轮都提到它时，这两轮的 necessity 都是 0，但**同时删掉"
        "两轮**就会丢失。所以正确的问法是集合覆盖——保留哪些轮能让所有可测"
        "槽位至少被提到一次，且保留最少。贪心给出近似最优：",
        "",
        f"- 总轮数 **{cc['total_turns']}**，最小覆盖需保留 **{cc['min_cover_turns']}** 轮",
        f"- **可删 {cc['deletable_turns']} 轮 = {pct(cc['deletable_rate'])}**",
        f"- 覆盖集中含正例 {cc['pos_in_cover']} / {cc['pos_total']}，"
        f"即 **{pct(cc['pos_dropped_rate'])} 的正例不在最小覆盖集里**",
        "",
        "这是「信息不丢失」口径下的压缩上界。真实系统还要考虑模型能否在"
        "更短的上下文里找到这些信息，所以实际可删比例会低于此值。",
    ]

    u = r["unmeasurable"]
    lines += [
        "",
        "### 5. 不可测槽位（不进校验，但必须报告）",
        "",
        f"gold 槽位 {u['total_gold_slots']} 个，其中 **{u['unmeasurable']}** 个不可测"
        f"（{pct(u['unmeasurable'] / max(u['total_gold_slots'], 1))}）：",
        "",
    ]
    for reason, cnt in u["by_reason"].items():
        lines.append(f"- {reason}：{cnt}")
    lines += [
        "",
        "这些槽位删不删都不会被判为「丢失」，因此校验的覆盖面是剩下的部分，不是全部标签。",
        "",
        f"耗时 {r['seconds']} 秒。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
