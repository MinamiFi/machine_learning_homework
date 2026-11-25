#!/usr/bin/env python3
"""把 runs/ 里的结果画成交付用的图（报告与 PPT 直接用）。

为什么单独写：数字在 md 里已经齐了，但报告和 PPT 要图；每次改数据后
手工重画必然出错，所以图也从 json 生成，和 `verify_docs.py` 用同一批产物。

产物落 `exp/figures/`，全部 300 dpi PNG（够 PPT 用，也够打印）。

    source exp/env.sh
    python3 exp/scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import exp_root, get_logger, path_of    # noqa: E402

log = get_logger("make_figures")

# 中文字体：容器里只有 Noto Sans CJK，没有 SimHei
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

FIG = path_of("figures")
FIG.mkdir(parents=True, exist_ok=True)

# 选择器配色：基线用灰系，我们的方法用蓝，零成本基线用橙
COLORS = {
    "B2": "#1f4e9c", "B2R": "#5b8dd9", "B1": "#8c8c8c",
    "B0": "#b3b3b3", "R0": "#d9822b", "R0A": "#e8a86a",
    "R1": "#c9c9c9", "R2": "#a6a6a6", "R3": "#bfbfbf", "LEN": "#e6b800",
    "NEEDLE": "#2e8b57",
}
LABELS = {
    "B2": "B2 向量检索+打分", "B2R": "B2R 检索+重排", "B1": "B1 仅向量检索",
    "B0": "B0 预算内最近优先", "R0": "R0 位置基线", "R0A": "R0A 绝对轮次",
    "R1": "R1 时间序", "R2": "R2 随机", "R3": "R3 前3轮", "LEN": "LEN 长度基线",
}


def _load(pattern: str, fname: str) -> tuple[Path, dict]:
    hits = sorted(path_of("runs").glob(pattern))
    if not hits:
        raise FileNotFoundError(f"找不到 run：{pattern}")
    d = hits[-1]
    return d, json.loads((d / fname).read_text("utf-8"))


# ---------------------------------------------------------------- 图 1

def fig_l1_by_length() -> None:
    """L1 主表：各选择器在三个长度上的 CR。这是报告的主图。"""
    run, d = _load("*_eval_l1_eval_l1_full2", "L1_selection.json")
    lengths = [128, 256, 512]
    sels = ["B2", "B2R", "B1", "LEN", "R0", "R1"]
    x = np.arange(len(lengths))
    w = 0.13

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(sels):
        vals = [d["by_length"][str(L)][s]["CR"] for L in lengths]
        ax.bar(x + (i - (len(sels) - 1) / 2) * w, vals, w,
               label=LABELS.get(s, s), color=COLORS.get(s, "#999999"),
               edgecolor="white", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - (len(sels) - 1) / 2) * w, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    # x 轴标注真实历史规模，不是场景数。L=512 的 90 个场景里有 71 个
    # 真的到 512 轮，其余受候选池限制落在 354–486 轮——这一点必须写在图上，
    # 否则「L=512 的 CR 还有 0.57」会被误读成「512 轮也扛得住」。
    st = d.get("scenario_tokens", {})
    full = {L: st.get(str(L), {}).get("n", 0) for L in lengths}
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"L={L}\n库 {d['by_length'][str(L)]['B2']['n_scenarios']} 场景"
        + (f"\n(实际满 {L} 轮 {full[L]}/{d['by_length'][str(L)]['B2']['n_scenarios']})"
           if 0 < full[L] < d["by_length"][str(L)]["B2"]["n_scenarios"] else "")
        for L in lengths])
    ax.set_ylabel("CR（关键信息是否被选入上下文）")
    ax.set_ylim(0, 1.12)
    ax.set_title("L1 选择质量：预算 2048 token 下的关键信息召回率", fontsize=12)
    ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig1_l1_cr_by_length.png")
    plt.close(fig)
    log.info("图1 → fig1_l1_cr_by_length.png（%s）", run.name)


# ---------------------------------------------------------------- 图 2

def fig_ablation() -> None:
    """消融：三个因素对 test AUC 的影响。关键是误差棒要画出来。"""
    run, d = _load("*_ablation", "ablation.json")
    rows = d["rows"]
    pos_auc = d["position_auc"]
    factors = [("num_layers", "网络层数"), ("lr", "学习率"),
               ("weight_decay", "L2 正则强度")]

    # 纵轴必须放大：三个因素的 AUC 跨度只有 0.0013，若用 0.78–0.96 的
    # 全范围画，会压成一条直线，看起来像"根本没测出差异"。实际情况是
    # 差异存在但远小于种子间波动（0.0007）与相对基线的优势（0.118）。
    all_means = [r["test_auc_mean"] for r in rows]
    lo, hi = min(all_means) - 0.004, max(all_means) + 0.004
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (fac, title) in zip(axes, factors):
        sub = [r for r in rows if r["factor"] == fac]
        xs = [str(r["value"]) for r in sub]
        ys = [r["test_auc_mean"] for r in sub]
        es = [r["test_auc_std"] for r in sub]
        ax.errorbar(range(len(xs)), ys, yerr=es, marker="o", capsize=5,
                    color=COLORS["B2"], linewidth=1.8, markersize=8,
                    elinewidth=1.6, capthick=1.6)
        # 标签放在误差棒顶端之上，否则会压在 cap 上
        span = hi - lo
        for i, (y, e) in enumerate(zip(ys, es)):
            ax.annotate(f"{y:.4f}", (i, y + e), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8.5,
                        annotation_clip=False)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if fac == "num_layers":
            ax.set_ylabel("test AUC（3 seeds 均值 ± std）")
        if fac == "lr":
            ax.set_xlabel("学习率")
        # 位置基线在纵轴之外，用文字说明而不是画一条看不见的线
        ax.text(0.5, 0.06, f"位置基线 {pos_auc:.4f} 在纵轴下方（低 {lo - pos_auc:.3f}）",
                transform=ax.transAxes, ha="center", fontsize=8.5,
                color=COLORS["R0"], style="italic")
    axes[0].set_xlabel("网络层数（对应参数 99.6k / 165.4k / 231.2k / 297.0k）", fontsize=8.5)
    fig.suptitle(
        "消融实验：纵轴放大后差异可见，但最大跨度 0.0013 与种子间标准差 0.0007 同量级，"
        "三个因素均无显著影响", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ablation.png")
    plt.close(fig)
    log.info("图2 → fig2_ablation.png（%s）", run.name)


# ---------------------------------------------------------------- 图 3

def fig_budget() -> None:
    """预算扫描：CR 随预算与长度变化。用来展示「极低预算下全都崩」。"""
    run, d = _load("*_budget", "budget_sweep.json")
    grid = d["grid"]
    budgets = d["budgets"]
    lengths = d["lengths"]
    sels = ["B2", "B1", "R0"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, L in zip(axes, lengths):
        for s in sels:
            ys, xs = [], []
            for B in budgets:
                k = f"L{L}_B{B}"
                if k in grid:
                    xs.append(B)
                    ys.append(grid[k]["results"][s]["CR"])
            ax.plot(xs, ys, marker="o", label=LABELS.get(s, s),
                    color=COLORS.get(s, "#999"), linewidth=1.9, markersize=6)
        bank = grid[f"L{L}_B{budgets[-1]}"]["bank_tokens"]
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_title(f"L={L}（库 {bank:,} token）", fontsize=11)
        ax.set_xlabel("预算 B（token）")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("CR")
    axes[0].legend(fontsize=9, loc="upper left")
    # 标题必须限定条件：L=128 时 B2 在 B=256 上仍赢（0.35 vs 0.15），
    # 只有 L>=256 才反过来（L=256: 0.10 vs 0.125；L=512: 0.05 vs 0.075）。
    # 笼统写「B=256 时打分器不如余弦」是错的。
    fig.suptitle(
        "预算扫描：L≥256 且预算只够放 8–9 个片段时，打分器反而不如余弦相似度",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_budget_sweep.png")
    plt.close(fig)
    log.info("图3 → fig3_budget_sweep.png（%s）", run.name)


# ---------------------------------------------------------------- 图 4

def fig_counterfactual() -> None:
    """反事实删除：三个 split 上删正例 vs 删负例的丢失率。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    splits = ["test", "dev", "train"]
    patterns = {"test": "*_counterfactual_test", "dev": "*_counterfactual_dev",
                "train": "*_counterfactual_train"}
    data = {}
    for sp in splits:
        run, d = _load(patterns[sp], "counterfactual.json")
        conds = d["conditions"]
        strat = d["position_stratified"]
        data[sp] = {
            "drop_pos": conds["drop_pos"]["pooled_loss_rate"],
            "drop_neg": conds["drop_neg"]["pooled_loss_rate"],
            "drop_rand": conds["drop_rand"]["pooled_loss_rate"],
            "strat_pos": strat["conditions"]["drop_pos"]["loss_rate"],
            "strat_neg": strat["conditions"]["drop_neg"]["loss_rate"],
            "ratio_paired": strat.get("ratio_paired"),
        }

    x = np.arange(len(splits))
    w = 0.26
    ax = axes[0]
    for i, (k, lab, col) in enumerate([
            ("drop_pos", "删 label=1（重要）", COLORS["B2"]),
            ("drop_rand", "删随机", COLORS["R2"]),
            ("drop_neg", "删 label=0（不重要）", COLORS["B1"])]):
        vals = [data[sp][k] for sp in splits]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col,
               edgecolor="white", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * w, v + 0.006, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([{"test": "test", "dev": "dev", "train": "train"}[s] for s in splits])
    ax.set_ylabel("信息丢失率")
    ax.set_title("等量删除后的信息丢失率", fontsize=11)
    # 图例放右上、纵轴留出空间：放左上会压住 test 的第一根柱子和标签
    ax.legend(fontsize=8.5, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.0), framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.40)

    ax = axes[1]
    xs = [data[sp]["strat_pos"] for sp in splits]
    ys = [data[sp]["strat_neg"] for sp in splits]
    ax.scatter(ys, xs, s=150, color=COLORS["B2"], zorder=3,
               edgecolor="white", linewidth=1.2)
    # 三个点挤在一起（x 都在 0.017–0.021），标签必须手动错开，
    # 自动 offset 会叠成一团。
    offsets = {"test": (-8, -34), "dev": (26, 6), "train": (-16, 20)}
    for sp, xx, yy in zip(splits, ys, xs):
        r = data[sp]["ratio_paired"]
        ax.annotate(f"{sp}  {r:.2f}×", (xx, yy), textcoords="offset points",
                    xytext=offsets[sp], fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                    linewidth=0.8, shrinkA=0, shrinkB=4))
    lim = max(max(xs), max(ys)) * 1.5
    ax.plot([0, lim], [0, lim], color="#bbbbbb", linestyle=":", linewidth=1,
            label="y = x（删除无差别）")
    ax.set_xlabel("删 label=0 的丢失率（同分母配对，turn 0–2）")
    ax.set_ylabel("删 label=1 的丢失率（同分母配对）")
    ax.set_title("位置混淆控制后：10–13 倍差距", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(ys) * 2.2)
    ax.set_ylim(0, max(xs) * 1.25)

    fig.suptitle("反事实删除校验：标签「重要」的轮次删掉后信息确实会丢", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_counterfactual.png")
    plt.close(fig)
    log.info("图4 → fig4_counterfactual.png")


# ---------------------------------------------------------------- 图 5

def fig_jga() -> None:
    """JGA 四条件对比：槽位级、可答子集、对话级三个口径。"""
    run, d = _load("*_jga_api_full", "jga.json")
    s = d["summary"]
    conds = ["full", "drop_neg", "drop_rand", "drop_pos"]
    names = ["全量上下文", "删不重要", "删随机", "删重要"]
    colors = ["#2e8b57", COLORS["B1"], COLORS["R2"], COLORS["B2"]]

    groups = [
        ("槽位级准确率", [s["slot_acc"][c] for c in conds]),
        ("可答子集准确率", [s["slot_acc_answerable"][c] for c in conds]),
        ("对话级 JGA", [s["jga"][c] for c in conds]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(groups))
    w = 0.19
    for i, (c, nm, col) in enumerate(zip(conds, names, colors)):
        vals = [g[1][i] for g in groups]
        ax.bar(x + (i - 1.5) * w, vals, w, label=nm, color=col,
               edgecolor="white", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1.5) * w, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("准确率 / JGA")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    p = s["paired_all"]["drop_pos-drop_neg"]
    bb = d.get("backbone") or run.name
    ax.set_title(
        f"JGA 口径（{bb}，{s['n_slot_rows']} 条槽位）：删重要 − 删不重要 = "
        f"{p['diff']:+.3f}，p<0.001", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_jga_conditions.png")
    plt.close(fig)
    log.info("图5 → fig5_jga_conditions.png（%s）", run.name)


# ---------------------------------------------------------------- 图 6

def fig_l2() -> None:
    """L2 任务完成率：三指标一起看（slot_acc / token / CR）。"""
    run, d = _load("*_eval_l2_l2_main_api2", "L2_generation.json")
    o = d["overall"]
    sels = ["B2", "B1", "R1", "B0", "R2", "R3", "R0", "B0FULL", "NEEDLE", "ORACLE"]
    acc = [o[x]["slot_acc"]["mean"] for x in sels]
    succ = [o[x]["scenario_success"]["mean"] for x in sels]
    ctx = [o[x]["context_tokens"] for x in sels]
    cr = [o[x]["cr"] for x in sels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # 左：slot_acc 与 scenario_success
    ax = axes[0]
    x = np.arange(len(sels))
    w = 0.38
    ax.bar(x - w / 2, acc, w, label="逐槽位准确率", color="#1f4e9c")
    ax.bar(x + w / 2, succ, w, label="场景成功率", color="#5b8dd9")
    for xi, (a, sc) in enumerate(zip(acc, succ)):
        ax.text(xi - w / 2, a + 0.012, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w / 2, sc + 0.012, f"{sc:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(sels, fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_title("任务完成率（90 场景 / 180 槽位）", fontsize=11)

    # 中：context tokens（log）
    ax = axes[1]
    cols = ["#1f4e9c" if x == "B2" else "#8c8c8c" for x in sels]
    cols = ["#2e8b57" if x == "ORACLE" else c for x, c in zip(sels, cols)]
    ax.barh(range(len(sels)), ctx, color=cols)
    for i, v in enumerate(ctx):
        ax.text(v * 1.06, i, f"{v:,.0f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(sels)))
    ax.set_yticklabels(sels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("进上下文的 token 数（对数轴）")
    ax.set_xlim(20, max(ctx) * 3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_title("token 消耗", fontsize=11)

    # 右：CR vs slot_acc 散点
    ax = axes[2]
    # CR=1.0 上有 4 个点（ORACLE/NEEDLE/B0FULL），标签必须手动错开
    # B0(cr=.167,acc=.317) 与 R1(cr=.167,acc=.328) 几乎同点，必须往相反方向拉
    offs = {"ORACLE": (10, 0), "NEEDLE": (-30, 16), "B0FULL": (10, -14),
            "B2": (8, 6), "B1": (8, 6), "B0": (-34, -16), "R2": (8, -10),
            "R3": (8, -10), "R0": (8, 4), "R1": (-34, 14)}
    for nm in sels:
        a = o[nm]["slot_acc"]["mean"]
        c = o[nm]["cr"]
        ax.scatter(c, a, s=70, color=COLORS.get(nm, "#2e8b57" if nm == "ORACLE" else "#a6a6a6"),
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(nm, (c, a), xytext=offs.get(nm, (5, 4)),
                    textcoords="offset points", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888")
                    if nm in ("NEEDLE", "B0FULL", "B0", "R1") else None)
    ax.set_xlabel("关键信息召回率 CR（needle 是否入选）")
    ax.set_ylabel("逐槽位准确率")
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_title("CR 高 ≠ 答得对", fontsize=11)

    p = d["paired"]["B2-B1"]
    fig.suptitle(
        f"L2 任务完成率（{d.get('backbone', run.name)}）：B2 准确率 {o['B2']['slot_acc']['mean']:.3f}，"
        f"但相对 B1 不显著（差 {p['diff']:+.3f}，p={p['p']:.3f}）；"
        f"ORACLE 只用 {o['ORACLE']['context_tokens']:,.0f} token 就到 "
        f"{o['ORACLE']['slot_acc']['mean']:.3f}", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_l2_task_success.png")
    plt.close(fig)
    log.info("图7 → fig7_l2_task_success.png（%s）", run.name)


def fig_score_dist() -> None:
    """打分分布：按 label 分组 + 按位置分组。

    两个必须注意的点：
    1. 打分器输出的是 **logits**（实测范围 −12.5 ~ 7.3），不是概率。
       画直方图前要么用原始 logits 并给足范围，要么过 sigmoid 转概率——
       直接当 0–1 画会把大半样本切掉。
    2. 解释「AUC 0.93 但 corr(score, 位置) = −0.63 不矛盾」：位置信号确实强，
       但同一位置区间内标签仍可分（分层 AUC 0.86–0.92）。
    """
    from collections import defaultdict

    import torch

    from src.memory.slice import load_segments
    from src.scripts_helpers import load_scorer

    model, _hist = load_scorer()
    segs = load_segments("test")
    npz = np.load(path_of("data", "processed", "memory", "test.minilm.npz"))
    emb, seg_ids = npz["emb"].astype(np.float32), npz["seg_ids"]
    row_of = {sid: i for i, sid in enumerate(seg_ids)}
    X = emb[[row_of[s.seg_id] for s in segs]]
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy().ravel()
    prob = 1.0 / (1.0 + np.exp(-logits))          # sigmoid，用于直方图

    y = np.array([s.label for s in segs])
    pos_of = {s.seg_id: i for i, s in enumerate(segs)}
    g: dict[str, list] = defaultdict(list)
    for s in segs:
        g[s.dialogue_id].append(s)
    rel = np.zeros(len(segs))
    for did, ss in g.items():
        ss = sorted(ss, key=lambda x: x.turn_idx)
        n = len(ss)
        for k, s in enumerate(ss):
            rel[pos_of[s.seg_id]] = k / (n - 1) if n > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(prob[y == 1], bins=bins, alpha=0.65,
            label=f"label=1 重要（{int(y.sum()):,}）",
            color=COLORS["B2"], density=True)
    ax.hist(prob[y == 0], bins=bins, alpha=0.65,
            label=f"label=0 不重要（{int((1 - y).sum()):,}）",
            color=COLORS["R0"], density=True)
    ax.set_xlabel("打分网络输出（sigmoid 后的概率）")
    ax.set_ylabel("密度")
    ax.set_title("按标签分组：两个分布有重叠但可区分", fontsize=11)
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    ax = axes[1]
    # 必须用**固定分箱**（0–.25 / .25–.5 / .5–.75 / .75–1），与
    # `verify_docs.py` 的 `strat.q*` 完全一致。曾用 np.quantile 分位数分箱，
    # 每个箱样本数相等但边界不同，算出的正例率与层内 AUC 和文档对不上
    # （图上 0.82/0.75/0.63/0.10，文档 0.819/0.748/0.664/0.131）。
    bins_rel = [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]
    labs = ["前 25%", "25–50%", "50–75%", "后 25%"]
    means, stds, prs, aucs = [], [], [], []
    from sklearn.metrics import roc_auc_score
    for lo, hi in bins_rel:
        m = (rel >= lo) & (rel < hi)
        means.append(float(prob[m].mean()))
        stds.append(float(prob[m].std()))
        prs.append(float(y[m].mean()))
        aucs.append(float(roc_auc_score(y[m], prob[m]))
                    if len(set(y[m])) > 1 else float("nan"))

    ax.bar(range(4), means, yerr=stds, capsize=5, color=COLORS["B2"],
           edgecolor="white", linewidth=0.6, alpha=0.9)
    top = max(m + sd for m, sd in zip(means, stds))
    for i, (mv, pr, au) in enumerate(zip(means, prs, aucs)):
        ax.text(i, mv + stds[i] + top * 0.03,
                f"均分 {mv:.2f}\n正例率 {pr:.2f}\n层内AUC {au:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs)
    ax.set_ylabel("平均打分（sigmoid 后）")
    ax.set_title("按对话内相对位置分箱（首因效应）", fontsize=11)
    ax.set_ylim(0, top * 1.45)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    r = float(np.corrcoef(prob, rel)[0, 1])
    fig.suptitle(
        f"打分分布：位置信号很强（corr = {r:.2f}），但同位置区间内仍可分"
        f"（层内 AUC 0.86–0.92）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_score_distribution.png")
    plt.close(fig)
    log.info("图6 → fig6_score_distribution.png（logits 经 sigmoid）")


def main() -> int:
    log.info("产物目录 %s", FIG)
    fig_l1_by_length()
    fig_ablation()
    fig_budget()
    fig_counterfactual()
    fig_jga()
    fig_l2()
    fig_score_dist()
    files = sorted(FIG.glob("*.png"))
    log.info("共 %d 张图", len(files))
    for f in files:
        log.info("  %s (%.0f KB)", f.name, f.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
