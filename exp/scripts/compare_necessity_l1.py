#!/usr/bin/env python3
"""两种标签口径的打分器，在**同一套 L1 场景**上比 CR。

为什么必须跑这个：`train_necessity.py` 只报 AUC，而 AUC 是在轮级标签上
算的。真正要回答的问题是「谁更会挑 needle」——这是 L1 的 CR 指标。
两个模型的 val AUC 高下并不能直接推出 CR 高下（尤其两者标签定义不同，
AUC 本来就不可比）。

用法：
    source exp/env.sh
    python exp/scripts/compare_necessity_l1.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import dump_json, get_logger, path_of, set_seed      # noqa: E402
from src.eval.runners import precompute_vectors, run_l1               # noqa: E402
from src.memory.embed import embed_split                              # noqa: E402
from src.memory.long_history import build_scenarios                   # noqa: E402
from src.memory.slice import load_segments                            # noqa: E402
from src.scorer.dataset import split_by_dialogue                      # noqa: E402
from src.scorer.train import train                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_necessity import build_labels, make_ds                     # noqa: E402

log = get_logger("compare_necessity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scorer.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--n-per-length", type=int, default=30)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--tag", default="necessity_l1")
    args = ap.parse_args()

    from src.common import load_yaml
    from src.eval.report import to_markdown

    set_seed(0)
    t0 = time.time()
    cfg = load_yaml(args.config)

    tr_segs = load_segments("train")
    tr_emb = embed_split("train", tr_segs)

    models = {}
    for name in ("intro", "nec"):
        labels = build_labels(tr_segs, name)
        ds = make_ds(tr_segs, tr_emb, labels, keep_unlabeled=False)
        tr, va = split_by_dialogue(ds, 0.1, cfg["train"].get("seed", 0))
        m, hist = train(cfg, tr, va, verbose=False)
        models[name] = m
        log.info("%s 训练完成：val_auc %.4f | 位置基线 %.4f | 超出 %+.4f",
                 name, hist.best_val_auc, hist.position_auc,
                 hist.margin_over_position)

    segs = load_segments(args.split)
    scenarios = build_scenarios(
        segs, lengths=tuple(args.lengths), n_per_length=args.n_per_length,
        seeds=(0,),
    )
    log.info("%s：%d 个场景", args.split, len(scenarios))
    vec_cache = precompute_vectors(scenarios)

    out: dict = {"split": args.split, "budget": args.budget,
                 "n_per_length": args.n_per_length, "by_length": {},
                 "per_scenario": []}
    for L in args.lengths:
        sub = [s for s in scenarios if f"_L{L}_" in s.scenario_id]
        if not sub:
            continue
        out["by_length"][str(L)] = {}
        per = {name: {} for name in ("intro", "nec")}
        for name in ("intro", "nec"):
            # 逐场景记录，才能做配对 bootstrap（两组用同一批场景）
            res = run_l1(sub, lambda sc, v, m=models[name]: _score(m, v),
                         name=name.upper(), budget=args.budget,
                         vec_cache=vec_cache)
            out["by_length"][str(L)][name] = res.summary()
            for sc, cr in zip(sub, res.cr):
                per[name][sc.scenario_id] = cr
        for sid, cr in per["intro"].items():
            out["per_scenario"].append(
                {"scenario_id": sid, "L": L,
                 "intro": cr, "nec": per["nec"][sid]})

    # 跨长度平均
    overall: dict = {}
    for name in ("intro", "nec"):
        vals = [out["by_length"][L][name]["CR"]
                for L in out["by_length"]]
        overall[name] = round(float(np.mean(vals)), 4)
    out["cr_overall"] = overall
    log.info("跨长度 CR：intro %.4f | nec %.4f", overall["intro"], overall["nec"])

    # 配对 bootstrap：两组用同一批场景，比的是逐场景差值
    a = np.array([r["intro"] for r in out["per_scenario"]])
    b = np.array([r["nec"] for r in out["per_scenario"]])
    d = a - b
    rng = np.random.RandomState(0)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)])
    out["paired"] = {
        "n": int(len(d)),
        "intro_mean": round(float(a.mean()), 4),
        "nec_mean": round(float(b.mean()), 4),
        "diff": round(float(d.mean()), 4),
        "ci_low": round(float(np.percentile(bs, 2.5)), 4),
        "ci_high": round(float(np.percentile(bs, 97.5)), 4),
        "p_two_sided": round(float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), 4),
        "intro_wins": int((d > 0).sum()),
        "nec_wins": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
    }
    log.info("配对差值 %+.4f CI[%+.4f, %+.4f] p=%.3f | intro 赢 %d / nec 赢 %d / 平 %d",
             d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
             out["paired"]["p_two_sided"], out["paired"]["intro_wins"],
             out["paired"]["nec_wins"], out["paired"]["ties"])

    run_dir = path_of("runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out, run_dir / "necessity_l1.json")
    (run_dir / "report.md").write_text(to_markdown(out), encoding="utf-8")
    log.info("完成，用时 %.1fs → %s", time.time() - t0, run_dir)
    return 0


def _score(model, vecs: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(vecs.astype(np.float32))).numpy().astype(np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
