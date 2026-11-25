"""把文档里引用的每个数字重新算一遍，逐项对照。

动机：写文档时容易凭印象填数。这里把「文档声称的值」和「脚本算出的值」
放在同一张表里比较，不一致就报 FAIL。文档改了数就要改这里的 EXPECTED。

用法：
    source exp/env.sh
    python exp/scripts/verify_docs.py            # 全部检查（约 1 分钟）
    python exp/scripts/verify_docs.py --quick    # 跳过检索召回（最慢的一项）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import get_logger, path_of          # noqa: E402
from src.memory.embed import embed_queries, embed_split  # noqa: E402
from src.memory.long_history import build_scenarios  # noqa: E402
from src.memory.slice import group_by_dialogue, load_segments  # noqa: E402
from src.scripts_helpers import load_scorer          # noqa: E402

log = get_logger("verify_docs")

# 文档里声称的值 → (实测值, 容差)。容差为 0 表示必须精确相等（整数/字符串）。
EXPECTED: dict[str, tuple] = {
    # ---- 数据规模 ----
    "test.段数": (8372, 0),
    "test.对话数": (1000, 0),
    "test.正例率": (0.5658, 5e-4),
    # ---- 打分器 ----
    "scorer.test_auc": (0.9332, 5e-4),
    "scorer.val_auc": (0.9284, 2e-3),
    # hist.position_auc 是 **val** 上的位置基线；test 上的 0.8172 要单独算。
    # 文档里两个数都出现，别混。
    "scorer.val_position_auc": (0.8065, 5e-4),
    "scorer.test_position_auc": (0.8172, 5e-4),
    "scorer.n_params": (165377, 0),
    # ---- 相关性 / 分层 AUC（L1 文档 §3）----
    "corr.score_relpos": (-0.6312, 5e-4),
    "corr.score_ntokens": (0.6331, 5e-4),
    "corr.relpos_label": (-0.5456, 5e-4),
    "strat.q0.auc": (0.8986, 5e-4),
    "strat.q1.auc": (0.8628, 5e-4),
    "strat.q2.auc": (0.8545, 5e-4),
    "strat.q3.auc": (0.9220, 5e-4),
    "strat.len0.auc": (0.9464, 5e-4),
    "strat.len1.auc": (0.9084, 5e-4),
    "strat.len2.auc": (0.8675, 5e-4),
    "strat.len3.auc": (0.8556, 5e-4),
    # ---- 全库打分耗时 ----
    # 容差放到 0.02：这是纯计时项，机器上有别的任务在跑（比如后台 JGA 占满
    # 2 核）时会从 0.014 涨到 0.02 甚至更多。它验的是「量级对不对」
    # （0.0x 秒 vs 分钟），不是「精确到毫秒」，收紧容差只会制造假失败。
    "score_all.seconds": (0.014, 2e-2),
    # ---- 检索召回（L1 文档 §2.2）----
    "recall.L128.top20": (0.4000, 5e-3),
    "recall.L128.top50": (0.6778, 5e-3),
    "recall.L256.top20": (0.3222, 5e-3),
    "recall.L256.top50": (0.4556, 5e-3),
    "recall.L512.top50": (0.3444, 5e-3),
    "recall.L128.top10": (0.3111, 5e-3),
    "recall.L128.top100": (0.9667, 5e-3),
    "recall.L256.top10": (0.2444, 5e-3),
    "recall.L256.top100": (0.6889, 5e-3),
    "recall.L512.top10": (0.1333, 5e-3),
    "recall.L512.top20": (0.2444, 5e-3),
    "recall.L512.top100": (0.5000, 5e-3),
}

# 从 runs/ 里读的结果文件，不重算（重算要十几分钟）
FROM_RUNS: dict[str, tuple[str, str, float]] = {
    # 键: (run 目录通配, JSON 内的路径, 容差)
    "L1.B2.CR.overall": ("*_eval_l1_eval_l1_full2", "overall/B2/CR", 5e-4),
    "L1.B1.CR.overall": ("*_eval_l1_eval_l1_full2", "overall/B1/CR", 5e-4),
    "L1.B2R.CR.overall": ("*_eval_l1_eval_l1_full2", "overall/B2R/CR", 5e-4),
    "L1.R0.CR.overall": ("*_eval_l1_eval_l1_full2", "overall/R0/CR", 5e-4),
    "L1.R2.CR.overall": ("*_eval_l1_eval_l1_full2", "overall/R2/CR", 5e-4),
    "L1.B0.CR.overall": ("*_eval_l1_eval_l1_full2", "overall/B0/CR", 5e-4),
    "L1.LEN.CR.overall": ("*_eval_l1_eval_l1_full2", "overall/LEN/CR", 5e-4),
    "L1.B2.CR.L128": ("*_eval_l1_eval_l1_full2", "by_length/128/B2/CR", 5e-4),
    "L1.B2.CR.L256": ("*_eval_l1_eval_l1_full2", "by_length/256/B2/CR", 5e-4),
    "L1.B2.CR.L512": ("*_eval_l1_eval_l1_full2", "by_length/512/B2/CR", 5e-4),
    # 消融（跑一次 7.5 分钟，这里只读结果文件）
    "abl.layers1.auc": ("*_ablation", "rows/0/test_auc_mean", 5e-4),
    "abl.layers2.auc": ("*_ablation", "rows/1/test_auc_mean", 5e-4),
    "abl.layers3.auc": ("*_ablation", "rows/2/test_auc_mean", 5e-4),
    "abl.layers4.auc": ("*_ablation", "rows/3/test_auc_mean", 5e-4),
    "abl.layers1.params": ("*_ablation", "rows/0/n_params", 0),
    "abl.layers4.params": ("*_ablation", "rows/3/n_params", 0),
    # ---- 反事实删除校验（docs/反事实删除校验.md）----
    # 只认带 position_stratified 的那次（早期版本没有位置控制）
    "cf.test.drop_pos": ("*_counterfactual_test", "conditions/drop_pos/pooled_loss_rate", 5e-4),
    "cf.test.drop_neg": ("*_counterfactual_test", "conditions/drop_neg/pooled_loss_rate", 5e-4),
    "cf.test.drop_neg_matched": ("*_counterfactual_test",
                                 "conditions/drop_neg_matched/pooled_loss_rate", 5e-4),
    "cf.test.drop_rand": ("*_counterfactual_test", "conditions/drop_rand/pooled_loss_rate", 5e-4),
    "cf.test.strat_pos": ("*_counterfactual_test",
                          "position_stratified/conditions/drop_pos/loss_rate", 5e-4),
    "cf.test.strat_neg": ("*_counterfactual_test",
                          "position_stratified/conditions/drop_neg/loss_rate", 5e-4),
    "cf.test.redundant_pos": ("*_counterfactual_test", "necessity/redundant_pos_rate", 5e-4),
    "cf.test.nec_label1": ("*_counterfactual_test", "necessity/mean_necessity_label1", 5e-4),
    "cf.test.nec_label0": ("*_counterfactual_test", "necessity/mean_necessity_label0", 5e-4),
    "cf.test.nec_auc": ("*_counterfactual_test",
                        "necessity/auc_necessity_predicts_label", 5e-4),
    "cf.test.n_pairs": ("*_counterfactual_test", "n_dialogues_usable", 0),
    "cf.test.unmeasurable": ("*_counterfactual_test", "unmeasurable/unmeasurable", 0),
    "cf.test.gold_slots": ("*_counterfactual_test", "unmeasurable/total_gold_slots", 0),
    "cf.dev.drop_pos": ("*_counterfactual_dev", "conditions/drop_pos/pooled_loss_rate", 5e-4),
    "cf.dev.strat_pos": ("*_counterfactual_dev",
                         "position_stratified/conditions/drop_pos/loss_rate", 5e-4),
    "cf.dev.strat_neg": ("*_counterfactual_dev",
                         "position_stratified/conditions/drop_neg/loss_rate", 5e-4),
    "cf.train.drop_pos": ("*_counterfactual_train", "conditions/drop_pos/pooled_loss_rate", 5e-4),
    "cf.train.strat_pos": ("*_counterfactual_train",
                           "position_stratified/conditions/drop_pos/loss_rate", 5e-4),
    "cf.train.strat_neg": ("*_counterfactual_train",
                           "position_stratified/conditions/drop_neg/loss_rate", 5e-4),
    "cf.test.ratio": ("*_counterfactual_test", "position_stratified/ratio", 5e-3),
    "cf.test.ratio_paired": ("*_counterfactual_test",
                             "position_stratified/ratio_paired", 5e-3),
    "cf.test.paired_pos": ("*_counterfactual_test",
                           "position_stratified/paired/conditions/drop_pos/loss_rate", 5e-4),
    "cf.test.paired_neg": ("*_counterfactual_test",
                           "position_stratified/paired/conditions/drop_neg/loss_rate", 5e-4),
    "cf.test.paired_n": ("*_counterfactual_test",
                         "position_stratified/paired/n_dialogues", 0),
    "cf.dev.ratio_paired": ("*_counterfactual_dev",
                            "position_stratified/ratio_paired", 5e-3),
    "cf.train.ratio_paired": ("*_counterfactual_train",
                              "position_stratified/ratio_paired", 5e-3),
    "cf.test.deletable_rate": ("*_counterfactual_test",
                               "coverage_ceiling/deletable_rate", 5e-4),
    "cf.test.pos_dropped_rate": ("*_counterfactual_test",
                                 "coverage_ceiling/pos_dropped_rate", 5e-4),
    "cf.dev.deletable_rate": ("*_counterfactual_dev",
                              "coverage_ceiling/deletable_rate", 5e-4),
    "cf.train.deletable_rate": ("*_counterfactual_train",
                                "coverage_ceiling/deletable_rate", 5e-4),
    # ---- necessity 标签对照（docs/反事实删除校验.md §8）----
    "nec.intro.val_auc": ("*_necessity", "by_label/intro/best_val_auc", 5e-4),
    "nec.nec.val_auc": ("*_necessity", "by_label/nec/best_val_auc", 5e-4),
    "nec.nec.pos_rate": ("*_necessity", "by_label/nec/pos_rate", 5e-4),
    "nec.nec.margin": ("*_necessity", "by_label/nec/margin", 5e-4),
    "nec.val_auc_on_intro.nec": ("*_necessity", "val_auc_on_intro_labels/nec", 5e-4),
    "necl1.intro.cr": ("*_necessity_l1", "cr_overall/intro", 5e-4),
    "necl1.nec.cr": ("*_necessity_l1", "cr_overall/nec", 5e-4),
    "necl1.paired.diff": ("*_necessity_l1", "paired/diff", 5e-4),
    "necl1.paired.ci_low": ("*_necessity_l1", "paired/ci_low", 5e-4),
    "necl1.paired.ci_high": ("*_necessity_l1", "paired/ci_high", 5e-4),
    "necl1.paired.p": ("*_necessity_l1", "paired/p_two_sided", 5e-4),
    "necl1.paired.ties": ("*_necessity_l1", "paired/ties", 0),
    "necl1.paired.n": ("*_necessity_l1", "paired/n", 0),
    "necl1.intro.recall": ("*_necessity_l1",
                           "avg:by_length/*/intro/Recall@budget", 5e-4),
    "necl1.nec.recall": ("*_necessity_l1",
                         "avg:by_length/*/nec/Recall@budget", 5e-4),
    "necl1.intro.ndcg": ("*_necessity_l1",
                         "avg:by_length/*/intro/NDCG@k", 5e-4),
    "necl1.paired.intro_wins": ("*_necessity_l1", "paired/intro_wins", 0),
    "necl1.paired.nec_wins": ("*_necessity_l1", "paired/nec_wins", 0),
    "necl1.L128.intro": ("*_necessity_l1", "by_length/128/intro/CR", 5e-4),
    "necl1.L128.nec": ("*_necessity_l1", "by_length/128/nec/CR", 5e-4),
    "necl1.L256.intro": ("*_necessity_l1", "by_length/256/intro/CR", 5e-4),
    "necl1.L256.nec": ("*_necessity_l1", "by_length/256/nec/CR", 5e-4),
    "necl1.L512.intro": ("*_necessity_l1", "by_length/512/intro/CR", 5e-4),
    "necl1.L512.nec": ("*_necessity_l1", "by_length/512/nec/CR", 5e-4),
    # ---- L2 任务完成率（API backbone） ----
    "l2.n_scenarios": ("*_eval_l2_l2_main_api2", "n_scenarios", 0),
    "l2.n_answerable": ("*_eval_l2_l2_main_api2", "n_answerable_slots", 0),
    "l2.B2.slot_acc": ("*_eval_l2_l2_main_api2", "overall/B2/slot_acc/mean", 5e-4),
    "l2.B1.slot_acc": ("*_eval_l2_l2_main_api2", "overall/B1/slot_acc/mean", 5e-4),
    "l2.B0.slot_acc": ("*_eval_l2_l2_main_api2", "overall/B0/slot_acc/mean", 5e-4),
    "l2.R0.slot_acc": ("*_eval_l2_l2_main_api2", "overall/R0/slot_acc/mean", 5e-4),
    "l2.R2.slot_acc": ("*_eval_l2_l2_main_api2", "overall/R2/slot_acc/mean", 5e-4),
    "l2.ORACLE.slot_acc": ("*_eval_l2_l2_main_api2", "overall/ORACLE/slot_acc/mean", 5e-4),
    "l2.B0FULL.slot_acc": ("*_eval_l2_l2_main_api2", "overall/B0FULL/slot_acc/mean", 5e-4),
    "l2.B2.succ": ("*_eval_l2_l2_main_api2", "overall/B2/scenario_success/mean", 5e-4),
    "l2.ORACLE.succ": ("*_eval_l2_l2_main_api2", "overall/ORACLE/scenario_success/mean", 5e-4),
    "l2.B2.ar": ("*_eval_l2_l2_main_api2", "overall/B2/ar/mean", 5e-4),
    "l2.B2.ctx": ("*_eval_l2_l2_main_api2", "overall/B2/context_tokens", 1e-2),
    "l2.ORACLE.ctx": ("*_eval_l2_l2_main_api2", "overall/ORACLE/context_tokens", 1e-2),
    "l2.B0FULL.ctx": ("*_eval_l2_l2_main_api2", "overall/B0FULL/context_tokens", 1e-2),
    "l2.B2.cr": ("*_eval_l2_l2_main_api2", "overall/B2/cr", 5e-4),
    "l2.B1.cr": ("*_eval_l2_l2_main_api2", "overall/B1/cr", 5e-4),
    "l2.B0.cr": ("*_eval_l2_l2_main_api2", "overall/B0/cr", 5e-4),
    "l2.paired.B2_B1.diff": ("*_eval_l2_l2_main_api2", "paired/B2-B1/diff", 5e-4),
    "l2.paired.B2_B1.p": ("*_eval_l2_l2_main_api2", "paired/B2-B1/p", 5e-4),
    "l2.paired.B2_R0.diff": ("*_eval_l2_l2_main_api2", "paired/B2-R0/diff", 5e-4),
    "l2.paired.B2_R0.p": ("*_eval_l2_l2_main_api2", "paired/B2-R0/p", 5e-4),
    "l2.L128.B2": ("*_eval_l2_l2_main_api2", "by_length/128/B2/slot_acc/mean", 5e-4),
    "l2.L512.B2": ("*_eval_l2_l2_main_api2", "by_length/512/B2/slot_acc/mean", 5e-4),
    "l2.L512.B1": ("*_eval_l2_l2_main_api2", "by_length/512/B1/slot_acc/mean", 5e-4),
    "jga.n_dialogues": ("*_jga_api_full", "summary/n_dialogues", 0),
    "jga.n_slot_rows": ("*_jga_api_full", "summary/n_slot_rows", 0),
    "jga.n_answerable": ("*_jga_api_full", "summary/n_answerable", 0),
    "jga.slot_acc.full": ("*_jga_api_full", "summary/slot_acc/full", 5e-4),
    "jga.slot_acc.drop_pos": ("*_jga_api_full", "summary/slot_acc/drop_pos", 5e-4),
    "jga.slot_acc.drop_neg": ("*_jga_api_full", "summary/slot_acc/drop_neg", 5e-4),
    "jga.slot_acc.drop_rand": ("*_jga_api_full", "summary/slot_acc/drop_rand", 5e-4),
    "jga.slot_acc_answerable.drop_pos": ("*_jga_api_full", "summary/slot_acc_answerable/drop_pos", 5e-4),
    "jga.slot_acc_answerable.drop_neg": ("*_jga_api_full", "summary/slot_acc_answerable/drop_neg", 5e-4),
    "jga.slot_acc_answerable.drop_rand": ("*_jga_api_full", "summary/slot_acc_answerable/drop_rand", 5e-4),
    "jga.jga.full": ("*_jga_api_full", "summary/jga/full", 5e-4),
    "jga.jga.drop_pos": ("*_jga_api_full", "summary/jga/drop_pos", 5e-4),
    "jga.jga.drop_neg": ("*_jga_api_full", "summary/jga/drop_neg", 5e-4),
    "jga.jga.drop_rand": ("*_jga_api_full", "summary/jga/drop_rand", 5e-4),
    "jga.paired.diff": ("*_jga_api_full", "summary/paired/diff", 5e-4),
    "jga.paired.ci_low": ("*_jga_api_full", "summary/paired/ci_low", 5e-4),
    "jga.paired.ci_high": ("*_jga_api_full", "summary/paired/ci_high", 5e-4),
    # 分槽位细分（文档 §4.4 表）。只登记几个有结论意义的，不逐个登记 19 行——
    # 表格是给人看的，自检要抓的是「抄错/算错」，不是把表格复制一遍。
    # 分槽位报的是扩样版（607 段 / 1332 条），不是主表那一版（120 段 / 258 条）。
    # 两版结论一致但样本量差 5.2 倍，文档 §4.4 用的是扩样版，这里必须对上。
    "jga.by_slot.n_types": ("*_jga_api_full", "len:summary/by_slot", 0),
    "jga.by_slot.train_day.full": ("*_jga_api_full", "summary/by_slot/train-day/slot_acc/full", 5e-4),
    "jga.by_slot.train_day.n": ("*_jga_api_full", "summary/by_slot/train-day/n", 0),
    "jga.by_slot.hotel_pricerange.full": ("*_jga_api_full", "summary/by_slot/hotel-pricerange/slot_acc/full", 5e-4),
    "jga.by_slot.restaurant_name.full": ("*_jga_api_full", "summary/by_slot/restaurant-name/slot_acc/full", 5e-4),
    "jga.by_slot.restaurant_name.drop_pos": ("*_jga_api_full", "summary/by_slot/restaurant-name/slot_acc/drop_pos", 5e-4),
    "jga.by_slot.attraction_name.drop_pos": ("*_jga_api_full", "summary/by_slot/attraction-name/slot_acc/drop_pos", 5e-4),
    "jga.by_slot.train_destination.n": ("*_jga_api_full", "summary/by_slot/train-destination/n", 0),
    # 扩样版主表口径（文档 §4.4 的"两版一致"依据）
    "jga.wide.n_dialogues": ("*_jga_api_full", "summary/n_dialogues", 0),
    "jga.wide.n_slot_rows": ("*_jga_api_full", "summary/n_slot_rows", 0),
    "jga.wide.drop_pos": ("*_jga_api_full", "summary/slot_acc/drop_pos", 5e-4),
    "jga.wide.drop_neg": ("*_jga_api_full", "summary/slot_acc/drop_neg", 5e-4),
    "jga.wide.diff": ("*_jga_api_full", "summary/paired_all/drop_pos-drop_neg/diff", 5e-4),
    # 模拟抽检 κ（docs/标签抽检.md §3）
    "spot.n": ("*_spot_check2_n200", "n", 0),
    "spot.label_vs_J2.kappa": ("*_spot_check2_n200", "label_vs_J2/kappa", 5e-4),
    "spot.label_vs_J2.agreement": ("*_spot_check2_n200", "label_vs_J2/agreement", 5e-4),
    "spot.label_vs_J2.ci_low": ("*_spot_check2_n200", "label_vs_J2/kappa_ci/0", 5e-4),
    "spot.label_vs_J2.ci_high": ("*_spot_check2_n200", "label_vs_J2/kappa_ci/1", 5e-4),
    "spot.label_vs_J3.kappa": ("*_spot_check2_n200", "label_vs_J3/kappa", 5e-4),
    "spot.label_vs_J3.agreement": ("*_spot_check2_n200", "label_vs_J3/agreement", 5e-4),
    "spot.label_vs_J3.ci_low": ("*_spot_check2_n200", "label_vs_J3/kappa_ci/0", 5e-4),
    "spot.label_vs_J3.ci_high": ("*_spot_check2_n200", "label_vs_J3/kappa_ci/1", 5e-4),
    "spot.label_vs_J3.tp": ("*_spot_check2_n200", "label_vs_J3/tp", 0),
    "spot.label_vs_J3.tn": ("*_spot_check2_n200", "label_vs_J3/tn", 0),
    "spot.label_vs_J3.fp": ("*_spot_check2_n200", "label_vs_J3/fp", 0),
    "spot.label_vs_J3.fn": ("*_spot_check2_n200", "label_vs_J3/fn", 0),
    "spot.label_vs_J3.precision": ("*_spot_check2_n200", "label_vs_J3/precision", 5e-4),
    "spot.label_vs_J3.recall": ("*_spot_check2_n200", "label_vs_J3/recall", 5e-4),
    "spot.J2_vs_J3.kappa": ("*_spot_check2_n200", "J2_vs_J3/kappa", 5e-4),
    "spot.J2_vs_J3.agreement": ("*_spot_check2_n200", "J2_vs_J3/agreement", 5e-4),
    "spot.J2_vs_J3.ci_low": ("*_spot_check2_n200", "J2_vs_J3/kappa_ci/0", 5e-4),
    "spot.J2_vs_J3.ci_high": ("*_spot_check2_n200", "J2_vs_J3/kappa_ci/1", 5e-4),
    "spot.csv_filled": ("*_spot_check2_n200", "csv_filled", 0),
    "spot.csv_disagree": ("*_spot_check2_n200", "csv_disagree", 0),
    "necl1.nec.ndcg": ("*_necessity_l1",
                       "avg:by_length/*/nec/NDCG@k", 5e-4),
}

EXPECTED_FROM_RUNS: dict[str, tuple] = {
    # ⚠️ 以下全部是 2026-11-24 修正 system 错位后的重算值（旧值已作废）
    "L1.B2.CR.overall": (0.7667, 5e-4),
    "L1.B1.CR.overall": (0.4778, 5e-4),
    "L1.B2R.CR.overall": (0.5, 5e-4),
    "L1.R0.CR.overall": (0.2778, 5e-4),
    "L1.R2.CR.overall": (0.2445, 5e-4),
    "L1.B0.CR.overall": (0.1667, 5e-4),
    "L1.LEN.CR.overall": (0.5555, 5e-4),
    "L1.B2.CR.L128": (1.0, 5e-4),
    "L1.B2.CR.L256": (0.7667, 5e-4),
    "L1.B2.CR.L512": (0.5333, 5e-4),
    "abl.layers1.auc": (0.9352, 5e-4),
    "abl.layers2.auc": (0.9340, 5e-4),
    "abl.layers3.auc": (0.9341, 5e-4),
    "abl.layers4.auc": (0.9339, 5e-4),
    "abl.layers1.params": (99585, 0),
    "abl.layers4.params": (296961, 0),
    # ---- 反事实删除校验（docs/反事实删除校验.md）----
    "cf.test.drop_pos": (0.2684, 5e-4),
    "cf.test.drop_neg": (0.0353, 5e-4),
    "cf.test.drop_neg_matched": (0.0349, 5e-4),
    "cf.test.drop_rand": (0.1911, 5e-4),
    "cf.test.strat_pos": (0.2074, 5e-4),
    "cf.test.strat_neg": (0.0213, 5e-4),
    "cf.test.redundant_pos": (0.4203, 5e-4),
    "cf.test.nec_label1": (0.9298, 5e-4),
    "cf.test.nec_label0": (0.1171, 5e-4),
    "cf.test.nec_auc": (0.7494, 5e-4),
    "cf.test.n_pairs": (672, 0),
    "cf.test.unmeasurable": (791, 0),
    "cf.test.gold_slots": (6941, 0),
    "cf.dev.drop_pos": (0.2655, 5e-4),
    "cf.dev.strat_pos": (0.2180, 5e-4),
    "cf.dev.strat_neg": (0.0172, 5e-4),
    "cf.train.drop_pos": (0.2836, 5e-4),
    "cf.train.strat_pos": (0.2174, 5e-4),
    "cf.train.strat_neg": (0.0217, 5e-4),
    # 配对版（同位置区间 + 同分母），文档 §3.3 与 §3.4
    "cf.test.ratio": (9.74, 5e-3),
    "cf.test.ratio_paired": (10.56, 5e-3),
    "cf.test.paired_pos": (0.2217, 5e-4),
    "cf.test.paired_neg": (0.0210, 5e-4),
    "cf.test.paired_n": (454, 0),
    "cf.dev.ratio_paired": (13.23, 5e-3),
    "cf.train.ratio_paired": (10.71, 5e-3),
    # 压缩上界（集合覆盖），文档 §4.4
    "cf.test.deletable_rate": (0.4924, 5e-4),
    "cf.test.pos_dropped_rate": (0.3334, 5e-4),
    "cf.dev.deletable_rate": (0.4827, 5e-4),
    "cf.train.deletable_rate": (0.5088, 5e-4),
    # ---- necessity 标签对照（docs/反事实删除校验.md §8）----
    "nec.intro.val_auc": (0.9284, 5e-4),
    "nec.nec.val_auc": (0.9002, 5e-4),
    "nec.nec.pos_rate": (0.3231, 5e-4),
    "nec.nec.margin": (0.1792, 5e-4),
    "nec.val_auc_on_intro.nec": (0.8948, 5e-4),
    "necl1.intro.cr": (0.7667, 5e-4),
    "necl1.nec.cr": (0.7778, 5e-4),
    "necl1.paired.diff": (-0.0111, 5e-4),
    "necl1.paired.ci_low": (-0.0889, 5e-4),
    "necl1.paired.ci_high": (0.0667, 5e-4),
    "necl1.paired.p": (0.896, 5e-4),
    "necl1.paired.ties": (75, 0),
    "necl1.paired.n": (90, 0),
    "necl1.intro.recall": (0.6051, 5e-4),
    "necl1.nec.recall": (0.5844, 5e-4),
    "necl1.intro.ndcg": (0.2114, 5e-4),
    "necl1.nec.ndcg": (0.2137, 5e-4),
    "necl1.paired.intro_wins": (7, 0),
    "necl1.paired.nec_wins": (8, 0),
    # needle 是否是某个 gold 槽位值的唯一文本支撑（§8.3 的原因分析）
    "necl1.needle_unique": (159, 0),
    "necl1.needle_no_unique": (21, 0),
    "necl1.L128.intro": (1.0, 5e-4),
    "necl1.L128.nec": (1.0, 5e-4),
    "necl1.L256.intro": (0.7667, 5e-4),
    "necl1.L256.nec": (0.8, 5e-4),
    "necl1.L512.intro": (0.5333, 5e-4),
    "necl1.L512.nec": (0.5333, 5e-4),
    # L2 JGA 口径校验（docs/L2_JGA校验.md §4）
    # ---- L2 任务完成率 ----
    "l2.n_scenarios": (90, 0),
    "l2.n_answerable": (152, 0),
    "l2.B2.slot_acc": (0.3778, 5e-4),
    "l2.B1.slot_acc": (0.3556, 5e-4),
    "l2.B0.slot_acc": (0.3167, 5e-4),
    "l2.R0.slot_acc": (0.2111, 5e-4),
    "l2.R2.slot_acc": (0.3056, 5e-4),
    "l2.ORACLE.slot_acc": (0.8444, 5e-4),
    "l2.B0FULL.slot_acc": (0.3222, 5e-4),
    "l2.B2.succ": (0.1667, 5e-4),
    "l2.ORACLE.succ": (0.7222, 5e-4),
    "l2.B2.ar": (0.4079, 5e-4),
    "l2.B2.ctx": (1488.9, 1e-2),
    "l2.ORACLE.ctx": (216.1, 1e-2),
    "l2.B0FULL.ctx": (7276.6, 1e-2),
    "l2.B2.cr": (0.7667, 5e-4),
    "l2.B1.cr": (0.4778, 5e-4),
    "l2.B0.cr": (0.1667, 5e-4),
    "l2.paired.B2_B1.diff": (0.0222, 5e-4),
    "l2.paired.B2_B1.p": (0.6964, 5e-4),
    "l2.paired.B2_R0.diff": (0.1667, 5e-4),
    "l2.paired.B2_R0.p": (0.0, 5e-4),
    "l2.L128.B2": (0.5, 5e-4),
    "l2.L512.B2": (0.3043, 5e-4),
    "l2.L512.B1": (0.3478, 5e-4),
    "jga.n_dialogues": (607, 0),
    "jga.n_slot_rows": (1332, 0),
    "jga.n_answerable": (1222, 0),
    "jga.slot_acc.full": (0.9174, 5e-4),
    "jga.slot_acc.drop_pos": (0.2477, 5e-4),
    "jga.slot_acc.drop_neg": (0.9204, 5e-4),
    "jga.slot_acc.drop_rand": (0.6839, 5e-4),
    "jga.slot_acc_answerable.drop_pos": (0.2676, 5e-4),
    "jga.slot_acc_answerable.drop_neg": (0.982, 5e-4),
    "jga.slot_acc_answerable.drop_rand": (0.7332, 5e-4),
    "jga.jga.full": (0.827, 5e-4),
    "jga.jga.drop_pos": (0.1598, 5e-4),
    "jga.jga.drop_neg": (0.8353, 5e-4),
    "jga.jga.drop_rand": (0.5338, 5e-4),
    "jga.paired.diff": (-0.6755, 5e-4),
    "jga.paired.ci_low": (-0.7133, 5e-4),
    "jga.paired.ci_high": (-0.6359, 5e-4),
    # ---- 分槽位细分（docs/L2_JGA校验.md §4.4）----
    "jga.by_slot.n_types": (29, 0),
    "jga.by_slot.train_day.full": (1.0, 5e-4),
    "jga.by_slot.train_day.n": (131, 0),
    "jga.by_slot.hotel_pricerange.full": (0.875, 5e-4),
    "jga.by_slot.restaurant_name.full": (0.9333, 5e-4),
    "jga.by_slot.restaurant_name.drop_pos": (0.15, 5e-4),
    "jga.by_slot.attraction_name.drop_pos": (0.1071, 5e-4),
    "jga.by_slot.train_destination.n": (119, 0),
    "jga.wide.n_dialogues": (607, 0),
    "jga.wide.n_slot_rows": (1332, 0),
    "jga.wide.drop_pos": (0.2477, 5e-4),
    "jga.wide.drop_neg": (0.9204, 5e-4),
    "jga.wide.diff": (-0.6755, 5e-4),
    # ---- 模拟抽检 κ（docs/标签抽检.md §3）----
    "spot.n": (200, 0),
    "spot.label_vs_J2.kappa": (0.43, 5e-4),
    "spot.label_vs_J2.agreement": (0.715, 5e-4),
    "spot.label_vs_J2.ci_low": (0.3051, 5e-4),
    "spot.label_vs_J2.ci_high": (0.55, 5e-4),
    "spot.label_vs_J3.kappa": (0.5, 5e-4),
    "spot.label_vs_J3.agreement": (0.75, 5e-4),
    "spot.label_vs_J3.ci_low": (0.3818, 5e-4),
    "spot.label_vs_J3.ci_high": (0.608, 5e-4),
    "spot.label_vs_J3.tp": (59, 0),
    "spot.label_vs_J3.tn": (91, 0),
    "spot.label_vs_J3.fp": (9, 0),
    "spot.label_vs_J3.fn": (41, 0),
    "spot.label_vs_J3.precision": (0.8676, 5e-4),
    "spot.label_vs_J3.recall": (0.59, 5e-4),
    "spot.J2_vs_J3.kappa": (0.7072, 5e-4),
    "spot.J2_vs_J3.agreement": (0.855, 5e-4),
    "spot.J2_vs_J3.ci_low": (0.6118, 5e-4),
    "spot.J2_vs_J3.ci_high": (0.7975, 5e-4),
    "spot.csv_filled": (50, 0),
    "spot.csv_disagree": (10, 0),
}


def _dig(d, path: str):
    """按 "/" 取字段。

    支持三种前缀写法：
      avg:by_length/*/intro/CR   → 对 by_length 下每个 key 取值再平均
      max:... / min:...          → 取最大 / 最小值
      len:summary/by_slot        → 取该容器的长度（用来验「槽位类型有几个」
                                    这类计数，否则 _dig 会返回整个 dict）
    """
    if path.startswith("len:"):
        return len(_dig(d, path[4:]))
    agg = None
    if path.startswith(("avg:", "max:", "min:")):
        agg, path = path.split(":", 1)
    parts = path.split("/")
    for i, k in enumerate(parts):
        if isinstance(d, dict):
            if k == "*":
                rest = "/".join(parts[i + 1:])
                vals = [_dig(v, rest) for v in d.values()]
                return {"avg": sum(vals) / len(vals),
                        "max": max(vals), "min": min(vals)}[agg]
            d = d[k]
        else:
            d = d[int(k)] if k.lstrip("-").isdigit() else d[k]
    return d


class Checker:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, actual, tol: float) -> None:
        exp = EXPECTED.get(name) or EXPECTED_FROM_RUNS.get(name)
        if exp is None:
            self.rows.append({"name": name, "expect": None, "actual": actual, "ok": None})
            return
        target, exp_tol = exp
        t = max(tol, exp_tol)
        ok = abs(float(actual) - float(target)) <= t
        self.rows.append({"name": name, "expect": target, "actual": actual, "ok": ok})

    def report(self) -> bool:
        w = max(len(r["name"]) for r in self.rows)
        bad = 0
        print()
        print(f"{'项目'.ljust(w)}  {'文档声称':>12}  {'实测':>12}  结果")
        print("-" * (w + 40))
        for r in self.rows:
            if r["expect"] is None:
                print(f"{r['name'].ljust(w)}  {'(未登记)':>12}  {r['actual']!s:>12}  跳过")
                continue
            mark = "OK" if r["ok"] else "FAIL"
            if not r["ok"]:
                bad += 1
            print(f"{r['name'].ljust(w)}  {r['expect']!s:>12}  {r['actual']!s:>12}  {mark}")
        print("-" * (w + 40))
        print(f"{len(self.rows)} 项检查，{bad} 项不一致")
        return bad == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过检索召回（约 40 秒）")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    t0 = time.time()
    ck = Checker()

    # ---------- 1. 数据 ----------
    log.info("[1/5] 数据规模与打分器")
    segs = load_segments(args.split)
    by_d = group_by_dialogue(segs)
    y = np.array([s.label for s in segs])
    ck.check(f"{args.split}.段数", len(segs), 0)
    ck.check(f"{args.split}.对话数", len(by_d), 0)
    ck.check(f"{args.split}.正例率", round(float(y.mean()), 4), 0)

    # ---------- 2. 打分器 ----------
    X = embed_split(args.split, segs)
    model, hist = load_scorer()
    ck.check("scorer.val_auc", round(hist.best_val_auc, 4), 0)
    ck.check("scorer.val_position_auc", round(hist.position_auc, 4), 0)
    ck.check("scorer.n_params", hist.n_params, 0)

    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    t = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        scores = model.predict(t).numpy().ravel()
    ck.check("scorer.test_auc", round(float(roc_auc_score(y, scores)), 4), 0)

    # test 上的位置基线：按对话内相对位置取负（首因），与 R0 选择器同一口径
    _pos = {s.seg_id: i for i, s in enumerate(segs)}
    _rel = np.zeros(len(segs))
    for _d, _g in by_d.items():
        _n = len(_g)
        for _k, _sg in enumerate(_g):
            _rel[_pos[_sg.seg_id]] = _k / (_n - 1) if _n > 1 else 0.0
    ck.check("scorer.test_position_auc",
             round(float(roc_auc_score(y, -_rel)), 4), 0)

    # 全库打分耗时（取 5 次中位数）
    times = []
    for _ in range(5):
        a = time.time()
        with torch.no_grad():
            model(t)
        times.append(time.time() - a)
    ck.check("score_all.seconds", round(sorted(times)[2], 3), 0)

    # ---------- 3. 相关性 / 分层 AUC ----------
    log.info("[2/5] 相关性、分层 AUC")
    # group_by_dialogue 返回的是 MemorySegment 列表（已按 turn_idx 排好），
    # 不是下标列表——这里曾把 segment 当下标用，TypeError。
    pos_of = {s.seg_id: i for i, s in enumerate(segs)}
    relpos = np.zeros(len(segs))
    for _d, group in by_d.items():
        n = len(group)
        for k, sg in enumerate(group):
            relpos[pos_of[sg.seg_id]] = k / (n - 1) if n > 1 else 0.0
    ntok = np.array([s.n_tokens for s in segs], float)

    ck.check("corr.score_relpos", round(float(np.corrcoef(scores, relpos)[0, 1]), 4), 0)
    ck.check("corr.score_ntokens", round(float(np.corrcoef(scores, ntok)[0, 1]), 4), 0)
    ck.check("corr.relpos_label", round(float(np.corrcoef(relpos, y)[0, 1]), 4), 0)

    for k, (lo, hi) in enumerate([(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]):
        m = (relpos >= lo) & (relpos < hi)
        ck.check(f"strat.q{k}.auc", round(float(roc_auc_score(y[m], scores[m])), 4), 0)

    q = np.quantile(ntok, [0, .25, .5, .75, 1.0])
    for k in range(4):
        m = (ntok >= q[k]) & (ntok <= q[k + 1]) if k == 3 else (ntok >= q[k]) & (ntok < q[k + 1])
        ck.check(f"strat.len{k}.auc", round(float(roc_auc_score(y[m], scores[m])), 4), 0)

    # ---------- 4. 从 runs/ 读 L1 结果 ----------
    log.info("[3/5] L1 结果文件")
    for name, (pattern, jpath, tol) in FROM_RUNS.items():
        # 消融只认修正后的那次（20261119_085551 配置写错段，结果无效）
        hits = [h for h in sorted(path_of("runs").glob(pattern))
                if not h.name.endswith("085551_ablation")]
        if name.startswith("abl."):
            fname = "ablation.json"
        elif name.startswith("cf."):
            fname = "counterfactual.json"
            # 早期版本没有位置控制，只认带 position_stratified 的那次
            hits = [h for h in hits
                    if "position_stratified" in json.loads(
                        (h / fname).read_text("utf-8"))]
        elif name.startswith("nec."):
            fname = "necessity.json"
        elif name.startswith("necl1."):
            fname = "necessity_l1.json"
        elif name.startswith("jga."):
            fname = "jga.json"
        elif name.startswith("spot."):
            fname = "spot_check2.json"
        elif name.startswith("l2."):
            fname = "L2_generation.json"
        else:
            fname = "L1_selection.json"
        if not hits:
            log.warning("找不到有效 run：%s", pattern)
            continue
        data = json.loads((hits[-1] / fname).read_text("utf-8"))
        ck.check(name, _dig(data, jpath), tol)

    # ---------- 5. 检索召回 ----------
    if not args.quick:
        log.info("[4/5] 检索召回（每个 L 90 场景，约 40 秒）")
        idx = {s.seg_id: i for i, s in enumerate(segs)}
        for L in (128, 256, 512):
            scs = build_scenarios(segs, lengths=(L,), n_per_length=30, seeds=(0, 1, 2))
            rec = {10: 0, 20: 0, 50: 0, 100: 0}
            n = 0
            for sc in scs:
                vi = np.array([idx[s.seg_id] for s in sc.bank])
                qv = embed_queries([sc.query])[0]
                order = np.argsort(-(X[vi] @ qv))
                local = [k for k, s in enumerate(sc.bank) if s.seg_id == sc.needle_id]
                if not local:
                    continue
                rank = int(np.where(order == local[0])[0][0])
                n += 1
                for k in rec:
                    if rank < k:
                        rec[k] += 1
            for k in rec:
                ck.check(f"recall.L{L}.top{k}", round(rec[k] / n, 4), 0)
    else:
        log.info("[4/5] 跳过检索召回（--quick）")

    # ---------- 6. needle 的唯一支撑（反事实文档 §8.3 的原因分析）----------
    log.info("[5/5] needle 唯一支撑统计（180 场景，约 5 秒）")
    from verify_counterfactual import measurable_slots

    scs180 = build_scenarios(segs, lengths=(128, 256, 512),
                             n_per_length=60, seeds=(0,))
    n_uniq = 0
    for sc in scs180:
        same = sorted((s for s in sc.bank if s.dialogue_id == sc.dialogue_id),
                      key=lambda s: s.turn_idx)
        sup = measurable_slots(same, same[-1].state)
        local = [i for i, s in enumerate(same) if s.seg_id == sc.needle_id]
        if any(len(v) == 1 and v[0] in local for v in sup.values()):
            n_uniq += 1
    ck.check("necl1.needle_unique", n_uniq, 0)
    ck.check("necl1.needle_no_unique", len(scs180) - n_uniq, 0)

    ok = ck.report()
    log.info("[5/5] 用时 %.1f 秒", time.time() - t0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
