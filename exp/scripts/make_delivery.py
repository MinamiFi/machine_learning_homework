#!/usr/bin/env python3
"""从 exp/ 重建 交付/ 目录（除两份手写文档外的全部内容）。

为什么需要这个脚本：交付包里 `code/`、`figures/`、`results/`、`data_samples/`、
`training_logs/`、`ablation/`、`报告/附件/` 都是 `exp/` 的副本或派生物。
把这些副本提交进 git 会让仓库里出现两份相同代码，读者无法判断哪份是源头。
所以：**`exp/` 是唯一源头，`交付/` 由本脚本生成**，生成物不进版本库。

手写的两份（不会被本脚本覆盖）：
  交付/报告/实验报告.md
  交付/交付说明.md

用法：
    python3 exp/scripts/make_delivery.py            # 重建
    python3 exp/scripts/make_delivery.py --check    # 只检查是否与 exp/ 一致
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import exp_root, get_logger, path_of  # noqa: E402

log = get_logger("make_delivery")

# verify_docs.py 与 make_figures.py 实际引用的 12 个权威 run（与根 .gitignore 白名单一致）
AUTHORITATIVE_RUNS = [
    "20261124_021731_scorer_scorer_base_s0_s0",
    "20261124_092829_eval_l1_eval_l1_full2",
    "20261124_022842_ablation",
    "20261124_093413_budget",
    "20261124_032401_counterfactual_counterfactual_test",
    "20261124_022143_counterfactual_counterfactual_dev",
    "20261124_022148_counterfactual_counterfactual_train",
    "20261124_093705_necessity_l1",
    "20261124_030223_necessity",
    "20261124_033153_spot_check2_n200",
    "20261124_081620_jga_api_full",
    "20261124_094210_eval_l2_l2_main_api2",
]

# 交付/training_logs 里放哪些运行日志
DELIVERED_LOGS = {
    "rerun_local.log": "L1 + 预算扫描 + necessity 全量重跑",
    "jga_api_full.log": "JGA 全量（5328 次生成，API）",
    "l2_main_api2.log": "L2 主表（180 槽位，API）",
    "rerun_checks.log": "反事实 / 抽检 / 消融等校验",
}

# 交付/报告/附件 里放哪些文档（源路径 → 交付名）
DELIVERED_DOCS = {
    "README.md": "项目README_v7.md",
    "exp/README.md": "实验环境说明.md",
    "exp/src/README.md": "代码模块设计.md",
    "exp/docs/L1实验结果.md": "L1_选择质量.md",
    "exp/docs/L2任务完成率.md": "L2_任务完成率.md",
    "exp/docs/L2_JGA校验.md": "标签校验_JGA.md",
    "exp/docs/反事实删除校验.md": "标签校验_反事实删除.md",
    "exp/docs/标签抽检.md": "标签校验_模拟抽检.md",
}

# 日志里的绝对路径替换（避免把本机目录结构带进仓库）。
# 仓库根在运行时解析，不写死，换机器也能用。
_SITE_RE = re.compile(r"(/[^\s:]*?)/site-packages")


def _rewrites() -> list[tuple[str, str]]:
    return [(str(exp_root().parent), "<REPO_ROOT>")]


def _sanitize_text(text: str) -> str:
    for a, b in _rewrites():
        text = text.replace(a, b)
    return _SITE_RE.sub(r"<PY_SITE>/site-packages", text)

# 密钥扫描：不写具体值（否则扫描器自身会被自己扫到），只按模式匹配
SECRET_PATTERNS = [
    re.compile(rb"sk-[0-9a-f]{32}"),          # 常见 API key 形态
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),     # GitHub PAT
    re.compile(rb"AKIA[0-9A-Z]{16}"),         # AWS
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")


def _rewrite_links(src: Path, dst: Path):
    """把 markdown 里的相对链接改写成从 dst 出发仍能解析的路径。

    交付/ 里的文档是从 exp/ 复制过来的，原链接（如 `exp/docs/L1实验结果.md`）
    是相对仓库根的，复制到 交付/报告/附件/ 后就会失效。这里按「先在原位置解析，
    再相对新位置重算」的方式改写，保证两份文档的链接都有效。
    """
    try:
        text = dst.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return

    def repl(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        if not target:
            return m.group(0)
        origin = (src.parent / target).resolve()
        if not origin.exists():
            return m.group(0)  # 原本就失效的链接不动
        new = os.path.relpath(origin, dst.parent.resolve())
        return f"[{label}]({new}{anchor})"

    new_text = _LINK_RE.sub(repl, text)
    if new_text != text:
        dst.write_text(new_text, encoding="utf-8")


def _copy_tree(src: Path, dst: Path, skip_names=("__pycache__",)):
    if not src.is_dir():
        log.warning("跳过（不存在）：%s", src)
        return 0
    n = 0
    for p in src.rglob("*"):
        if any(s in p.parts for s in skip_names):
            continue
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        log.warning("跳过（不存在）：%s", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _sanitize(path: Path):
    """替换日志里的本机绝对路径。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IsADirectoryError):
        return
    new = _sanitize_text(text)
    if new != text:
        path.write_text(new, encoding="utf-8")


def build(check_only: bool = False) -> int:
    root = exp_root().parent
    out = root / "交付"
    if not check_only:
        out.mkdir(exist_ok=True)

    made = {}

    # 1) 代码
    if not check_only:
        code = out / "code"
        n = 0
        n += _copy_tree(path_of("src"), code / "src")
        n += _copy_tree(path_of("scripts"), code / "scripts")
        n += _copy_tree(path_of("configs"), code / "configs")
        for f in ("requirements.txt", "requirements-torch.txt", "requirements.lock.txt", "env.sh"):
            if _copy_file(exp_root() / f, code / f):
                n += 1
        made["code"] = n

    # 2) 图
    if not check_only:
        made["figures"] = _copy_tree(path_of("figures"), out / "figures")

    # 3) 权威 runs
    if not check_only:
        n = 0
        for r in AUTHORITATIVE_RUNS:
            d = path_of("runs") / r
            if not d.is_dir():
                log.warning("run 目录不存在：%s", r)
                continue
            n += _copy_tree(d, out / "results" / "runs" / r)
        made["results"] = n

    # 4) 训练日志
    if not check_only:
        tl = out / "training_logs"
        tl.mkdir(parents=True, exist_ok=True)
        scorer = path_of("runs") / AUTHORITATIVE_RUNS[0]
        for f in ("train_history.steps.jsonl", "train_history.json"):
            _copy_file(scorer / f, tl / f)
        _copy_file(scorer / "result.json", tl / "scorer_result.json")
        n = 0
        for fname in DELIVERED_LOGS:
            if _copy_file(path_of("logs") / fname, tl / fname):
                _sanitize(tl / fname)
                n += 1
        made["training_logs"] = n

    # 5) 消融
    if not check_only:
        ab = out / "ablation"
        ab.mkdir(parents=True, exist_ok=True)
        _copy_file(path_of("runs") / "20261124_022842_ablation" / "ablation.json", ab / "ablation.json")
        _copy_file(path_of("runs") / "20261124_093413_budget" / "budget_sweep.json", ab / "budget_sweep.json")
        _copy_file(path_of("figures") / "fig2_ablation.png", ab / "消融_三因素.png")
        _copy_file(path_of("figures") / "fig3_budget_sweep.png", ab / "预算扫描.png")

    # 6) 数据样本
    if not check_only:
        ds = out / "data_samples"
        ds.mkdir(parents=True, exist_ok=True)
        for f in ("train", "dev", "test"):
            _copy_file(path_of("data", "raw", "multiwoz22", f"{f}.parquet"), ds / f"{f}.parquet")
        for f in ("sample.jsonl", "stats.json"):
            _copy_file(path_of("data", "processed", "multiwoz22", f), ds / f)
        for src_name, dst_name in (("README.md", "数据说明.md"), ("标签说明.md", "标签说明.md")):
            src = path_of("data") / src_name
            if _copy_file(src, ds / dst_name):
                _rewrite_links(src, ds / dst_name)
        _copy_file(path_of("docs") / "标签抽检表.csv", ds / "标签抽检表.csv")

    # 7) 报告附件
    if not check_only:
        att = out / "报告" / "附件"
        att.mkdir(parents=True, exist_ok=True)
        for src_rel, name in DELIVERED_DOCS.items():
            src = root / src_rel
            if _copy_file(src, att / name):
                _rewrite_links(src, att / name)
        _copy_file(root / "题目原图.jpg", att / "题目原图.jpg")

    # 8) 报告 docx
    md = out / "报告" / "实验报告.md"
    if md.is_file():
        docx = out / "报告" / "实验报告.docx"
        if check_only:
            log.info("（check 模式跳过 docx 生成）")
        else:
            subprocess.run(
                [sys.executable, str(path_of("scripts") / "md2docx.py"), str(md), str(docx)],
                check=True,
                capture_output=True,
            )
            made["docx"] = 1
    else:
        log.warning("缺少手写报告 %s，跳过 docx 生成", md)

    # 9) 权限统一
    if not check_only:
        for p in out.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)

    # 10) 密钥扫描
    hits = []
    if not check_only:
        for p in out.rglob("*"):
            if p.is_file():
                try:
                    data = p.read_bytes()
                except OSError:
                    continue
                if any(pat.search(data) for pat in SECRET_PATTERNS):
                    hits.append(str(p.relative_to(root)))
    if hits:
        log.error("交付目录发现密钥，必须处理：%s", hits)
        return 1

    if check_only:
        log.info("check 模式：未写入任何文件")
    else:
        log.info("交付目录已重建：%s", {k: v for k, v in made.items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="从 exp/ 重建 交付/")
    ap.add_argument("--check", action="store_true", help="只检查，不写文件")
    args = ap.parse_args()
    return build(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
