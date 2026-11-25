"""报告生成：结果 → markdown 表格。"""

from __future__ import annotations

import json


def to_table(rows: list[dict], cols: list[str] | None = None) -> str:
    """dict 列表 → markdown 表格。"""
    if not rows:
        return "（空）"
    cols = cols or list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join(
        "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        for r in rows
    )
    return "\n".join([head, sep, body])


def to_markdown(results: dict, title: str = "实验结果", notes: str = "") -> str:
    """结果字典 → 完整 markdown 报告。"""
    lines = [f"# {title}", ""]
    if notes:
        lines += [notes, ""]

    rows = []
    for name, r in results.items():
        if hasattr(r, "summary"):
            rows.append(r.summary())
        elif isinstance(r, dict):
            rows.append({"name": name, **r})

    lines += [to_table(rows), ""]
    return "\n".join(lines)


def save_json(obj, p) -> None:
    from ..common import dump_json

    dump_json(obj, p)
