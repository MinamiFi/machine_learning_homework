#!/usr/bin/env python3
"""把实验报告 Markdown 转成 docx（无 pandoc 环境下的轻量转换器）。

只支持本项目报告用到的语法：标题、段落、粗体、行内代码、围栏代码块、
无序/有序列表、管道表格、水平线、引用块。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CJK = "微软雅黑"
MONO = "Consolas"


def set_cjk(run, font=CJK):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), font)
    rf.set(qn("w:ascii"), font)
    rf.set(qn("w:hAnsi"), font)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(par, text: str, base_bold=False, size=None):
    """处理 **粗体** 与 `行内代码`。"""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            r = par.add_run(piece[2:-2])
            r.bold = True
            set_cjk(r)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            r.font.name = MONO
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
        else:
            r = par.add_run(piece)
            r.bold = base_bold
            set_cjk(r)
        if size:
            r.font.size = Pt(size)


def add_code_block(doc, lines):
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.6)
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(8)
    par.paragraph_format.line_spacing = 1.0
    # 浅灰底纹
    pPr = par._p.get_or_add_pPr()
    shd = pPr.makeelement(qn("w:shd"), {})
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F4F4F4")
    pPr.append(shd)
    for i, ln in enumerate(lines):
        r = par.add_run(ln)
        r.font.name = MONO
        r.font.size = Pt(8.5)
        if i < len(lines) - 1:
            r.add_break()


def split_row(line: str):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line)) and "-" in line


def add_table(doc, rows):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            par = c.paragraphs[0]
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
            add_inline(par, cell, base_bold=(i == 0), size=8.5)
            if i == 0:
                for r in par.runs:
                    r.bold = True
    doc.add_paragraph()


def convert(md_path: Path, docx_path: Path):
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    doc = Document()
    # 页面与正文样式
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    st = doc.styles["Normal"]
    st.font.name = CJK
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    st.paragraph_format.line_spacing = 1.45
    st.paragraph_format.space_after = Pt(6)

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]

        # 围栏代码块
        if ln.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, buf)
            continue

        # 表格
        if ln.strip().startswith("|") and i + 1 < n and is_sep(lines[i + 1]):
            rows = [split_row(ln)]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            add_table(doc, rows)
            continue

        # 水平线
        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", ln):
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            pbdr = pPr.makeelement(qn("w:pBdr"), {})
            bottom = pbdr.makeelement(qn("w:bottom"), {})
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "CCCCCC")
            pbdr.append(bottom)
            pPr.append(pbdr)
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(title)
                r.bold = True
                r.font.size = Pt(20)
                set_cjk(r)
                p.paragraph_format.space_after = Pt(14)
            else:
                h = doc.add_heading(level=min(level - 1, 4))
                h.text = ""
                r = h.add_run(title)
                r.bold = True
                r.font.size = Pt({2: 15, 3: 13, 4: 11.5, 5: 10.5}.get(level, 10.5))
                r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
                set_cjk(r)
                h.paragraph_format.space_before = Pt(10)
                h.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # 引用块
        if ln.strip().startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            for b in buf:
                if not b.strip():
                    continue
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.8)
                p.paragraph_format.space_after = Pt(2)
                add_inline(p, b)
                for r in p.runs:
                    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            continue

        # 列表
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", ln)
        if m:
            indent = len(m.group(1)) // 2
            ordered = m.group(2)[0].isdigit()
            style = "List Number" if ordered else "List Bullet"
            p = doc.add_paragraph(style=style)
            p.paragraph_format.left_indent = Cm(0.75 + 0.6 * indent)
            p.paragraph_format.space_after = Pt(2)
            add_inline(p, m.group(3))
            i += 1
            continue

        # 空行
        if not ln.strip():
            i += 1
            continue

        # 普通段落
        p = doc.add_paragraph()
        add_inline(p, ln.strip())
        i += 1

    doc.save(docx_path)
    print(f"OK {docx_path}  ({docx_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".docx")
    convert(src, dst)
