"""Build the public English technical PDF from its Markdown source only.

This documentation-only command never opens data/, models, or evaluation output.
Requires ReportLab. Visual QA is performed separately after rendering the PDF.
"""
from __future__ import annotations

import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph,
    Preformatted, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#132b46")
TEAL = colors.HexColor("#147d83")
PALE = colors.HexColor("#edf5f5")
GRAY = colors.HexColor("#526171")


def register_fonts():
    """Embed available Windows fonts, with portable built-in fallbacks."""
    directory = Path("C:/Windows/Fonts")
    for regular, bold, italic in (
        ("calibri.ttf", "calibrib.ttf", "calibrii.ttf"),
        ("arial.ttf", "arialbd.ttf", "ariali.ttf"),
    ):
        if all((directory / name).is_file() for name in (regular, bold, italic)):
            for label, name in zip(("NM", "NM-Bold", "NM-Italic"), (regular, bold, italic)):
                pdfmetrics.registerFont(TTFont(label, str(directory / name)))
            pdfmetrics.registerFontFamily("NM", normal="NM", bold="NM-Bold", italic="NM-Italic", boldItalic="NM-Bold")
            body = "NM"
            break
    else:
        body = "Helvetica"
    mono = "Courier"
    if (directory / "consola.ttf").is_file():
        pdfmetrics.registerFont(TTFont("NM-Mono", str(directory / "consola.ttf")))
        mono = "NM-Mono"
    return body, mono


def inline(text, mono):
    """Translate the limited public-source Markdown syntax to safe XML."""
    pattern = r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\(https?://[^)]+\))"
    pieces = []
    for token in re.split(pattern, text):
        if token.startswith("`") and token.endswith("`"):
            code = token[1:-1]
            escaped = html.escape(code)
            pieces.append(f'<font name="{mono}" size="8.8">{escaped}</font>')
        elif token.startswith("**") and token.endswith("**"):
            pieces.append(f"<b>{html.escape(token[2:-2])}</b>")
        elif token.startswith("["):
            match = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", token)
            if match:
                label, url = match.groups()
                pieces.append(f'<link href="{html.escape(url, quote=True)}" color="#147d83"><u>{html.escape(label)}</u></link>')
            else:
                pieces.append(html.escape(token))
        else:
            pieces.append(html.escape(token))
    return "".join(pieces)


def build(source: Path, output: Path):
    body, mono = register_fonts()
    styles = {
        "body": ParagraphStyle("Body", fontName=body, fontSize=10.7, leading=14.4, textColor=NAVY, spaceAfter=8, alignment=TA_LEFT),
        "title": ParagraphStyle("Title", fontName=body, fontSize=28, leading=32, textColor=NAVY, spaceAfter=4),
        "heading": ParagraphStyle("Heading", fontName=body, fontSize=15, leading=18, textColor=TEAL, spaceBefore=10, spaceAfter=9, keepWithNext=True),
        "cell": ParagraphStyle("Cell", fontName=body, fontSize=10, leading=12.8, textColor=NAVY),
        "code": ParagraphStyle("Code", fontName=mono, fontSize=8.8, leading=12.8, textColor=NAVY, backColor=PALE, borderPadding=8, spaceAfter=10),
    }
    lines = source.read_text(encoding="utf-8").splitlines()
    if sum(line.strip() == "<!-- page -->" for line in lines) != 3:
        raise ValueError("The four-page source must contain exactly three page separators")
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- page -->":
            story.append(PageBreak())
            i += 1
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:], mono), styles["title"]))
            i += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:], mono), styles["heading"]))
            i += 1
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(block), styles["code"]))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
                    rows.append([Paragraph(inline(cell, mono), styles["cell"]) for cell in cells])
                i += 1
            width = A4[0] - 96
            table = Table(rows, colWidths=[width * 0.34, width * 0.66], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d8e3e8")),
            ]))
            story.extend((table, Spacer(1, 10)))
        else:
            bullet = line.startswith("- ")
            paragraph = [line[2:] if bullet else line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "```", "<!--")):
                paragraph.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(paragraph), mono), styles["body"], bulletText="\u2022" if bullet else None))

    def decorate(canvas, doc):
        if doc.page > 4:
            raise ValueError("Content overflow: expected four pages; revise source/layout")
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(0.8)
        canvas.line(48, A4[1] - 35, A4[0] - 48, A4[1] - 35)
        canvas.setFont(body, 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(48, A4[1] - 27, "NORTHMIND  /  TECHNICAL REPORT")
        canvas.drawString(48, 27, "More Than a Hack 2027 | 17 November 2026")
        canvas.drawRightString(A4[0] - 48, 27, f"Page {doc.page}/4")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(output), pagesize=A4, leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48, title="Northmind - Algorithmic football possession", author="Northmind", subject="Technical method, evidence and reproduction")
    doc.addPageTemplates(PageTemplate(id="technical", frames=[Frame(48, 48, A4[0] - 96, A4[1] - 96, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=decorate))
    doc.build(story)


def main() -> int:
    source = ROOT / "docs/submission/Northmind.en.md"
    if not source.is_file():
        source = ROOT / "Northmind.md"
    build(source, ROOT / "output/pdf/Northmind.pdf")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
