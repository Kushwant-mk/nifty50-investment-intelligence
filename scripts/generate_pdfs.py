from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _clean(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "")
    )


def markdown_to_report_pdf(markdown_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallBullet", parent=styles["BodyText"], leftIndent=18, firstLineIndent=-10, spaceAfter=4))
    story = []

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 0.08 * inch))
            continue
        if line.startswith("# "):
            story.append(Paragraph(_clean(line[2:]), styles["Title"]))
            story.append(Spacer(1, 0.15 * inch))
        elif line.startswith("## "):
            story.append(Paragraph(_clean(line[3:]), styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"- {_clean(line[2:])}", styles["SmallBullet"]))
        else:
            story.append(Paragraph(_clean(line), styles["BodyText"]))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48)
    doc.build(story)


def markdown_to_slide_pdf(markdown_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name="SlideTitle", parent=styles["Title"], fontSize=26, leading=30, spaceAfter=18)
    body_style = ParagraphStyle(name="SlideBody", parent=styles["BodyText"], fontSize=16, leading=22, spaceAfter=10)
    bullet_style = ParagraphStyle(name="SlideBullet", parent=body_style, leftIndent=24, firstLineIndent=-14)
    story = []
    first_slide = True

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("# "):
            continue
        if line.startswith("## "):
            if not first_slide:
                story.append(PageBreak())
            first_slide = False
            story.append(Spacer(1, 0.45 * inch))
            story.append(Paragraph(_clean(line[3:]), title_style))
        elif line.startswith("- "):
            story.append(Paragraph(f"- {_clean(line[2:])}", bullet_style))
        else:
            story.append(Paragraph(_clean(line), body_style))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=64,
        leftMargin=64,
        topMargin=44,
        bottomMargin=44,
    )
    doc.build(story)


def main() -> None:
    markdown_to_report_pdf(DOCS / "REPORT.md", DOCS / "REPORT.pdf")
    markdown_to_slide_pdf(DOCS / "PRESENTATION.md", DOCS / "PRESENTATION.pdf")
    print("Generated docs/REPORT.pdf and docs/PRESENTATION.pdf")


if __name__ == "__main__":
    main()
