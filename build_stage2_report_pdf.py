"""
Build ECS 170 Stage 2 report PDF (template-style sections 1–3.7).
Requires: pip install reportlab (inside project venv).
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT.parent / "ECS170_Stage2_Report_Saahith_Kalakuntla.pdf"

GITHUB = "https://github.com/SKalakuntla1/ECS_170_Project"


def load_metrics() -> list[tuple[str, float, float, float, float]]:
    rows = []
    specs = [
        ("results_baseline.json", "Baseline: [128,64], Adam, lr=0.001"),
        ("results_deeper.json", "Deeper: [256,128,64], Adam, lr=0.001"),
        ("results_sgd.json", "Baseline shape, SGD (momentum 0.9)"),
    ]
    for fname, label in specs:
        p = ROOT / fname
        m = json.loads(p.read_text())["metrics"]
        rows.append(
            (
                label,
                m["accuracy"],
                m["precision_macro"],
                m["recall_macro"],
                m["f1_macro"],
            )
        )
    return rows


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=14,
        textColor=colors.HexColor("#1a1a1a"),
    )
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=11, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#000"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#000"))
    body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=11)

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="ECS 170 Stage 2 Report",
    )
    story: list = []

    story.append(Paragraph("ECS 170 — Artificial Intelligence — Spring 2026", title))
    story.append(Paragraph("Course Project: Stage 2 Report", title))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Team information</b>", h1))
    team_data = [
        ["Team name", "Solo (one student)"],
        ["Student 1", "Saahith Kalakuntla"],
        ["Student 1 ID", "920895078"],
        ["Student 1 Email", "skalakuntla@ucdavis.edu"],
    ]
    t = Table(team_data, colWidths=[1.35 * inch, 4.7 * inch])
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Section 1: Task description</b>", h1))
    story.append(
        Paragraph(
            "This project is digit classification. We use the Stage 2 course data. Each row is one digit image stored as "
            "<b>784</b> numbers (a 28 by 28 grayscale image flattened). The label is <b>0 through 9</b>. The model learns to predict the digit from those inputs.",
            body,
        )
    )

    story.append(Paragraph("<b>Section 2: Model description</b>", h1))
    story.append(
        Paragraph(
            "We use a fully connected neural network (MLP) in PyTorch. The script is <b>stage2_train.py</b> in the public GitHub repo (Section 3.4). "
            "Each hidden block is: <b>Linear → ReLU → Dropout</b>. The output layer has <b>10</b> logits. Training uses <b>cross-entropy loss</b> "
            "(PyTorch <i>CrossEntropyLoss</i>). The baseline uses hidden sizes <b>128</b> then <b>64</b>.",
            body,
        )
    )
    arch = ROOT / "figure_architecture.png"
    if arch.is_file():
        story.append(Spacer(1, 4))
        story.append(Image(str(arch), width=6.2 * inch, height=1.38 * inch))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Section 3: Experiment settings</b>", h1))

    story.append(Paragraph("<b>3.1 Dataset description</b>", h2))
    story.append(
        Paragraph(
            "We use <b>train.csv</b> and <b>test.csv</b> in <b>stage_2_data</b>. There is no header row. Column <b>0</b> is the label. "
            "Columns <b>1–784</b> are pixels. We normalize using the <b>training</b> mean and standard deviation only, then apply the same transform to the test file. "
            "We do not split the data again; the course provides fixed train and test files.",
            body,
        )
    )

    story.append(Paragraph("<b>3.2 Detailed experimental setups</b>", h2))
    story.append(
        Paragraph(
            "<b>Baseline:</b> hidden layers [128, 64], dropout 0.2, batch size 64, 50 epochs, Adam with learning rate 0.001, cross-entropy loss. "
            "<b>Device:</b> the code picks CUDA, Apple MPS, or CPU automatically. "
            "<b>Ablations:</b> a deeper/wider network [256, 128, 64] with Adam, and the baseline shape with SGD (momentum 0.9) instead of Adam.",
            body,
        )
    )

    story.append(Paragraph("<b>3.3 Evaluation metrics</b>", h2))
    story.append(
        Paragraph(
            "We report <b>accuracy</b> and <b>macro</b> precision, recall, and F1 using scikit-learn with <i>average='macro'</i>, so each digit class counts equally.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Accuracy</b> is the fraction of test examples where the prediction matches the true label. "
            "<b>Macro precision</b> averages per-class precision. <b>Macro recall</b> averages per-class recall. "
            "<b>Macro F1</b> averages per-class F1 scores.",
            body,
        )
    )

    story.append(Paragraph("<b>3.4 Source code</b>", h2))
    story.append(Paragraph(f"Public GitHub repository (no login required):<br/><b><font color='blue'>{GITHUB}</font></b>", body))
    story.append(
        Paragraph(
            "The repository includes <b>stage2_train.py</b>, <b>plot_history.py</b>, <b>generate_report_assets.py</b>, <b>requirements_stage2.txt</b>, and <b>README.md</b>.",
            body,
        )
    )

    story.append(Paragraph("<b>3.5 Training convergence plot</b>", h2))
    story.append(
        Paragraph(
            "Below: epoch on the horizontal axis. The left plot shows training and test cross-entropy loss. The right plot shows test accuracy for the baseline. "
            "The third plot compares test accuracy across all three runs. Loss decreases and then levels off, which matches normal gradient-based training.",
            body,
        )
    )
    p1 = ROOT / "figure_convergence_baseline.png"
    p2 = ROOT / "figure_test_accuracy_compare.png"
    if p1.is_file():
        story.append(Image(str(p1), width=6.4 * inch, height=2.55 * inch))
        story.append(Spacer(1, 6))
    if p2.is_file():
        story.append(Image(str(p2), width=5.8 * inch, height=3.15 * inch))

    story.append(Paragraph("<b>3.6 Model performance</b>", h2))
    story.append(Paragraph("Test set results on <b>stage_2_data</b> (macro precision, recall, F1 from our training script).", body))

    metric_rows = load_metrics()
    table_data = [["Configuration", "Accuracy", "Prec (macro)", "Recall (macro)", "F1 (macro)"]]
    for label, acc, pr, rc, f1 in metric_rows:
        table_data.append([label, fmt4(acc), fmt4(pr), fmt4(rc), fmt4(f1)])
    perf = Table(table_data, colWidths=[2.35 * inch, 0.85 * inch, 0.95 * inch, 0.95 * inch, 0.85 * inch])
    perf.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(Spacer(1, 4))
    story.append(perf)

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>3.7 Ablation studies</b>", h2))
    story.append(
        Paragraph(
            "We changed network depth/width and the optimizer. Other settings stayed the same. On <b>this</b> run, the deeper [256,128,64] model "
            "reached the <b>highest</b> test accuracy. The baseline Adam model was a bit lower. SGD was in the middle. Small gaps can change with random seed and training noise.",
            body,
        )
    )

    story.append(Spacer(1, 12))
    story.append(Paragraph("<i>Font: 10 pt, single-spaced body text. Report generated from measured JSON results and figures.</i>", small))

    doc.build(story)
    print("Wrote", OUT_PDF)


if __name__ == "__main__":
    build_pdf()
