"""
Build ECS 170 Stage 3 report PDF (template-style sections 1–3.7).
Run from ecs170-stage3 with JSON results + figures present.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT.parent / "ECS170_Stage3_Report_Saahith_Kalakuntla.pdf"
GITHUB = "https://github.com/SKalakuntla1/ECS_170_Project/tree/main/stage3"


def load_row(path: Path, label: str) -> tuple[str, float, float, float, float]:
    m = json.loads(path.read_text())["metrics"]
    return (label, m["accuracy"], m["precision_macro"], m["recall_macro"], m["f1_macro"])


def epoch_info(path: Path) -> int:
    return len(json.loads(path.read_text())["history"]["epoch"])


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a1a"),
    )
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=11, spaceBefore=8, spaceAfter=5)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10, spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=11)

    mnist_p = ROOT / "results_mnist_stage3_pdf.json"
    orl_p = ROOT / "results_orl_stage3_pdf.json"
    cifar_p = ROOT / "results_cifar_stage3_pdf.json"
    wide_p = ROOT / "results_cifar_wide_stage3_pdf.json"
    for p in (mnist_p, orl_p, cifar_p, wide_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p.name}. Run training commands in README first.")

    em = epoch_info(mnist_p)
    eo = epoch_info(orl_p)
    ec = epoch_info(cifar_p)
    ew = epoch_info(wide_p)

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="ECS 170 Stage 3 Report",
    )
    story: list = []

    story.append(Paragraph("ECS 170 — Artificial Intelligence — Spring 2026", title))
    story.append(Paragraph("Course Project: Stage 3 Report", title))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Team information</b>", h1))
    team = Table(
        [
            ["Team name", "Solo (one student)"],
            ["Student 1", "Saahith Kalakuntla"],
            ["Student 1 ID", "920895078"],
            ["Student 1 Email", "skalakuntla@ucdavis.edu"],
        ],
        colWidths=[1.32 * inch, 4.75 * inch],
    )
    team.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(team)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Section 1: Task description</b>", h1))
    story.append(
        Paragraph(
            "Stage 3 is <b>image classification</b> with <b>convolutional neural networks (CNNs)</b>. We train three separate models on "
            "<b>MNIST</b> (handwritten digits 0–9, 28×28 grayscale), <b>ORL</b> (40 people, 112×92 face images stored as RGB with identical channels—we use one channel), "
            "and <b>CIFAR-10</b> (10 object classes, 32×32 color). Each dataset already has fixed <b>train</b> and <b>test</b> splits from the course.",
            body,
        )
    )

    story.append(Paragraph("<b>Section 2: Model description</b>", h1))
    story.append(
        Paragraph(
            "Each dataset uses a custom CNN in PyTorch (<b>models.py</b>). Blocks use convolution, ReLU, and max pooling; CIFAR also uses <b>BatchNorm2d</b>. "
            "We end with <b>global average pooling</b> and a linear classifier. MNIST and CIFAR have <b>10</b> outputs; ORL has <b>40</b> outputs (one per person). Training uses "
            "<b>Adam</b> and <b>cross-entropy</b> loss (<i>train_cnn.py</i>).",
            body,
        )
    )
    arch = ROOT / "figure_stage3_architectures.png"
    if arch.is_file():
        story.append(Image(str(arch), width=6.5 * inch, height=1.65 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Section 3: Experiment settings</b>", h1))
    story.append(Paragraph("<b>3.1 Dataset description</b>", h2))
    story.append(
        Paragraph(
            "We load instructor pickles <b>MNIST</b>, <b>ORL</b>, and <b>CIFAR</b> from <b>stage_3_data/</b>. Each file has <b>train</b> and <b>test</b> lists of dictionaries with "
            "<b>image</b> arrays and integer <b>label</b>. Images are converted to NCHW tensors with pixels scaled to <b>[0, 1]</b>. ORL labels 1–40 are shifted to 0–39 for training.",
            body,
        )
    )

    story.append(Paragraph("<b>3.2 Detailed experimental setups</b>", h2))
    story.append(
        Paragraph(
            f"<b>MNIST:</b> batch size 128, Adam lr=1e-3, <b>{em}</b> epochs, default CNN. "
            f"<b>ORL:</b> batch size 32, Adam lr=3e-4, <b>{eo}</b> epochs (small dataset). "
            f"<b>CIFAR:</b> batch size 128, Adam lr=1e-3, <b>{ec}</b> epochs, default CNN. "
            f"<b>Ablation:</b> CIFAR <b>wide</b> variant (wider channels), <b>{ew}</b> epochs, same lr and batch size. "
            "Device: CUDA, Apple MPS, or CPU (auto).",
            body,
        )
    )

    story.append(Paragraph("<b>3.3 Evaluation metrics</b>", h2))
    story.append(
        Paragraph(
            "We report <b>accuracy</b> and <b>macro</b> precision, recall, and F1 (scikit-learn, <i>average='macro'</i> with all class indices) so each class is weighted equally.",
            body,
        )
    )

    story.append(Paragraph("<b>3.4 Source code</b>", h2))
    story.append(Paragraph(f"Public GitHub (no login):<br/><b><font color='blue'>{GITHUB}</font></b>", body))
    story.append(
        Paragraph(
            "Folder <b>stage3/</b> includes <b>datasets.py</b>, <b>models.py</b>, <b>train_cnn.py</b>, <b>plot_history.py</b>, <b>README.md</b>.",
            body,
        )
    )

    story.append(Paragraph("<b>3.5 Training convergence plot</b>", h2))
    story.append(
        Paragraph(
            "Epoch on the horizontal axis. For each dataset we show training vs test cross-entropy loss and test accuracy. "
            "An extra plot shows the CIFAR wide ablation run.",
            body,
        )
    )
    for png, h in [
        ("figure_stage3_mnist.png", 2.15 * inch),
        ("figure_stage3_orl.png", 2.15 * inch),
        ("figure_stage3_cifar.png", 2.15 * inch),
        ("figure_stage3_cifar_wide.png", 2.15 * inch),
    ]:
        fp = ROOT / png
        if fp.is_file():
            story.append(Image(str(fp), width=6.45 * inch, height=h))

    story.append(Paragraph("<b>3.6 Model performance</b>", h2))
    rows = [
        load_row(mnist_p, "MNIST, default CNN"),
        load_row(orl_p, "ORL, default CNN"),
        load_row(cifar_p, "CIFAR-10, default CNN"),
        load_row(wide_p, "CIFAR-10, wide variant (ablation)"),
    ]
    td = [["Dataset / config", "Acc", "P(macro)", "R(macro)", "F1(macro)"]]
    for label, acc, pr, rc, f1 in rows:
        td.append([label, fmt4(acc), fmt4(pr), fmt4(rc), fmt4(f1)])
    perf = Table(td, colWidths=[2.55 * inch, 0.78 * inch, 0.88 * inch, 0.88 * inch, 0.88 * inch])
    perf.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(Spacer(1, 3))
    story.append(perf)

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>3.7 Ablation studies</b>", h2))
    m_def = json.loads(cifar_p.read_text())["metrics"]["accuracy"]
    m_wide = json.loads(wide_p.read_text())["metrics"]["accuracy"]
    cmp_txt = "improved" if m_wide > m_def else "was lower than" if m_wide < m_def else "matched"
    story.append(
        Paragraph(
            f"We changed CIFAR channel width using the <b>wide</b> flag in <b>models.py</b> (more feature maps in each block). "
            f"Compared to the default CIFAR CNN on this run, the wide model test accuracy <b>{cmp_txt}</b> the default "
            f"({fmt4(m_def)} vs {fmt4(m_wide)}). ORL remains harder with only 360 training images and 40 classes.",
            body,
        )
    )

    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>10 pt body, single-spaced. Metrics and curves from local JSON training logs.</i>", small))

    doc.build(story)
    print("Wrote", OUT_PDF)


if __name__ == "__main__":
    build_pdf()
