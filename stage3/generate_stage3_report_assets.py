"""Figures for Stage 3 PDF: CNN architecture overview + convergence plots from JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ROOT = Path(__file__).resolve().parent


def draw_cnn_overview(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.4))
    titles = ["MNIST CNN\n1×28×28", "ORL CNN\n1×112×92", "CIFAR CNN\n3×32×32"]
    flows = [
        "Conv→ReLU→Pool\n×3 blocks\nGAP → 10 classes",
        "Conv→ReLU→Pool\n×4 blocks\nGAP → 40 people",
        "Conv+BN→ReLU\nPool ×2\nGAP → 10 classes",
    ]
    for ax, title, txt in zip(axes, titles, flows):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0.08, 0.12), 0.84, 0.76, boxstyle="round,pad=0.02", facecolor="#eef6ff", edgecolor="#333"))
        ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(0.5, 0.38, txt, ha="center", va="center", fontsize=7.5)
    fig.suptitle("Convolutional networks used (PyTorch)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run(json_path: Path, out_path: Path) -> None:
    data = json.loads(json_path.read_text())
    h = data["history"]
    e = h["epoch"]
    ds = data["dataset"].upper()
    var = data.get("variant", "default")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    axes[0].plot(e, h["train_loss"], label="train")
    axes[0].plot(e, h["test_loss"], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("CE loss")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"{ds} loss ({var})")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(e, h["test_accuracy"], color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test acc")
    axes[1].set_title(f"{ds} test accuracy")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_cnn_overview(ROOT / "figure_stage3_architectures.png")
    pairs = [
        ("results_mnist_stage3_pdf.json", "figure_stage3_mnist.png"),
        ("results_orl_stage3_pdf.json", "figure_stage3_orl.png"),
        ("results_cifar_stage3_pdf.json", "figure_stage3_cifar.png"),
        ("results_cifar_wide_stage3_pdf.json", "figure_stage3_cifar_wide.png"),
    ]
    for jname, png in pairs:
        p = ROOT / jname
        if p.is_file():
            plot_run(p, ROOT / png)
            print("wrote", png)
    print("wrote figure_stage3_architectures.png")


if __name__ == "__main__":
    main()
