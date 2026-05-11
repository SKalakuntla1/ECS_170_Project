"""Generate architecture diagram and convergence figures for the Stage 2 PDF report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, w, text, h=1.8):
        r = mpatches.FancyBboxPatch(
            (x, 0.6), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", linewidth=1.2, edgecolor="#333", facecolor="#e8f4fc"
        )
        ax.add_patch(r)
        ax.text(x + w / 2, 0.6 + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)

    box(0.2, 1.6, "Input\n784 features")
    box(2.3, 1.4, "Linear\n128 + ReLU\nDropout 0.2")
    box(4.2, 1.4, "Linear\n64 + ReLU\nDropout 0.2")
    box(6.1, 1.6, "Linear\n10 logits\n(digits 0–9)")
    for x0, x1 in [(1.8, 2.3), (3.7, 4.2), (5.6, 6.1)]:
        ax.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.set_title("MLP architecture (baseline: hidden 128 → 64)", fontsize=11, pad=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_convergence(json_path: Path, out_path: Path) -> None:
    data = json.loads(json_path.read_text())
    h = data["history"]
    epochs = h["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, h["train_loss"], label="train loss")
    axes[0].plot(epochs, h["test_loss"], label="test loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Loss vs epoch (baseline)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, h["test_accuracy"], color="tab:green", label="test accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Test accuracy vs epoch (baseline)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_test_accuracy(paths_labels: list[tuple[Path, str]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for path, label in paths_labels:
        data = json.loads(path.read_text())
        h = data["history"]
        ax.plot(h["epoch"], h["test_accuracy"], label=label, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("Test accuracy vs epoch (all runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    draw_architecture(root / "figure_architecture.png")
    plot_baseline_convergence(root / "results_baseline.json", root / "figure_convergence_baseline.png")
    plot_compare_test_accuracy(
        [
            (root / "results_baseline.json", "Baseline [128,64] Adam"),
            (root / "results_deeper.json", "Deeper [256,128,64] Adam"),
            (root / "results_sgd.json", "Baseline shape SGD"),
        ],
        root / "figure_test_accuracy_compare.png",
    )
    print("Wrote figure_architecture.png, figure_convergence_baseline.png, figure_test_accuracy_compare.png")


if __name__ == "__main__":
    main()
