"""Plot learning curves from train_cnn.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=str)
    ap.add_argument("--out", type=str, default="convergence.png")
    args = ap.parse_args()
    data = json.loads(Path(args.json_path).read_text())
    h = data["history"]
    e = h["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(e, h["train_loss"], label="train loss")
    axes[0].plot(e, h["test_loss"], label="test loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{data['dataset']} — loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(e, h["test_accuracy"], color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title(f"{data['dataset']} — test accuracy ({data.get('variant','')})")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
