"""Plot learning curves from stage2_train.py JSON output."""

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
    epochs = h["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, h["train_loss"], label="train CE loss")
    axes[0].plot(epochs, h["test_loss"], label="test CE loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Cross-entropy vs epoch")

    axes[1].plot(epochs, h["test_accuracy"], label="test accuracy", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].set_title("Test accuracy vs epoch")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
