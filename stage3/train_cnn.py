"""Train and evaluate CNNs for ECS 170 Stage 3 (MNIST, ORL, CIFAR)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from datasets import build_tensors
from models import build_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def macro_metrics(y_true, y_pred, num_classes: int) -> dict[str, float]:
    avg = "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average=avg, zero_division=0, labels=list(range(num_classes)))),
        "recall_macro": float(recall_score(y_true, y_pred, average=avg, zero_division=0, labels=list(range(num_classes)))),
        "f1_macro": float(f1_score(y_true, y_pred, average=avg, zero_division=0, labels=list(range(num_classes)))),
    }


def run_epoch(model, loader, device, train: bool, optimizer, criterion):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    n = 0
    all_y, all_p = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
        all_y.append(yb.detach().cpu())
        all_p.append(logits.argmax(1).detach().cpu())

    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()
    return total_loss / max(n, 1), y, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="stage_3_data")
    ap.add_argument("--dataset", type=str, required=True, choices=["mnist", "orl", "cifar"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--variant", type=str, default="default", help="default | wide | heavy_dropout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    data_root = Path(args.data_root)

    train_ds, test_ds, num_classes = build_tensors(data_root, args.dataset)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    model = build_model(args.dataset, args.variant).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"epoch": [], "train_loss": [], "test_loss": [], "test_accuracy": []}

    for ep in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, device, True, optimizer, criterion)
        te_loss, y_te, p_te = run_epoch(model, test_loader, device, False, optimizer, criterion)
        acc = float(accuracy_score(y_te, p_te))
        history["epoch"].append(float(ep))
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["test_accuracy"].append(acc)

    _, y_te, p_te = run_epoch(model, test_loader, device, False, optimizer, criterion)
    metrics = macro_metrics(y_te, p_te, num_classes)

    out = {
        "dataset": args.dataset,
        "variant": args.variant,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "num_classes": num_classes,
        },
        "metrics": metrics,
        "history": history,
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
