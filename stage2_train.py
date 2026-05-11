"""
ECS 170 Stage 2: MLP digit classification (tabular 784-dim features).
Expects stage_2_data/train.csv and stage_2_data/test.csv with no header:
  column 0 = label (0-9), columns 1-784 = pixel intensities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    y = data[:, 0].astype(np.int64)
    x = data[:, 1:].astype(np.float32)
    return x, y


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], num_classes: int, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def macro_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_one_config(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    hidden: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    optimizer_name: str,
    momentum: float,
    device: torch.device,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLP(train_x.shape[1], hidden, num_classes=10, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            te_logits = model(test_x.to(device))
            te_loss = float(criterion(te_logits, test_y.to(device)).item())
            preds = te_logits.argmax(dim=1).cpu().numpy()
            acc = float(accuracy_score(test_y.numpy(), preds))

        history["epoch"].append(float(ep))
        history["train_loss"].append(train_loss)
        history["test_loss"].append(te_loss)
        history["test_accuracy"].append(acc)

    model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    y_np = test_y.numpy()
    metrics = {
        "accuracy": float(accuracy_score(y_np, preds)),
        **macro_prf(y_np, preds),
    }
    return {
        "config": {
            "hidden": hidden,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "optimizer": optimizer_name,
            "momentum": momentum,
        },
        "metrics": metrics,
        "history": history,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="stage_2_data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden", type=str, default="128,64", help="Comma-separated hidden widths")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. "
            "Copy the instructor Stage 2 CSVs into stage_2_data/ next to this script."
        )

    x_tr, y_tr = load_csv(train_path)
    x_te, y_te = load_csv(test_path)

    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True) + 1e-8
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    device = pick_device()
    hidden = [int(s) for s in args.hidden.split(",") if s.strip()]

    train_x = torch.from_numpy(x_tr)
    train_y = torch.from_numpy(y_tr)
    test_x = torch.from_numpy(x_te)
    test_y = torch.from_numpy(y_te)

    result = train_one_config(
        train_x,
        train_y,
        test_x,
        test_y,
        hidden=hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        device=device,
    )

    print(json.dumps(result, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
