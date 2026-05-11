"""Load Stage 3 pickle datasets (MNIST, ORL, CIFAR) into torch tensors."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import TensorDataset

DatasetName = Literal["mnist", "orl", "cifar"]


def _load_pickle(data_root: Path, name: str) -> dict:
    path = data_root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing pickle file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_tensors(
    data_root: Path,
    dataset: DatasetName,
) -> tuple[TensorDataset, TensorDataset, int]:
    """
    Returns train_dataset, test_dataset, num_classes.
    Image tensors are float32, shape NCHW, labels int64 in 0..num_classes-1.
    """
    root = Path(data_root)
    if dataset == "mnist":
        raw = _load_pickle(root, "MNIST")
        xs, ys = [], []
        for ex in raw["train"]:
            im = np.asarray(ex["image"], dtype=np.float32) / 255.0
            xs.append(im[None, ...])
            ys.append(int(ex["label"]))
        x_tr = torch.from_numpy(np.stack(xs, axis=0))
        y_tr = torch.tensor(ys, dtype=torch.long)
        xs, ys = [], []
        for ex in raw["test"]:
            im = np.asarray(ex["image"], dtype=np.float32) / 255.0
            xs.append(im[None, ...])
            ys.append(int(ex["label"]))
        x_te = torch.from_numpy(np.stack(xs, axis=0))
        y_te = torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x_tr, y_tr), TensorDataset(x_te, y_te), 10

    if dataset == "orl":
        raw = _load_pickle(root, "ORL")
        xs, ys = [], []
        for ex in raw["train"]:
            im = np.asarray(ex["image"], dtype=np.float32) / 255.0
            r = im[:, :, 0:1]
            r = np.transpose(r, (2, 0, 1))
            xs.append(r)
            ys.append(int(ex["label"]) - 1)
        x_tr = torch.from_numpy(np.stack(xs, axis=0))
        y_tr = torch.tensor(ys, dtype=torch.long)
        xs, ys = [], []
        for ex in raw["test"]:
            im = np.asarray(ex["image"], dtype=np.float32) / 255.0
            r = im[:, :, 0:1]
            r = np.transpose(r, (2, 0, 1))
            xs.append(r)
            ys.append(int(ex["label"]) - 1)
        x_te = torch.from_numpy(np.stack(xs, axis=0))
        y_te = torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x_tr, y_tr), TensorDataset(x_te, y_te), 40

    if dataset == "cifar":
        raw = _load_pickle(root, "CIFAR")
        xs, ys = [], []
        for ex in raw["train"]:
            im = np.asarray(ex["image"], dtype=np.float32) / 255.0
            im = np.transpose(im, (2, 0, 1))
            xs.append(im)
            ys.append(int(ex["label"]))
        x_tr = torch.from_numpy(np.stack(xs, axis=0))
        y_tr = torch.tensor(ys, dtype=torch.long)
        xs, ys = [], []
        for ex in raw["test"]:
            im = np.asarray(ex["image"], dtype=np.float32) / 255.0
            im = np.transpose(im, (2, 0, 1))
            xs.append(im)
            ys.append(int(ex["label"]))
        x_te = torch.from_numpy(np.stack(xs, axis=0))
        y_te = torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x_tr, y_tr), TensorDataset(x_te, y_te), 10

    raise ValueError(dataset)
