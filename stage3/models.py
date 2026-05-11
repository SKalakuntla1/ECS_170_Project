"""CNN architectures for Stage 3 datasets."""

from __future__ import annotations

import torch.nn as nn


class CNNMNIST(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CNNORL(nn.Module):
    """112x92 grayscale (1 channel). 40 person classes."""

    def __init__(self, num_classes: int = 40, dropout: float = 0.4, widen: int = 1):
        super().__init__()
        c = 16 * widen
        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        feat_dim = c * 4
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CNNCIFAR(nn.Module):
    """32x32 RGB."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25, channels: int = 64):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(
    dataset: str,
    variant: str = "default",
) -> nn.Module:
    ds = dataset.lower()
    if ds == "mnist":
        d = 0.35 if variant == "heavy_dropout" else 0.25
        return CNNMNIST(num_classes=10, dropout=d)
    if ds == "orl":
        w = 2 if variant == "wide" else 1
        d = 0.5 if variant == "heavy_dropout" else 0.25
        return CNNORL(num_classes=40, dropout=d, widen=w)
    if ds == "cifar":
        ch = 96 if variant == "wide" else 64
        d = 0.35 if variant == "heavy_dropout" else 0.25
        return CNNCIFAR(num_classes=10, dropout=d, channels=ch)
    raise ValueError(dataset)
