"""
engine/model.py

Minimal, practical image -> K-dim regression model (K=50 by default).

Design goals (Week1):
- Simple and reliable baseline (ResNet backbone + linear head)
- Configurable backbone and pretrained flag
- Clean forward returning (N, K) float tensor

Usage:
    from engine.model import build_model
    model = build_model(backbone="resnet18", pretrained=True, output_dim=50)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
)


BackboneName = Literal[
    "resnet18",
    "resnet50",
]


class ImageRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, feat_dim: int, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feat_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y = self.head(z)
        return y


def _build_resnet(backbone: BackboneName, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    if backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    raise ValueError(f"Unsupported backbone: {backbone}")


def build_model(
    backbone: BackboneName = "resnet18",
    pretrained: bool = True,
    output_dim: int = 50,
) -> nn.Module:
    encoder, feat_dim = _build_resnet(backbone, pretrained)
    return ImageRegressor(encoder, feat_dim, output_dim)
