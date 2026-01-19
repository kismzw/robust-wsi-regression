"""
engine/trainer.py

Minimal training utilities for image -> K-dim regression (K=50).
- MSE loss (optimize)
- Primary metric: mean Spearman (evaluate)
- Optional AMP
- Supports dry_run (limit number of batches)

Expected batch format from engine/data.py:
    batch = {"x": Tensor[N,3,H,W], "y": Tensor[N,K], "path": list[str]}

Usage (from train.py):
    from engine.trainer import train_one_epoch, evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from engine.metrics import compute_metrics


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Train for one epoch and return aggregated stats.

    Args:
        scaler: pass GradScaler if using AMP on CUDA, else None
        max_batches: if set, only run this many batches (dry_run)
    """
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    n_batches = 0

    use_amp = scaler is not None

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {
        "train_loss": avg_loss,
        "train_batches": float(n_batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    primary_metric: str = "spearman",
    secondary_metrics: Optional[list[str]] = None,
    include_samplewise: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate on validation loader.

    Returns:
        dict containing:
          - val_loss
          - val_<metric>
    """
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    n_batches = 0

    y_true_all = []
    y_pred_all = []

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

        y_true_all.append(_to_numpy(y))
        y_pred_all.append(_to_numpy(pred))

    avg_loss = total_loss / max(n_batches, 1)

    y_true_np = np.concatenate(y_true_all, axis=0) if y_true_all else np.zeros((0, 0), dtype=np.float64)
    y_pred_np = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.zeros((0, 0), dtype=np.float64)

    metrics = compute_metrics(
        y_true_np,
        y_pred_np,
        primary=primary_metric,
        secondary=secondary_metrics or [],
        include_samplewise=include_samplewise,
    )

    out: Dict[str, float] = {"val_loss": avg_loss, "val_batches": float(n_batches)}
    for k, v in metrics.items():
        out[f"val_{k}"] = float(v)
    return out


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_amp_scaler(enabled: bool) -> Optional[torch.cuda.amp.GradScaler]:
    # AMP only makes sense on CUDA
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None
