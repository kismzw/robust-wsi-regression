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
import torch.distributed as dist

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
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """
    Train for one epoch and return aggregated stats.

    Args:
        scaler: pass GradScaler if using AMP on CUDA, else None
        max_batches: if set, only run this many batches (dry_run)
    """
    model.train()
    assert grad_accum_steps >= 1, "grad_accum_steps must be >= 1"
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    use_amp = scaler is not None

    # zero grad before starting
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)


        if use_amp:
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
            # scale loss for accumulation
            loss_to_bp = loss / grad_accum_steps
            scaler.scale(loss_to_bp).backward()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss_to_bp = loss / grad_accum_steps
            loss_to_bp.backward()

        bs = int(y.shape[0])
        total_loss += float(loss.detach().cpu().item()) * bs
        n_batches += 1
        n_samples += bs

        # Step only every grad_accum_steps micro-batches
        do_step = ((step + 1) % grad_accum_steps == 0)

        if do_step:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
    # Handle leftover grads if number of batches not divisible by grad_accum_steps
    if n_batches > 0 and (n_batches % grad_accum_steps != 0):
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    avg_loss = total_loss / max(n_samples, 1)
    return {
        "train_loss": avg_loss,
        "train_batches": float(n_batches),
        "train_samples": float(n_samples),
    }


@torch.no_grad()
def _dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _gather_numpy(arr: np.ndarray) -> np.ndarray:
    """All-gather numpy arrays across ranks (variable batch sizes supported)."""
    if not _dist_is_on():
        return arr
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, arr)
    return np.concatenate(gathered, axis=0) if len(gathered) > 0 else arr


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

    DDP behavior:
      - val loss: aggregated across ranks via all_reduce
      - metrics: y_true/y_pred gathered across ranks; computed on rank0 only
      - non-rank0 returns {} (so train.py can guard printing/saving with is_main)
    """
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    y_true_all = []
    y_pred_all = []

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y)

        bs = int(y.shape[0])
        total_loss += float(loss.detach().cpu().item()) * bs
        n_batches += 1
        n_samples += bs

        y_true_all.append(_to_numpy(y))
        y_pred_all.append(_to_numpy(pred))

    # --- aggregate loss across ranks ---
    if _dist_is_on():
        loss_tensor = torch.tensor(
            [total_loss, float(n_samples), float(n_batches)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss_global = float(loss_tensor[0].item())
        n_samples_global = float(loss_tensor[1].item())
        n_batches_global = float(loss_tensor[2].item())
    else:
        total_loss_global = total_loss
        n_samples_global = float(n_samples)
        n_batches_global = float(n_batches)

    avg_loss = total_loss_global / max(n_samples_global, 1.0)

    # --- gather predictions across ranks for global metrics ---
    y_true_np = np.concatenate(y_true_all, axis=0) if y_true_all else np.zeros((0, 0), dtype=np.float64)
    y_pred_np = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.zeros((0, 0), dtype=np.float64)

    y_true_np = _gather_numpy(y_true_np)
    y_pred_np = _gather_numpy(y_pred_np)

    # DistributedSampler(drop_last=False) pads with duplicates; trim to real dataset size.
    n_total = len(loader.dataset)
    y_true_np = y_true_np[:n_total]
    y_pred_np = y_pred_np[:n_total]

    # compute metrics on rank0 only in DDP
    if _dist_is_on() and dist.get_rank() != 0:
        return {}

    metrics = compute_metrics(
        y_true_np,
        y_pred_np,
        primary=primary_metric,
        secondary=secondary_metrics or [],
        include_samplewise=include_samplewise,
    )

    out: Dict[str, float] = {
        "val_loss": float(avg_loss),
        "val_batches": float(n_batches_global),
        "val_samples": float(n_samples_global),
    }
    for k, v in metrics.items():
        # be robust if some metric returns arrays (optional)
        if isinstance(v, (np.ndarray, list)):
            v = float(np.mean(v))
        out[f"val_{k}"] = float(v)
    return out


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_amp_scaler(enabled: bool) -> Optional[torch.cuda.amp.GradScaler]:
    # AMP only makes sense on CUDA
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None
