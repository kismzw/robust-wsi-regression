#!/usr/bin/env python3
"""
train.py

Config-driven training entrypoint for image -> K-dim regression (K=50).
- Reproducible seeding
- Group-aware split consumption (from splits/*.csv)
- Train-only target normalization (optional)
- Primary metric: Spearman (recommended for log-scale expression)
- --dry_run to validate the entire pipeline quickly

Usage:
  python train.py --config configs/base.yaml
  python train.py --config configs/base.yaml --dry_run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import yaml

from engine.data import build_dataloaders
# from engine.io import ensure_run_dir, save_json, save_text, save_yaml  # optional if you have it
from engine.model import build_model
from engine.seed import set_seed
from engine.trainer import build_amp_scaler, build_optimizer, evaluate, train_one_epoch
from typing import Optional

# -------------------------
# Minimal IO helpers (self-contained)
# -------------------------
def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    m = unwrap_model(model)
    m.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt

def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def get_dist_info():
    if not is_distributed():
        return 0, 0, 1  # local_rank, rank, world_size
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, rank, world_size


def _now_run_name() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _save_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# -------------------------
# Config
# -------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping/dict.")
    return cfg


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict (used for overrides if needed)."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--run_dir", type=str, default="", help="Optional run directory. Default: results/<timestamp>/")
    p.add_argument("--dry_run", action="store_true", help="Run only a couple of batches for sanity check.")
    p.add_argument("--max_train_batches", type=int, default=0, help="Override max train batches (0 = no limit).")
    p.add_argument("--max_val_batches", type=int, default=0, help="Override max val batches (0 = no limit).")
    p.add_argument("--resume",type=str, default="", help="Optional checkpoint path to resume from.")
    p.add_argument("--resume_strict", action="store_true", help="Use strict=True when loading model weights.")
    return p.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    local_rank, rank, world_size = get_dist_info()
    is_main = (rank == 0)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    if is_distributed():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


    # Resolve run directory
    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / _now_run_name()
    if is_main:
        _ensure_dir(run_dir)

    ddp_barrier() # wait for main process to create dir
    
    # Save resolved config early
    cfg_resolved = dict(cfg)
    cfg_resolved["runtime"] = {
        "device": str(device),
        "dry_run": bool(args.dry_run),
        "seed": seed,
        "distributed": bool(is_distributed()),
        "rank" : rank,
        "world_size": world_size,
    }

    if is_main:
        _save_yaml(run_dir / "config_resolved.yaml", cfg_resolved)

    # Build dataloaders
    data_cfg = cfg.get("data", {})
    dls = build_dataloaders(
        metadata_csv=data_cfg["csv"],
        split_csv=data_cfg["split"],
        image_size=int(data_cfg.get("image_size", 224)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        target_dim=int(data_cfg.get("target_dim", 50)),
        normalize_target=bool(data_cfg.get("normalize_target", True)),
        seed=seed,
        distributed=is_distributed(),
        rank=rank,
        world_size=world_size,
    )
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]
    train_sampler = dls.get("train_sampler", None)
    artifacts = dls["artifacts"]

    # Build model
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "resnet18")
    pretrained = bool(model_cfg.get("pretrained", True))
    output_dim = int(model_cfg.get("output_dim", data_cfg.get("target_dim", 50)))

    model = build_model(backbone=backbone, pretrained=pretrained, output_dim=output_dim)
    model.to(device)

    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )

    # Optimizer / AMP
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    amp = bool(train_cfg.get("amp", True))

    optimizer = build_optimizer(model, lr=lr, weight_decay=wd)
    scaler = build_amp_scaler(enabled=amp)

    # Metrics config
    eval_cfg = cfg.get("eval", {})
    primary_metric = str(eval_cfg.get("primary_metric", "spearman"))
    secondary_metrics = list(eval_cfg.get("secondary_metrics", ["pearson"]))
    include_samplewise = bool(eval_cfg.get("include_samplewise", False))

    # Batch limiting for dry_run / overrides
    if args.dry_run:
        max_train_batches = 2
        max_val_batches = 2
        epochs = min(epochs, 1)
    else:
        max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None
        max_val_batches = args.max_val_batches if args.max_val_batches > 0 else None

    # Training loop
    log_lines = []
    best_primary = float("-inf")
    start_epoch = 1
    ckpt_dir = run_dir / "checkpoints"
    if is_main:
        _ensure_dir(ckpt_dir)
    ddp_barrier()  # ensure ckpt_dir exists

    best_ckpt_path = ckpt_dir / "best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        
        # All ranks load. Rank 0 prints.
        ckpt = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            strict=args.resume_strict,
        )
        # epoch is the last completed epoch
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        # Restore best_primary if available
        primary_key = f"val_{primary_metric}"
        ckpt_stats = ckpt.get("stats", {}) if isinstance(ckpt.get("stats", {}), dict) else {}
        if primary_key in ckpt_stats:
            try:
                best_primary = float(ckpt_stats[primary_key])
            except Exception:
                best_primary = float("-inf")
        # If you store best_primary explicitly, this will override:
        if ckpt.get("best_primary") is not None:
            try:
                best_primary = float(ckpt["best_primary"])
            except Exception:
                pass

        ddp_barrier()
        if is_main:
            print(f"[RESUME] Loaded checkpoint from {resume_path} (start_epoch {start_epoch}, best_primary {best_primary:.6f})")
    for epoch in range(1, epochs + 1):
        if epoch < start_epoch:
            continue
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_batches=max_train_batches,
        )
        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            include_samplewise=include_samplewise,
            max_batches=max_val_batches,
        )

        # Merge stats
        stats = {"epoch": epoch, **tr, **va}

        # Console logging (simple)
        if is_main:
            primary_key = f"val_{primary_metric}"
            primary_val = stats.get(primary_key, float("nan"))
            line = (
                f"Epoch {epoch:03d} | "
                f"train_loss={stats['train_loss']:.4f} | "
                f"val_loss={stats.get('val_loss',float('nan')):.4f} | "
                f"{primary_key}={primary_val:.4f}"
            )
            print(line)
            log_lines.append(line)

            m = unwrap_model(model)

            try:
                primary_val_f = float(primary_val)
            except Exception:
                primary_val_f = None

            improved = (primary_val_f is not None) and (primary_val_f == primary_val_f) and (primary_val_f > best_primary)
            if improved:
                best_primary = primary_val_f

            # Save last checkpoint each epoch (small repo; ok)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": m.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "config": cfg_resolved,
                    "artifacts": artifacts,
                    "stats": stats,
                    "best_primary": best_primary,
                },
                last_ckpt_path,
            )

            # Save best checkpoint by primary metric
            if improved:
                torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": m.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                            "config": cfg_resolved,
                            "artifacts": artifacts,
                            "stats": stats,
                            "best_primary": best_primary,
                        },
                        best_ckpt_path,
                    )

            # Save metrics snapshot
            _save_json(run_dir / "metrics.json", {"best_primary": best_primary, "last": stats})
        # Keep ranks in sync
        ddp_barrier()
    # Write log file
    if is_main:
        _save_text(run_dir / "log.txt", "\n".join(log_lines) + "\n")

        print(f"[OK] Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
