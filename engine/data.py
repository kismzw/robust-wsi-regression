"""
engine/data.py

Dataset + split handling + target normalization for image -> K-dim regression (K=50).

Key design points:
- Uses a persisted split CSV (created by make_splits.py) to ensure reproducibility.
- Enforces group-aware split by consuming the split file rather than re-splitting.
- Optionally normalizes targets (z-score) using TRAIN split statistics only.
- Keeps I/O and normalization artifacts in a returned dict so the trainer can save them.

Expected files:
- metadata CSV with columns: path, group_id, y0..y49 (or y{0..K-1})
- split CSV with columns: path, group_id, split (train|val)

Typical usage:
    from engine.data import build_dataloaders
    dls = build_dataloaders(
        metadata_csv="data/metadata.csv",
        split_csv="splits/split_seed42.csv",
        image_size=224,
        batch_size=32,
        num_workers=4,
        target_dim=50,
        normalize_target=True,
        seed=42
    )
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


# -------------------------
# Utilities
# -------------------------

def _ensure_exists(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p


def _target_cols(target_dim: int) -> List[str]:
    return [f"y{i}" for i in range(target_dim)]


def _worker_init_fn(worker_id: int) -> None:
    """
    Make DataLoader workers deterministic-ish. (Still depends on upstream seed setting.)
    Use this together with engine/seed.py's set_seed().
    """
    # torch.initial_seed() already differs per worker; use it to seed numpy/random.
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


@dataclass
class TargetScaler:
    mean: np.ndarray  # shape (K,)
    std: np.ndarray   # shape (K,)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_y_train(y_train: np.ndarray, eps: float = 1e-8) -> "TargetScaler":
        mean = y_train.mean(axis=0)
        std = y_train.std(axis=0)
        std = np.where(std < eps, 1.0, std)  # avoid division by ~0 for near-constant genes
        return TargetScaler(mean=mean, std=std)


# -------------------------
# Dataset
# -------------------------

class ImageRegressionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_dim: int,
        image_transform: Optional[transforms.Compose] = None,
        target_scaler: Optional[TargetScaler] = None,
        path_col: str = "path",
    ):
        self.df = df.reset_index(drop=True)
        self.target_dim = target_dim
        self.path_col = path_col
        self.y_cols = _target_cols(target_dim)
        self.tfm = image_transform
        self.scaler = target_scaler

        missing = [c for c in [self.path_col] + self.y_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataframe: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_path = str(row[self.path_col])
        # Load image
        img = Image.open(img_path).convert("RGB")

        if self.tfm is not None:
            x = self.tfm(img)
        else:
            # Default: ToTensor only (not recommended for real training)
            x = transforms.ToTensor()(img)

        y = row[self.y_cols].to_numpy(dtype=np.float32)  # (K,)
        if self.scaler is not None:
            y = self.scaler.transform(y.astype(np.float64)).astype(np.float32)

        return {
            "x": x,                                   # torch.FloatTensor (C,H,W)
            "y": torch.from_numpy(y),                 # torch.FloatTensor (K,)
            "path": img_path,
        }


# -------------------------
# Builders
# -------------------------

def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, val_transform).
    Keep this simple for Week1; you can add augmentations later.
    """
    train_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # You can optionally add normalization if you want:
        # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return train_tfm, val_tfm


def load_merged_metadata(
    metadata_csv: str | Path,
    split_csv: str | Path,
    path_col: str = "path",
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Merge metadata with split assignments by path.
    """
    metadata_csv = _ensure_exists(metadata_csv)
    split_csv = _ensure_exists(split_csv)

    meta = pd.read_csv(metadata_csv)
    split = pd.read_csv(split_csv)

    for col in (path_col,):
        if col not in meta.columns:
            raise ValueError(f"metadata CSV missing required column '{col}'")
        if col not in split.columns:
            raise ValueError(f"split CSV missing required column '{col}'")

    if split_col not in split.columns:
        raise ValueError(f"split CSV missing required column '{split_col}' (expected train|val)")

    # Keep only necessary split columns to avoid accidental conflicts
    split = split[[path_col, split_col]].copy()

    df = meta.merge(split, on=path_col, how="inner")
    if len(df) == 0:
        raise ValueError(
            "Merged dataframe is empty. Ensure split paths match metadata paths exactly."
        )

    # Validate split values
    bad = sorted(set(df[split_col].unique()) - {"train", "val"})
    if bad:
        raise ValueError(f"Invalid split values found: {bad}. Expected only 'train' or 'val'.")

    return df


def build_target_scaler(
    df: pd.DataFrame,
    target_dim: int,
    split_col: str = "split",
) -> TargetScaler:
    """
    Fit TargetScaler using TRAIN rows only.
    """
    y_cols = _target_cols(target_dim)
    train_df = df[df[split_col] == "train"]
    if len(train_df) < 2:
        raise ValueError("Need at least 2 training samples to compute target normalization stats.")

    y_train = train_df[y_cols].to_numpy(dtype=np.float64)
    return TargetScaler.from_y_train(y_train)


def build_dataloaders(
    metadata_csv: str | Path,
    split_csv: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    target_dim: int = 50,
    normalize_target: bool = True,
    seed: int = 42,
    path_col: str = "path",
    split_col: str = "split",
    # --- DDP knobs ---
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    """
    High-level builder that returns dataloaders and fit artifacts (e.g., scaler stats).
    """
    # Load + merge
    df = load_merged_metadata(metadata_csv, split_csv, path_col=path_col, split_col=split_col)

    # Build transforms
    train_tfm, val_tfm = build_transforms(image_size=image_size)

    # Optional target normalization (train-only)
    scaler = build_target_scaler(df, target_dim=target_dim, split_col=split_col) if normalize_target else None

    # Split dataframes
    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()

    if len(val_df) == 0:
        raise ValueError("Validation set is empty. Adjust val_ratio or check split file.")
    if len(train_df) == 0:
        raise ValueError("Training set is empty. Adjust val_ratio or check split file.")

    # Create datasets
    train_ds = ImageRegressionDataset(
        train_df, target_dim=target_dim, image_transform=train_tfm, target_scaler=scaler, path_col=path_col
    )
    val_ds = ImageRegressionDataset(
        val_df, target_dim=target_dim, image_transform=val_tfm, target_scaler=scaler, path_col=path_col
    )

    # Pin memory only helps when CUDA is available (avoids warning on CPU)
    pin_memory = torch.cuda.is_available()

    # Samplers (DDP)
    train_sampler = None
    val_sampler = None
    shuffle_train = True

    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        shuffle_train = False  # sampler does shuffling

    # Deterministic-ish DataLoader generator (single-process shuffle order)
    # In DDP, shuffling is controlled by DistributedSampler + set_epoch(epoch),
    # so we only use generator for the non-distributed case.
    generator = None
    if not distributed:
        g = torch.Generator()
        g.manual_seed(seed)
        generator = g

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )

    artifacts: Dict[str, Any] = {
        "target_scaler": scaler.to_dict() if scaler is not None else None,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "target_dim": target_dim,
        "image_size": image_size,
        "distributed": distributed,
        "world_size": world_size,
    }

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_sampler": train_sampler,  # ★追加：train.pyでset_epochに使う
        "val_sampler": val_sampler,
        "artifacts": artifacts,
        "df_merged": df,  # debugging; optional
    }
