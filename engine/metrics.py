"""
engine/metrics.py

Metrics for high-dimensional regression (K=50 by default).

Primary:
- Mean Spearman correlation (gene-wise average), suitable for log-scale targets.

Secondary (optional):
- Mean Pearson correlation (gene-wise average), for reference.

Notes:
- Spearman/Pearson can be undefined if a target dimension is constant.
  We handle this by returning NaN for that dimension and taking nanmean.
- Expected inputs:
    y_true: (N, K) array-like
    y_pred: (N, K) array-like
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _to_2d_numpy(x) -> np.ndarray:
    """Convert torch.Tensor / list / np.ndarray to float64 numpy array (N, K)."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (N, K), got shape={x.shape}")
    return x.astype(np.float64, copy=False)


def _rankdata_1d(a: np.ndarray) -> np.ndarray:
    """
    Compute average ranks for a 1D array (ties get average rank).
    This is a minimal replacement for scipy.stats.rankdata(method="average").
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("rankdata expects 1D input")
    n = a.size
    if n == 0:
        return a.astype(np.float64)

    # stable sort to make ties deterministic
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        # find tie run in sorted order
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        # average rank for ties; ranks are 1..n
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def _safe_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation for 1D arrays; returns NaN if undefined."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    # If constant vector -> undefined correlation
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def mean_pearson(y_true, y_pred) -> float:
    """
    Gene-wise mean Pearson correlation across K dimensions.
    Returns nanmean across dimensions (ignores NaNs from constant genes).
    """
    yt = _to_2d_numpy(y_true)
    yp = _to_2d_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")

    k = yt.shape[1]
    cors = np.empty(k, dtype=np.float64)
    for i in range(k):
        cors[i] = _safe_corr_1d(yt[:, i], yp[:, i])
    return float(np.nanmean(cors))


def mean_spearman(y_true, y_pred) -> float:
    """
    Gene-wise mean Spearman correlation across K dimensions.
    Spearman is Pearson correlation on ranks (average ranks for ties).
    Returns nanmean across dimensions (ignores NaNs from constant genes).
    """
    yt = _to_2d_numpy(y_true)
    yp = _to_2d_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")

    k = yt.shape[1]
    cors = np.empty(k, dtype=np.float64)
    for i in range(k):
        rt = _rankdata_1d(yt[:, i])
        rp = _rankdata_1d(yp[:, i])
        cors[i] = _safe_corr_1d(rt, rp)
    return float(np.nanmean(cors))


def samplewise_spearman(y_true, y_pred) -> float:
    """
    Sample-wise Spearman: for each sample n, compute Spearman correlation
    between the K-dim true and predicted vectors, then average over samples.

    This is often informative for expression-vector predictions.
    """
    yt = _to_2d_numpy(y_true)
    yp = _to_2d_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")

    n = yt.shape[0]
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        rt = _rankdata_1d(yt[i, :])
        rp = _rankdata_1d(yp[i, :])
        scores[i] = _safe_corr_1d(rt, rp)
    return float(np.nanmean(scores))


@dataclass(frozen=True)
class MetricsConfig:
    primary: str = "spearman"
    secondary: Tuple[str, ...] = ("pearson",)


def compute_metrics(
    y_true,
    y_pred,
    primary: str = "spearman",
    secondary: Optional[Iterable[str]] = ("pearson",),
    include_samplewise: bool = False,
) -> Dict[str, float]:
    """
    Compute a dict of metrics.

    Args:
        y_true, y_pred: (N, K) arrays or torch tensors
        primary: "spearman" or "pearson"
        secondary: iterable of metric names
        include_samplewise: if True, also compute sample-wise Spearman

    Returns:
        Dict[str, float] mapping metric_name -> value
    """
    primary = primary.lower()
    secondary = tuple([m.lower() for m in (secondary or ())])

    out: Dict[str, float] = {}

    def _calc(name: str) -> float:
        if name == "spearman":
            return mean_spearman(y_true, y_pred)
        if name == "pearson":
            return mean_pearson(y_true, y_pred)
        if name in ("sample_spearman", "samplewise_spearman"):
            return samplewise_spearman(y_true, y_pred)
        raise ValueError(f"Unknown metric: {name}")

    out[primary] = _calc(primary)

    for m in secondary:
        if m == primary:
            continue
        out[m] = _calc(m)

    if include_samplewise:
        out["samplewise_spearman"] = _calc("samplewise_spearman")

    return out
