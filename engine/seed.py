"""
engine/seed.py

Reproducibility utilities:
- Set global seeds (Python / NumPy / PyTorch)
- Make cuDNN deterministic (as much as practical)
- Provide a DataLoader worker_init_fn for deterministic worker seeding

Notes:
- Perfect bitwise reproducibility on GPU is not always guaranteed across hardware/versions.
- This module aims for stable metrics and consistent runs given the same config/seed.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed: random seed
        deterministic: if True, configure PyTorch for more deterministic behavior
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Extra: hash-based ops / python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuDNN settings (affects convs etc.)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch deterministic algorithms (may raise if an op has no deterministic version)
        # Keep it off by default in many projects; here we enable cautiously via env flag.
        # If you want strict determinism, set STRICT_DETERMINISM=1 in env.
        if os.environ.get("STRICT_DETERMINISM", "0") == "1":
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker init function.
    Ensures each worker has a deterministic NumPy/random seed derived from torch initial seed.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator seeded for deterministic DataLoader shuffling.

    Usage:
        g = build_generator(seed)
        DataLoader(..., shuffle=True, generator=g)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
