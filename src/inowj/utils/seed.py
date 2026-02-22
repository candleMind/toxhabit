"""Random seed utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for `random`, `numpy`, and `torch`.

    Args:
        seed: Global seed value.
        deterministic: If True, configures PyTorch for deterministic behavior
            (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a torch device string.

    Args:
        device: "auto", "cpu", or "cuda".

    Returns:
        Torch device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
