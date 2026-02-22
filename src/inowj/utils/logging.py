"""Logging helpers (console + optional Weights & Biases)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WandbConfig:
    """Configuration for optional W&B logging."""

    enabled: bool = False
    project: str = ""
    run_name: str = ""
    config: dict[str, Any] | None = None


def maybe_init_wandb(cfg: WandbConfig) -> Any | None:
    """Initialize W&B if enabled.

    Uses `WANDB_API_KEY` from the environment if required by your setup.

    Returns:
        The W&B run object if initialized, otherwise None.
    """
    if not cfg.enabled:
        return None

    import wandb

    run = wandb.init(
        project=cfg.project,
        name=cfg.run_name or None,
        config=cfg.config or None,
        reinit=True,
    )
    return run


def maybe_wandb_log(run: Any | None, metrics: dict[str, Any]) -> None:
    """Log metrics to W&B if a run exists."""
    if run is None:
        return
    run.log(metrics)
