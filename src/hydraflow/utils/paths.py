"""Run-directory helpers and shared artifact filenames."""

from __future__ import annotations

import os

# Filenames written into every run dir (alongside Hydra's automatic `.hydra/`).
PREPROCESSING_STATE = "preprocessing_state.npz"
POSTERIOR_SAMPLES = "posterior.npz"
LOSS_PLOT = "loss.png"


def get_run_dir() -> str:
    """Return the current Hydra run output dir (works regardless of the ``job.chdir`` setting)."""
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.output_dir


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
