"""Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose leading axis
is the number of simulations (one (parameters, observation) pair per row).
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from hydrabflow.utils.logging import get_logger

log = get_logger(__name__)
Dataset = Dict[str, np.ndarray]


def save_dataset(path: str, data: Dataset) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **data)
    n = len(next(iter(data.values()))) if data else 0
    log.info("Saved dataset (%d rows, keys=%s) -> %s", n, list(data), path)


def load_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate it first (e.g. `hydrabflow-simulate`)."
        )
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n = len(next(iter(data.values()))) if data else 0
    log.info("Loaded dataset (%d rows, keys=%s) <- %s", n, list(data), path)
    return data


def concatenate_chunks(chunks: list[Dataset]) -> Dataset:
    """Concatenate a list of dataset dicts along the leading (simulation) axis."""
    if not chunks:
        return {}
    keys = chunks[0].keys()
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
