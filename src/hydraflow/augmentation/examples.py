"""Example augmentations. Use as templates for problem-specific ones.

Augmentations run per batch during training and should be cheap and stochastic. They receive and
return a dict of arrays (the batch). Register with ``@register_augmentation("name")`` and list the
name under ``augmentation.steps`` in config.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from hydraflow.augmentation.registry import Augmentation, register_augmentation


@register_augmentation("gaussian_noise")
def gaussian_noise(params: Mapping) -> Augmentation:
    """Add zero-mean Gaussian noise to one observable key.

    Config params: ``noise_key`` (which key to perturb, default ``"x"``) and ``noise_scale``
    (std, default ``0.0``). This is intentionally trivial — a working example of the contract.
    """
    key = params.get("noise_key", "x")
    scale = float(params.get("noise_scale", 0.0))

    def _apply(batch: dict) -> dict:
        if scale > 0.0 and key in batch:
            x = np.asarray(batch[key])
            batch[key] = x + np.random.normal(0.0, scale, size=x.shape).astype(x.dtype)
        return batch

    return _apply
