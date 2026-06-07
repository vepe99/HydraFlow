"""Per-batch, stochastic augmentations applied inside ``workflow.fit_offline``.

Each augmentation is a callable ``batch -> batch`` (a dict of arrays). They are resolved by name
from the registry and composed in config order. This is the counterpart to the preprocessing
module: augmentations are random and re-drawn every epoch, preprocessing is deterministic and
applied once. The template ships an example; add your own with ``@register_augmentation``.
"""

from hydrabflow.augmentation import examples as _examples  # noqa: F401 (self-registers)
from hydrabflow.augmentation.registry import build_augmentations, register_augmentation

__all__ = ["build_augmentations", "register_augmentation"]
