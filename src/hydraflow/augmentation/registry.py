"""Name -> augmentation-factory registry and builder.

An augmentation factory receives the shared ``params`` mapping from config and returns a callable
``batch -> batch``. Returning a closure lets augmentations capture configuration (noise scale,
target keys, ...) while staying compatible with BayesFlow's ``fit_offline(augmentations=[...])``.
"""

from __future__ import annotations

from typing import Callable, Dict, List

# factory(params) -> (batch -> batch)
Augmentation = Callable[[dict], dict]
_REGISTRY: Dict[str, Callable[..., Augmentation]] = {}


def register_augmentation(name: str):
    def _wrap(factory: Callable[..., Augmentation]) -> Callable[..., Augmentation]:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"Augmentation '{name}' already registered")
        _REGISTRY[name] = factory
        return factory

    return _wrap


def build_augmentations(cfg) -> List[Augmentation]:
    """Build the ordered augmentation list from ``cfg.augmentation`` (an ``AugmentationConfig``)."""
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.params, resolve=True) if OmegaConf.is_config(cfg.params) else dict(cfg.params)
    augs: List[Augmentation] = []
    for name in cfg.steps:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown augmentation '{name}'. Registered: {sorted(_REGISTRY)}")
        augs.append(_REGISTRY[name](params))
    return augs


def available_augmentations() -> list[str]:
    return sorted(_REGISTRY)
