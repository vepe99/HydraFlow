"""Name -> preprocessing-step registry and pipeline builder."""

from __future__ import annotations

from typing import Callable, Dict

from hydrabflow.preprocessing.base import PreprocessPipeline, PreprocessStep

# A factory takes the per-step keyword params from config and returns a PreprocessStep instance.
_REGISTRY: Dict[str, Callable[..., PreprocessStep]] = {}


def register_step(name: str):
    """Register a step factory (usually the step class itself) under ``name``."""

    def _wrap(factory: Callable[..., PreprocessStep]) -> Callable[..., PreprocessStep]:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"Preprocess step '{name}' already registered")
        _REGISTRY[name] = factory
        return factory

    return _wrap


def build_pipeline(cfg) -> PreprocessPipeline:
    """Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``PreprocessingConfig``).

    Each entry in ``cfg.steps`` is a mapping ``{name: <key>, ...params}``; the remaining keys are
    passed to the registered factory as keyword args.
    """
    from omegaconf import OmegaConf

    steps: list[PreprocessStep] = []
    for entry in cfg.steps:
        entry = OmegaConf.to_container(entry, resolve=True) if OmegaConf.is_config(entry) else dict(entry)
        name = entry.pop("name")
        if name not in _REGISTRY:
            raise KeyError(f"Unknown preprocess step '{name}'. Registered: {sorted(_REGISTRY)}")
        steps.append(_REGISTRY[name](**entry))
    return PreprocessPipeline(steps)


def available_steps() -> list[str]:
    return sorted(_REGISTRY)
