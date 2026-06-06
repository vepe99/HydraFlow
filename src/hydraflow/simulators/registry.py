"""Name -> simulator-class registry.

New simulators self-register with the ``@register_simulator("name")`` decorator, so the rest of
the pipeline resolves them by the ``simulator.name`` config field without any if/elif chains or
infrastructure edits.
"""

from __future__ import annotations

from typing import Dict, Type

from hydraflow.simulators.base import BaseSimulator

_REGISTRY: Dict[str, Type[BaseSimulator]] = {}


def register_simulator(name: str):
    """Class decorator registering a :class:`BaseSimulator` subclass under ``name``."""

    def _wrap(cls: Type[BaseSimulator]) -> Type[BaseSimulator]:
        if not issubclass(cls, BaseSimulator):
            raise TypeError(f"{cls!r} must subclass BaseSimulator")
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Simulator name '{name}' already registered to {_REGISTRY[name]!r}")
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_simulator(cfg) -> BaseSimulator:
    """Instantiate the simulator selected by ``cfg.simulator`` (a ``SimulatorConfig``)."""
    name = cfg.name
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown simulator '{name}'. Registered: {sorted(_REGISTRY)}. "
            "Did you import the module that defines it?"
        )
    return _REGISTRY[name](params=cfg.params)


def available_simulators() -> list[str]:
    return sorted(_REGISTRY)
