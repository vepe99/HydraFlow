"""Skeleton simulator: the intentional stub shipped with the template.

It declares a 2-parameter / single-observable interface so the config, adapter, and networks all
compose end-to-end, but every forward call raises ``NotImplementedError`` telling you exactly
where to plug in your physics. Replace this file (or, better, add your own module + a
``conf/simulator/<name>.yaml``) with a real :class:`BaseSimulator`.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.simulators.registry import register_simulator

_FILL_ME_IN = (
    "SkeletonSimulator is a placeholder. Implement your forward model: create a BaseSimulator "
    "subclass (e.g. src/hydrabflow/simulators/my_sim.py), decorate it with "
    "@register_simulator('my_sim'), implement sample_prior() and simulate(), and point "
    "conf/simulator/my_sim.yaml at it. See README.md."
)


@register_simulator("skeleton")
class SkeletonSimulator(BaseSimulator):
    @property
    def parameter_names(self) -> list[str]:
        return ["theta1", "theta2"]

    @property
    def observable_keys(self) -> list[str]:
        return ["x"]

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        raise NotImplementedError(_FILL_ME_IN)

    def simulate(
        self, params: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError(_FILL_ME_IN)
