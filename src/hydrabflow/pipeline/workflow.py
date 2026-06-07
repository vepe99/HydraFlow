"""Assemble a single-level BayesFlow workflow from config.

Uses ``bf.BasicWorkflow`` (a ``ContinuousApproximator`` under the hood) — no compositional /
hierarchical machinery. Combines the adapter, the summary network, and the inference network.
"""

from __future__ import annotations

from typing import Any

from hydrabflow.networks.factory import build_inference_network, build_summary_network
from hydrabflow.pipeline.adapter import build_adapter


def build_workflow(cfg) -> Any:
    """Build a ``bf.BasicWorkflow`` from the root ``cfg``."""
    import bayesflow as bf
    from omegaconf import OmegaConf

    adapter = build_adapter(cfg.adapter)
    summary_network = build_summary_network(cfg.model.summary_network)
    inference_network = build_inference_network(cfg.model.inference_network)

    standardize = list(OmegaConf.to_container(cfg.training.standardize, resolve=True))

    return bf.BasicWorkflow(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize=standardize,
    )
