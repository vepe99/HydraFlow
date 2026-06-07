"""Adapter / network / workflow construction. Skipped if bayesflow isn't installed."""

from __future__ import annotations

import pytest

bf = pytest.importorskip("bayesflow")


def test_build_adapter(cfg):
    from hydrabflow.pipeline.adapter import build_adapter

    adapter = build_adapter(cfg.adapter)
    assert adapter is not None


def test_build_networks(cfg):
    from hydrabflow.networks.factory import build_inference_network, build_summary_network

    assert build_summary_network(cfg.model.summary_network) is not None
    assert build_inference_network(cfg.model.inference_network) is not None


def test_build_workflow(cfg):
    from hydrabflow.pipeline.workflow import build_workflow

    workflow = build_workflow(cfg)
    assert hasattr(workflow, "approximator")
