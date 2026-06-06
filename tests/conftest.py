"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))


def compose_cfg(overrides=None):
    """Compose the root config with the structured schemas registered.

    Provides the mandatory ``adapter.inference_variables`` so composition resolves.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from hydraflow.config import register_configs

    register_configs()
    base = ["adapter.inference_variables=[theta1,theta2]"]
    if overrides:
        base += list(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        return compose(config_name="config", overrides=base)


@pytest.fixture
def cfg():
    return compose_cfg()


@pytest.fixture
def compose():
    """Expose the composer so tests can build configs with custom overrides."""
    return compose_cfg
