"""Build the BayesFlow ``Adapter`` from ``AdapterConfig``.

The adapter is the structural (non-stochastic) transform that maps raw dataset keys to the roles
BayesFlow expects: ``inference_variables`` (the target), ``summary_variables`` (fed to the
summary network), and ``inference_conditions`` (direct conditions). Generalizes the reference's
hand-written adapter chain (main_train_new_rotationcurve_agama.py:108-122).

Single observable (default): the one ``summary_variables`` key is renamed to the BayesFlow role.
Fusion seam: with multiple keys, ``group`` them into ``summary_variables`` and build one summary
backbone per key in ``networks.factory`` (left commented below — uncomment + adjust to enable).
"""

from __future__ import annotations

from typing import Any, List


def _as_list(x) -> List[str]:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(x):
        return list(OmegaConf.to_container(x, resolve=True))
    return list(x)


def build_adapter(cfg) -> Any:
    """Construct ``bf.adapters.Adapter`` from ``cfg`` (an ``AdapterConfig``)."""
    import bayesflow as bf

    inference_variables = _as_list(cfg.inference_variables)
    summary_variables = _as_list(cfg.summary_variables)
    inference_conditions = _as_list(cfg.inference_conditions)
    drop = _as_list(cfg.drop)

    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(inference_variables, into="inference_variables")
    )

    if drop:
        adapter = adapter.drop(drop)

    if len(summary_variables) == 1:
        adapter = adapter.rename(summary_variables[0], "summary_variables")
    elif len(summary_variables) > 1:
        # --- Fusion seam -------------------------------------------------------------------- #
        # Group multiple observables into summary_variables; build a FusionNetwork (one backbone
        # per key) in networks.factory to consume them.
        adapter = adapter.group(summary_variables, into="summary_variables")
        # ------------------------------------------------------------------------------------ #

    if inference_conditions:
        adapter = adapter.concatenate(inference_conditions, into="inference_conditions")

    return adapter
