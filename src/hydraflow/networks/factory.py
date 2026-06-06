"""Build BayesFlow networks from structured dataclass configs (no ``_target_``).

These are thin, opinionated wrappers around ``bayesflow.networks``. They translate the scalar
hyperparameters in ``SummaryNetworkConfig`` / ``InferenceNetworkConfig`` into the constructor
kwargs each network expects (e.g. expanding ``num_blocks`` into per-block tuples). ``bayesflow``
is imported lazily so config-only contexts (and the test suite) don't require the backend.

Multi-observable **fusion** is the documented extension seam: when an adapter groups several
observable keys into ``summary_variables``, build one summary net per key here and combine them
with ``bayesflow.networks.FusionNetwork``. The default single-observable path returns one net.
"""

from __future__ import annotations

from typing import Any


def build_summary_network(cfg) -> Any:
    """Return a single BayesFlow summary network for ``cfg`` (a ``SummaryNetworkConfig``)."""
    import bayesflow as bf

    t = cfg.type
    blocks = int(cfg.num_blocks)

    if t == "set_transformer":
        return bf.networks.SetTransformer(
            summary_dim=int(cfg.summary_dim),
            embed_dims=(int(cfg.embed_dim),) * blocks,
            num_heads=(int(cfg.num_heads),) * blocks,
            mlp_depths=(int(cfg.mlp_depth),) * blocks,
            mlp_widths=(int(cfg.mlp_width),) * blocks,
            dropout=float(cfg.dropout),
        )
    if t == "time_series_transformer":
        return bf.networks.TimeSeriesTransformer(
            summary_dim=int(cfg.summary_dim),
            embed_dims=(int(cfg.embed_dim),) * blocks,
            num_heads=(int(cfg.num_heads),) * blocks,
        )
    if t == "deep_set":
        return bf.networks.DeepSet(
            summary_dim=int(cfg.summary_dim),
            dropout=float(cfg.dropout),
        )
    raise ValueError(
        f"Unknown summary_network.type '{t}'. "
        "Expected one of: set_transformer, time_series_transformer, deep_set."
    )


def build_inference_network(cfg) -> Any:
    """Return a BayesFlow inference (posterior) network for ``cfg`` (an ``InferenceNetworkConfig``)."""
    import bayesflow as bf

    widths = [int(cfg.mlp_width)] * int(cfg.mlp_depth)
    t = cfg.type

    if t == "flow_matching":
        return bf.networks.FlowMatching(
            subnet_kwargs={"widths": widths, "dropout": float(cfg.dropout)},
        )
    if t == "diffusion":
        return bf.networks.DiffusionModel(
            subnet_kwargs={
                "widths": widths,
                "time_embedding_dim": int(cfg.time_embedding_dim),
            },
        )
    raise ValueError(
        f"Unknown inference_network.type '{t}'. Expected one of: flow_matching, diffusion."
    )
