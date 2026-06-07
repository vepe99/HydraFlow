"""Deterministic, whole-dataset preprocessing.

This is the half of data handling that is NOT augmentation: transforms applied once to the full
dataset (NaN cleaning, train/val split, z-score standardization), fitted on the training split,
and saved to the run dir so evaluation / real-data inference replay the exact same transform.

Import side effect: the shipped steps register themselves.
"""

from hydrabflow.preprocessing import standardize as _standardize  # noqa: F401 (self-registers)
from hydrabflow.preprocessing import steps as _steps  # noqa: F401 (self-registers)
from hydrabflow.preprocessing.base import PreprocessPipeline, PreprocessStep
from hydrabflow.preprocessing.registry import build_pipeline, register_step

__all__ = [
    "PreprocessPipeline",
    "PreprocessStep",
    "build_pipeline",
    "register_step",
]
