"""HydraFlow: a reusable BayesFlow + Hydra SBI pipeline template.

Importing the package pins the Keras backend to JAX (see :mod:`hydraflow.utils.backend`)
*before* any keras/bayesflow import can happen, so downstream modules can import keras safely.
"""

from hydraflow.utils import backend as _backend  # noqa: F401  (side effect: sets KERAS_BACKEND)

__version__ = "0.1.0"
