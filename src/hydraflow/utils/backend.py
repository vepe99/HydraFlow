"""Pin the Keras backend before keras/bayesflow are imported anywhere.

BayesFlow runs on Keras 3, which selects its compute backend from the ``KERAS_BACKEND``
environment variable *at import time*. This template defaults to JAX (matching the reference
project and giving fast, vectorizable simulators). Importing :mod:`hydraflow` imports this module
first, guaranteeing the variable is set before the first ``import keras``.

To override (e.g. to ``torch``), set ``KERAS_BACKEND`` in your shell before launching.
"""

from __future__ import annotations

import os

DEFAULT_BACKEND = "jax"


def set_backend(backend: str = DEFAULT_BACKEND) -> str:
    """Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active backend."""
    os.environ.setdefault("KERAS_BACKEND", backend)
    return os.environ["KERAS_BACKEND"]


# Side effect on import: pin the backend.
ACTIVE_BACKEND = set_backend()
