"""Minimal logging helper so all pipeline stages log consistently."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "hydrabflow") -> logging.Logger:
    """Return a configured logger. Hydra also installs its own handlers; this is a safe default."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
