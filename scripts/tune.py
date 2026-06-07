#!/usr/bin/env python
"""Entry point: hyperparameter tuning. Thin wrapper over hydrabflow.pipeline.tune."""

from hydrabflow.pipeline.tune import cli

if __name__ == "__main__":
    cli()
