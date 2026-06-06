#!/usr/bin/env python
"""Entry point: hyperparameter tuning. Thin wrapper over hydraflow.pipeline.tune."""

from hydraflow.pipeline.tune import cli

if __name__ == "__main__":
    cli()
