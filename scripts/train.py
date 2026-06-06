#!/usr/bin/env python
"""Entry point: training. Thin wrapper over hydraflow.pipeline.train."""

from hydraflow.pipeline.train import cli

if __name__ == "__main__":
    cli()
