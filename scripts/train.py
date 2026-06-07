#!/usr/bin/env python
"""Entry point: training. Thin wrapper over hydrabflow.pipeline.train."""

from hydrabflow.pipeline.train import cli

if __name__ == "__main__":
    cli()
