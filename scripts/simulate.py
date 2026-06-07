#!/usr/bin/env python
"""Entry point: dataset generation. Thin wrapper over hydrabflow.pipeline.simulate."""

from hydrabflow.pipeline.simulate import cli

if __name__ == "__main__":
    cli()
