#!/usr/bin/env python
"""Entry point: dataset generation. Thin wrapper over hydraflow.pipeline.simulate."""

from hydraflow.pipeline.simulate import cli

if __name__ == "__main__":
    cli()
