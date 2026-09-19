#!/usr/bin/env python3
"""Protocol-driven entry point for controlled cross-species train-CV."""

from pipeline_common import run_cli
from run_vdss_cross_species_transfer_traincv import main


if __name__ == "__main__":
    run_cli(main)
