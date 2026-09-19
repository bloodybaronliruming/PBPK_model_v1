#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet python scripts/run_gate1b_nested_scaffold_mlp.py
