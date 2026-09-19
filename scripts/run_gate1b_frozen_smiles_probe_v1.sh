#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

PYTHONPATH=scripts conda run --no-capture-output -n oneadmet \
  python scripts/run_gate1b_frozen_smiles_probe.py \
  --output results/benchmarks/gate1b_frozen_smiles_probe_v1 \
  --threads 4 \
  --seed 20260916 \
  --verify-reload
