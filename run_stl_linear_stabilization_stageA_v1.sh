#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

PYTHONPATH=scripts conda run --no-capture-output -n oneadmet \
  python scripts/run_stl_benchmark_screen.py \
  --protocol data/public_development/stl_benchmark_protocol_v1 \
  --output results/benchmarks/stl_stageA_linear_stabilization_rdkit2d_v1 \
  --algorithms ridge,elasticnet \
  --feature-set rdkit2d \
  --scope train_cv \
  --candidate-profile linear_stabilization \
  --trials-per-algorithm 2 \
  --verify-reload \
  --seed 20260916 \
  --threads 8
