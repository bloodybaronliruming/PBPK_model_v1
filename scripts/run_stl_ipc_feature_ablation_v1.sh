#!/usr/bin/env bash
set -euo pipefail

# Pre-registered, train-CV-only Ipc descriptor ablation of the six Stage-A
# ExtraTrees leaders.  This script intentionally has a fixed, new output path.
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet \
  python scripts/run_stl_ipc_feature_ablation.py \
  --output results/benchmarks/stl_ipc_feature_ablation_v1
