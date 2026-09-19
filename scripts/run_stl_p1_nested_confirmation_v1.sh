#!/usr/bin/env bash
set -euo pipefail

# Frozen P1 registry: 60 candidates × 5 outer folds × 4 inner scaffold fits,
# followed by 3-seed refits of each outer-fold winner.
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet \
  python scripts/run_stl_p1_nested_confirmation.py \
  --verify-reload \
  --output results/benchmarks/stl_p1_nested_confirmation_v1
