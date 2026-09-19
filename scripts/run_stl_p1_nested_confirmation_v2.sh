#!/usr/bin/env bash
set -euo pipefail

# Corrected fu parent target plus matched Stage-A control:
# 60 candidate configurations × 5 outer folds × 4 inner fits, followed by
# 3-seed P1 and Stage-A-control refits per outer fold.
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet \
  python scripts/run_stl_p1_nested_confirmation.py \
  --verify-reload \
  --output results/benchmarks/stl_p1_nested_confirmation_v2
