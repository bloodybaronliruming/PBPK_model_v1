#!/usr/bin/env bash
set -euo pipefail

# Fixed, train-CV-only diagnostic: 2 tasks × 4 sample fractions × 3 seeds × 5 folds.
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet \
  python scripts/run_stl_papp_fu_learning_curve.py \
  --output results/analysis/stl_papp_fu_learning_curve_v1
