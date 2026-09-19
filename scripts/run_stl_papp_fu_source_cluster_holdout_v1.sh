#!/usr/bin/env bash
set -euo pipefail

# Fixed source-cluster and scaffold-isolated diagnostic: 2 tasks × 5 folds × 3 seeds.
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet \
  python scripts/run_stl_papp_fu_source_cluster_holdout.py \
  --output results/analysis/stl_papp_fu_source_cluster_holdout_v1
