#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

output="results/benchmarks/gate3_b6_physchem_formal_cells_v3/cl_human_systemic_iv_fold0"
conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_outer_fold.py \
  --task CL__human__systemic_iv --outer-fold 0 --threads 8 --output "$output"

test -f "$output/complete.json"
echo "Gate 3 B6 first formal cell v3 completed"
