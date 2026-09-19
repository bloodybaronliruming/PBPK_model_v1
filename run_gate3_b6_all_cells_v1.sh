#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

authorization="data/public_development/gate3_b6_formal_run_v4"
batch_root="results/benchmarks/gate3_b6_physchem_formal_cells_v3"

conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_cell_batch.py \
  --authorization "$authorization" --batch-root "$batch_root" \
  --threads 8 --initial-cell-seconds 132 --check-only

conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_cell_batch.py \
  --authorization "$authorization" --batch-root "$batch_root" \
  --threads 8 --initial-cell-seconds 132

test -f "$batch_root/batch_complete.json"
echo "Gate 3 B6 all formal cells completed"
