#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_cell_batch.py \
  --authorization data/public_development/gate3_b6_formal_run_v3 \
  --batch-root results/benchmarks/gate3_b6_physchem_formal_cells_v2 \
  --threads 8 --check-only

conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_cell_batch.py \
  --authorization data/public_development/gate3_b6_formal_run_v3 \
  --batch-root results/benchmarks/gate3_b6_physchem_formal_cells_v2 \
  --threads 8

test -f results/benchmarks/gate3_b6_physchem_formal_cells_v2/batch_complete.json
echo "Gate 3 B6 remaining formal cells completed"
