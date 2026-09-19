#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shahab/miniconda3/envs/oneadmet/bin/python}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -c 'import numpy; print("Python environment check passed:", numpy.__version__)'

for seed in 2027 2028; do
  run_name="v15_rdkit2d_seed${seed}"
  if [[ -f "models/stl/Thalf__human__terminal_iv/${run_name}/complete.json" ]]; then
    echo "Skipping completed seed: ${seed}"
    continue
  fi
  "$PYTHON_BIN" scripts/train_Thalf.py \
    --datasets-dir data/processed_v15/datasets \
    --splits-dir data/processed_v15/splits \
    --algorithms ridge,rf,extratrees \
    --feature-set rdkit2d \
    --trials 20 \
    --threads 12 \
    --seed "$seed" \
    --run-name "$run_name"
done
