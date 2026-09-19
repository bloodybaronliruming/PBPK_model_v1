#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shahab/miniconda3/envs/oneadmet/bin/python}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -c 'import numpy; print("Python environment check passed:", numpy.__version__)'

# The existing v15_obach_seed2026 run is the ecfp4_rdkit2d reference.
for feature_set in ecfp4 rdkit2d mechanism2d; do
  run_name="v15_${feature_set}_seed2026"
  if [[ -f "models/stl/Thalf__human__terminal_iv/${run_name}/complete.json" ]]; then
    echo "Skipping completed feature set: ${feature_set}"
    continue
  fi
  "$PYTHON_BIN" scripts/train_Thalf.py \
    --datasets-dir data/processed_v15/datasets \
    --splits-dir data/processed_v15/splits \
    --algorithms ridge,rf,extratrees \
    --feature-set "$feature_set" \
    --trials 20 \
    --threads 12 \
    --seed 2026 \
    --run-name "$run_name"
done
