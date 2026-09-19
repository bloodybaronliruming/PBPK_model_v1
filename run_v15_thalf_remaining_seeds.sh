#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shahab/miniconda3/envs/oneadmet/bin/python}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -c 'import numpy; print("Python environment check passed:", numpy.__version__)'

for seed in 2027 2028; do
  "$PYTHON_BIN" scripts/train_Thalf.py \
    --datasets-dir data/processed_v15/datasets \
    --splits-dir data/processed_v15/splits \
    --algorithms ridge,rf,extratrees \
    --trials 20 \
    --threads 12 \
    --seed "$seed" \
    --run-name "v15_obach_seed${seed}"
done
