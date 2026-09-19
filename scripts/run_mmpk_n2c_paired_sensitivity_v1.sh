#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n oneadmet python scripts/run_mmpk_n2c_paired_sensitivity.py \
  --device cuda --threads 8 --confirm-formal-paired-sensitivity
