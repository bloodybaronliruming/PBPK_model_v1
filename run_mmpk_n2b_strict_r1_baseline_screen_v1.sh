#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n oneadmet python scripts/run_mmpk_n2b_strict_baseline_screen.py \
  --device cuda --threads 8 --confirm-formal-r1-screen
