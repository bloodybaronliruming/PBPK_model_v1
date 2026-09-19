#!/usr/bin/env bash
# Public human-plasma baseline only.  This script never opens source-test labels.
set -euo pipefail

conda run --no-capture-output -n oneadmet python scripts/train_oneadmet_public_thalf.py \
  --algorithms ridge,extratrees \
  --feature-set rdkit2d \
  --trials 2 \
  --threads 12
