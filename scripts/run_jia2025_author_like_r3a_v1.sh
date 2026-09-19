#!/usr/bin/env bash
set -euo pipefail
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet python scripts/train_jia2025_author_like_r3a.py
