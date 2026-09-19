#!/usr/bin/env bash
set -euo pipefail
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet python scripts/analyze_jia2025_r3d_bootstrap.py
