#!/usr/bin/env bash
set -euo pipefail

# Frozen formal VDss comparison: 5 outer folds × 3 seeds × 3 routes.
# Fail fast instead of silently starting the long run on CPU.
cd /home/shahab/MTL-model-4-1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if ! conda run --no-capture-output -n oneadmet \
  python -c 'import sys, torch; print("cuda_available=", torch.cuda.is_available()); print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); sys.exit(0 if torch.cuda.is_available() else 1)'
then
  echo "CUDA is not visible in the oneadmet environment; formal training was not started." >&2
  exit 1
fi
conda run --no-capture-output -n oneadmet \
  python scripts/run_vdss_cross_species_transfer_traincv.py \
  --protocol data/public_development/vdss_cross_species_transfer_protocol_v4 \
  --device cuda \
  --output results/benchmarks/vdss_cross_species_transfer_traincv_v1
