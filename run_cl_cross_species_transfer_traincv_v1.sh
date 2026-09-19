#!/usr/bin/env bash
set -euo pipefail

# Frozen CL comparison: 5 outer folds x 3 seeds x 3 routes.
cd /home/shahab/MTL-model-4-1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if ! conda run --no-capture-output -n oneadmet \
  python -c 'import sys, torch; print("cuda_available=", torch.cuda.is_available()); print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); sys.exit(0 if torch.cuda.is_available() else 1)'
then
  echo "CUDA is not visible in the oneadmet environment; formal CL training was not started." >&2
  exit 1
fi
conda run --no-capture-output -n oneadmet \
  python scripts/run_cross_species_transfer_traincv.py \
  --protocol data/public_development/cl_cross_species_transfer_protocol_v1 \
  --device cuda \
  --output results/benchmarks/cl_cross_species_transfer_traincv_v1
