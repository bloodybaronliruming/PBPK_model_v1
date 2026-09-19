#!/usr/bin/env bash
set -euo pipefail

# Frozen VDss source-perturbation stress test: 5 document groups x 3 seeds x 3 routes.
# Fail fast instead of silently starting the formal run on CPU.
cd /home/shahab/MTL-model-4-1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if ! conda run --no-capture-output -n oneadmet \
  python -c 'import sys, torch; print("cuda_available=", torch.cuda.is_available()); print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); sys.exit(0 if torch.cuda.is_available() else 1)'
then
  echo "CUDA is not visible in the oneadmet environment; formal source stress test was not started." >&2
  exit 1
fi
conda run --no-capture-output -n oneadmet \
  python scripts/run_vdss_source_document_holdout.py \
  --source-protocol data/public_development/vdss_source_generalization_protocol_v1 \
  --transfer-protocol data/public_development/vdss_cross_species_transfer_protocol_v4 \
  --device cuda \
  --output results/analysis/vdss_source_document_holdout_v1
