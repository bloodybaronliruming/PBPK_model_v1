#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if ! conda run --no-capture-output -n oneadmet \
  python -c 'import sys, torch; print("cuda_available=", torch.cuda.is_available()); print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); sys.exit(0 if torch.cuda.is_available() else 1)'
then
  echo "CUDA is not visible in the oneadmet environment; formal two-group training was not started." >&2
  exit 1
fi
conda run --no-capture-output -n oneadmet \
  python scripts/run_multitask_two_group_traincv.py \
  --device cuda \
  --output results/benchmarks/multitask_two_group_traincv_v1
