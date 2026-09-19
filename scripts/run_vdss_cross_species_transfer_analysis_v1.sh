#!/usr/bin/env bash
set -euo pipefail

# Run only after the formal train-CV complete marker has been verified.
cd /home/shahab/MTL-model-4-1
conda run --no-capture-output -n oneadmet \
  python scripts/analyze_vdss_cross_species_transfer_traincv.py \
  --traincv results/benchmarks/vdss_cross_species_transfer_traincv_v1 \
  --protocol data/public_development/vdss_cross_species_transfer_protocol_v4 \
  --output results/analysis/vdss_cross_species_transfer_traincv_analysis_v1
