#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

PYTHONPATH=scripts conda run --no-capture-output -n oneadmet \
  python scripts/run_gate1b_dmpnn_screen.py \
  --stl-protocol data/public_development/stl_benchmark_protocol_v1 \
  --multimodal-protocol data/public_development/gate1b_multimodal_protocol_v2 \
  --graph-manifest data/public_development/gate1b_graph_manifest_v1 \
  --output results/benchmarks/gate1b_dmpnn_stage1_v1 \
  --profile stage1 \
  --device cuda \
  --threads 4 \
  --seed 20260916 \
  --verify-reload
