#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shahab/miniconda3/envs/oneadmet/bin/python}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -c 'import numpy; print("Python environment check passed:", numpy.__version__)'

"$PYTHON_BIN" scripts/audit_endpoint_sources.py \
  --output data/processed_v12/audit \
  --papp-decision-registry results/analysis/papp_provenance_decisions_v2.csv \
  --papp-unresolved-policy exclude \
  --human-evidence-registry results/analysis/human_evidence_decisions_v10.csv

"$PYTHON_BIN" scripts/rebuild_endpoint_datasets.py \
  --audit-dir data/processed_v12/audit \
  --output data/processed_v12/datasets

"$PYTHON_BIN" scripts/build_split_manifest.py \
  --datasets-dir data/processed_v12/datasets \
  --cache-path data/processed_v12/_cache/structure_groups.sqlite \
  --output data/processed_v12/splits

"$PYTHON_BIN" scripts/verify_dataset_version.py \
  --baseline-dir data/processed_v11 \
  --candidate-dir data/processed_v12 \
  --expect-delta Thalf__human__terminal_iv:train:-21:-12 \
  --expect-delta Thalf__human__terminal_iv:val:-3:-1 \
  --expect-delta Thalf__human__terminal_iv:test:0:0 \
  --expect-delta F__human__absolute_oral:train:0:0 \
  --expect-delta F__human__absolute_oral:val:0:0 \
  --expect-delta F__human__absolute_oral:test:0:0

"$PYTHON_BIN" scripts/audit_human_evidence_expansion.py \
  --audit-dir data/processed_v12/audit \
  --output results/analysis/human_evidence_expansion_v12
