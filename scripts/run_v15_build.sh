#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shahab/miniconda3/envs/oneadmet/bin/python}"
REGISTRY="results/analysis/obach_thalf_audit_v1/human_evidence_decisions_v13.csv"

cd "$ROOT_DIR"

"$PYTHON_BIN" -c 'import numpy; print("Python environment check passed:", numpy.__version__)'

"$PYTHON_BIN" scripts/audit_endpoint_sources.py \
  --output data/processed_v15/audit \
  --papp-decision-registry results/analysis/papp_provenance_decisions_v2.csv \
  --papp-unresolved-policy exclude \
  --human-evidence-registry "$REGISTRY"

"$PYTHON_BIN" scripts/rebuild_endpoint_datasets.py \
  --audit-dir data/processed_v15/audit \
  --output data/processed_v15/datasets

mkdir -p data/processed_v15/_cache
if [[ ! -f data/processed_v15/_cache/structure_groups.sqlite && -f data/processed_v14/_cache/structure_groups.sqlite ]]; then
  cp data/processed_v14/_cache/structure_groups.sqlite data/processed_v15/_cache/structure_groups.sqlite
fi

"$PYTHON_BIN" scripts/build_split_manifest.py \
  --datasets-dir data/processed_v15/datasets \
  --cache-path data/processed_v15/_cache/structure_groups.sqlite \
  --output data/processed_v15/splits

"$PYTHON_BIN" scripts/verify_dataset_version.py \
  --baseline-dir data/processed_v14 \
  --candidate-dir data/processed_v15 \
  --expect-delta Thalf__human__terminal_iv:train:530:509 \
  --expect-delta Thalf__human__terminal_iv:val:53:50 \
  --expect-delta Thalf__human__terminal_iv:test:60:60 \
  --expect-delta F__human__absolute_oral:train:0:0 \
  --expect-delta F__human__absolute_oral:val:0:0 \
  --expect-delta F__human__absolute_oral:test:0:0

"$PYTHON_BIN" scripts/audit_human_evidence_expansion.py \
  --audit-dir data/processed_v15/audit \
  --output results/analysis/human_evidence_expansion_v15
