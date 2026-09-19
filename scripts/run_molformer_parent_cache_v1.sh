#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

PYTHONPATH=scripts conda run --no-capture-output -n oneadmet \
  python scripts/build_molformer_parent_embedding_cache.py \
  --batch-size 64 \
  --output data/public_development/molformer_parent_embedding_cache_v1
