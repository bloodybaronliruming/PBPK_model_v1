#!/usr/bin/env bash
set -euo pipefail
cd /home/shahab/MTL-model-4-1
if [[ $# -ne 1 ]]; then echo "Usage: $0 <log-path>" >&2; exit 2; fi
log="$1"; pidfile="$log.pid"
if [[ ! -f "$log" ]]; then echo "Missing log: $log" >&2; exit 1; fi
if [[ -f "$pidfile" ]] && kill -0 "$(<"$pidfile")" 2>/dev/null; then echo "status=RUNNING pid=$(<"$pidfile") log=$log"; else echo "status=FINISHED_OR_FAILED log=$log"; fi
tail -n 30 "$log"
if [[ -f results/analysis/jia2025_author_like_r3d_bootstrap_v1/complete.json ]]; then echo "complete_marker=results/analysis/jia2025_author_like_r3d_bootstrap_v1/complete.json"; fi
