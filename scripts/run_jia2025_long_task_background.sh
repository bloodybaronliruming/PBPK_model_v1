#!/usr/bin/env bash
set -euo pipefail
cd /home/shahab/MTL-model-4-1
if [[ $# -ne 1 || "$1" != "scripts/run_jia2025_author_like_r3a_v1.sh" ]]; then
  echo "Usage: $0 scripts/run_jia2025_author_like_r3a_v1.sh" >&2; exit 2
fi
log_dir="results/pipeline_logs/jia2025_long_tasks"; mkdir -p "$log_dir"
stamp="$(date +%Y%m%d_%H%M%S)"; log="$log_dir/jia2025_author_like_r3a_v1_${stamp}.log"; pidfile="$log.pid"
nohup bash "$1" >"$log" 2>&1 < /dev/null & pid=$!; echo "$pid" > "$pidfile"
echo "Started Jia 2025 R3a background task"; echo "PID: $pid"; echo "Log: $log"; echo "PID file: $pidfile"; echo "Check: kill -0 $pid 2>/dev/null && echo RUNNING || echo FINISHED"
