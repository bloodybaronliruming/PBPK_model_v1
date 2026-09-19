#!/usr/bin/env bash
set -euo pipefail
cd /home/shahab/MTL-model-4-1
log_dir="results/pipeline_logs/jia2025_long_tasks"; mkdir -p "$log_dir"
stamp="$(date +%Y%m%d_%H%M%S)"; log="$log_dir/jia2025_r3d_bootstrap_v1_${stamp}.log"; pidfile="$log.pid"
nohup bash scripts/run_jia2025_r3d_bootstrap_v1.sh >"$log" 2>&1 < /dev/null & pid=$!; echo "$pid" > "$pidfile"
echo "Started Jia 2025 R3d bootstrap background task"; echo "PID: $pid"; echo "Log: $log"; echo "PID file: $pidfile"; echo "Check: bash scripts/check_jia2025_r3d_bootstrap_task.sh $log"
