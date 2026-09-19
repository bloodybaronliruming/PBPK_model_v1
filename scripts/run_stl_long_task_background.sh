#!/usr/bin/env bash
set -euo pipefail

# Launch one registered long STL job detached from the terminal.

cd /home/shahab/MTL-model-4-1

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <registered STL/Gate1B long-task script>" >&2
  exit 2
fi

task_script="$1"
case "$task_script" in
  scripts/run_stl_ecfp4_stageA_v1.sh|scripts/run_stl_combined_stageA_v1.sh|scripts/run_gate1b_dmpnn_stage1_v1.sh|scripts/run_molformer_parent_cache_v1.sh|scripts/run_gate1b_frozen_smiles_probe_v1.sh|scripts/run_gate1b_frozen_smiles_adaptation_v1.sh|scripts/run_gate1b_fu_physical_alignment_v1.sh|scripts/run_gate1b_nested_scaffold_mlp_v1.sh|scripts/run_stl_ipc_feature_ablation_v1.sh|scripts/run_stl_papp_fu_learning_curve_v1.sh|scripts/run_stl_papp_fu_source_cluster_holdout_v1.sh|scripts/run_stl_p1_nested_confirmation_v1.sh|scripts/run_stl_p1_nested_confirmation_v2.sh|scripts/run_vdss_cross_species_transfer_traincv_v1.sh|scripts/run_vdss_source_document_holdout_v1.sh|scripts/run_cl_cross_species_transfer_traincv_v1.sh|scripts/run_thalf_cross_species_transfer_traincv_v1.sh|scripts/run_fu_cross_species_transfer_traincv_v1.sh|scripts/run_fu_cross_species_transfer_traincv_v2.sh|scripts/run_multitask_shared_private_traincv_v1.sh|scripts/run_multitask_shared_private_traincv_v2.sh|scripts/run_multitask_gradient_affinity_v1.sh|scripts/run_multitask_two_group_traincv_v1.sh|scripts/run_gate3_b6_first_formal_cell_v1.sh|scripts/run_gate3_b6_first_formal_cell_v2.sh|scripts/run_gate3_b6_remaining_cells_v1.sh|scripts/run_gate3_b6_first_formal_cell_v3.sh|scripts/run_gate3_b6_all_cells_v1.sh|scripts/run_mmpk_n2b_strict_r1_baseline_screen_v1.sh|scripts/run_mmpk_n2c_paired_sensitivity_v1.sh) ;;
  *) echo "Refusing unregistered long task: $task_script" >&2; exit 2 ;;
esac

if [[ ! -f "$task_script" ]]; then
  echo "Missing task script: $task_script" >&2
  exit 1
fi

log_dir="results/pipeline_logs/stl_long_tasks"
mkdir -p "$log_dir"
stamp="$(date +%Y%m%d_%H%M%S)"
task_name="$(basename "$task_script" .sh)"
log_path="$log_dir/${task_name}_${stamp}.log"
pid_path="$log_path.pid"

nohup bash "$task_script" >"$log_path" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pid_path"

echo "Started background STL task"
echo "PID: $pid"
echo "Log: $log_path"
echo "PID file: $pid_path"
echo "Check: kill -0 $pid 2>/dev/null && echo RUNNING || echo FINISHED"
echo "Status command: bash scripts/check_stl_long_task.sh $log_path"
