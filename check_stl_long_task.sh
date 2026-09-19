#!/usr/bin/env bash
set -euo pipefail

cd /home/shahab/MTL-model-4-1

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 results/pipeline_logs/stl_long_tasks/<task>.log" >&2
  exit 2
fi

log_path="$1"
pid_path="$log_path.pid"
if [[ ! -f "$log_path" ]]; then
  echo "Missing log: $log_path" >&2
  exit 1
fi

if [[ -f "$pid_path" ]]; then
  pid="$(<"$pid_path")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "status=RUNNING pid=$pid log=$log_path"
  else
    echo "status=FINISHED_OR_FAILED pid=$pid log=$log_path"
  fi
else
  echo "status=UNKNOWN_NO_PID_FILE log=$log_path"
fi

echo "--- last 20 log lines ---"
tail -n 20 "$log_path"

if [[ "$log_path" == *run_stl_ecfp4_stageA_v1_*.log ]]; then
  output="results/benchmarks/stl_stageA_ecfp4_v1"
elif [[ "$log_path" == *run_stl_combined_stageA_v1_*.log ]]; then
  output="results/benchmarks/stl_stageA_ecfp4_rdkit2d_v1"
elif [[ "$log_path" == *run_gate1b_dmpnn_stage1_v1_*.log ]]; then
  output="results/benchmarks/gate1b_dmpnn_stage1_v1"
elif [[ "$log_path" == *run_molformer_parent_cache_v1_*.log ]]; then
  output="data/public_development/molformer_parent_embedding_cache_v1"
elif [[ "$log_path" == *run_gate1b_frozen_smiles_probe_v1_*.log ]]; then
  output="results/benchmarks/gate1b_frozen_smiles_probe_v1"
elif [[ "$log_path" == *run_gate1b_frozen_smiles_adaptation_v1_*.log ]]; then
  output="results/benchmarks/gate1b_frozen_smiles_adaptation_v1"
elif [[ "$log_path" == *run_gate1b_fu_physical_alignment_v1_*.log ]]; then
  output="results/benchmarks/gate1b_fu_physical_alignment_v1"
elif [[ "$log_path" == *run_gate1b_nested_scaffold_mlp_v1_*.log ]]; then
  output="results/benchmarks/gate1b_nested_scaffold_mlp_v1"
elif [[ "$log_path" == *run_stl_ipc_feature_ablation_v1_*.log ]]; then
  output="results/benchmarks/stl_ipc_feature_ablation_v1"
elif [[ "$log_path" == *run_stl_papp_fu_learning_curve_v1_*.log ]]; then
  output="results/analysis/stl_papp_fu_learning_curve_v1"
elif [[ "$log_path" == *run_stl_papp_fu_source_cluster_holdout_v1_*.log ]]; then
  output="results/analysis/stl_papp_fu_source_cluster_holdout_v1"
elif [[ "$log_path" == *run_stl_p1_nested_confirmation_v1_*.log ]]; then
  output="results/benchmarks/stl_p1_nested_confirmation_v1"
elif [[ "$log_path" == *run_stl_p1_nested_confirmation_v2_*.log ]]; then
  output="results/benchmarks/stl_p1_nested_confirmation_v2"
elif [[ "$log_path" == *run_vdss_cross_species_transfer_traincv_v1_*.log ]]; then
  output="results/benchmarks/vdss_cross_species_transfer_traincv_v1"
elif [[ "$log_path" == *run_vdss_source_document_holdout_v1_*.log ]]; then
  output="results/analysis/vdss_source_document_holdout_v1"
elif [[ "$log_path" == *run_cl_cross_species_transfer_traincv_v1_*.log ]]; then
  output="results/benchmarks/cl_cross_species_transfer_traincv_v1"
elif [[ "$log_path" == *run_thalf_cross_species_transfer_traincv_v1_*.log ]]; then
  output="results/benchmarks/thalf_cross_species_transfer_traincv_v1"
elif [[ "$log_path" == *run_fu_cross_species_transfer_traincv_v1_*.log ]]; then
  output="results/benchmarks/fu_cross_species_transfer_traincv_v1"
elif [[ "$log_path" == *run_fu_cross_species_transfer_traincv_v2_*.log ]]; then
  output="results/benchmarks/fu_cross_species_transfer_traincv_v2"
elif [[ "$log_path" == *run_multitask_shared_private_traincv_v1_*.log ]]; then
  output="results/benchmarks/multitask_shared_private_traincv_v1"
elif [[ "$log_path" == *run_multitask_shared_private_traincv_v2_*.log ]]; then
  output="results/benchmarks/multitask_shared_private_traincv_v2"
elif [[ "$log_path" == *run_multitask_gradient_affinity_v1_*.log ]]; then
  output="results/analysis/multitask_gradient_affinity_v1"
elif [[ "$log_path" == *run_multitask_two_group_traincv_v1_*.log ]]; then
  output="results/benchmarks/multitask_two_group_traincv_v1"
elif [[ "$log_path" == *run_gate3_b6_first_formal_cell_v1_*.log ]]; then
  output="results/benchmarks/gate3_b6_physchem_formal_cells_v1/cl_human_systemic_iv_fold0"
elif [[ "$log_path" == *run_gate3_b6_first_formal_cell_v2_*.log ]]; then
  output="results/benchmarks/gate3_b6_physchem_formal_cells_v2/cl_human_systemic_iv_fold0"
elif [[ "$log_path" == *run_gate3_b6_remaining_cells_v1_*.log ]]; then
  output="results/benchmarks/gate3_b6_physchem_formal_cells_v2"
elif [[ "$log_path" == *run_gate3_b6_first_formal_cell_v3_*.log ]]; then
  output="results/benchmarks/gate3_b6_physchem_formal_cells_v3/cl_human_systemic_iv_fold0"
elif [[ "$log_path" == *run_gate3_b6_all_cells_v1_*.log ]]; then
  output="results/benchmarks/gate3_b6_physchem_formal_cells_v3"
elif [[ "$log_path" == *run_mmpk_n2b_strict_r1_baseline_screen_v1_*.log ]]; then
  output="results/benchmarks/mmpk_n2b_strict_r1_baseline_screen_v1"
elif [[ "$log_path" == *run_mmpk_n2c_paired_sensitivity_v1_*.log ]]; then
  output="results/benchmarks/mmpk_n2c_paired_sensitivity_v1"
else
  output=""
fi
if [[ ( "$log_path" == *run_gate3_b6_remaining_cells_v1_*.log || "$log_path" == *run_gate3_b6_all_cells_v1_*.log ) && -f "$output/batch_complete.json" ]]; then
  echo "complete_marker=$output/batch_complete.json"
elif [[ -n "$output" && -f "$output/complete.json" ]]; then
  echo "complete_marker=$output/complete.json"
fi
