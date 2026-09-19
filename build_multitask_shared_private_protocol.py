#!/usr/bin/env python3
"""Freeze the finite shared-encoder versus fully-private multitask ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


ROUTES = [
    {
        "route_id": "fully_private_task_networks",
        "encoder_sharing": "none",
        "private_heads": "one encoder and one head per task",
        "purpose": "matched neural control for representation-sharing attribution",
    },
    {
        "route_id": "shared_encoder_private_heads",
        "encoder_sharing": "all_registered_tasks",
        "private_heads": "one head per task",
        "purpose": "test finite shared molecular representation with task-specific outputs",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--p1-freeze", type=Path, default=ROOT / "data/public_development/stl_p1_decision_freeze_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/multitask_shared_private_protocol_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.interface / "complete.json", args.interface / "task_target_transforms_train_only.csv",
        args.interface / "tiered_task_loss_schedule.csv", args.interface / "training_records.csv",
        args.stl / "complete.json", args.stl / "benchmark_train_records.csv",
        args.p1_freeze / "complete.json", args.p1_freeze / "p1_decision_registry.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    if any(m.get("test_labels_read", False) for m in [interface_meta, stl_meta, p1_meta]):
        raise ValueError("Protocol inputs must remain test-closed")

    transforms = pd.read_csv(args.interface / "task_target_transforms_train_only.csv")
    schedule = pd.read_csv(args.interface / "tiered_task_loss_schedule.csv")
    records = pd.read_csv(
        args.interface / "training_records.csv", dtype=str, keep_default_na=False,
        usecols=["row_id", "task_id", "split", "parent_id", "scaffold_group"],
    )
    tasks = sorted(transforms.task_id.astype(str).unique())
    if len(tasks) != 45 or set(records.task_id) != set(tasks):
        raise ValueError("Expected the frozen 45-task interface")
    human = transforms.loc[transforms.head_selection_authorized.astype(bool)].copy()
    if len(human) != 6 or set(human.endpoint) != {"CL", "CLint", "Papp", "Thalf", "VDss", "fu"}:
        raise ValueError("Expected the six authorized human evaluation heads")
    if schedule.task_id.duplicated().any() or set(schedule.task_id) != set(tasks):
        raise ValueError("Task schedule does not uniquely cover all tasks")

    target_rows = []
    for row in transforms.itertuples(index=False):
        bounded = row.endpoint in {"F", "fu"}
        target_rows.append({
            "task_id": row.task_id,
            "endpoint": row.endpoint,
            "canonical_unit": row.canonical_unit,
            "input_target_transform": row.target_transform,
            "parent_aggregation": "mean_record_level_expit_physical" if row.endpoint == "fu" else "mean_record_target",
            "target_standardization": "none_physical_fraction" if bounded else "taskwise_zscore_fit_on_retained_outer_train_parents",
            "model_output": "bounded_sigmoid_fraction" if bounded else "unbounded_standardized_target",
            "training_loss": "physical_MAE" if bounded else "task_standardized_Huber",
            "primary_metric": "physical_MAE" if row.endpoint == "fu" else "transformed_RMSE",
            "head_selection_authorized": bool(row.head_selection_authorized),
            "role": "human_primary_evaluation" if bool(row.head_selection_authorized) else "representation_auxiliary_only",
        })
    target_registry = pd.DataFrame(target_rows)
    p1 = pd.read_csv(args.p1_freeze / "p1_decision_registry.csv")
    baselines = p1.loc[p1.task_id.isin(set(human.task_id))].copy()
    if len(baselines) != 6 or not baselines.reference_for_next_stage.eq("corrected_stageA_matched_control").all():
        raise ValueError("Six corrected Stage-A references are not frozen")

    stl = pd.read_csv(
        args.stl / "benchmark_train_records.csv", dtype=str, keep_default_na=False,
        usecols=["row_id", "task_id", "parent_id", "scaffold_group", "inner_fold_id"],
    )
    stl = stl.loc[stl.task_id.isin(set(human.task_id))].copy()
    stl["inner_fold_id"] = pd.to_numeric(stl.inner_fold_id, errors="raise").astype(int)
    isolation_rows = []
    train_records = records.loc[records.split.eq("train")].copy()
    for fold in range(5):
        evaluation = stl.loc[stl.inner_fold_id.eq(fold)]
        eval_parents, eval_scaffolds = set(evaluation.parent_id), set(evaluation.scaffold_group)
        retained = train_records.loc[
            ~train_records.parent_id.isin(eval_parents) & ~train_records.scaffold_group.isin(eval_scaffolds)
        ]
        if set(retained.task_id) != set(tasks):
            raise ValueError(f"Global fold isolation removed a task in fold {fold}")
        isolation_rows.append({
            "outer_fold": fold, "evaluation_tasks": evaluation.task_id.nunique(),
            "evaluation_parents_union": len(eval_parents), "evaluation_scaffolds_union": len(eval_scaffolds),
            "retained_tasks": retained.task_id.nunique(), "retained_records": len(retained),
            "retained_parents": retained.parent_id.nunique(),
            "post_filter_parent_overlap": int(retained.parent_id.isin(eval_parents).sum()),
            "post_filter_scaffold_overlap": int(retained.scaffold_group.isin(eval_scaffolds).sum()),
        })
    isolation = pd.DataFrame(isolation_rows)
    if isolation[["post_filter_parent_overlap", "post_filter_scaffold_overlap"]].to_numpy().max() != 0:
        raise ValueError("Protocol isolation simulation failed")

    routes = pd.DataFrame(ROUTES)
    routes["feature_set"] = "rdkit2d"
    routes["architecture"] = "MLP_128_64_GELU_layernorm"
    routes["formal_outer_folds"] = 5
    routes["formal_seeds"] = "20260917;20260918;20260919"
    routes["phase1_epochs"] = 20
    routes["phase2_epochs"] = 10
    routes["batch_size"] = 128
    routes["learning_rate"] = 0.0003
    routes["weight_decay"] = 0.001
    routes["architecture_selection_authorized"] = False
    protocol = {
        "schema_version": 1,
        "routes": routes.route_id.tolist(),
        "tasks": tasks,
        "human_evaluation_tasks": sorted(human.task_id.tolist()),
        "exploratory_human_F_task": "F__human__absolute_oral",
        "feature_set": "rdkit2d",
        "outer_fold_isolation": "union the six human evaluation parent/scaffold sets, then purge every registered task",
        "evaluation_label_lifecycle": "read each human outer-fold target only after every route and seed has fit and predicted",
        "feature_preprocessing": "identical task-specific imputation and scaling fit only on each retained outer-train task for both routes",
        "sampling": "mandatory one-batch coverage for every active task per epoch, followed by an equal number of batches sampled from the frozen tier-then-task weights",
        "phase1": "all 45 tasks",
        "phase2": "six authorized human heads only",
        "human_F_policy": "representation auxiliary and exploratory output only; prohibited from selection",
        "fixed_validation_status": "closed",
        "test_status": "closed",
        "formal_scope": "five outer folds x three seeds x two architectures",
        "advance_gate": {
            "minimum_mean_improvement_vs_fully_private_pct": 2.0,
            "minimum_noninferior_outer_folds": 3,
            "paired_parent_bootstrap_upper_95ci_below_zero": True,
            "minimum_endpoints_passing": 3,
            "no_endpoint_relative_harm_over_pct": 5.0,
        },
    }
    json.dumps(protocol, allow_nan=False)
    if args.check_only:
        print(f"Shared/private protocol ready: tasks={len(tasks)} human_heads={len(human)} routes=2 folds=5 seeds=3")
        return
    with stage_output(args.output) as out:
        routes.to_csv(out / "route_registry.csv", index=False)
        target_registry.to_csv(out / "task_target_registry.csv", index=False)
        schedule[[
            "task_id", "tier_loss_weight", "tasks_in_tier", "within_tier_task_weight",
            "phase2_human_refinement", "phase2_task_weight",
        ]].to_csv(out / "task_sampling_registry.csv", index=False)
        isolation.to_csv(out / "outer_fold_global_isolation_feasibility.csv", index=False)
        baselines.to_csv(out / "corrected_stageA_references.csv", index=False)
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Finite multitask shared/private ablation\n\n"
            "The comparison is train-CV-only and globally purges the union of all six human evaluation parent/scaffold sets from all 45 tasks. "
            "fu and F use physical parent targets with sigmoid outputs; fixed validation/test remain closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "multitask_shared_private_protocol",
            inputs={
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_complete_sha256": sha256(args.stl / "complete.json"),
                "p1_freeze_complete_sha256": sha256(args.p1_freeze / "complete.json"),
            },
            tasks=45, human_evaluation_heads=6, routes=2, outer_folds=5, seeds=3,
            validation_target_file_opened=False, test_labels_read=False, model_fitted=False,
            architecture_selection_authorized=False, partial=False,
        )
    print(f"Shared/private protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
