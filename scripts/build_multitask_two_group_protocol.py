#!/usr/bin/env python3
"""Freeze the gradient-informed fu/non-fu two-group multitask protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from multitask_two_group_common import (
    FU_GROUP, NON_FU_GROUP, ROUTE_ID, encoder_parameter_count, private_head_parameter_count,
)
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, default=ROOT / "data/public_development/multitask_shared_private_protocol_v2")
    parser.add_argument("--traincv", type=Path, default=ROOT / "results/benchmarks/multitask_shared_private_traincv_v2")
    parser.add_argument("--traincv-analysis", type=Path, default=ROOT / "results/analysis/multitask_shared_private_traincv_analysis_v2")
    parser.add_argument("--affinity", type=Path, default=ROOT / "results/analysis/multitask_gradient_affinity_v1")
    parser.add_argument("--affinity-report", type=Path, default=ROOT / "results/analysis/multitask_gradient_affinity_report_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/multitask_two_group_protocol_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    checkpoint = args.traincv / "models/fold0__seed20260917__shared_encoder_private_heads.pt"
    required = [
        args.base_protocol / "complete.json", args.base_protocol / "protocol.json",
        args.base_protocol / "route_registry.csv", args.base_protocol / "task_target_registry.csv",
        args.base_protocol / "task_sampling_registry.csv", args.base_protocol / "corrected_stageA_references.csv",
        args.base_protocol / "outer_fold_global_isolation_feasibility.csv",
        args.traincv / "complete.json", checkpoint,
        args.traincv_analysis / "complete.json", args.traincv_analysis / "decision.json",
        args.affinity / "complete.json", args.affinity_report / "complete.json",
        args.affinity_report / "next_architecture_candidate.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    base_meta = verify_stage(args.base_protocol, "multitask_shared_private_protocol")
    train_meta = verify_stage(args.traincv, "multitask_shared_private_traincv")
    analysis_meta = verify_stage(args.traincv_analysis, "multitask_shared_private_traincv_analysis")
    affinity_meta = verify_stage(args.affinity, "multitask_gradient_affinity")
    report_meta = verify_stage(args.affinity_report, "multitask_gradient_affinity_report")
    metas = [base_meta, train_meta, analysis_meta, affinity_meta, report_meta]
    if any(meta.get("test_labels_read", False) for meta in metas):
        raise ValueError("Two-group protocol inputs must remain test-closed")
    if not train_meta.get("full_configuration") or not affinity_meta.get("full_configuration"):
        raise ValueError("Two-group protocol requires both full formal predecessor runs")

    decision = json.loads((args.traincv_analysis / "decision.json").read_text(encoding="utf-8"))
    if decision.get("shared_architecture_advanced") or decision.get("maximum_endpoint_relative_harm_pct", 0) < 5:
        raise ValueError("Formal shared/private result does not justify conflict-guided grouping")
    candidate = pd.read_csv(args.affinity_report / "next_architecture_candidate.csv")
    if len(candidate) != 1 or candidate.iloc[0].candidate != ROUTE_ID:
        raise ValueError("Affinity report does not nominate the frozen two-group candidate")
    if candidate[["architecture_training_authorized", "fixed_validation_authorized", "test_authorized"]].to_numpy(bool).any():
        raise ValueError("Affinity report improperly authorizes training or evaluation")

    base = json.loads((args.base_protocol / "protocol.json").read_text(encoding="utf-8"))
    target = pd.read_csv(args.base_protocol / "task_target_registry.csv")
    sampling = pd.read_csv(args.base_protocol / "task_sampling_registry.csv")
    tasks = list(map(str, base["tasks"]))
    human_tasks = list(map(str, base["human_evaluation_tasks"]))
    if len(tasks) != 45 or len(human_tasks) != 6 or set(target.task_id) != set(tasks):
        raise ValueError("Frozen base task registry is incomplete")
    sampling["phase2_human_refinement"] = sampling.task_id.isin(human_tasks)
    sampling["phase2_task_weight"] = sampling.phase2_human_refinement.astype(float) / len(human_tasks)
    if sampling.loc[sampling.task_id.eq("F__human__absolute_oral"), "phase2_human_refinement"].iloc[0]:
        raise ValueError("Human F must not enter phase-2 refinement")
    group_registry = target[["task_id", "endpoint", "role"]].copy()
    group_registry["encoder_group"] = group_registry.endpoint.map(lambda value: FU_GROUP if value == "fu" else NON_FU_GROUP)
    group_registry["routing_basis"] = group_registry.endpoint.map(
        lambda value: "endpoint_equals_fu" if value == "fu" else "endpoint_not_fu"
    )
    counts = group_registry.encoder_group.value_counts().to_dict()
    if counts != {NON_FU_GROUP: 36, FU_GROUP: 9}:
        raise ValueError(f"Unexpected task grouping: {counts}")
    if group_registry.loc[group_registry.task_id.eq("F__human__absolute_oral"), "role"].iloc[0] != "representation_auxiliary_only":
        raise ValueError("Human F must remain selection-prohibited")

    locked = torch.load(checkpoint, map_location="cpu", weights_only=False)
    n_features = int(locked["n_features"])
    if locked["width"] != 128 or locked["latent"] != 64 or locked["task_order"] != tasks:
        raise ValueError("Locked full-shared comparator architecture differs from expectation")
    shared_encoder = encoder_parameter_count(n_features, 128, 64)
    grouped_encoders = 2 * encoder_parameter_count(n_features, 64, 64)
    heads = private_head_parameter_count(64, len(tasks))
    budget = pd.DataFrame([
        {"route_id": "shared_encoder_private_heads", "encoders": 1, "encoder_width": 128, "latent": 64,
         "encoder_parameters": shared_encoder, "private_head_parameters": heads,
         "total_trainable_parameters": shared_encoder + heads, "role": "locked_full_shared_comparator"},
        {"route_id": ROUTE_ID, "encoders": 2, "encoder_width": 64, "latent": 64,
         "encoder_parameters": grouped_encoders, "private_head_parameters": heads,
         "total_trainable_parameters": grouped_encoders + heads, "role": "single_new_candidate"},
    ])
    ratio = float(budget.loc[budget.route_id.eq(ROUTE_ID), "total_trainable_parameters"].iloc[0] /
                  budget.loc[budget.route_id.eq("shared_encoder_private_heads"), "total_trainable_parameters"].iloc[0])
    if abs(ratio - 1.0) > 0.01:
        raise ValueError("Two-group route is not within 1% of the full-shared parameter budget")

    route = pd.DataFrame([{
        "route_id": ROUTE_ID, "encoder_sharing": "within_fu_family_and_within_non_fu_family_only",
        "private_heads": "one head per task", "feature_set": "rdkit2d",
        "architecture": "two_MLP_64_64_GELU_layernorm_encoders",
        "encoder_width": 64, "latent": 64,
        "formal_outer_folds": 5, "formal_seeds": "20260917;20260918;20260919",
        "phase1_epochs": 20, "phase2_epochs": 10, "batch_size": 128,
        "learning_rate": 0.0003, "weight_decay": 0.001,
        "total_parameter_ratio_vs_full_shared": ratio,
        "architecture_selection_authorized": False,
    }])
    comparators = pd.DataFrame([
        {"comparator": "shared_encoder_private_heads", "source": str(args.traincv),
         "complete_sha256": sha256(args.traincv / "complete.json"), "retrain": False},
        {"comparator": "fully_private_task_networks", "source": str(args.traincv),
         "complete_sha256": sha256(args.traincv / "complete.json"), "retrain": False},
        {"comparator": "corrected_stageA_STL", "source": str(args.base_protocol / "corrected_stageA_references.csv"),
         "complete_sha256": sha256(args.base_protocol / "complete.json"), "retrain": False},
    ])
    protocol = {
        "schema_version": 1,
        "candidate_route": ROUTE_ID,
        "tasks": tasks,
        "human_evaluation_tasks": human_tasks,
        "groups": {FU_GROUP: group_registry.loc[group_registry.encoder_group.eq(FU_GROUP), "task_id"].tolist(),
                   NON_FU_GROUP: group_registry.loc[group_registry.encoder_group.eq(NON_FU_GROUP), "task_id"].tolist()},
        "grouping_basis": "locked formal gradient-affinity pattern: cross-species fu supportive and human fu broadly conflicting with non-fu tasks",
        "capacity_control": "two width-64 latent-64 encoders versus the locked width-128 latent-64 full-shared encoder; identical private heads",
        "total_parameter_ratio_vs_full_shared": ratio,
        "feature_set": "rdkit2d",
        "outer_fold_isolation": base["outer_fold_isolation"],
        "evaluation_label_lifecycle": "formal runner may read each outer-fold target only after all candidate models for that fold fit and predict; smoke reads no evaluation targets",
        "feature_preprocessing": base["feature_preprocessing"],
        "sampling": base["sampling"],
        "phase1": "all 45 tasks with frozen task-level sampling weights",
        "phase2": "six authorized human heads only; human F excluded",
        "human_F_policy": base["human_F_policy"],
        "formal_scope": "five outer folds x three seeds x one new capacity-matched grouped route",
        "fixed_validation_status": "closed",
        "test_status": "closed",
        "mechanistic_gate": {
            "fu_minimum_improvement_vs_full_shared_pct": 5.0,
            "fu_minimum_noninferior_outer_folds": 4,
            "fu_paired_bootstrap_upper_95ci_below_zero": True,
            "fu_maximum_harm_vs_fully_private_pct": 2.0,
            "maximum_other_endpoint_harm_vs_full_shared_pct": 2.0,
        },
        "model_candidacy_gate": {
            "minimum_improvement_vs_corrected_stageA_pct": 2.0,
            "minimum_noninferior_outer_folds": 3,
            "paired_parent_bootstrap_upper_95ci_below_zero": True,
            "source_sensitivity_required_after_gate": True,
        },
        "deferred_methods": ["PCGrad", "GradNorm", "MMoE", "additional_task_group_search"],
        "architecture_selection_authorized": False,
    }
    json.dumps(protocol, allow_nan=False)
    if args.check_only:
        print(f"Two-group protocol ready: tasks=45 fu=9 non_fu=36 parameter_ratio={ratio:.6f}")
        return
    with stage_output(args.output) as out:
        route.to_csv(out / "route_registry.csv", index=False)
        group_registry.to_csv(out / "task_group_registry.csv", index=False)
        target.to_csv(out / "task_target_registry.csv", index=False)
        sampling.to_csv(out / "task_sampling_registry.csv", index=False)
        budget.to_csv(out / "parameter_budget.csv", index=False)
        comparators.to_csv(out / "locked_comparator_registry.csv", index=False)
        for name in ["corrected_stageA_references.csv", "outer_fold_global_isolation_feasibility.csv"]:
            pd.read_csv(args.base_protocol / name).to_csv(out / name, index=False)
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Gradient-informed two-group multitask protocol\n\n"
            "The only new route separates all fu tasks from all non-fu tasks at the encoder while retaining private heads. "
            "Two width-64 encoders keep the total parameter budget within 1% of the locked width-128 full-shared comparator. "
            "Human F remains representation-only; fixed validation/test stay closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "multitask_two_group_protocol",
            inputs={
                "base_protocol_complete_sha256": sha256(args.base_protocol / "complete.json"),
                "traincv_complete_sha256": sha256(args.traincv / "complete.json"),
                "traincv_analysis_complete_sha256": sha256(args.traincv_analysis / "complete.json"),
                "affinity_complete_sha256": sha256(args.affinity / "complete.json"),
                "affinity_report_complete_sha256": sha256(args.affinity_report / "complete.json"),
            },
            tasks=45, fu_tasks=9, non_fu_tasks=36, human_evaluation_heads=6,
            candidate_routes=1, outer_folds=5, seeds=3,
            total_parameter_ratio_vs_full_shared=ratio,
            model_fitted=False, evaluation_labels_read=False,
            validation_target_file_opened=False, test_labels_read=False,
            architecture_selection_authorized=False, partial=False,
        )
    print(f"Two-group multitask protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
