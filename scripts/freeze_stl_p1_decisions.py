#!/usr/bin/env python3
"""Freeze corrected P1 decisions and the strong STL reference registry.

This is a read-only decision stage. It verifies the complete Stage-A audit,
corrected P1 nested confirmation, matched comparison, and finite candidate
registry. It fits no model and opens no validation or test target file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stagea",
        type=Path,
        default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2",
    )
    parser.add_argument(
        "--p1",
        type=Path,
        default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=ROOT / "results/analysis/stl_p1_matched_comparison_v1",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "data/public_development/stl_p1_candidate_registry_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/public_development/stl_p1_decision_freeze_v1",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    required = [
        args.stagea / "complete.json",
        args.stagea / "endpoint_decisions.csv",
        args.stagea / "frozen_stageA_shortlist.csv",
        args.p1 / "complete.json",
        args.p1 / "fold_isolation_audit.csv",
        args.p1 / "reload_checks.csv",
        args.comparison / "complete.json",
        args.comparison / "overall_matched_comparison.csv",
        args.comparison / "outer_fold_matched_comparison.csv",
        args.comparison / "selection_stability.csv",
        args.candidates / "complete.json",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)

    stagea_meta = verify_stage(args.stagea, "stl_stageA_three_view_audit")
    p1_meta = verify_stage(args.p1, "stl_p1_nested_confirmation")
    comparison_meta = verify_stage(args.comparison, "stl_p1_matched_comparison")
    candidate_meta = verify_stage(args.candidates, "stl_p1_candidate_registry")
    for label, meta in {
        "Stage-A": stagea_meta,
        "P1": p1_meta,
        "comparison": comparison_meta,
        "candidate registry": candidate_meta,
    }.items():
        if meta.get("test_labels_read", False):
            raise ValueError(f"{label} is not test-closed")
    if stagea_meta.get("validation_labels_read", False):
        raise ValueError("Stage-A audit opened validation labels")
    if p1_meta.get("validation_target_file_opened", False) or comparison_meta.get(
        "validation_target_file_opened", False
    ):
        raise ValueError("P1 evidence opened a validation target file")
    if p1_meta.get("partial", True) or comparison_meta.get("partial", True):
        raise ValueError("Only complete P1 evidence can be frozen")
    if p1_meta.get("inner_fits") != 1200 or p1_meta.get("final_fits") != 180:
        raise ValueError("Unexpected corrected P1 fit count")

    isolation = pd.read_csv(args.p1 / "fold_isolation_audit.csv")
    if len(isolation) != 30:
        raise ValueError("Expected six tasks x five outer-fold isolation rows")
    for column in ["outer_parent_overlap", "outer_scaffold_overlap", "inner_scaffold_overlap_max"]:
        if not isolation[column].eq(0).all():
            raise ValueError(f"Non-zero P1 isolation audit: {column}")
    if isolation["evaluation_targets_used_in_inner_selection"].astype(str).str.lower().ne("false").any():
        raise ValueError("Outer evaluation targets entered inner selection")

    reloads = pd.read_csv(args.p1 / "reload_checks.csv")
    if len(reloads) != 12 or reloads["max_abs_difference"].max() > 1e-12:
        raise ValueError("P1 reload audit failed")

    decisions = pd.read_csv(args.stagea / "endpoint_decisions.csv")
    leaders = pd.read_csv(args.stagea / "frozen_stageA_shortlist.csv")
    leaders = leaders.loc[leaders["shortlist_position"].eq(1)].copy()
    comparison = pd.read_csv(args.comparison / "overall_matched_comparison.csv")
    folds = pd.read_csv(args.comparison / "outer_fold_matched_comparison.csv")
    stability = pd.read_csv(args.comparison / "selection_stability.csv")

    if any(frame["task_id"].nunique() != 6 for frame in [decisions, leaders, comparison, folds, stability]):
        raise ValueError("Every P1 decision input must cover exactly six tasks")
    if len(decisions) != 6 or len(leaders) != 6 or len(comparison) != 6 or len(folds) != 30:
        raise ValueError("Unexpected P1 decision input cardinality")
    if set(decisions.task_id) != set(leaders.task_id) or set(decisions.task_id) != set(comparison.task_id):
        raise ValueError("Task sets differ across P1 decision inputs")

    leader_cols = [
        "task_id",
        "endpoint",
        "algorithm",
        "feature_set",
        "candidate_id",
        "parameters",
        "primary_metric_name",
        "primary_metric",
        "records",
        "parents",
    ]
    registry = leaders[leader_cols].rename(
        columns={
            "primary_metric": "stageA_original_train_cv_primary_metric",
            "candidate_id": "stageA_candidate_id",
            "algorithm": "stageA_algorithm",
            "feature_set": "stageA_feature_set",
            "parameters": "stageA_parameters",
        }
    )
    registry = registry.merge(comparison, on=["task_id", "endpoint", "primary_metric_name"], validate="one_to_one")

    fold_summary = (
        folds.assign(p1_noninferior=lambda x: x.primary_metric_p1 <= x.primary_metric_control)
        .groupby("task_id", as_index=False)
        .agg(
            outer_folds=("outer_fold", "nunique"),
            folds_p1_noninferior=("p1_noninferior", "sum"),
        )
    )
    stability_summary = (
        stability.groupby("task_id", as_index=False)
        .agg(
            selected_candidate_configurations=("candidate_uid", "nunique"),
            maximum_folds_for_one_configuration=("outer_folds_selected", "max"),
        )
    )
    registry = registry.merge(fold_summary, on="task_id", validate="one_to_one")
    registry = registry.merge(stability_summary, on="task_id", validate="one_to_one")
    registry["improvement_pct"] = -registry["relative_change_pct"]
    registry["meets_2pct_improvement"] = registry["improvement_pct"] >= 2.0
    registry["majority_outer_folds_noninferior"] = registry["folds_p1_noninferior"] >= 3
    registry["bootstrap_supports_improvement"] = registry["paired_bootstrap_difference_95ci_high"] < 0
    registry["single_configuration_stable_all_folds"] = registry["maximum_folds_for_one_configuration"] == 5
    registry["p1_advance_decision"] = "reject_no_preregistered_advance"
    registry["reference_for_next_stage"] = "corrected_stageA_matched_control"
    registry["next_stage_role"] = "strong_STL_reference_for_transfer_and_shared_private_comparison"
    registry["fixed_validation_status"] = "closed"
    registry["test_status"] = "closed"
    registry["p1_v1_status"] = "historical_only_invalid_fu_parent_aggregation_for_performance_conclusions"
    registry["p1_v2_status"] = "formal_corrected_train_cv_evidence"

    advance_gate = (
        registry["meets_2pct_improvement"]
        & registry["majority_outer_folds_noninferior"]
        & registry["bootstrap_supports_improvement"]
        & registry["single_configuration_stable_all_folds"]
    )
    if advance_gate.any() or registry.p1_advance_decision.ne("reject_no_preregistered_advance").any():
        raise ValueError("P1 decision does not match the preregistered advance gate")
    if registry.task_id.duplicated().any() or len(registry) != 6:
        raise ValueError("Frozen P1 decision registry must contain one row per task")

    lifecycle = {
        "schema_version": 1,
        "formal_p1_result": "stl_p1_nested_confirmation_v2",
        "historical_non_decisional_result": "stl_p1_nested_confirmation_v1",
        "strong_stl_reference": "corrected_stageA_matched_control",
        "p1_candidates_advanced": 0,
        "next_comparison_routes": [
            "corrected_stageA_STL",
            "controlled_cross_species_transfer",
            "shared_encoder_private_task_heads",
        ],
        "fixed_validation_status": "closed_until_finite_architecture_shortlist_is_frozen",
        "test_status": "closed_until_final_model_and_analysis_plan_are_locked",
        "forbidden_actions": [
            "use_p1_v1_for_performance_conclusions",
            "expand_tree_hyperparameter_search_on_the_same_outer_cv",
            "open_fixed_validation_or_test_during_protocol_or_smoke",
            "mix_mean_logit_fu_with_mean_physical_fu_aggregation",
        ],
    }

    if args.check_only:
        print("P1 decision freeze ready: tasks=6 advanced=0 reference=corrected_stageA_matched_control")
        return

    with stage_output(args.output) as out:
        registry.to_csv(out / "p1_decision_registry.csv", index=False)
        leaders[leader_cols].to_csv(out / "stageA_leader_source_registry.csv", index=False)
        (out / "lifecycle_contract.json").write_text(
            json.dumps(lifecycle, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "README.md").write_text(
            "# Corrected P1 decision freeze\n\n"
            "All six finite P1 searches are rejected for advancement. Corrected Stage-A matched controls are "
            "the strong STL references for controlled cross-species transfer and shared-encoder/private-head "
            "comparisons. P1 v1 is historical only because its fu parent aggregation did not follow the frozen "
            "physical-space rule. This stage fits no model and opens no validation or test targets.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "stl_p1_decision_freeze",
            inputs={
                "stageA_complete_sha256": sha256(args.stagea / "complete.json"),
                "p1_v2_complete_sha256": sha256(args.p1 / "complete.json"),
                "matched_comparison_complete_sha256": sha256(args.comparison / "complete.json"),
                "candidate_registry_complete_sha256": sha256(args.candidates / "complete.json"),
            },
            tasks=6,
            p1_candidates_advanced=0,
            strong_stl_reference="corrected_stageA_matched_control",
            validation_target_file_opened=False,
            validation_rows_evaluated=False,
            test_labels_read=False,
            model_fitted=False,
            data_modified=False,
            final_model_selection_authorized=False,
            next_stage="cross_species_and_multitask_input_freeze_audit",
            partial=False,
        )
    print(f"P1 decision freeze: {args.output}")


if __name__ == "__main__":
    run_cli(main)
