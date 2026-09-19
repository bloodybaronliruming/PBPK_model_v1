#!/usr/bin/env python3
"""Freeze the bounded Gate-3 B6 nested physchem late-fusion protocol.

This metadata-only stage fits no model.  It binds the train-only physchem
audit, corrected Stage-A references, two finite auxiliary producer candidates,
three fully nested downstream routes, and pre-registered stop/advance rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


HUMAN_TASKS = [
    "CL__human__systemic_iv",
    "CLint__human__microsome",
    "Papp__human__caco2_ab",
    "Thalf__human__terminal_iv",
    "VDss__human__steady_state_iv",
    "fu__human__plasma",
]
AUXILIARY_TASKS = ["experimental_logP", "experimental_logD_pH7_4"]
OUTER_FOLDS = 5
INNER_PRODUCER_FOLDS = 5
META_ALPHA_GRID = "0.1;1.0;10.0"


def build_producer_registry() -> pd.DataFrame:
    rows = []
    for task in AUXILIARY_TASKS:
        for feature_set in ["leakage_safe_rdkit2d", "ecfp4_plus_leakage_safe_rdkit2d"]:
            rows.append(
                {
                    "auxiliary_task": task,
                    "producer_id": f"{task}__extra_trees__{feature_set}",
                    "algorithm": "ExtraTreesRegressor",
                    "feature_set": feature_set,
                    "n_estimators": 300,
                    "max_features": 0.5,
                    "min_samples_leaf": 1,
                    "random_state": 20260917,
                    "target_unit": "log10_coefficient",
                    "parent_aggregation": "arithmetic_mean_target_with_parent_equal_weight",
                    "target_standardization": "fit_on_current_producer_train_parents_only",
                    "selection_scope": "auxiliary_train_internal_scaffold_cv_only",
                    "selection_metric": "parent_level_RMSE",
                    "tie_break": "if_RMSE_within_1pct_choose_leakage_safe_rdkit2d",
                    "source_test_allowed_for_selection": False,
                    "downstream_PK_performance_allowed_for_selection": False,
                    "reliability_policy": "all_labels_R3_parent_equal_weight_tier_is_not_a_predictor",
                }
            )
    return pd.DataFrame(rows)


def build_ancestry_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_role": "auxiliary_outer_train_crossfit",
                "recipient": "PK outer-training parent",
                "producer_training_scope": "audit-clean auxiliary source-train only",
                "required_exclusion": "recipient parent and scaffold plus every auxiliary scaffold assigned to recipient producer fold",
                "fold_construction": "five balanced scaffold groups on the union of eligible auxiliary and downstream outer-training structures; no target used for grouping",
                "required_saved_ancestry": "downstream_task|outer_fold|producer_fold|recipient_parent|recipient_scaffold|producer_train_parent_hash|producer_train_scaffold_hash|model_hash",
                "measured_physchem_allowed": False,
            },
            {
                "prediction_role": "auxiliary_outer_evaluation_inference",
                "recipient": "PK outer-evaluation parent",
                "producer_training_scope": "all audit-clean auxiliary source-train remaining after downstream outer-fold parent/scaffold purge",
                "required_exclusion": "every parent and scaffold in the downstream outer-evaluation fold",
                "fold_construction": "no producer-fold selection on outer-evaluation structures",
                "required_saved_ancestry": "downstream_task|outer_fold|recipient_parent|recipient_scaffold|producer_train_parent_hash|producer_train_scaffold_hash|model_hash",
                "measured_physchem_allowed": False,
            },
            {
                "prediction_role": "stageA_outer_train_crossfit",
                "recipient": "PK outer-training parent used to fit the late-fusion meta-model",
                "producer_training_scope": "PK outer-training folds excluding the recipient inner scaffold fold",
                "required_exclusion": "recipient parent and scaffold",
                "fold_construction": "reuse frozen PK scaffold fold groups among the four outer-training folds; no hyperparameter reselection",
                "required_saved_ancestry": "downstream_task|outer_fold|inner_fold|stageA_train_parent_hash|stageA_train_scaffold_hash|stageA_model_hash",
                "measured_physchem_allowed": False,
            },
            {
                "prediction_role": "stageA_outer_evaluation_inference",
                "recipient": "PK outer-evaluation parent",
                "producer_training_scope": "all PK outer-training parents with the corrected Stage-A configuration",
                "required_exclusion": "every parent and scaffold in the downstream outer-evaluation fold",
                "fold_construction": "frozen downstream outer fold",
                "required_saved_ancestry": "downstream_task|outer_fold|stageA_train_parent_hash|stageA_train_scaffold_hash|stageA_model_hash",
                "measured_physchem_allowed": False,
            },
            {
                "prediction_role": "meta_model_fit_and_evaluation",
                "recipient": "PK outer-training residuals then untouched PK outer-evaluation parents",
                "producer_training_scope": "outer-training crossfit Stage-A and auxiliary predictions only",
                "required_exclusion": "outer-evaluation labels cannot be opened before every route has fit and predicted",
                "fold_construction": "ridge alpha selected only inside outer-training scaffold groups from the frozen three-value grid",
                "required_saved_ancestry": "downstream_task|outer_fold|route_id|meta_train_row_hash|selected_alpha|meta_model_hash",
                "measured_physchem_allowed": False,
            },
        ]
    )


def build_routes() -> pd.DataFrame:
    common = {
        "stageA_direct_path": True,
        "meta_model": "ridge_residual_correction",
        "meta_alpha_grid": META_ALPHA_GRID,
        "meta_selection": "outer_train_scaffold_groups_only",
        "formal_outer_folds": OUTER_FOLDS,
        "measured_physchem_PK_input": False,
        "fixed_validation_authorized": False,
        "test_authorized": False,
    }
    rows = [
        {
            **common,
            "route_id": "C0_corrected_stageA_reference",
            "role": "locked_primary_structure_reference",
            "late_branch_inputs": "none",
            "availability_masks": "none",
            "meta_model": "none",
            "meta_alpha_grid": "none",
            "meta_selection": "none",
            "residual_correction_fitted": False,
            "scientific_question": "frozen strong STL reference",
        },
        {
            **common,
            "route_id": "C1_capacity_matched_structure_only_late_control",
            "role": "capacity_matched_negative_control",
            "late_branch_inputs": "fold-local standardized RDKit MolLogP|TPSA computed from structure only",
            "availability_masks": "two explicit structure-feature availability masks",
            "residual_correction_fitted": True,
            "scientific_question": "does extra late-fusion capacity alone improve Stage-A",
        },
        {
            **common,
            "route_id": "B6_nested_physchem_oof_late_fusion",
            "role": "single_Gate3_candidate",
            "late_branch_inputs": "nested predicted experimental logP|nested predicted experimental logD pH7.4",
            "availability_masks": "two explicit producer availability masks; nonfinite predictions hard-fail rather than silently impute",
            "residual_correction_fitted": True,
            "scientific_question": "does independent experimental physchem supervision add information beyond structure and equal fusion capacity",
        },
    ]
    return pd.DataFrame(rows)


def endpoint_fusion_registry(stagea: pd.DataFrame) -> pd.DataFrame:
    result = stagea[
        [
            "task_id",
            "endpoint",
            "stageA_algorithm",
            "stageA_feature_set",
            "stageA_candidate_id",
            "stageA_parameters",
            "primary_metric_name",
            "parents",
        ]
    ].copy()
    result["fusion_target_space"] = result.endpoint.map(
        lambda endpoint: "physical_fraction_parent_mean" if endpoint == "fu" else "frozen_transformed_target_space"
    )
    result["stageA_outer_train_prediction"] = "strict inner-scaffold crossfit using the frozen Stage-A configuration"
    result["stageA_outer_evaluation_prediction"] = "fit frozen Stage-A configuration on all outer-training parents"
    result["meta_target"] = "observed_parent_target_minus_crossfit_stageA_prediction"
    result["output_constraint"] = result.endpoint.map(
        lambda endpoint: "clip_to_open_unit_interval_after_residual_addition" if endpoint == "fu" else "unbounded"
    )
    result["human_F_included"] = False
    return result


def capacity_control_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comparison_axis": "direct_structure_path",
                "C1_structure_only_control": "crossfit corrected Stage-A prediction",
                "B6_physchem_candidate": "same crossfit corrected Stage-A prediction",
                "matched": True,
            },
            {
                "comparison_axis": "late_branch_dimension",
                "C1_structure_only_control": "two label-free computed structure scalars plus two masks",
                "B6_physchem_candidate": "two nested auxiliary predictions plus two masks",
                "matched": True,
            },
            {
                "comparison_axis": "meta_learner",
                "C1_structure_only_control": f"ridge residual; alpha in {META_ALPHA_GRID}",
                "B6_physchem_candidate": f"ridge residual; alpha in {META_ALPHA_GRID}",
                "matched": True,
            },
            {
                "comparison_axis": "training_and_evaluation_membership",
                "C1_structure_only_control": "same outer/inner parent and scaffold folds",
                "B6_physchem_candidate": "same outer/inner parent and scaffold folds",
                "matched": True,
            },
            {
                "comparison_axis": "randomness_and_metric",
                "C1_structure_only_control": "deterministic ridge and endpoint primary metric",
                "B6_physchem_candidate": "deterministic ridge and endpoint primary metric",
                "matched": True,
            },
        ]
    )


def advance_gates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A1", "B6_vs_corrected_StageA", "mean primary error reduction >= 2%", True),
            ("A2", "B6_vs_corrected_StageA", "at least 3 of 5 outer folds nonworse", True),
            ("A3", "B6_vs_corrected_StageA", "paired-parent bootstrap error-difference 95% CI upper bound < 0", True),
            ("A4", "B6_vs_capacity_control", "B6 primary error is lower and paired-parent bootstrap 95% CI upper bound < 0", True),
            ("A5", "ancestry_and_reload", "all saved parent/scaffold overlaps are zero; predictions finite; reload tolerance <= 1e-12", True),
            ("A6", "sensitivity", "low-similarity and repeated-parent subsets show no >2% relative harm when sufficiently powered", True),
            ("A7", "source_sensitivity", "required only after A1-A6; unavailable per-record provenance must remain an explicit limitation", False),
        ],
        columns=["gate_id", "comparison", "criterion", "required_for_primary_advance"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physchem-audit",
        type=Path,
        default=ROOT / "data/public_development/physchem_auxiliary_feasibility_audit_v3",
    )
    parser.add_argument(
        "--gate2b",
        type=Path,
        default=ROOT / "data/public_development/gate2b_decision_freeze_gate3_audit_v1",
    )
    parser.add_argument(
        "--p1-freeze",
        type=Path,
        default=ROOT / "data/public_development/stl_p1_decision_freeze_v1",
    )
    parser.add_argument(
        "--stl",
        type=Path,
        default=ROOT / "data/public_development/stl_train_only_protocol_v1",
    )
    parser.add_argument(
        "--multimodal",
        type=Path,
        default=ROOT / "data/public_development/gate1b_multimodal_protocol_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/public_development/gate3_b6_physchem_protocol_v2",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    required = [
        args.physchem_audit / "complete.json",
        args.physchem_audit / "train_record_audit.csv",
        args.physchem_audit / "task_definition_registry.csv",
        args.physchem_audit / "target_distribution_audit.csv",
        args.physchem_audit / "outer_fold_purge_feasibility.csv",
        args.physchem_audit / "descriptor_leakage_registry.csv",
        args.physchem_audit / "gate_decision.csv",
        args.physchem_audit / "provenance_limitations.csv",
        args.gate2b / "complete.json",
        args.gate2b / "gate3_candidate_registry.csv",
        args.p1_freeze / "complete.json",
        args.p1_freeze / "p1_decision_registry.csv",
        args.stl / "complete.json",
        args.stl / "benchmark_train_records.csv",
        args.multimodal / "complete.json",
        args.multimodal / "oof_ancestry_contract.csv",
        args.multimodal / "leakage_prohibition_registry.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    audit_meta = verify_stage(args.physchem_audit, "physchem_auxiliary_feasibility_audit")
    gate_meta = verify_stage(args.gate2b, "gate2b_decision_freeze_gate3_audit")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    multimodal_meta = verify_stage(args.multimodal, "gate1b_multimodal_input_protocol")
    metas = [audit_meta, gate_meta, p1_meta, stl_meta, multimodal_meta]
    if any(meta.get("test_labels_read", False) for meta in metas):
        raise ValueError("A protocol input reports protected test-label access")
    if any(meta.get("validation_labels_read", False) or meta.get("validation_target_file_opened", False) for meta in metas):
        raise ValueError("A protocol input reports fixed-validation target access")
    if audit_meta.get("decision") != "CONDITIONAL_PASS":
        raise ValueError("Physchem feasibility audit did not conditionally pass")
    if audit_meta.get("source_test_target_values_used") != 0 or audit_meta.get("source_test_target_values_published"):
        raise ValueError("Physchem feasibility audit used or published source-test values")
    if audit_meta.get("cohort_construction_authorized") or audit_meta.get("model_training_authorized"):
        raise ValueError("Upstream audit unexpectedly pre-authorized cohort construction or training")

    gates = pd.read_csv(args.physchem_audit / "gate_decision.csv")
    hard = gates.loc[~gates.gate_id.eq("G7_per_record_provenance")]
    provenance = gates.loc[gates.gate_id.eq("G7_per_record_provenance")]
    if len(hard) != 6 or not hard.passed.astype(bool).all() or len(provenance) != 1 or bool(provenance.iloc[0].passed):
        raise ValueError("Physchem conditional-pass gate pattern differs from the frozen expectation")

    definitions = pd.read_csv(args.physchem_audit / "task_definition_registry.csv")
    distribution = pd.read_csv(args.physchem_audit / "target_distribution_audit.csv")
    if set(definitions.task_id) != set(AUXILIARY_TASKS) or set(distribution.task_id) != set(AUXILIARY_TASKS):
        raise ValueError("Physchem audit must contain exactly logP and logD")
    if definitions.per_record_doi_or_assay_available.astype(bool).any():
        raise ValueError("Protocol expects the material R3 per-record provenance limitation")
    if distribution.structure_failure_records.sum() or distribution.high_conflict_duplicate_parents.sum():
        raise ValueError("Unresolved structure or high-conflict duplicate rows require a new feasibility version")

    fold_budget = pd.read_csv(args.physchem_audit / "outer_fold_purge_feasibility.csv")
    endpoint_budget = fold_budget.loc[fold_budget.downstream_scope.isin(HUMAN_TASKS)].copy()
    if len(endpoint_budget) != len(HUMAN_TASKS) * OUTER_FOLDS * len(AUXILIARY_TASKS):
        raise ValueError("Endpoint-specific physchem fold budget is incomplete")
    if set(endpoint_budget.downstream_scope) != set(HUMAN_TASKS) or set(endpoint_budget.outer_fold) != set(range(5)):
        raise ValueError("Endpoint/fold membership differs from the frozen six-task/five-fold scope")
    if not endpoint_budget.scale_gate_passed.astype(bool).all() or not endpoint_budget.isolation_gate_passed.astype(bool).all():
        raise ValueError("A physchem endpoint/fold resource cell failed scale or isolation")
    if endpoint_budget[["post_filter_parent_overlap", "post_filter_scaffold_overlap"]].to_numpy().max() != 0:
        raise ValueError("A physchem endpoint/fold resource cell retains overlap")

    candidate = pd.read_csv(args.gate2b / "gate3_candidate_registry.csv")
    if len(candidate) != 1 or candidate.iloc[0].candidate_id != "G3_B6_nested_physchem_oof_late_fusion":
        raise ValueError("Gate 2B freeze does not nominate the expected B6 candidate")
    if not bool(candidate.iloc[0].protocol_construction_authorized):
        raise ValueError("B6 protocol construction is not authorized")
    if bool(candidate.iloc[0].cohort_construction_authorized) or bool(candidate.iloc[0].model_training_authorized):
        raise ValueError("Gate 2B must not pre-authorize cohort construction or training")

    stagea = pd.read_csv(args.p1_freeze / "p1_decision_registry.csv")
    stagea = stagea.loc[stagea.task_id.isin(HUMAN_TASKS)].copy()
    if set(stagea.task_id) != set(HUMAN_TASKS) or len(stagea) != len(HUMAN_TASKS):
        raise ValueError("Corrected Stage-A references do not cover exactly six authorized tasks")
    if not stagea.reference_for_next_stage.eq("corrected_stageA_matched_control").all():
        raise ValueError("A Stage-A reference is not frozen as the next-stage comparator")
    if set(stagea.stageA_algorithm) != {"extra_trees"}:
        raise ValueError("This protocol expects the frozen ExtraTrees Stage-A leaders")

    train = pd.read_csv(
        args.physchem_audit / "train_record_audit.csv",
        usecols=["task_id", "source_split", "future_clean_pool", "audit_only_not_cohort_authorized"],
    )
    if set(train.source_split) != {"train"} or not train.audit_only_not_cohort_authorized.astype(bool).all():
        raise ValueError("Physchem audit record lifecycle differs from expectation")
    clean = train.loc[train.future_clean_pool.astype(bool)].groupby("task_id").size().to_dict()
    if clean != {"experimental_logD_pH7_4": 3649, "experimental_logP": 4559}:
        raise ValueError(f"Unexpected audit-clean population: {clean}")

    producer = build_producer_registry()
    ancestry = build_ancestry_contract()
    routes = build_routes()
    endpoint_registry = endpoint_fusion_registry(stagea)
    capacity = capacity_control_contract()
    advance = advance_gates()
    leakage = pd.read_csv(args.physchem_audit / "descriptor_leakage_registry.csv")
    leakage["applies_to"] = "auxiliary_producer_only"
    global_leakage = pd.DataFrame(
        [
            ("all", "source_id|doc_id|assay_id", "prohibited_predictor", "prevents source shortcut"),
            ("all", "reliability_tier", "prohibited_predictor", "stratification/reporting only"),
            ("all", "split|outer_fold|producer_fold|test_membership", "prohibited_predictor", "administrative metadata"),
            ("PK", "measured_logP|measured_logD", "prohibited_PK_input", "strict nested predictions only"),
            ("pKa", "predicted_database_pKa", "prohibited_experimental_label", "pKa remains outside Gate 3 B6"),
        ],
        columns=["target_family", "excluded_feature", "rule", "reason"],
    )
    global_leakage["feature_view"] = "all"
    global_leakage["audit_status"] = "protocol_frozen"
    global_leakage["applies_to"] = "full_pipeline"
    leakage = pd.concat(
        [
            leakage[
                ["target_family", "excluded_feature", "rule", "reason", "feature_view", "audit_status", "applies_to"]
            ],
            global_leakage[
                ["target_family", "excluded_feature", "rule", "reason", "feature_view", "audit_status", "applies_to"]
            ],
        ],
        ignore_index=True,
    )

    evidence = pd.DataFrame(
        [
            {
                "input_stage": "physchem_auxiliary_feasibility_audit_v3",
                "stage": audit_meta["stage"],
                "complete_sha256": sha256(args.physchem_audit / "complete.json"),
                "role": "conditional_pass_train_side_logP_logD_evidence",
            },
            {
                "input_stage": "gate2b_decision_freeze_gate3_audit_v1",
                "stage": gate_meta["stage"],
                "complete_sha256": sha256(args.gate2b / "complete.json"),
                "role": "single_B6_protocol_authorization",
            },
            {
                "input_stage": "stl_p1_decision_freeze_v1",
                "stage": p1_meta["stage"],
                "complete_sha256": sha256(args.p1_freeze / "complete.json"),
                "role": "six_corrected_StageA_references",
            },
            {
                "input_stage": "stl_train_only_protocol_v1",
                "stage": stl_meta["stage"],
                "complete_sha256": sha256(args.stl / "complete.json"),
                "role": "six_task_five_fold_train_only_membership",
            },
            {
                "input_stage": "gate1b_multimodal_protocol_v2",
                "stage": multimodal_meta["stage"],
                "complete_sha256": sha256(args.multimodal / "complete.json"),
                "role": "upstream_OOF_and_multimodal_leakage_contract",
            },
        ]
    )

    delegation = pd.DataFrame(
        [
            ("protocol_freeze", "metadata only", "<5 min", "CPU", "assistant", False),
            ("train_only_cohort_build", "structure/feature materialization", "5-15 min", "CPU", "assistant", False),
            ("one_fold_smoke", "two producers plus three downstream routes on bounded parents", "10-30 min", "CPU", "assistant_then_delegate_if_over_15min", False),
            ("formal_nested_crossfit", "6 tasks x 5 outer folds x producer/meta fits", "hours", "CPU_or_GPU_after_benchmark", "user_background_task", True),
        ],
        columns=["stage", "scope", "estimated_wall_time", "preferred_device", "executor", "long_task"],
    )

    lifecycle = {
        "physchem_input": "only train_record_audit.csv rows with source_split=train and future_clean_pool=true",
        "source_test": "membership count retained as metadata; target values never used for producer selection, fitting, or reporting",
        "fixed_validation": "closed; no target-bearing validation file may be opened",
        "test": "closed",
        "human_F": "excluded from Gate 3 B6 selection and claims",
        "pKa": "deferred",
        "measured_physchem_on_PK_rows": "prohibited",
        "R3_policy": "public aggregate evidence; no individually source-verified claim and no source-cluster generalization claim",
        "current_stage_authority": "protocol metadata only",
        "next_stage_authority": "versioned train-only cohort construction and bounded engineering smoke",
        "formal_training_authorized": False,
        "fixed_validation_authorized": False,
        "test_authorized": False,
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": "G3_B6_nested_physchem_oof_late_fusion",
        "human_tasks": HUMAN_TASKS,
        "auxiliary_tasks": AUXILIARY_TASKS,
        "outer_folds": OUTER_FOLDS,
        "inner_producer_folds": INNER_PRODUCER_FOLDS,
        "auxiliary_candidate_rule": "two finite ExtraTrees feature views; choose per auxiliary task only by internal scaffold-CV RMSE",
        "duplicate_policy": "retain record audit; aggregate target by arithmetic parent mean with one parent one training weight",
        "late_fusion": "corrected Stage-A direct prediction plus deterministic ridge residual correction",
        "capacity_control": "same meta learner, folds and late-branch dimension; label-free computed MolLogP and TPSA replace predicted experimental physchem",
        "candidate_inputs": "nested predicted experimental logP and logD plus explicit availability masks",
        "nonfinite_policy": "hard failure; no silent imputation",
        "selection_scope": "one preregistered Gate-3 family and no post-result architecture expansion",
        "endpoint_specific_decisions": True,
        "advance_rule": "A1-A6 must all pass per endpoint; A7 is a mandatory reported limitation/sensitivity trigger rather than a primary gate",
        "source_sensitivity": "cannot be claimed from current per-record provenance; register limitation and rerun only if mapping is later obtained",
        "cohort_construction_authorized_next": True,
        "bounded_smoke_authorized_after_cohort": True,
        "formal_model_training_authorized": False,
        "fixed_validation_status": "closed",
        "test_status": "closed",
    }
    json.dumps({"protocol": protocol, "lifecycle": lifecycle}, allow_nan=False)

    if args.check_only:
        print(
            "Gate 3 B6 protocol ready: "
            f"tasks={len(HUMAN_TASKS)} auxiliary={len(AUXILIARY_TASKS)} "
            f"producer_candidates={len(producer)} routes={len(routes)} fold_cells={len(endpoint_budget)}"
        )
        return

    inputs = {str(path.resolve()): sha256(path) for path in required}
    with stage_output(args.output) as out:
        evidence.to_csv(out / "input_evidence_registry.csv", index=False)
        definitions.to_csv(out / "auxiliary_task_registry.csv", index=False)
        producer.to_csv(out / "auxiliary_producer_registry.csv", index=False)
        ancestry.to_csv(out / "nested_prediction_ancestry_contract.csv", index=False)
        routes.to_csv(out / "downstream_route_registry.csv", index=False)
        endpoint_registry.to_csv(out / "endpoint_fusion_registry.csv", index=False)
        capacity.to_csv(out / "capacity_matched_control_contract.csv", index=False)
        endpoint_budget.sort_values(["downstream_scope", "outer_fold", "auxiliary_task"]).to_csv(
            out / "outer_fold_resource_budget.csv", index=False
        )
        leakage.to_csv(out / "feature_leakage_contract.csv", index=False)
        advance.to_csv(out / "endpoint_advance_gates.csv", index=False)
        pd.read_csv(args.physchem_audit / "provenance_limitations.csv").to_csv(
            out / "provenance_limitations.csv", index=False
        )
        delegation.to_csv(out / "compute_delegation_plan.csv", index=False)
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "lifecycle_contract.json").write_text(
            json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / "README.md").write_text(
            "# Gate 3 B6 nested physchem protocol v2\n\n"
            "Metadata-only preregistration for six endpoint-specific Stage-A-anchored late-fusion tests. "
            "Only nested predictions of experimental logP/logD may enter PK rows. C0 is corrected Stage-A; "
            "C1 is an equal-capacity structure-only late control; B6 is the single new candidate. The current "
            "stage authorizes a versioned train-only cohort and bounded smoke next, not formal training or fixed validation/test.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "gate3_b6_physchem_protocol",
            inputs=inputs,
            partial=False,
            human_tasks=len(HUMAN_TASKS),
            auxiliary_tasks=len(AUXILIARY_TASKS),
            producer_candidates=len(producer),
            downstream_routes=len(routes),
            outer_folds=OUTER_FOLDS,
            inner_producer_folds=INNER_PRODUCER_FOLDS,
            outer_fold_resource_cells=len(endpoint_budget),
            minimum_parents_after_purge=int(endpoint_budget.parents_after_fold_purge.min()),
            minimum_scaffolds_after_purge=int(endpoint_budget.scaffolds_after_fold_purge.min()),
            model_fitted=False,
            evaluation_labels_read=False,
            validation_target_file_opened=False,
            test_labels_read=False,
            train_only_cohort_construction_authorized=True,
            bounded_smoke_authorized_after_cohort=True,
            formal_model_training_authorized=False,
            fixed_validation_authorized=False,
            test_authorized=False,
        )
    print(f"Gate 3 B6 physchem protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
