#!/usr/bin/env python3
"""Freeze the finite VDss controlled cross-species transfer protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


HUMAN_TASK = "VDss__human__steady_state_iv"
ANIMAL_TASKS = [
    "VDss__dog__steady_state_iv",
    "VDss__monkey__steady_state_iv",
    "VDss__mouse__steady_state_iv",
    "VDss__rat__steady_state_iv",
]
ROUTES = [
    {
        "route_id": "human_only_neural_control",
        "training_phases": "human_outer_train_only",
        "species_conditioning": True,
        "private_heads": "human_only",
        "purpose": "matched neural control for transfer attribution",
    },
    {
        "route_id": "cross_species_pretrain_human_finetune",
        "training_phases": "four_animal_tasks_pretrain_then_human_outer_train_finetune",
        "species_conditioning": True,
        "private_heads": "animal_pretrain_heads_plus_human_private_head",
        "purpose": "test whether animal supervision improves the human representation",
    },
    {
        "route_id": "species_conditioned_joint_shared_encoder",
        "training_phases": "task_balanced_joint_animal_and_human_outer_train",
        "species_conditioning": True,
        "private_heads": "one_private_head_per_species_including_human",
        "purpose": "test simultaneous shared representation with explicit species identity",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-audit",
        type=Path,
        default=ROOT / "data/public_development/cross_species_multitask_input_audit_v1",
    )
    parser.add_argument(
        "--p1-freeze",
        type=Path,
        default=ROOT / "data/public_development/stl_p1_decision_freeze_v1",
    )
    parser.add_argument(
        "--interface",
        type=Path,
        default=ROOT / "data/public_development/multitask_pk_training_interface_v5",
    )
    parser.add_argument(
        "--stl",
        type=Path,
        default=ROOT / "data/public_development/stl_train_only_protocol_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/public_development/vdss_cross_species_transfer_protocol_v4",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    required = [
        args.input_audit / "complete.json",
        args.input_audit / "human_transfer_readiness.csv",
        args.input_audit / "auxiliary_task_eligibility.csv",
        args.input_audit / "outer_fold_transfer_summary.csv",
        args.p1_freeze / "complete.json",
        args.p1_freeze / "p1_decision_registry.csv",
        args.interface / "complete.json",
        args.interface / "task_target_transforms_train_only.csv",
        args.stl / "complete.json",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    audit_meta = verify_stage(args.input_audit, "cross_species_multitask_input_audit")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(meta.get("test_labels_read", False) for meta in [audit_meta, p1_meta, interface_meta, stl_meta]):
        raise ValueError("Protocol inputs must remain test-closed")
    if audit_meta.get("validation_target_file_opened") or audit_meta.get("model_fitted"):
        raise ValueError("Input audit must be label-blind and fit-free")
    if audit_meta.get("first_transfer_pilot") != HUMAN_TASK:
        raise ValueError("The frozen input audit did not select VDss as the first pilot")

    readiness = pd.read_csv(args.input_audit / "human_transfer_readiness.csv")
    vdss = readiness.loc[readiness.human_task_id.eq(HUMAN_TASK)]
    if len(vdss) != 1 or vdss.iloc[0].transfer_readiness != "ready_for_controlled_transfer_pilot":
        raise ValueError("VDss is not ready for controlled transfer")
    if int(vdss.iloc[0].minimum_fold_retained_direct_records) < 500:
        raise ValueError("VDss direct auxiliary support is below the frozen minimum")

    eligibility = pd.read_csv(args.input_audit / "auxiliary_task_eligibility.csv")
    direct = eligibility.loc[
        eligibility.human_task_id.eq(HUMAN_TASK) & eligibility.direct_label_transfer_authorized.eq(True)
    ].copy()
    if set(direct.auxiliary_task_id) != set(ANIMAL_TASKS):
        raise ValueError("VDss direct-transfer task set differs from the frozen four-species set")
    if not direct.auxiliary_unit.eq("L/kg").all() or not direct.auxiliary_system.eq("steady_state_iv").all():
        raise ValueError("VDss animal definitions are not aligned")

    transforms = pd.read_csv(args.interface / "task_target_transforms_train_only.csv")
    selected_tasks = [HUMAN_TASK, *ANIMAL_TASKS]
    task_registry = transforms.loc[transforms.task_id.isin(selected_tasks), [
        "task_id", "endpoint", "canonical_unit", "evidence_role", "reliability_tier",
        "train_records", "validation_records", "target_transform", "reporting_inverse",
        "standardization", "model_output", "head_selection_authorized",
    ]].copy()
    if len(task_registry) != 5 or set(task_registry.task_id) != set(selected_tasks):
        raise ValueError("Training interface does not uniquely cover the five VDss tasks")
    if not task_registry.endpoint.eq("VDss").all() or not task_registry.canonical_unit.eq("L/kg").all():
        raise ValueError("VDss endpoint/unit mismatch")
    if not task_registry.target_transform.eq("log10").all():
        raise ValueError("VDss protocol requires aligned log10 targets")

    p1 = pd.read_csv(args.p1_freeze / "p1_decision_registry.csv")
    baseline = p1.loc[p1.task_id.eq(HUMAN_TASK)]
    if len(baseline) != 1 or baseline.iloc[0].reference_for_next_stage != "corrected_stageA_matched_control":
        raise ValueError("Corrected Stage-A VDss reference is not frozen")

    routes = pd.DataFrame(ROUTES)
    routes["feature_set"] = "rdkit2d"
    routes["encoder"] = "MLP_128_64_GELU_layernorm"
    routes["loss"] = "task_standardized_Huber"
    routes["human_primary_metric"] = "transformed_RMSE"
    routes["formal_outer_folds"] = 5
    routes["formal_seeds"] = "20260917;20260918;20260919"
    routes["pretrain_epochs"] = 40
    routes["human_or_joint_epochs"] = 30
    routes["batch_size"] = 128
    routes["learning_rate"] = 0.0003
    routes["weight_decay"] = 0.001
    routes["task_balancing"] = "equal_optimizer_steps_per_task_per_epoch"
    routes["steps_per_task_reference"] = "ceil(retained_human_outer_train_parents / batch_size)"
    routes["early_stopping"] = False
    routes["seed_aggregation"] = "arithmetic_mean_prediction_in_transformed_space"
    routes["architecture_selection_authorized"] = False

    fold_budget = pd.read_csv(args.input_audit / "outer_fold_transfer_summary.csv")
    fold_budget = fold_budget.loc[fold_budget.human_task_id.eq(HUMAN_TASK)].copy()
    if len(fold_budget) != 5 or fold_budget[["post_filter_parent_overlap_all", "post_filter_scaffold_overlap_all"]].to_numpy().max() != 0:
        raise ValueError("VDss fold resource audit is incomplete or contaminated")

    contract = {
        "schema_version": 4,
        "human_task": HUMAN_TASK,
        "animal_tasks": ANIMAL_TASKS,
        "species_order": ["human", "dog", "monkey", "mouse", "rat"],
        "feature_set": "rdkit2d",
        "parent_aggregation": "mean transformed target within task and canonical parent",
        "fold_filter": "remove all rows whose parent OR scaffold overlaps the human outer-evaluation fold",
        "target_transform": "log10 already encoded upstream; z-score fit independently within each task from retained train parents only",
        "human_evaluation": "frozen five-fold human scaffold CV only",
        "smoke_scope": "outer fold 0, one seed, bounded parent counts and epochs; engineering only",
        "formal_scope": "five outer folds x three fixed seeds x three routes",
        "randomization": "same registered seed and initialization within an outer fold across all three routes",
        "task_balancing": "each active task receives ceil(retained human outer-training parents / batch size) optimizer steps per epoch; tasks are independently reshuffled and cycled",
        "epoch_policy": "fixed registered epochs without early stopping or evaluation-label monitoring",
        "feature_preprocessing": "imputation and scaling fit only on structures used by the current route in the retained outer-training fold",
        "seed_aggregation": "arithmetic mean of the three fixed-seed predictions in transformed target space before primary scoring",
        "evaluation_label_lifecycle": "outer-evaluation targets are read only after every route and seed for that outer fold has finished fitting and prediction",
        "primary_transfer_claim": "compare transfer routes against matched human-only neural control",
        "main_model_replacement_claim": "requires improvement over corrected Stage-A STL in addition to human-only neural control",
        "advance_gate": {
            "minimum_mean_improvement_vs_human_only_pct": 2.0,
            "minimum_noninferior_outer_folds": 3,
            "paired_parent_bootstrap_repeats": 2000,
            "paired_parent_bootstrap_upper_95ci_below_zero": True,
            "low_similarity_maximum_relative_harm_pct": 2.0,
            "low_similarity_definition": "bottom fold-specific quintile of exact maximum ECFP4 Tanimoto to retained human outer-training parents",
            "source_unseen_minimum_parents_for_gate": 30,
            "source_unseen_if_underpowered": "descriptive_only_and_source_cluster_sensitivity_required_before_fixed_validation",
        },
        "stageA_replacement_gate": {
            "minimum_improvement_pct": 2.0,
            "minimum_noninferior_outer_folds": 3,
            "paired_parent_bootstrap_upper_95ci_below_zero": True,
            "low_similarity_maximum_relative_harm_pct": 2.0,
        },
        "fixed_validation_status": "closed",
        "test_status": "closed",
        "source_id_as_model_input": False,
    }

    if args.check_only:
        print("VDss transfer protocol ready: routes=3 tasks=5 folds=5 seeds=3")
        return

    with stage_output(args.output) as out:
        routes.to_csv(out / "route_registry.csv", index=False)
        task_registry.to_csv(out / "task_registry.csv", index=False)
        fold_budget.to_csv(out / "outer_fold_resource_budget.csv", index=False)
        baseline.to_csv(out / "corrected_stageA_reference.csv", index=False)
        (out / "protocol.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "README.md").write_text(
            "# VDss controlled cross-species transfer protocol\n\n"
            "Three finite neural routes use identical RDKit2D inputs, human outer folds, and registered budgets. "
            "Animal tasks match steady-state IV VDss and L/kg across dog, monkey, mouse, and rat. Every route "
            "removes human evaluation parents and scaffolds before fitting. Protocol v4 freezes human-reference equal-step "
            "task balancing, same-seed route initialization, fixed epochs, and transformed-space seed ensembling. "
            "It also fixes bootstrap/fold/low-similarity decision thresholds and treats the very small ordinary-CV "
            "source-unseen subset as descriptive, requiring a later source-cluster sensitivity before fixed validation. "
            "Fixed validation/test remain closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "vdss_cross_species_transfer_protocol",
            inputs={
                "input_audit_complete_sha256": sha256(args.input_audit / "complete.json"),
                "p1_freeze_complete_sha256": sha256(args.p1_freeze / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_train_only_complete_sha256": sha256(args.stl / "complete.json"),
            },
            human_task=HUMAN_TASK,
            animal_tasks=4,
            routes=3,
            outer_folds=5,
            seeds=3,
            validation_target_file_opened=False,
            test_labels_read=False,
            model_fitted=False,
            architecture_selection_authorized=False,
            partial=False,
        )
    print(f"VDss transfer protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
