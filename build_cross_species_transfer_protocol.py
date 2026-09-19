#!/usr/bin/env python3
"""Freeze an endpoint-configured controlled cross-species transfer protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


ENDPOINT_CONFIGS = {
    "CL": {
        "human_task": "CL__human__systemic_iv",
        "animal_tasks": [
            "CL__dog__systemic_iv",
            "CL__monkey__systemic_iv",
            "CL__mouse__systemic_iv",
            "CL__rat__systemic_iv",
        ],
        "system": "systemic_iv",
        "canonical_unit": "L/h/kg",
        "target_transform": "log10",
        "reporting_inverse": "power10",
        "primary_metric": "transformed_RMSE",
        "parent_aggregation": "mean_transformed_target_within_task_and_parent",
        "target_standardization": "taskwise_zscore_fit_on_retained_train_parents",
        "model_output": "unbounded_standardized_target",
        "training_loss": "task_standardized_Huber",
    },
    "Thalf": {
        "human_task": "Thalf__human__terminal_iv",
        "animal_tasks": [
            "Thalf__dog__terminal_iv",
            "Thalf__monkey__terminal_iv",
            "Thalf__mouse__terminal_iv",
            "Thalf__rat__terminal_iv",
        ],
        "system": "terminal_iv",
        "canonical_unit": "h",
        "target_transform": "log10",
        "reporting_inverse": "power10",
        "primary_metric": "transformed_RMSE",
        "parent_aggregation": "mean_transformed_target_within_task_and_parent",
        "target_standardization": "taskwise_zscore_fit_on_retained_train_parents",
        "model_output": "unbounded_standardized_target",
        "training_loss": "task_standardized_Huber",
    },
    "fu": {
        "human_task": "fu__human__plasma",
        "animal_tasks": [
            "fu__dog__plasma",
            "fu__monkey__plasma",
            "fu__mouse__plasma",
            "fu__rat__plasma",
        ],
        "system": "plasma",
        "canonical_unit": "fraction",
        "target_transform": "logit",
        "reporting_inverse": "expit",
        "primary_metric": "physical_MAE",
        "parent_aggregation": "mean_record_level_expit_physical",
        "target_standardization": "none_physical_fraction",
        "model_output": "bounded_sigmoid_fraction",
        "training_loss": "physical_MAE",
    },
}

ROUTES = [
    {
        "route_id": "human_only_neural_control",
        "training_phases": "human_outer_train_only",
        "private_heads": "human_only",
        "purpose": "matched neural control for transfer attribution",
    },
    {
        "route_id": "cross_species_pretrain_human_finetune",
        "training_phases": "four_animal_tasks_pretrain_then_human_outer_train_finetune",
        "private_heads": "animal_pretrain_heads_plus_human_private_head",
        "purpose": "test whether animal supervision improves the human representation",
    },
    {
        "route_id": "species_conditioned_joint_shared_encoder",
        "training_phases": "task_balanced_joint_animal_and_human_outer_train",
        "private_heads": "one_private_head_per_species_including_human",
        "purpose": "test simultaneous shared representation with explicit species identity",
    },
]


def source_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = frame.loc[frame.doc_id.astype(str).ne(""), ["parent_id", "doc_id"]].drop_duplicates()
    adjacency: dict[str, set[str]] = {}
    for row in edges.itertuples(index=False):
        parent, document = f"p:{row.parent_id}", f"d:{row.doc_id}"
        adjacency.setdefault(parent, set()).add(document)
        adjacency.setdefault(document, set()).add(parent)
    seen, rows = set(), []
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack, parents, documents = [node], set(), set()
        seen.add(node)
        while stack:
            current = stack.pop()
            (parents if current.startswith("p:") else documents).add(current[2:])
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        rows.append({
            "component_parent_count": len(parents),
            "component_document_count": len(documents),
            "document_ids": ";".join(sorted(documents)),
        })
    components = pd.DataFrame(rows).sort_values(
        ["component_parent_count", "component_document_count"], ascending=False
    ).reset_index(drop=True)
    components.insert(0, "source_component_rank", components.index + 1)
    document_counts = edges.groupby("doc_id", as_index=False).agg(
        parent_document_edges=("parent_id", "size"), parents=("parent_id", "nunique")
    ).sort_values(["parents", "doc_id"], ascending=[False, True])
    return components, document_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=sorted(ENDPOINT_CONFIGS), default="CL")
    parser.add_argument("--input-audit", type=Path,
                        default=ROOT / "data/public_development/cross_species_multitask_input_audit_v1")
    parser.add_argument("--p1-freeze", type=Path,
                        default=ROOT / "data/public_development/stl_p1_decision_freeze_v1")
    parser.add_argument("--interface", type=Path,
                        default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path,
                        default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/public_development/cl_cross_species_transfer_protocol_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = ENDPOINT_CONFIGS[args.endpoint]
    human_task = config["human_task"]
    animal_tasks = config["animal_tasks"]
    selected_tasks = [human_task, *animal_tasks]

    required = [
        args.input_audit / "complete.json", args.input_audit / "human_transfer_readiness.csv",
        args.input_audit / "auxiliary_task_eligibility.csv", args.input_audit / "outer_fold_transfer_summary.csv",
        args.p1_freeze / "complete.json", args.p1_freeze / "p1_decision_registry.csv",
        args.interface / "complete.json", args.interface / "task_target_transforms_train_only.csv",
        args.stl / "complete.json", args.stl / "benchmark_train_records.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    audit_meta = verify_stage(args.input_audit, "cross_species_multitask_input_audit")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    metas = [audit_meta, p1_meta, interface_meta, stl_meta]
    if any(meta.get("test_labels_read", False) for meta in metas):
        raise ValueError("Protocol inputs must remain test-closed")
    if audit_meta.get("validation_target_file_opened") or audit_meta.get("model_fitted"):
        raise ValueError("Input audit must remain fit-free and validation-closed")

    readiness = pd.read_csv(args.input_audit / "human_transfer_readiness.csv")
    ready = readiness.loc[readiness.human_task_id.eq(human_task)]
    if len(ready) != 1 or ready.iloc[0].transfer_readiness != "ready_for_controlled_transfer_pilot":
        raise ValueError(f"{human_task} is not ready for controlled transfer")
    eligibility = pd.read_csv(args.input_audit / "auxiliary_task_eligibility.csv")
    direct = eligibility.loc[
        eligibility.human_task_id.eq(human_task)
        & eligibility.direct_label_transfer_authorized.eq(True)
    ].copy()
    if set(direct.auxiliary_task_id) != set(animal_tasks):
        raise ValueError("Direct-transfer task set differs from the configured four-species set")
    if not direct.auxiliary_system.eq(config["system"]).all() \
            or not direct.auxiliary_unit.eq(config["canonical_unit"]).all():
        raise ValueError("Direct-transfer system/unit definitions are not aligned")

    transforms = pd.read_csv(args.interface / "task_target_transforms_train_only.csv")
    columns = [
        "task_id", "endpoint", "canonical_unit", "evidence_role", "reliability_tier",
        "train_records", "validation_records", "target_transform", "reporting_inverse",
        "standardization", "model_output", "head_selection_authorized",
    ]
    task_registry = transforms.loc[transforms.task_id.isin(selected_tasks), columns].copy()
    if len(task_registry) != 5 or set(task_registry.task_id) != set(selected_tasks):
        raise ValueError("Training interface does not uniquely cover the configured tasks")
    if not task_registry.endpoint.eq(args.endpoint).all() \
            or not task_registry.canonical_unit.eq(config["canonical_unit"]).all() \
            or not task_registry.target_transform.eq(config["target_transform"]).all() \
            or not task_registry.reporting_inverse.eq(config["reporting_inverse"]).all():
        raise ValueError("Configured endpoint/unit/transform is not aligned across species")

    p1 = pd.read_csv(args.p1_freeze / "p1_decision_registry.csv")
    baseline = p1.loc[p1.task_id.eq(human_task)].copy()
    if len(baseline) != 1 or baseline.iloc[0].reference_for_next_stage != "corrected_stageA_matched_control":
        raise ValueError("Corrected Stage-A reference is not frozen")

    fold_budget = pd.read_csv(args.input_audit / "outer_fold_transfer_summary.csv")
    fold_budget = fold_budget.loc[fold_budget.human_task_id.eq(human_task)].copy()
    if len(fold_budget) != 5 or fold_budget[[
        "post_filter_parent_overlap_all", "post_filter_scaffold_overlap_all"
    ]].to_numpy().max() != 0:
        raise ValueError("Outer-fold resource audit is incomplete or contaminated")
    fold_budget["human_reference_steps_per_epoch"] = fold_budget.target_human_outer_train_parents.map(
        lambda value: math.ceil(int(value) / 128)
    )
    steps = fold_budget.human_reference_steps_per_epoch
    fold_budget["human_only_total_optimizer_updates"] = steps * 30
    fold_budget["pretrain_finetune_total_optimizer_updates"] = steps * (4 * 40 + 30)
    fold_budget["joint_total_optimizer_updates"] = steps * (5 * 30)
    fold_budget["human_supervision_updates_each_route"] = steps * 30

    human_source = pd.read_csv(
        args.stl / "benchmark_train_records.csv", dtype=str, keep_default_na=False,
        usecols=["task_id", "parent_id", "doc_id"],
    )
    human_source = human_source.loc[human_source.task_id.eq(human_task)].copy()
    components, document_counts = source_components(human_source)
    human_parents = int(human_source.parent_id.nunique())
    largest_component_parents = int(components.component_parent_count.max())
    largest_component_fraction = largest_component_parents / human_parents
    component_fivefold_feasible = bool(len(components) >= 5 and largest_component_fraction <= 0.5)

    routes = pd.DataFrame(ROUTES)
    routes["species_conditioning"] = True
    routes["feature_set"] = "rdkit2d"
    routes["encoder"] = "MLP_128_64_GELU_layernorm"
    routes["loss"] = config["training_loss"]
    routes["model_output"] = config["model_output"]
    routes["target_standardization"] = config["target_standardization"]
    routes["human_primary_metric"] = config["primary_metric"]
    routes["formal_outer_folds"] = 5
    routes["formal_seeds"] = "20260917;20260918;20260919"
    routes["pretrain_epochs"] = 40
    routes["human_or_joint_epochs"] = 30
    routes["batch_size"] = 128
    routes["learning_rate"] = 0.0003
    routes["weight_decay"] = 0.001
    routes["task_balancing"] = "equal_optimizer_steps_per_task_per_epoch"
    routes["steps_per_task_reference"] = "ceil(retained_human_outer_train_parents / batch_size)"
    routes["human_supervision_budget_matched"] = True
    routes["total_optimizer_budget_matched"] = False
    routes["early_stopping"] = False
    routes["seed_aggregation"] = (
        "arithmetic_mean_prediction_in_physical_fraction_space"
        if config["model_output"] == "bounded_sigmoid_fraction"
        else "arithmetic_mean_prediction_in_transformed_space"
    )
    routes["architecture_selection_authorized"] = False

    contract = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "human_task": human_task,
        "animal_tasks": animal_tasks,
        "tasks": selected_tasks,
        "species_order": ["human", "dog", "monkey", "mouse", "rat"],
        "system": config["system"],
        "canonical_unit": config["canonical_unit"],
        "feature_set": "rdkit2d",
        "target_transform": config["target_transform"],
        "upstream_target_encoding": f"{config['target_transform']} already encoded upstream",
        "reporting_inverse": config["reporting_inverse"],
        "human_primary_metric": config["primary_metric"],
        "parent_aggregation": config["parent_aggregation"],
        "target_standardization": config["target_standardization"],
        "model_output": config["model_output"],
        "training_loss": config["training_loss"],
        "fold_filter": "remove every row whose parent OR scaffold overlaps the human outer-evaluation fold",
        "human_evaluation": "frozen five-fold human scaffold CV only",
        "routes": [row["route_id"] for row in ROUTES],
        "randomization": "same registered seed and initialization within an outer fold across routes",
        "task_balancing": "each active task receives human-reference optimizer steps per epoch",
        "compute_budget_interpretation": "human supervision is matched; total optimizer updates differ by design because transfer routes consume animal supervision",
        "epoch_policy": "fixed registered epochs without evaluation-label monitoring",
        "feature_preprocessing": "imputation and scaling fit only on current-route retained outer-training structures",
        "seed_aggregation": (
            "arithmetic mean of the three fixed-seed predictions in physical fraction space before primary scoring"
            if config["model_output"] == "bounded_sigmoid_fraction"
            else "arithmetic mean of the three fixed-seed predictions in transformed target space before primary scoring"
        ),
        "evaluation_label_lifecycle": "outer targets are read only after all registered routes and seeds have predicted",
        "smoke_scope": "outer fold 0, one seed, bounded parents and epochs; engineering only",
        "formal_scope": "five outer folds x three fixed seeds x three routes",
        "advance_gate": {
            "minimum_mean_improvement_vs_human_only_pct": 2.0,
            "minimum_noninferior_outer_folds": 3,
            "paired_parent_bootstrap_repeats": 2000,
            "paired_parent_bootstrap_upper_95ci_below_zero": True,
            "low_similarity_maximum_relative_harm_pct": 2.0,
        },
        "stageA_replacement_gate": {
            "minimum_improvement_pct": 2.0,
            "minimum_noninferior_outer_folds": 3,
            "paired_parent_bootstrap_upper_95ci_below_zero": True,
            "low_similarity_maximum_relative_harm_pct": 2.0,
        },
        "source_graph": {
            "human_parents": human_parents,
            "human_documents": int(human_source.doc_id.replace("", pd.NA).nunique()),
            "connected_components": len(components),
            "largest_component_parents": largest_component_parents,
            "largest_component_fraction": largest_component_fraction,
            "balanced_component_fivefold_feasible": component_fivefold_feasible,
            "interpretation": "source sensitivity requires a separately registered unbalanced stress test when infeasible",
        },
        "fixed_validation_status": "closed",
        "test_status": "closed",
        "source_id_as_model_input": False,
    }
    json.dumps(contract, allow_nan=False)
    if args.check_only:
        print(
            f"{args.endpoint} transfer protocol ready: tasks=5 routes=3 folds=5 seeds=3 "
            f"auxiliary_records={int(ready.iloc[0].direct_transfer_train_records_before)} "
            f"source_components={len(components)} largest_fraction={largest_component_fraction:.3f}"
        )
        return

    with stage_output(args.output) as out:
        routes.to_csv(out / "route_registry.csv", index=False)
        task_registry.to_csv(out / "task_registry.csv", index=False)
        fold_budget.to_csv(out / "outer_fold_resource_budget.csv", index=False)
        baseline.to_csv(out / "corrected_stageA_reference.csv", index=False)
        components.to_csv(out / "source_component_feasibility.csv", index=False)
        document_counts.to_csv(out / "human_document_counts.csv", index=False)
        (out / "endpoint_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / "protocol.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / "README.md").write_text(
            f"# {args.endpoint} controlled cross-species transfer protocol\n\n"
            "Three registered routes share frozen human outer folds, train-fold-only feature preprocessing, "
            f"endpoint-specific target handling ({config['target_standardization']}; "
            f"{config['model_output']}; {config['training_loss']}) and matched human supervision. "
            "Transfer routes use additional animal optimizer updates by design; the total compute budget is "
            "therefore reported rather than described as matched. Fixed validation/test remain closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "cross_species_transfer_protocol",
            inputs={
                "input_audit_complete_sha256": sha256(args.input_audit / "complete.json"),
                "p1_freeze_complete_sha256": sha256(args.p1_freeze / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_train_only_complete_sha256": sha256(args.stl / "complete.json"),
            },
            endpoint=args.endpoint, human_task=human_task, animal_tasks=4, routes=3,
            outer_folds=5, seeds=3, human_parents=human_parents,
            source_connected_components=len(components),
            largest_source_component_parents=largest_component_parents,
            largest_source_component_fraction=largest_component_fraction,
            balanced_source_component_fivefold_feasible=component_fivefold_feasible,
            validation_target_file_opened=False, test_labels_read=False, model_fitted=False,
            architecture_selection_authorized=False, partial=False,
        )
    print(f"{args.endpoint} transfer protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
