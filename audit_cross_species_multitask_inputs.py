#!/usr/bin/env python3
"""Freeze cross-species and multitask input eligibility under human outer folds.

The audit is label-blind: it parses identifiers, task metadata, and train
membership only. For each of the six authorized human tasks and five frozen
outer folds, every potential multitask input row is removed when its canonical
parent or scaffold overlaps the human evaluation fold. No model is fitted and
fixed validation/test targets are never opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
RELATED_ENDPOINTS = {"CL": {"CL_related"}, "Thalf": {"Thalf_related"}}
PILOT_PRIORITY = {
    "VDss__human__steady_state_iv": 1,
    "CL__human__systemic_iv": 2,
    "Thalf__human__terminal_iv": 3,
    "fu__human__plasma": 4,
    "CLint__human__microsome": 5,
    "Papp__human__caco2_ab": 6,
}

RECORD_COLUMNS = [
    "row_id",
    "task_id",
    "split",
    "endpoint",
    "species",
    "system",
    "canonical_unit",
    "doc_id",
    "parent_id",
    "scaffold_group",
    "evidence_role",
    "reliability_tier",
    "evidence_quality_tier",
]
STL_COLUMNS = [
    "row_id",
    "task_id",
    "parent_id",
    "scaffold_group",
    "doc_id",
    "inner_fold_id",
]


def relation_for(target: pd.Series, auxiliary: pd.Series) -> str:
    target_task_id = target.get("task_id", target.name)
    auxiliary_task_id = auxiliary.get("task_id", auxiliary.name)
    if auxiliary_task_id == target_task_id:
        return "target_human_task"
    if auxiliary.evidence_role == "strict_A_curated_development":
        if auxiliary_task_id == "F__human__absolute_oral":
            return "protected_human_F_representation_only"
        return "other_human_task_shared_representation"
    if auxiliary.evidence_role == "auxiliary_C_cross_species":
        if auxiliary.endpoint == target.endpoint and auxiliary.canonical_unit == target.canonical_unit:
            if auxiliary.system == target.system:
                return "direct_cross_species_same_endpoint_system_unit"
            return "adjacent_system_same_endpoint_unit_sensitivity"
        return "other_cross_species_shared_representation"
    if auxiliary.endpoint in RELATED_ENDPOINTS.get(target.endpoint, set()):
        return "published_related_endpoint_representation_only"
    return "other_published_shared_representation"


def unique_nonempty(values: pd.Series) -> int:
    cleaned = values.astype(str).str.strip()
    return int(cleaned.loc[cleaned.ne("")].nunique())


def readiness_gate(min_records: int, species: int) -> str:
    if min_records >= 500 and species >= 3:
        return "ready_for_controlled_transfer_pilot"
    if min_records >= 100 and species >= 2:
        return "limited_transfer_feasibility"
    if min_records >= 30:
        return "exploratory_only_underpowered"
    return "not_ready_for_direct_transfer"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "data/public_development/multitask_pk_24h_v6",
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
        "--p1-freeze",
        type=Path,
        default=ROOT / "data/public_development/stl_p1_decision_freeze_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/public_development/cross_species_multitask_input_audit_v1",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    records_path = args.interface / "training_records.csv"
    registry_path = args.cohort / "task_registry.csv"
    stl_path = args.stl / "benchmark_train_records.csv"
    required = [
        args.cohort / "complete.json",
        args.interface / "complete.json",
        args.stl / "complete.json",
        args.p1_freeze / "complete.json",
        records_path,
        registry_path,
        stl_path,
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    cohort_meta = verify_stage(args.cohort, "public_multitask_pk_development_cohort")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    for name, meta in {
        "cohort": cohort_meta,
        "interface": interface_meta,
        "STL": stl_meta,
        "P1 freeze": p1_meta,
    }.items():
        if meta.get("test_labels_read", False):
            raise ValueError(f"{name} input is not test-closed")
    if p1_meta.get("p1_candidates_advanced") != 0:
        raise ValueError("Input audit requires the frozen no-P1-advance decision")
    if interface_meta.get("cross_split_parent_overlap") != 0 or interface_meta.get("cross_split_scaffold_overlap") != 0:
        raise ValueError("Training interface contains a global split overlap")

    # Deliberately omit target_value and smiles. Validation rows are discarded
    # by membership without ever parsing a target column.
    records = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=RECORD_COLUMNS)
    records = records.loc[records.split.eq("train")].copy()
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    stl = pd.read_csv(stl_path, dtype=str, keep_default_na=False, usecols=STL_COLUMNS)

    if records.row_id.duplicated().any() or stl.row_id.duplicated().any():
        raise ValueError("Duplicate row IDs in an audited input")
    if records[["parent_id", "scaffold_group", "task_id"]].eq("").any().any():
        raise ValueError("Blank parent, scaffold, or task identity")
    if not records.reliability_tier.eq(records.evidence_quality_tier).all():
        raise ValueError("Reliability columns disagree")
    if set(registry.task_id) != set(records.task_id) or len(registry) != 45:
        raise ValueError("Task registry and interface train records must cover the same 45 tasks")
    if set(HUMAN_TASKS) != set(stl.task_id.unique()):
        raise ValueError("Frozen STL folds do not cover exactly the six authorized human tasks")
    stl["inner_fold_id"] = pd.to_numeric(stl.inner_fold_id, errors="raise").astype(int)
    if set(stl.inner_fold_id) != set(range(5)):
        raise ValueError("Expected frozen outer folds 0..4")
    stl_pairs = stl[["row_id", "task_id"]]
    interface_pairs = records[["row_id", "task_id"]]
    if len(stl_pairs.merge(interface_pairs, on=["row_id", "task_id"], how="left", indicator=True).query("_merge != 'both'")):
        raise ValueError("Frozen human train rows are not embedded in the interface")

    registry_num = registry.copy()
    for column in ["train_records", "validation_records", "development_records", "development_molecules", "sources"]:
        registry_num[column] = pd.to_numeric(registry_num[column], errors="raise").astype(int)
    train_counts = records.groupby("task_id").size()
    if not registry_num.set_index("task_id").train_records.eq(train_counts).all():
        raise ValueError("Task registry train counts do not match the immutable interface copy")

    inventory_rows = []
    for task_id, group in records.groupby("task_id", sort=True):
        meta = registry_num.loc[registry_num.task_id.eq(task_id)].iloc[0]
        inventory_rows.append({
            "task_id": task_id,
            "endpoint": meta.endpoint,
            "species": meta.species,
            "system": meta.system,
            "canonical_unit": meta.canonical_unit,
            "evidence_role": meta.evidence_role,
            "reliability_tier": meta.reliability_tier,
            "train_records": len(group),
            "train_parents": group.parent_id.nunique(),
            "train_scaffolds": group.scaffold_group.nunique(),
            "reported_source_clusters": unique_nonempty(group.doc_id),
            "missing_source_records": int(group.doc_id.astype(str).str.strip().eq("").sum()),
            "head_selection_authorized": str(meta.head_selection_authorized).lower() == "true",
            "input_status": (
                "authorized_human_target"
                if task_id in HUMAN_TASKS
                else "protected_human_F_no_selection"
                if task_id == "F__human__absolute_oral"
                else "auxiliary_only"
            ),
        })
    inventory = pd.DataFrame(inventory_rows)

    target_meta = registry_num.set_index("task_id").loc[HUMAN_TASKS].reset_index()
    auxiliary_registry = registry_num.loc[~registry_num.task_id.isin(HUMAN_TASKS)].copy()
    eligibility_rows = []
    for _, target in target_meta.iterrows():
        for _, auxiliary in auxiliary_registry.iterrows():
            relation = relation_for(target, auxiliary)
            eligibility_rows.append({
                "human_task_id": target.task_id,
                "human_endpoint": target.endpoint,
                "human_system": target.system,
                "human_unit": target.canonical_unit,
                "auxiliary_task_id": auxiliary.task_id,
                "auxiliary_endpoint": auxiliary.endpoint,
                "auxiliary_species": auxiliary.species,
                "auxiliary_system": auxiliary.system,
                "auxiliary_unit": auxiliary.canonical_unit,
                "evidence_role": auxiliary.evidence_role,
                "reliability_tier": auxiliary.reliability_tier,
                "relation_to_human_task": relation,
                "direct_label_transfer_authorized": relation == "direct_cross_species_same_endpoint_system_unit",
                "shared_representation_authorized_after_fold_exclusion": True,
                "selection_role": (
                    "primary_transfer_candidate"
                    if relation == "direct_cross_species_same_endpoint_system_unit"
                    else "sensitivity_only"
                    if relation == "adjacent_system_same_endpoint_unit_sensitivity"
                    else "representation_only"
                ),
            })
    eligibility = pd.DataFrame(eligibility_rows)

    task_meta = registry_num.set_index("task_id")
    fold_rows = []
    summary_rows = []
    for human_task in HUMAN_TASKS:
        target = task_meta.loc[human_task]
        for fold in range(5):
            evaluation = stl.loc[stl.task_id.eq(human_task) & stl.inner_fold_id.eq(fold)]
            if evaluation.empty:
                raise ValueError(f"Empty human evaluation fold: {human_task} fold={fold}")
            evaluation_parents = set(evaluation.parent_id)
            evaluation_scaffolds = set(evaluation.scaffold_group)
            evaluation_sources = set(evaluation.doc_id.astype(str).str.strip()) - {""}
            classified = records.copy()
            classified["input_relation"] = classified.task_id.map(
                lambda task: relation_for(target, task_meta.loc[task])
            )
            parent_conflict = classified.parent_id.isin(evaluation_parents)
            scaffold_conflict = classified.scaffold_group.isin(evaluation_scaffolds)
            source_overlap = classified.doc_id.astype(str).str.strip().isin(evaluation_sources)
            classified["parent_conflict"] = parent_conflict
            classified["scaffold_conflict"] = scaffold_conflict
            classified["source_overlap"] = source_overlap
            classified["excluded"] = parent_conflict | scaffold_conflict
            for relation, group in classified.groupby("input_relation", sort=True):
                removed = group.loc[group.excluded]
                retained = group.loc[~group.excluded]
                fold_rows.append({
                    "human_task_id": human_task,
                    "human_endpoint": target.endpoint,
                    "outer_fold": fold,
                    "input_relation": relation,
                    "records_before": len(group),
                    "records_removed_parent": int(group.parent_conflict.sum()),
                    "records_removed_scaffold_only": int((group.scaffold_conflict & ~group.parent_conflict).sum()),
                    "records_removed_total": int(group.excluded.sum()),
                    "records_after": len(retained),
                    "parents_after": retained.parent_id.nunique(),
                    "scaffolds_after": retained.scaffold_group.nunique(),
                    "reported_source_clusters_after": unique_nonempty(retained.doc_id),
                    "source_overlap_records_report_only": int(group.source_overlap.sum()),
                    "source_overlap_removed_by_default": False,
                    "post_filter_parent_overlap": int(retained.parent_id.isin(evaluation_parents).sum()),
                    "post_filter_scaffold_overlap": int(retained.scaffold_group.isin(evaluation_scaffolds).sum()),
                })
            retained_all = classified.loc[~classified.excluded]
            direct = classified.loc[
                classified.input_relation.eq("direct_cross_species_same_endpoint_system_unit")
            ]
            direct_retained = direct.loc[~direct.excluded]
            target_retained = classified.loc[
                classified.input_relation.eq("target_human_task") & ~classified.excluded
            ]
            expected_target_train_parents = set(
                stl.loc[stl.task_id.eq(human_task) & stl.inner_fold_id.ne(fold), "parent_id"]
            )
            if set(target_retained.parent_id) != expected_target_train_parents:
                raise ValueError(f"Target outer-train parent mismatch: {human_task} fold={fold}")
            summary_rows.append({
                "human_task_id": human_task,
                "human_endpoint": target.endpoint,
                "outer_fold": fold,
                "evaluation_records": len(evaluation),
                "evaluation_parents": len(evaluation_parents),
                "evaluation_scaffolds": len(evaluation_scaffolds),
                "evaluation_reported_source_clusters": len(evaluation_sources),
                "target_human_outer_train_records": len(target_retained),
                "target_human_outer_train_parents": target_retained.parent_id.nunique(),
                "direct_transfer_records_before": len(direct),
                "direct_transfer_records_removed": int(direct.excluded.sum()),
                "direct_transfer_records_after": len(direct_retained),
                "direct_transfer_parents_after": direct_retained.parent_id.nunique(),
                "direct_transfer_scaffolds_after": direct_retained.scaffold_group.nunique(),
                "direct_transfer_species_after": direct_retained.species.nunique(),
                "all_multitask_train_records_before": len(classified),
                "all_multitask_train_records_removed": int(classified.excluded.sum()),
                "all_multitask_train_records_after": len(retained_all),
                "all_multitask_tasks_after": retained_all.task_id.nunique(),
                "post_filter_parent_overlap_all": int(retained_all.parent_id.isin(evaluation_parents).sum()),
                "post_filter_scaffold_overlap_all": int(retained_all.scaffold_group.isin(evaluation_scaffolds).sum()),
            })
    fold_audit = pd.DataFrame(fold_rows)
    fold_summary = pd.DataFrame(summary_rows)
    if len(fold_summary) != 30:
        raise ValueError("Expected six human tasks x five outer folds")
    if fold_audit[["post_filter_parent_overlap", "post_filter_scaffold_overlap"]].to_numpy().max() != 0:
        raise ValueError("Fold-filtered input relation retains a human evaluation overlap")
    if fold_summary[["post_filter_parent_overlap_all", "post_filter_scaffold_overlap_all"]].to_numpy().max() != 0:
        raise ValueError("Fold-filtered multitask input retains a human evaluation overlap")

    readiness_rows = []
    for human_task in HUMAN_TASKS:
        target = task_meta.loc[human_task]
        task_folds = fold_summary.loc[fold_summary.human_task_id.eq(human_task)]
        direct_tasks = eligibility.loc[
            eligibility.human_task_id.eq(human_task) & eligibility.direct_label_transfer_authorized,
            "auxiliary_task_id",
        ]
        direct_meta = registry_num.loc[registry_num.task_id.isin(direct_tasks)]
        min_records = int(task_folds.direct_transfer_records_after.min())
        species = int(direct_meta.species.nunique())
        gate = readiness_gate(min_records, species)
        readiness_rows.append({
            "human_task_id": human_task,
            "endpoint": target.endpoint,
            "system": target.system,
            "canonical_unit": target.canonical_unit,
            "direct_transfer_tasks": len(direct_tasks),
            "direct_transfer_species": species,
            "direct_transfer_train_records_before": int(task_folds.direct_transfer_records_before.iloc[0]),
            "minimum_fold_retained_direct_records": min_records,
            "maximum_fold_direct_records_removed": int(task_folds.direct_transfer_records_removed.max()),
            "minimum_fold_retained_direct_parents": int(task_folds.direct_transfer_parents_after.min()),
            "minimum_fold_retained_direct_scaffolds": int(task_folds.direct_transfer_scaffolds_after.min()),
            "transfer_readiness": gate,
            "recommended_pilot_priority": PILOT_PRIORITY[human_task],
            "recommended_use": (
                "first_controlled_transfer_pilot"
                if human_task == "VDss__human__steady_state_iv"
                else "subsequent_controlled_transfer"
                if gate == "ready_for_controlled_transfer_pilot"
                else "shared_representation_only_until_more_direct_data"
            ),
            "scientific_note": (
                "exact steady-state IV system/unit across four species; unbounded log target"
                if human_task == "VDss__human__steady_state_iv"
                else "species must remain explicit; evaluate only on held-out human folds"
                if gate == "ready_for_controlled_transfer_pilot"
                else "insufficient exact endpoint/system/unit auxiliary support"
            ),
        })
    readiness = pd.DataFrame(readiness_rows).sort_values("recommended_pilot_priority")

    contract = {
        "schema_version": 1,
        "human_targets": HUMAN_TASKS,
        "folds": [0, 1, 2, 3, 4],
        "target_columns_parsed": False,
        "fixed_validation_membership_used_for_training": False,
        "direct_transfer_rule": "same endpoint + same system + same canonical unit + non-human species",
        "adjacent_system_rule": "sensitivity only; never pooled as the same label definition",
        "published_related_rule": "representation only; no direct human label transfer claim",
        "outer_fold_filter": "remove every train row across all tasks if parent OR scaffold overlaps the human outer-evaluation fold",
        "source_overlap_rule": "report only in ordinary scaffold CV; source IDs are not model inputs; use separately preregistered source-cluster sensitivity for source extrapolation",
        "human_F": "protected; no head selection or performance claim",
        "first_transfer_pilot": "VDss__human__steady_state_iv",
        "architecture_routes_after_audit": [
            "corrected_stageA_STL_reference",
            "human_only_neural_control",
            "cross_species_pretrain_then_human_finetune",
            "species_conditioned_shared_encoder_human_private_head",
            "shared_encoder_private_endpoint_heads",
        ],
        "validation_status": "closed",
        "test_status": "closed",
    }
    summary = {
        "tasks": int(len(inventory)),
        "train_records": int(len(records)),
        "human_outer_tasks": 6,
        "outer_fold_audits": 30,
        "tasks_ready_for_controlled_transfer": int(
            readiness.transfer_readiness.eq("ready_for_controlled_transfer_pilot").sum()
        ),
        "first_transfer_pilot": "VDss__human__steady_state_iv",
        "minimum_tasks_retained_in_any_fold": int(fold_summary.all_multitask_tasks_after.min()),
        "maximum_post_filter_parent_overlap": int(fold_summary.post_filter_parent_overlap_all.max()),
        "maximum_post_filter_scaffold_overlap": int(fold_summary.post_filter_scaffold_overlap_all.max()),
        "validation_target_file_opened": False,
        "validation_rows_evaluated": False,
        "test_labels_read": False,
        "model_fitted": False,
        "data_modified": False,
    }

    if args.check_only:
        print(
            "Cross-species/multitask input audit ready: "
            f"tasks={summary['tasks']} folds={summary['outer_fold_audits']} "
            f"transfer_ready={summary['tasks_ready_for_controlled_transfer']}"
        )
        return

    with stage_output(args.output) as out:
        inventory.to_csv(out / "task_input_inventory.csv", index=False)
        eligibility.to_csv(out / "auxiliary_task_eligibility.csv", index=False)
        fold_audit.to_csv(out / "outer_fold_input_isolation_audit.csv", index=False)
        fold_summary.to_csv(out / "outer_fold_transfer_summary.csv", index=False)
        readiness.to_csv(out / "human_transfer_readiness.csv", index=False)
        (out / "input_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "README.md").write_text(
            "# Cross-species and multitask input freeze audit\n\n"
            "This label-blind stage inventories 45 train tasks and simulates every six-task x five-fold human "
            "outer evaluation. Any row from any task that shares the held-out human parent or scaffold is removed. "
            "Direct transfer requires the same endpoint, system, and canonical unit; adjacent systems are sensitivity "
            "only and published related endpoints are representation only. Fixed validation/test targets are not opened.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "cross_species_multitask_input_audit",
            inputs={
                "cohort_complete_sha256": sha256(args.cohort / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_train_only_complete_sha256": sha256(args.stl / "complete.json"),
                "p1_decision_freeze_complete_sha256": sha256(args.p1_freeze / "complete.json"),
            },
            **summary,
            next_stage="controlled_cross_species_transfer_protocol_and_smoke",
            partial=False,
        )
    print(f"Cross-species/multitask input audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
