#!/usr/bin/env python3
"""Build the versioned train-only Gate-3 B6 experimental physchem cohort.

This stage materializes only audit-authorized source-train logP/logD records,
parent-level targets, leakage-safe structural features, auxiliary scaffold
folds, and endpoint-specific outer-fold purge memberships.  It fits no model
and never opens source-test, fixed-validation, or test target files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger, rdBase
from rdkit.Chem import Descriptors

from build_stl_benchmark_protocol import canonical_parent_smiles
from dmpk_toolkit import featurize
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
N_FOLDS = 5
SEED = 20260917
MAX_RDKIT_NONFINITE_CELL_FRACTION = 1e-4
MAX_RDKIT_AFFECTED_PARENT_FRACTION = 1e-3


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def reconstruct_parent_smiles(records: pd.DataFrame) -> pd.DataFrame:
    """Recreate canonical parent structures and verify their frozen hashes."""
    unique = records[["source_smiles", "parent_id"]].drop_duplicates().copy()
    identities = []
    hashes = []
    for row in unique.itertuples(index=False):
        identity, parent_hash = canonical_parent_smiles(str(row.source_smiles))
        identities.append(identity)
        hashes.append(parent_hash)
    unique["canonical_parent_smiles"] = identities
    unique["reconstructed_parent_id"] = hashes
    if not unique.parent_id.eq(unique.reconstructed_parent_id).all():
        bad = unique.loc[~unique.parent_id.eq(unique.reconstructed_parent_id)].head(5)
        raise ValueError(f"Canonical parent reconstruction differs from audit identity:\n{bad}")
    identities_per_parent = unique.groupby("parent_id").canonical_parent_smiles.nunique()
    if identities_per_parent.gt(1).any():
        raise ValueError("One audited parent maps to multiple canonical parent structures")
    return unique[["source_smiles", "parent_id", "canonical_parent_smiles"]]


def build_parent_targets(records: pd.DataFrame) -> pd.DataFrame:
    required = {
        "task_id",
        "source_record_id",
        "parent_id",
        "canonical_parent_smiles",
        "scaffold_group",
        "target_value",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Cohort records lack columns: {sorted(missing)}")
    if not np.isfinite(records.target_value.to_numpy(float)).all():
        raise ValueError("Train-only physchem records contain non-finite targets")
    identity = records.groupby(["task_id", "parent_id"], sort=True).agg(
        canonical_parent_smiles=("canonical_parent_smiles", "nunique"),
        scaffold_count=("scaffold_group", "nunique"),
    )
    if identity.canonical_parent_smiles.gt(1).any() or identity.scaffold_count.gt(1).any():
        raise ValueError("A task-parent has conflicting canonical structure or scaffold identity")
    parent = records.groupby(["task_id", "parent_id"], sort=True, as_index=False).agg(
        canonical_parent_smiles=("canonical_parent_smiles", "first"),
        scaffold_group=("scaffold_group", "first"),
        target_value=("target_value", "mean"),
        record_count=("source_record_id", "size"),
        target_std_population=("target_value", lambda values: float(values.std(ddof=0))),
        target_min=("target_value", "min"),
        target_max=("target_value", "max"),
        source_record_ids=("source_record_id", lambda values: ";".join(sorted(map(str, values)))),
    )
    parent["target_aggregation"] = "arithmetic_mean_of_source_train_records"
    parent["training_weight"] = 1.0
    parent["canonical_unit"] = "log10_coefficient"
    parent["reliability_tier"] = "R3_public_aggregate_per_record_provenance_unavailable"
    parent["cohort_role"] = "auxiliary_train_only"
    return parent


def build_feature_manifest(parent_targets: pd.DataFrame) -> pd.DataFrame:
    consistency = parent_targets.groupby("parent_id", sort=True).agg(
        canonical_parent_smiles=("canonical_parent_smiles", "nunique"),
        scaffold_group=("scaffold_group", "nunique"),
    )
    if consistency.max().max() != 1:
        raise ValueError("Cross-task parent identity or scaffold mismatch")
    manifest = parent_targets.groupby("parent_id", sort=True, as_index=False).agg(
        canonical_parent_smiles=("canonical_parent_smiles", "first"),
        scaffold_group=("scaffold_group", "first"),
        auxiliary_tasks=("task_id", lambda values: ";".join(sorted(set(values)))),
        auxiliary_task_count=("task_id", "nunique"),
    )
    manifest["feature_index"] = np.arange(len(manifest), dtype=int)
    return manifest


def assign_auxiliary_scaffold_folds(
    parent_targets: pd.DataFrame, feature_manifest: pd.DataFrame, n_folds: int = N_FOLDS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign target-value-blind scaffold groups while balancing both tasks."""
    membership = parent_targets[["task_id", "parent_id", "scaffold_group"]].drop_duplicates()
    scaffold_union = feature_manifest.groupby("scaffold_group").parent_id.nunique().rename("union_parents")
    task_counts = (
        membership.groupby(["scaffold_group", "task_id"]).parent_id.nunique().unstack(fill_value=0)
    )
    for task in AUXILIARY_TASKS:
        if task not in task_counts:
            task_counts[task] = 0
    groups = pd.concat([scaffold_union, task_counts[AUXILIARY_TASKS]], axis=1).reset_index()
    if len(groups) < n_folds:
        raise ValueError("Insufficient independent auxiliary scaffolds for five folds")
    groups["tie_break"] = groups.scaffold_group.map(
        lambda value: int.from_bytes(__import__("hashlib").sha256(f"{SEED}|{value}".encode()).digest()[:8], "big")
    )
    groups = groups.sort_values(
        ["union_parents", *AUXILIARY_TASKS, "tie_break"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    totals = np.array(
        [groups.union_parents.sum(), *[groups[task].sum() for task in AUXILIARY_TASKS]], dtype=float
    )
    loads = np.zeros((n_folds, len(totals)), dtype=float)
    scaffold_assignment: dict[str, int] = {}
    for row in groups.itertuples(index=False):
        addition = np.array(
            [row.union_parents, *[getattr(row, task) for task in AUXILIARY_TASKS]], dtype=float
        )
        scores = []
        for fold in range(n_folds):
            projected = loads.copy()
            projected[fold] += addition
            normalized = projected / totals
            scores.append((float(np.square(normalized).sum()), float(normalized.max()), fold))
        chosen = min(scores)[2]
        scaffold_assignment[str(row.scaffold_group)] = int(chosen)
        loads[chosen] += addition
    membership["auxiliary_fold"] = membership.scaffold_group.map(scaffold_assignment).astype(int)
    if set(membership.auxiliary_fold) != set(range(n_folds)):
        raise ValueError("Auxiliary scaffold assignment does not cover all five folds")
    if membership.groupby("scaffold_group").auxiliary_fold.nunique().max() != 1:
        raise ValueError("An auxiliary scaffold crosses internal folds")
    summary = membership.groupby(["task_id", "auxiliary_fold"], as_index=False).agg(
        parents=("parent_id", "nunique"), scaffolds=("scaffold_group", "nunique")
    )
    summary["assignment_inputs"] = "structure_identity_and_task_membership_only_no_target_values"
    summary["assignment_scope"] = "auxiliary_candidate_selection_only; downstream_crossfit_is_outer_fold_specific"
    return membership.sort_values(["task_id", "auxiliary_fold", "parent_id"]), summary


def build_outer_fold_purge(
    parent_targets: pd.DataFrame, stl_membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize exact train-side eligibility for all 60 endpoint/fold/task cells."""
    required = {"task_id", "parent_id", "scaffold_group", "inner_fold_id"}
    if required - set(stl_membership.columns):
        raise ValueError("STL fold membership schema is incomplete")
    if set(stl_membership.task_id) != set(HUMAN_TASKS) or set(stl_membership.inner_fold_id) != set(range(N_FOLDS)):
        raise ValueError("STL membership differs from the frozen six-task/five-fold scope")
    memberships = []
    summaries = []
    for downstream_task in HUMAN_TASKS:
        downstream = stl_membership.loc[stl_membership.task_id.eq(downstream_task)].drop_duplicates(
            ["parent_id", "scaffold_group", "inner_fold_id"]
        )
        for outer_fold in range(N_FOLDS):
            evaluation = downstream.loc[downstream.inner_fold_id.eq(outer_fold)]
            eval_parents = set(evaluation.parent_id.astype(str))
            eval_scaffolds = set(evaluation.scaffold_group.astype(str))
            for auxiliary_task in AUXILIARY_TASKS:
                task = parent_targets.loc[parent_targets.task_id.eq(auxiliary_task)].copy()
                task["downstream_task"] = downstream_task
                task["outer_fold"] = outer_fold
                task["evaluation_parent_overlap"] = task.parent_id.isin(eval_parents)
                task["evaluation_scaffold_overlap"] = task.scaffold_group.isin(eval_scaffolds)
                task["retained_for_outer_fold"] = ~(
                    task.evaluation_parent_overlap | task.evaluation_scaffold_overlap
                )
                task["exclusion_reason"] = np.select(
                    [
                        task.evaluation_parent_overlap & task.evaluation_scaffold_overlap,
                        task.evaluation_parent_overlap,
                        task.evaluation_scaffold_overlap,
                    ],
                    ["evaluation_parent_and_scaffold", "evaluation_parent", "evaluation_scaffold"],
                    default="retained_train_only_auxiliary",
                )
                retained = task.loc[task.retained_for_outer_fold]
                summaries.append(
                    {
                        "downstream_scope": downstream_task,
                        "outer_fold": outer_fold,
                        "auxiliary_task": auxiliary_task,
                        "evaluation_parents": evaluation.parent_id.nunique(),
                        "evaluation_scaffolds": evaluation.scaffold_group.nunique(),
                        "records_before_fold_purge": int(task.record_count.sum()),
                        "parents_before_fold_purge": task.parent_id.nunique(),
                        "scaffolds_before_fold_purge": task.scaffold_group.nunique(),
                        "records_after_fold_purge": int(retained.record_count.sum()),
                        "parents_after_fold_purge": retained.parent_id.nunique(),
                        "scaffolds_after_fold_purge": retained.scaffold_group.nunique(),
                        "record_retention_fraction": float(retained.record_count.sum() / task.record_count.sum()),
                        "post_filter_parent_overlap": int(retained.parent_id.isin(eval_parents).sum()),
                        "post_filter_scaffold_overlap": int(retained.scaffold_group.isin(eval_scaffolds).sum()),
                    }
                )
                memberships.append(
                    task[
                        [
                            "downstream_task",
                            "outer_fold",
                            "task_id",
                            "parent_id",
                            "scaffold_group",
                            "record_count",
                            "evaluation_parent_overlap",
                            "evaluation_scaffold_overlap",
                            "retained_for_outer_fold",
                            "exclusion_reason",
                        ]
                    ].rename(columns={"task_id": "auxiliary_task"})
                )
    return pd.concat(memberships, ignore_index=True), pd.DataFrame(summaries)


def compare_resource_budget(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    keys = ["downstream_scope", "outer_fold", "auxiliary_task"]
    integers = [
        "evaluation_parents",
        "evaluation_scaffolds",
        "records_before_fold_purge",
        "parents_before_fold_purge",
        "scaffolds_before_fold_purge",
        "records_after_fold_purge",
        "parents_after_fold_purge",
        "scaffolds_after_fold_purge",
        "post_filter_parent_overlap",
        "post_filter_scaffold_overlap",
    ]
    expected = expected.loc[expected.downstream_scope.isin(HUMAN_TASKS), keys + integers + ["record_retention_fraction"]]
    merged = actual.merge(expected, on=keys, suffixes=("_actual", "_expected"), validate="one_to_one")
    if len(merged) != len(HUMAN_TASKS) * N_FOLDS * len(AUXILIARY_TASKS):
        raise ValueError("Outer-fold resource budget does not contain exactly 60 cells")
    for column in integers:
        if not merged[f"{column}_actual"].astype(int).eq(merged[f"{column}_expected"].astype(int)).all():
            raise ValueError(f"Rebuilt outer-fold resource differs from protocol: {column}")
    if not np.allclose(
        merged.record_retention_fraction_actual,
        merged.record_retention_fraction_expected,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Rebuilt outer-fold retention fractions differ from protocol")


def materialize_features(
    feature_manifest: pd.DataFrame, leakage: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, pd.DataFrame, dict]:
    excluded = sorted(set(leakage.excluded_feature.astype(str)))
    all_names = [name for name, _ in Descriptors._descList]
    if not set(excluded) <= set(all_names):
        raise ValueError(f"Frozen leakage exclusions are absent from current RDKit: {sorted(set(excluded) - set(all_names))}")
    allowed = [name for name in all_names if name not in set(excluded)]
    if set(allowed) & set(excluded) or len(allowed) + len(excluded) != len(all_names):
        raise ValueError("Leakage-safe descriptor partition is inconsistent")
    x, names = featurize(
        feature_manifest.canonical_parent_smiles.tolist(),
        descriptor_names=allowed,
        progress=True,
        feature_set="ecfp4_rdkit2d",
    )
    if names != allowed or x.shape != (len(feature_manifest), 2048 + len(allowed)):
        raise ValueError("Materialized feature layout differs from the frozen registry")
    ecfp4 = x[:, :2048].astype(np.float32, copy=False)
    rdkit2d = x[:, 2048:].astype(np.float32, copy=False)
    nonfinite_ecfp = int((~np.isfinite(ecfp4)).sum())
    nonfinite_rdkit = int((~np.isfinite(rdkit2d)).sum())
    if nonfinite_ecfp:
        raise ValueError(f"ECFP4 materialization produced {nonfinite_ecfp} non-finite cells")
    if np.isinf(rdkit2d).any():
        raise ValueError("RDKit2D contains infinity; only explicit rare NaN calculation failures are permitted")
    missing_rows, missing_columns = np.where(np.isnan(rdkit2d))
    affected_parents = len(set(missing_rows.tolist()))
    cell_fraction = nonfinite_rdkit / rdkit2d.size
    parent_fraction = affected_parents / len(rdkit2d)
    if (
        cell_fraction > MAX_RDKIT_NONFINITE_CELL_FRACTION
        or parent_fraction > MAX_RDKIT_AFFECTED_PARENT_FRACTION
    ):
        raise ValueError(
            "RDKit2D calculation-failure rate exceeds the preregistered cohort gate: "
            f"cells={cell_fraction:.3g}, parents={parent_fraction:.3g}"
        )
    missingness = pd.DataFrame(
        [
            {
                "parent_id": feature_manifest.iloc[row].parent_id,
                "feature_index": int(feature_manifest.iloc[row].feature_index),
                "descriptor_name": allowed[column],
                "raw_cached_value": "NaN",
                "reason": "RDKit_descriptor_calculation_nonfinite",
                "downstream_policy": "median_fit_on_current_producer_training_parents_only",
                "missingness_is_predictor": False,
            }
            for row, column in zip(missing_rows, missing_columns)
        ]
    )
    subset = np.linspace(0, len(feature_manifest) - 1, num=min(32, len(feature_manifest)), dtype=int)
    repeat, repeat_names = featurize(
        feature_manifest.iloc[subset].canonical_parent_smiles.tolist(),
        descriptor_names=allowed,
        progress=False,
        feature_set="ecfp4_rdkit2d",
    )
    repeat_expected = x[subset].astype(np.float64)
    repeat_actual = repeat.astype(np.float64)
    if not np.array_equal(np.isnan(repeat_expected), np.isnan(repeat_actual)):
        raise ValueError("Deterministic feature repeat audit changed the NaN mask")
    finite = np.isfinite(repeat_expected) & np.isfinite(repeat_actual)
    repeat_difference = np.abs(repeat_actual[finite] - repeat_expected[finite])
    max_repeat_difference = float(repeat_difference.max(initial=0.0))
    if repeat_names != allowed or max_repeat_difference > 0:
        raise ValueError(f"Deterministic feature repeat audit failed: {max_repeat_difference:.3g}")
    quality = pd.DataFrame(
        [
            {
                "feature_view": "ecfp4",
                "rows": len(ecfp4),
                "dimensions": ecfp4.shape[1],
                "nonfinite_cells": nonfinite_ecfp,
                "minimum": float(ecfp4.min()),
                "maximum": float(ecfp4.max()),
            },
            {
                "feature_view": "leakage_safe_rdkit2d",
                "rows": len(rdkit2d),
                "dimensions": rdkit2d.shape[1],
                "nonfinite_cells": nonfinite_rdkit,
                "minimum": float(np.nanmin(rdkit2d)),
                "maximum": float(np.nanmax(rdkit2d)),
            },
        ]
    )
    repeat_audit = {
        "repeat_rows": int(len(subset)),
        "repeat_indices": subset.tolist(),
        "maximum_absolute_difference": max_repeat_difference,
        "nan_mask_identical": True,
        "exact_repeat_passed": True,
    }
    return ecfp4, rdkit2d, allowed, quality, missingness, repeat_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "data/public_development/physchem_auxiliary_feasibility_audit_v3",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "data/public_development/gate3_b6_physchem_protocol_v2",
    )
    parser.add_argument(
        "--stl",
        type=Path,
        default=ROOT / "data/public_development/stl_train_only_protocol_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/public_development/gate3_b6_physchem_cohort_v1",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    required = [
        args.audit / "complete.json",
        args.audit / "train_record_audit.csv",
        args.audit / "task_definition_registry.csv",
        args.audit / "descriptor_leakage_registry.csv",
        args.audit / "gate_decision.csv",
        args.protocol / "complete.json",
        args.protocol / "protocol.json",
        args.protocol / "lifecycle_contract.json",
        args.protocol / "outer_fold_resource_budget.csv",
        args.protocol / "auxiliary_producer_registry.csv",
        args.stl / "complete.json",
        args.stl / "benchmark_train_records.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    audit_meta = verify_stage(args.audit, "physchem_auxiliary_feasibility_audit")
    protocol_meta = verify_stage(args.protocol, "gate3_b6_physchem_protocol")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if audit_meta.get("decision") != "CONDITIONAL_PASS" or audit_meta.get("source_test_target_values_used") != 0:
        raise ValueError("Physchem audit does not authorize the frozen next step")
    if not protocol_meta.get("train_only_cohort_construction_authorized"):
        raise ValueError("Gate-3 protocol does not authorize train-only cohort construction")
    if protocol_meta.get("formal_model_training_authorized"):
        raise ValueError("Gate-3 protocol unexpectedly authorizes formal model training")
    if any(meta.get("test_labels_read", False) for meta in [audit_meta, protocol_meta, stl_meta]):
        raise ValueError("An upstream stage reports protected test-label access")
    if protocol_meta.get("validation_target_file_opened", False):
        raise ValueError("Gate-3 protocol reports fixed-validation target access")
    protocol = json.loads((args.protocol / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("human_tasks") != HUMAN_TASKS or protocol.get("auxiliary_tasks") != AUXILIARY_TASKS:
        raise ValueError("Gate-3 task scope differs from the cohort builder")
    if protocol.get("formal_model_training_authorized"):
        raise ValueError("Protocol JSON unexpectedly authorizes formal training")
    if args.check_only:
        print("Gate 3 B6 physchem cohort inputs valid; feature materialization not started.")
        return

    audit_records = pd.read_csv(args.audit / "train_record_audit.csv", keep_default_na=False)
    required_record_columns = {
        "task_id",
        "source_row",
        "source_smiles",
        "target_value",
        "source_record_id",
        "parent_id",
        "scaffold_group",
        "standardization_error",
        "high_conflict_duplicate",
        "overlap_any_protected_parent",
        "overlap_any_protected_scaffold",
        "future_clean_pool",
        "audit_only_not_cohort_authorized",
        "source_split",
        "source_doi",
        "provenance_tier",
    }
    if required_record_columns - set(audit_records.columns):
        raise ValueError("Physchem audit record schema is incomplete")
    clean = audit_records.loc[as_bool(audit_records.future_clean_pool)].copy()
    if set(clean.task_id) != set(AUXILIARY_TASKS) or set(clean.source_split) != {"train"}:
        raise ValueError("Clean cohort contains an unexpected task or source split")
    forbidden = (
        clean.standardization_error.astype(str).str.len().gt(0)
        | as_bool(clean.high_conflict_duplicate)
        | as_bool(clean.overlap_any_protected_parent)
        | as_bool(clean.overlap_any_protected_scaffold)
    )
    if forbidden.any() or not as_bool(clean.audit_only_not_cohort_authorized).all():
        raise ValueError("Audit clean pool lifecycle or exclusion flags differ from expectation")
    observed = clean.groupby("task_id").size().to_dict()
    if observed != {"experimental_logD_pH7_4": 3649, "experimental_logP": 4559}:
        raise ValueError(f"Audit clean pool counts changed: {observed}")

    identities = reconstruct_parent_smiles(clean)
    clean = clean.merge(identities, on=["source_smiles", "parent_id"], validate="many_to_one")
    clean["dataset_doi"] = clean.source_doi
    clean["per_record_source_available"] = False
    clean["cohort_role"] = "auxiliary_train_only"
    clean["cohort_authorized_by"] = "gate3_b6_physchem_protocol_v2"
    parent_targets = build_parent_targets(clean)
    feature_manifest = build_feature_manifest(parent_targets)
    auxiliary_folds, auxiliary_fold_summary = assign_auxiliary_scaffold_folds(parent_targets, feature_manifest)
    feature_manifest = feature_manifest.merge(
        auxiliary_folds[["parent_id", "auxiliary_fold"]].drop_duplicates(),
        on="parent_id",
        validate="one_to_one",
    )
    parent_targets = parent_targets.merge(
        feature_manifest[["parent_id", "feature_index", "auxiliary_fold"]], on="parent_id", validate="many_to_one"
    )
    clean = clean.merge(
        feature_manifest[["parent_id", "feature_index", "auxiliary_fold"]], on="parent_id", validate="many_to_one"
    )

    stl_membership = pd.read_csv(
        args.stl / "benchmark_train_records.csv",
        usecols=["task_id", "parent_id", "scaffold_group", "inner_fold_id"],
        dtype={"task_id": str, "parent_id": str, "scaffold_group": str, "inner_fold_id": int},
    ).drop_duplicates()
    outer_membership, outer_summary = build_outer_fold_purge(parent_targets, stl_membership)
    expected_budget = pd.read_csv(args.protocol / "outer_fold_resource_budget.csv")
    compare_resource_budget(outer_summary, expected_budget)
    if outer_summary[["post_filter_parent_overlap", "post_filter_scaffold_overlap"]].to_numpy().max() != 0:
        raise ValueError("An endpoint-specific auxiliary view retains evaluation overlap")
    outer_membership = outer_membership.merge(
        feature_manifest[["parent_id", "feature_index"]], on="parent_id", validate="many_to_one"
    )

    leakage = pd.read_csv(args.audit / "descriptor_leakage_registry.csv")
    ecfp4, rdkit2d, descriptor_names, feature_quality, feature_missingness, repeat_audit = (
        materialize_features(feature_manifest, leakage)
    )
    excluded = sorted(set(leakage.excluded_feature.astype(str)))
    descriptor_registry = pd.DataFrame(
        [
            {
                "descriptor_name": name,
                "feature_index_within_rdkit2d": index if name in descriptor_names else -1,
                "included_in_leakage_safe_rdkit2d": name in descriptor_names,
                "exclusion_reason": "direct_computed_logP_derived_leakage" if name in excluded else "",
            }
            for index, name in enumerate([name for name, _ in Descriptors._descList])
        ]
    )
    # Recompute indices after exclusions; excluded descriptors remain -1.
    allowed_indices = {name: index for index, name in enumerate(descriptor_names)}
    descriptor_registry["feature_index_within_rdkit2d"] = descriptor_registry.descriptor_name.map(
        allowed_indices
    ).fillna(-1).astype(int)

    task_registry = pd.read_csv(args.audit / "task_definition_registry.csv")
    task_counts = parent_targets.groupby("task_id", as_index=False).agg(
        cohort_parents=("parent_id", "nunique"),
        cohort_scaffolds=("scaffold_group", "nunique"),
        source_train_records_retained=("record_count", "sum"),
        repeated_parents=("record_count", lambda values: int((values > 1).sum())),
        target_min=("target_value", "min"),
        target_median=("target_value", "median"),
        target_max=("target_value", "max"),
    )
    task_registry = task_registry.merge(task_counts, on="task_id", validate="one_to_one")
    task_registry["cohort_role"] = "R3_train_only_auxiliary"
    task_registry["parent_target_aggregation"] = "arithmetic_mean"
    task_registry["parent_training_weight"] = 1.0
    task_registry["model_training_authorized_this_stage"] = False

    feature_registry = {
        "feature_rows": len(feature_manifest),
        "parent_order": "feature_index ascending in feature_parent_manifest.csv",
        "ecfp4": {
            "array": "ecfp4",
            "dimensions": 2048,
            "radius": 2,
            "dtype": "float32",
        },
        "leakage_safe_rdkit2d": {
            "array": "leakage_safe_rdkit2d",
            "dimensions": len(descriptor_names),
            "descriptor_names": descriptor_names,
            "excluded_descriptor_names": excluded,
            "dtype": "float32",
        },
        "ecfp4_plus_leakage_safe_rdkit2d": {
            "arrays": ["ecfp4", "leakage_safe_rdkit2d"],
            "dimensions": 2048 + len(descriptor_names),
        },
        "preprocessing": "raw cache retains rare explicit NaN calculation failures; median imputation and any scaling must be fitted on the current producer-training parents only; an all-missing training column is a hard failure",
        "missingness": {
            "cells": int(len(feature_missingness)),
            "affected_parents": int(feature_missingness.parent_id.nunique()) if len(feature_missingness) else 0,
            "cell_fraction_gate": MAX_RDKIT_NONFINITE_CELL_FRACTION,
            "affected_parent_fraction_gate": MAX_RDKIT_AFFECTED_PARENT_FRACTION,
            "missingness_is_predictor": False,
        },
        "canonicalization": "FragmentParent + Uncharger + canonical tautomer + nonstereo; frozen parent hash verified",
        "rdkit_version": rdBase.rdkitVersion,
        "repeat_audit": repeat_audit,
    }
    lifecycle = {
        "records": "only physchem audit v3 rows with source_split=train and future_clean_pool=true",
        "source_test": "not opened; no source-test membership or target value enters this cohort",
        "fixed_validation": "closed; no target-bearing validation file opened",
        "test": "closed",
        "outer_evaluation_membership": "parent/scaffold/fold identifiers only; no downstream target read",
        "measured_physchem_on_PK_rows": "prohibited",
        "reliability": "R3 public aggregate; tier is reporting/weighting metadata and never a predictor",
        "parent_weight": "one parent one weight after arithmetic target aggregation",
        "current_stage_authority": "cohort and deterministic feature materialization only",
        "next_stage_authority": "one outer-fold bounded engineering smoke",
        "formal_training_authorized": False,
        "fixed_validation_authorized": False,
        "test_authorized": False,
    }
    json.dumps({"feature_registry": feature_registry, "lifecycle": lifecycle}, allow_nan=False)
    input_hashes = {str(path.resolve()): sha256(path) for path in required}

    with stage_output(args.output) as out:
        clean.sort_values(["task_id", "source_row", "source_record_id"]).to_csv(
            out / "train_record_lineage.csv", index=False
        )
        parent_targets.sort_values(["task_id", "parent_id"]).to_csv(out / "parent_targets.csv", index=False)
        feature_manifest.sort_values("feature_index").to_csv(out / "feature_parent_manifest.csv", index=False)
        np.savez_compressed(
            out / "parent_features_float32.npz",
            ecfp4=ecfp4,
            leakage_safe_rdkit2d=rdkit2d,
        )
        auxiliary_folds.to_csv(out / "auxiliary_scaffold_fold_membership.csv", index=False)
        auxiliary_fold_summary.to_csv(out / "auxiliary_scaffold_fold_summary.csv", index=False)
        outer_membership.sort_values(
            ["downstream_task", "outer_fold", "auxiliary_task", "parent_id"]
        ).to_csv(out / "outer_fold_auxiliary_parent_membership.csv", index=False)
        outer_summary.sort_values(["downstream_scope", "outer_fold", "auxiliary_task"]).to_csv(
            out / "outer_fold_purge_summary.csv", index=False
        )
        descriptor_registry.to_csv(out / "descriptor_registry.csv", index=False)
        leakage.to_csv(out / "descriptor_leakage_exclusions.csv", index=False)
        feature_quality.to_csv(out / "feature_quality_audit.csv", index=False)
        feature_missingness.to_csv(out / "feature_missingness_registry.csv", index=False)
        task_registry.to_csv(out / "task_registry.csv", index=False)
        (out / "feature_registry.json").write_text(
            json.dumps(feature_registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / "lifecycle_contract.json").write_text(
            json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / "README.md").write_text(
            "# Gate 3 B6 train-only experimental physchem cohort v1\n\n"
            "Audit-authorized OneADMET source-train experimental logP/logD records, parent targets, "
            "leakage-safe ECFP4/RDKit2D features, five auxiliary scaffold folds, and all 60 endpoint-specific "
            "outer-fold purge views. This stage fits no model and opens no source-test, fixed-validation, or "
            "test targets. It authorizes one bounded outer-fold engineering smoke next, not formal training.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "gate3_b6_physchem_cohort",
            inputs=input_hashes,
            partial=False,
            auxiliary_tasks=len(AUXILIARY_TASKS),
            source_train_records=int(len(clean)),
            parent_task_rows=int(len(parent_targets)),
            unique_parents=int(len(feature_manifest)),
            feature_views=2,
            ecfp4_dimensions=int(ecfp4.shape[1]),
            leakage_safe_rdkit2d_dimensions=int(rdkit2d.shape[1]),
            excluded_descriptors=len(excluded),
            auxiliary_folds=N_FOLDS,
            outer_fold_resource_cells=int(len(outer_summary)),
            minimum_parents_after_purge=int(outer_summary.parents_after_fold_purge.min()),
            minimum_scaffolds_after_purge=int(outer_summary.scaffolds_after_fold_purge.min()),
            maximum_post_filter_parent_overlap=int(outer_summary.post_filter_parent_overlap.max()),
            maximum_post_filter_scaffold_overlap=int(outer_summary.post_filter_scaffold_overlap.max()),
            feature_nonfinite_cells=int(feature_quality.nonfinite_cells.sum()),
            feature_missingness_affected_parents=(
                int(feature_missingness.parent_id.nunique()) if len(feature_missingness) else 0
            ),
            feature_missingness_explicit=True,
            feature_imputation_scope="current_producer_training_parents_only",
            feature_repeat_maximum_absolute_difference=repeat_audit["maximum_absolute_difference"],
            model_fitted=False,
            source_test_target_file_opened=False,
            evaluation_labels_read=False,
            validation_target_file_opened=False,
            test_labels_read=False,
            bounded_one_fold_smoke_authorized=True,
            formal_model_training_authorized=False,
            fixed_validation_authorized=False,
            test_authorized=False,
        )
    print(f"Gate 3 B6 physchem cohort: {args.output}")


if __name__ == "__main__":
    run_cli(main)
