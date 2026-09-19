#!/usr/bin/env python3
"""Audit train-side OneADMET logP/logD data before any Gate-3 cohort is built.

This stage is deliberately an audit, not a cohort builder.  It publishes only
source-train target values, quarantines structural/duplicate conflicts, tests
fold-local purge feasibility against the frozen human-PK folds, and keeps all
fixed validation/test and external-reserve labels closed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import RDLogger

from build_split_manifest import structure_groups
from pipeline_common import (
    ROOT,
    finish_stage,
    run_cli,
    sha256,
    stable_id,
    stage_output,
    startup_self_check,
    verify_stage,
)


RAW_TASKS = {
    "experimental_logP": "PUBLIC___LogPow_Public.csv",
    "experimental_logD_pH7_4": "PUBLIC___LogD74_Public.csv",
}
TASK_DEFINITIONS = {
    "experimental_logP": {
        "definition": "experimental/as-published octanol-water partition coefficient",
        "condition": "nominally neutral-species logP; per-record method/ionization metadata unavailable",
    },
    "experimental_logD_pH7_4": {
        "definition": "experimental/as-published distribution coefficient at pH 7.4",
        "condition": "pH 7.4 is encoded at task level; per-record buffer/method metadata unavailable",
    },
}
MIN_PARENTS_AFTER_PURGE = 1000
MIN_SCAFFOLDS_AFTER_PURGE = 100
MAX_STRUCTURE_FAILURE_FRACTION = 0.05
DUPLICATE_CONFLICT_RANGE = 1.0

# Failures remain explicit in structure_failures.csv; suppressing RDKit's
# repetitive stderr messages keeps batch audits readable.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(str).str.lower().eq("true")


def normalize_structures(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = records[["source_smiles"]].drop_duplicates().copy()
    parents, scaffolds, errors = [], [], []
    for smiles in unique.source_smiles:
        try:
            parent, scaffold = structure_groups(smiles)
            parents.append(parent)
            scaffolds.append(scaffold)
            errors.append("")
        except (RuntimeError, ValueError) as exc:
            parents.append("")
            scaffolds.append("")
            errors.append(f"{type(exc).__name__}: {exc}")
    unique["parent_id"] = parents
    unique["scaffold_group"] = scaffolds
    unique["standardization_error"] = errors
    return records.merge(unique, on="source_smiles", validate="many_to_one"), unique


def read_source_train_records(raw: Path, registry: pd.DataFrame, chunk_size: int) -> tuple[pd.DataFrame, dict]:
    expected = {
        task_id: int(registry.loc[registry.task_name.eq(raw_name), "train_measurements"].iloc[0])
        for task_id, raw_name in RAW_TASKS.items()
    }
    pieces = []
    source_rows = {"train": 0, "test": 0}
    usecols = ["SMILES", "Set", *RAW_TASKS.values()]
    for chunk in pd.read_csv(raw, usecols=usecols, chunksize=chunk_size, low_memory=False):
        split = chunk.Set.fillna("<NA>").astype(str)
        unexpected = set(split) - {"train", "test"}
        if unexpected:
            raise ValueError(f"Unexpected OneADMET Set values: {sorted(unexpected)}")
        source_rows["train"] += int(split.eq("train").sum())
        source_rows["test"] += int(split.eq("test").sum())
        # Source-test cells share the same wide CSV container, but are neither
        # transformed nor counted below and can never reach an output artifact.
        train = chunk.loc[split.eq("train")].copy()
        for task_id, raw_name in RAW_TASKS.items():
            selected = train.loc[train[raw_name].notna(), ["SMILES", raw_name]].copy()
            if selected.empty:
                continue
            selected["target_value"] = pd.to_numeric(selected[raw_name], errors="raise")
            selected["source_row"] = selected.index.astype(int)
            selected["task_id"] = task_id
            selected = selected.rename(columns={"SMILES": "source_smiles"})
            pieces.append(selected[["task_id", "source_row", "source_smiles", "target_value"]])
    records = pd.concat(pieces, ignore_index=True)
    observed = records.groupby("task_id").size().to_dict()
    if observed != expected:
        raise ValueError(f"Source-train measurement counts differ from frozen registry: {observed} != {expected}")
    if not np.isfinite(records.target_value).all():
        raise ValueError("Non-finite train-side physchem target")
    records["source_record_id"] = [
        stable_id(f"oneadmet-physchem|{task}|{row}|{smiles}")
        for task, row, smiles in zip(records.task_id, records.source_row, records.source_smiles)
    ]
    return records, source_rows


def duplicate_audit(records: pd.DataFrame) -> pd.DataFrame:
    valid = records.loc[records.standardization_error.eq("")].copy()
    grouped = valid.groupby(["task_id", "parent_id"], as_index=False).agg(
        records=("source_record_id", "size"),
        unique_source_smiles=("source_smiles", "nunique"),
        scaffolds=("scaffold_group", "nunique"),
        target_mean=("target_value", "mean"),
        target_median=("target_value", "median"),
        target_std=("target_value", lambda value: float(value.std(ddof=0))),
        target_min=("target_value", "min"),
        target_max=("target_value", "max"),
    )
    grouped["target_range"] = grouped.target_max - grouped.target_min
    grouped["is_repeated_parent"] = grouped.records.gt(1)
    grouped["high_conflict_duplicate"] = grouped.is_repeated_parent & grouped.target_range.gt(
        DUPLICATE_CONFLICT_RANGE
    )
    grouped["future_handling"] = np.select(
        [grouped.high_conflict_duplicate, grouped.is_repeated_parent],
        ["quarantine_pending_protocol", "retain_with_parent_level_aggregation_only_after_protocol"],
        default="single_parent_record",
    )
    return grouped


def protected_reference_registry(
    splits_path: Path,
    project_tasks_path: Path,
    boulton_path: Path,
    public_f_path: Path,
    decisions_path: Path,
    project_source_path: Path,
    pkdb_path: Path,
    pubmed_path: Path,
    pksmart_path: Path,
) -> pd.DataFrame:
    rows = []

    def add(group: str, member_id: str, parent: str, scaffold: str, role: str) -> None:
        if str(parent).strip() and str(scaffold).strip():
            rows.append(
                {
                    "protected_group": group,
                    "member_id": str(member_id),
                    "parent_id": str(parent),
                    "scaffold_group": str(scaffold),
                    "protected_role": role,
                }
            )

    manifest = pd.read_csv(splits_path, dtype=str, keep_default_na=False)
    eligible = bool_series(manifest, "eligible")
    for row in manifest.loc[eligible & manifest.split.isin(["val", "test"])].itertuples(index=False):
        add(f"fixed_project_{row.split}", row.molecule_id, row.parent_id, row.scaffold_group, row.split)

    project_tasks = pd.read_csv(project_tasks_path, dtype=str, keep_default_na=False)
    strict = project_tasks.loc[project_tasks.task_id.eq("F__human__absolute_oral"), ["molecule_id", "split"]]
    strict = strict.merge(
        manifest[["molecule_id", "parent_id", "scaffold_group"]], on="molecule_id", validate="many_to_one"
    ).drop_duplicates("molecule_id")
    for row in strict.itertuples(index=False):
        add("strict_F_frozen", row.molecule_id, row.parent_id, row.scaffold_group, row.split)

    for path, group, id_column in [
        (boulton_path, "boulton_reserve", "candidate_id"),
        (public_f_path, "public_F_B2_unassigned", "record_id"),
    ]:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for row in frame.itertuples(index=False):
            add(group, getattr(row, id_column), row.parent_id, row.scaffold_group, "reserved_or_unassigned")

    decisions = pd.read_csv(decisions_path, dtype=str, keep_default_na=False)
    accepted = decisions.loc[decisions.eligibility_decision.eq("accepted")].copy()
    mapped = {}

    accepted_molecules = set(accepted.molecule_id.astype(str))
    source = pd.read_csv(project_source_path, usecols=["molecule_id", "smiles"], dtype=str)
    source = source.loc[source.molecule_id.astype(str).isin(accepted_molecules)].drop_duplicates("molecule_id")
    for row in source.itertuples(index=False):
        try:
            mapped[str(row.molecule_id)] = structure_groups(str(row.smiles))
        except (RuntimeError, ValueError):
            pass
    for path, key_column in [(pkdb_path, "candidate_id"), (pubmed_path, "candidate_id")]:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for row in frame.itertuples(index=False):
            mapped[str(getattr(row, key_column))] = (str(row.parent_id), str(row.scaffold_group))
    pksmart = pd.read_csv(pksmart_path, dtype=str, keep_default_na=False)
    for row in pksmart.itertuples(index=False):
        mapped[str(row.candidate_id)] = (str(row.parent_id), str(row.scaffold_group))

    missing = []
    for row in accepted.itertuples(index=False):
        keys = [str(row.candidate_id), str(row.molecule_id)]
        match = next((mapped[key] for key in keys if key in mapped), None)
        if match is None:
            missing.append(str(row.candidate_id))
        else:
            add("accepted_Thalf_external", row.candidate_id, match[0], match[1], "formal_external_reserve")
    if missing:
        raise ValueError(f"Accepted external records lack structure mapping: {missing}")
    external_count = sum(row["protected_group"] == "accepted_Thalf_external" for row in rows)
    if external_count != len(accepted):
        raise ValueError("Accepted external protection count mismatch")
    return pd.DataFrame(rows).drop_duplicates().sort_values(
        ["protected_group", "member_id"], ignore_index=True
    )


def add_record_audit_flags(
    records: pd.DataFrame, duplicates: pd.DataFrame, protected: pd.DataFrame
) -> pd.DataFrame:
    result = records.copy()
    conflict = set(duplicates.loc[duplicates.high_conflict_duplicate, "parent_id"])
    result["high_conflict_duplicate"] = result.parent_id.isin(conflict)
    all_parents = set(protected.parent_id)
    all_scaffolds = set(protected.scaffold_group)
    result["overlap_any_protected_parent"] = result.parent_id.isin(all_parents)
    result["overlap_any_protected_scaffold"] = result.scaffold_group.isin(all_scaffolds)
    result["future_clean_pool"] = (
        result.standardization_error.eq("")
        & ~result.high_conflict_duplicate
        & ~result.overlap_any_protected_parent
        & ~result.overlap_any_protected_scaffold
    )
    result["audit_only_not_cohort_authorized"] = True
    result["source_split"] = "train"
    result["source_doi"] = "10.1021/acs.jmedchem.6c00049"
    result["provenance_tier"] = "R3_public_aggregate_per_record_provenance_unavailable"
    return result


def protected_overlap_audit(records: pd.DataFrame, protected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = records.loc[records.standardization_error.eq("")]
    for task_id, task in valid.groupby("task_id"):
        for group, reference in protected.groupby("protected_group"):
            parent = task.parent_id.isin(set(reference.parent_id))
            scaffold = task.scaffold_group.isin(set(reference.scaffold_group))
            rows.append(
                {
                    "task_id": task_id,
                    "protected_group": group,
                    "reference_members": reference.member_id.nunique(),
                    "overlap_records_any": int((parent | scaffold).sum()),
                    "overlap_parents": int(task.loc[parent, "parent_id"].nunique()),
                    "overlap_scaffolds": int(task.loc[scaffold, "scaffold_group"].nunique()),
                    "future_action": "exclude_parent_or_scaffold_before_any_cohort",
                }
            )
    return pd.DataFrame(rows)


def outer_fold_audit(records: pd.DataFrame, stl_path: Path) -> pd.DataFrame:
    stl = pd.read_csv(
        stl_path,
        usecols=["task_id", "parent_id", "scaffold_group", "inner_fold_id"],
        dtype={"task_id": str, "parent_id": str, "scaffold_group": str, "inner_fold_id": int},
    ).drop_duplicates()
    if stl.task_id.nunique() != 6 or set(stl.inner_fold_id) != set(range(5)):
        raise ValueError("Frozen human-PK outer-fold membership differs from the six-task/five-fold contract")
    base = records.loc[records.future_clean_pool].copy()
    rows = []
    scopes = sorted(stl.task_id.unique()) + ["ALL_PK_TASKS_UNION"]
    for scope in scopes:
        scoped = stl if scope == "ALL_PK_TASKS_UNION" else stl.loc[stl.task_id.eq(scope)]
        for fold in range(5):
            evaluation = scoped.loc[scoped.inner_fold_id.eq(fold)]
            eval_parents = set(evaluation.parent_id)
            eval_scaffolds = set(evaluation.scaffold_group)
            for auxiliary_task, task in base.groupby("task_id"):
                conflict_parent = task.parent_id.isin(eval_parents)
                conflict_scaffold = task.scaffold_group.isin(eval_scaffolds)
                retained = task.loc[~conflict_parent & ~conflict_scaffold]
                post_parent = int(retained.parent_id.isin(eval_parents).sum())
                post_scaffold = int(retained.scaffold_group.isin(eval_scaffolds).sum())
                rows.append(
                    {
                        "downstream_scope": scope,
                        "outer_fold": fold,
                        "auxiliary_task": auxiliary_task,
                        "evaluation_parents": evaluation.parent_id.nunique(),
                        "evaluation_scaffolds": evaluation.scaffold_group.nunique(),
                        "records_before_fold_purge": len(task),
                        "parents_before_fold_purge": task.parent_id.nunique(),
                        "scaffolds_before_fold_purge": task.scaffold_group.nunique(),
                        "records_after_fold_purge": len(retained),
                        "parents_after_fold_purge": retained.parent_id.nunique(),
                        "scaffolds_after_fold_purge": retained.scaffold_group.nunique(),
                        "record_retention_fraction": len(retained) / len(task),
                        "post_filter_parent_overlap": post_parent,
                        "post_filter_scaffold_overlap": post_scaffold,
                        "scale_gate_passed": retained.parent_id.nunique() >= MIN_PARENTS_AFTER_PURGE
                        and retained.scaffold_group.nunique() >= MIN_SCAFFOLDS_AFTER_PURGE,
                        "isolation_gate_passed": post_parent == 0 and post_scaffold == 0,
                    }
                )
    return pd.DataFrame(rows)


def task_distribution(records: pd.DataFrame, duplicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_id, group in records.groupby("task_id"):
        values = group.target_value
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        duplicate_task = duplicates.loc[duplicates.task_id.eq(task_id)]
        rows.append(
            {
                "task_id": task_id,
                "source_train_records": len(group),
                "valid_structure_records": int(group.standardization_error.eq("").sum()),
                "structure_failure_records": int(group.standardization_error.ne("").sum()),
                "unique_parents": group.loc[group.standardization_error.eq(""), "parent_id"].nunique(),
                "unique_scaffolds": group.loc[group.standardization_error.eq(""), "scaffold_group"].nunique(),
                "future_clean_pool_records": int(group.future_clean_pool.sum()),
                "future_clean_pool_parents": group.loc[group.future_clean_pool, "parent_id"].nunique(),
                "future_clean_pool_scaffolds": group.loc[group.future_clean_pool, "scaffold_group"].nunique(),
                "target_min": values.min(),
                "target_q01": values.quantile(0.01),
                "target_median": values.median(),
                "target_q99": values.quantile(0.99),
                "target_max": values.max(),
                "robust_3iqr_outlier_records": int(((values < low) | (values > high)).sum()),
                "repeated_parents": int(duplicate_task.is_repeated_parent.sum()),
                "high_conflict_duplicate_parents": int(duplicate_task.high_conflict_duplicate.sum()),
                "high_conflict_threshold_log_units": DUPLICATE_CONFLICT_RANGE,
            }
        )
    return pd.DataFrame(rows)


def build_figure(folds: pd.DataFrame) -> plt.Figure:
    summary = folds.groupby("auxiliary_task", as_index=False).agg(
        clean_parents=("parents_before_fold_purge", "max"),
        worst_fold_parents=("parents_after_fold_purge", "min"),
        clean_scaffolds=("scaffolds_before_fold_purge", "max"),
        worst_fold_scaffolds=("scaffolds_after_fold_purge", "min"),
    )
    labels = ["Experimental logD\n(pH 7.4)" if "logD" in task else "Experimental logP" for task in summary.auxiliary_task]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    colors = ["#4C78A8", "#F58518"]
    for axis, clean, worst, threshold, title, ylabel in [
        (axes[0], "clean_parents", "worst_fold_parents", MIN_PARENTS_AFTER_PURGE, "Parent-level feasibility", "Unique parents"),
        (axes[1], "clean_scaffolds", "worst_fold_scaffolds", MIN_SCAFFOLDS_AFTER_PURGE, "Scaffold-level feasibility", "Unique scaffolds"),
    ]:
        axis.bar(x - 0.18, summary[clean], width=0.36, color=colors[0], label="Clean audit pool")
        axis.bar(x + 0.18, summary[worst], width=0.36, color=colors[1], label="Worst outer-fold purge")
        axis.axhline(threshold, color="#C44E52", linestyle="--", linewidth=1.2, label=f"Gate threshold ({threshold:,})")
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    labels_legend[0] = "Gate thresholds (1,000 parents; 100 scaffolds)"
    fig.legend(handles, labels_legend, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Leakage-safe auxiliary physchem data remain viable after protected purging", fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Measured logP/logD values remain auxiliary labels only; downstream PK rows may receive nested predictions only.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.82])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/OneADMET/data/raw/oneADMET.csv")
    parser.add_argument("--oneadmet", type=Path, default=ROOT / "data/external/oneadmet_auxiliary_v2")
    parser.add_argument("--gate2b", type=Path, default=ROOT / "data/public_development/gate2b_decision_freeze_gate3_audit_v1")
    parser.add_argument("--stl", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--project-splits", type=Path, default=ROOT / "data/processed_v15/splits/split_manifest.csv")
    parser.add_argument("--project-tasks", type=Path, default=ROOT / "data/processed_v15/datasets/task_records.csv")
    parser.add_argument("--project-source", type=Path, default=ROOT / "data/processed_v15/audit/source_records.csv")
    parser.add_argument("--boulton", type=Path, default=ROOT / "data/literature/absolute_f_increment_view_v1/increment_records.csv")
    parser.add_argument("--public-f", type=Path, default=ROOT / "data/literature/public_f_B2_increment_view_v2/B2_increment_records.csv")
    parser.add_argument("--external-decisions", type=Path, default=ROOT / "results/analysis/thalf_external_validation_decisions_v24.csv")
    parser.add_argument("--pkdb-candidates", type=Path, default=ROOT / "results/analysis/pkdb_external_review_batch_v2/candidate_records_blinded.csv")
    parser.add_argument("--pubmed-candidates", type=Path, default=ROOT / "results/analysis/pubmed_external_independent_batch_v3/candidate_records_blinded.csv")
    parser.add_argument("--pksmart-mapping", type=Path, default=ROOT / "results/analysis/pksmart_external_chembl_mapping_v2/candidate_mapping_blinded.csv")
    parser.add_argument("--leakage", type=Path, default=ROOT / "results/analysis/endpoint_expansion_feasibility_v1/physchem_feature_leakage_exclusions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/physchem_auxiliary_feasibility_audit_v3")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    required = [
        args.raw,
        args.oneadmet / "complete.json",
        args.oneadmet / "task_registry.csv",
        args.gate2b / "complete.json",
        args.gate2b / "gate3_candidate_registry.csv",
        args.stl / "complete.json",
        args.stl / "benchmark_train_records.csv",
        args.project_splits,
        args.project_tasks,
        args.project_source,
        args.boulton,
        args.public_f,
        args.external_decisions,
        args.pkdb_candidates,
        args.pubmed_candidates,
        args.pksmart_mapping,
        args.leakage,
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    oneadmet_meta = verify_stage(args.oneadmet, "oneadmet_auxiliary")
    gate_meta = verify_stage(args.gate2b, "gate2b_decision_freeze_gate3_audit")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if sha256(args.raw) != oneadmet_meta["inputs"]["raw_sha256"]:
        raise ValueError("Raw OneADMET file differs from the frozen audited source")
    gate3 = pd.read_csv(args.gate2b / "gate3_candidate_registry.csv")
    if len(gate3) != 1 or not bool(gate3.iloc[0].protocol_construction_authorized):
        raise ValueError("Gate 3 protocol-only authorization is absent")
    if bool(gate3.iloc[0].cohort_construction_authorized) or bool(gate3.iloc[0].model_training_authorized):
        raise ValueError("This audit cannot run after cohort/training was pre-authorized")
    if gate_meta.get("validation_labels_read") or gate_meta.get("test_labels_read") or stl_meta.get("test_labels_read"):
        raise ValueError("An upstream stage violates the closed-target lifecycle")
    if args.check_only:
        print("Physchem auxiliary feasibility inputs valid; no target scan performed.")
        return

    registry = pd.read_csv(args.oneadmet / "task_registry.csv")
    for raw_name in RAW_TASKS.values():
        if len(registry.loc[registry.task_name.eq(raw_name)]) != 1:
            raise ValueError(f"Missing or duplicate frozen OneADMET task: {raw_name}")
    records, source_rows = read_source_train_records(args.raw, registry, args.chunk_size)
    records, structures = normalize_structures(records)
    duplicates = duplicate_audit(records)
    protected = protected_reference_registry(
        args.project_splits,
        args.project_tasks,
        args.boulton,
        args.public_f,
        args.external_decisions,
        args.project_source,
        args.pkdb_candidates,
        args.pubmed_candidates,
        args.pksmart_mapping,
    )
    records = add_record_audit_flags(records, duplicates, protected)
    protected_audit = protected_overlap_audit(records, protected)
    folds = outer_fold_audit(records, args.stl / "benchmark_train_records.csv")
    distribution = task_distribution(records, duplicates)

    task_rows = []
    for task_id, raw_name in RAW_TASKS.items():
        frozen = registry.loc[registry.task_name.eq(raw_name)].iloc[0]
        task_rows.append(
            {
                "task_id": task_id,
                "raw_task_name": raw_name,
                "definition": TASK_DEFINITIONS[task_id]["definition"],
                "condition_semantics": TASK_DEFINITIONS[task_id]["condition"],
                "canonical_unit": "log10_coefficient",
                "source_doi": oneadmet_meta["source_doi"],
                "source_train_records": int(frozen.train_measurements),
                "source_test_records_reserved": int(frozen.test_measurements),
                "per_record_doi_or_assay_available": False,
                "reliability_tier": "R3_public_aggregate_per_record_provenance_unavailable",
                "source_test_targets_published_or_used": False,
            }
        )
    task_definitions = pd.DataFrame(task_rows)

    leakage = pd.read_csv(args.leakage)
    expected_features = {"MolLogP", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", *{f"SlogP_VSA{i}" for i in range(1, 13)}}
    if not expected_features <= set(leakage.excluded_feature):
        raise ValueError("Frozen leakage exclusions do not cover all direct computed-logP descriptors")
    leakage["audit_status"] = "required_exclusion_confirmed"

    cross_task_parents = set(records.loc[records.task_id.eq("experimental_logP"), "parent_id"]) & set(
        records.loc[records.task_id.eq("experimental_logD_pH7_4"), "parent_id"]
    )
    cross_task = pd.DataFrame(
        [
            {
                "left_task": "experimental_logP",
                "right_task": "experimental_logD_pH7_4",
                "shared_parents": len(cross_task_parents - {""}),
                "policy": "retain_separate_heads; shared identity is audited but labels are never pooled",
            }
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "limitation_id": "P1",
                "severity": "material_conditional",
                "finding": "OneADMET supplies a dataset DOI and task-level labels but no per-record DOI/assay provenance for these two tasks.",
                "consequence": "Cannot claim individually source-verified experimental labels or perform source-cluster holdout.",
                "required_mitigation": "Use R3 reliability, report the limitation, retain the source-provided split, and run a future source-enriched sensitivity if mapping becomes available.",
            },
            {
                "limitation_id": "P2",
                "severity": "controlled",
                "finding": "logP neutral-state and logD pH 7.4 semantics are available at task level only.",
                "consequence": "Per-record temperature, method, buffer and ionization context cannot be modeled.",
                "required_mitigation": "Keep tasks separate and never reinterpret logD as logP.",
            },
        ]
    )

    structure_failure_fraction = float(records.standardization_error.ne("").mean())
    numeric_gate = bool(np.isfinite(records.target_value).all())
    count_gate = all(
        len(records.loc[records.task_id.eq(task)])
        == int(registry.loc[registry.task_name.eq(raw_name), "train_measurements"].iloc[0])
        for task, raw_name in RAW_TASKS.items()
    )
    structure_gate = structure_failure_fraction <= MAX_STRUCTURE_FAILURE_FRACTION
    isolation_gate = bool(folds.isolation_gate_passed.all())
    scale_gate = bool(folds.scale_gate_passed.all())
    protected_removal_gate = not records.loc[records.future_clean_pool, [
        "overlap_any_protected_parent", "overlap_any_protected_scaffold"
    ]].any().any()
    hard_pass = numeric_gate and count_gate and structure_gate and isolation_gate and scale_gate and protected_removal_gate
    decision = "CONDITIONAL_PASS" if hard_pass else "STOP"
    gate = pd.DataFrame(
        [
            ("G1_frozen_counts", count_gate, "source-train counts exactly match the frozen registry"),
            ("G2_numeric_finite", numeric_gate, "all published audit labels are finite numeric source-train values"),
            ("G3_structure_quality", structure_gate, f"structure failures <= {MAX_STRUCTURE_FAILURE_FRACTION:.0%}"),
            ("G4_protected_exclusion", protected_removal_gate, "future clean pool has zero protected parent/scaffold overlap"),
            ("G5_outer_fold_isolation", isolation_gate, "zero parent/scaffold overlap after every endpoint/fold purge"),
            ("G6_worst_fold_scale", scale_gate, f">={MIN_PARENTS_AFTER_PURGE} parents and >={MIN_SCAFFOLDS_AFTER_PURGE} scaffolds in every audit cell"),
            ("G7_per_record_provenance", False, "material limitation: per-record DOI/assay provenance unavailable"),
        ],
        columns=["gate_id", "passed", "criterion"],
    )
    gate["overall_decision"] = decision
    gate["cohort_construction_authorized"] = False
    gate["model_training_authorized"] = False

    summary = {
        "decision": decision,
        "source_train_records": int(len(records)),
        "source_test_rows_membership_observed": source_rows["test"],
        "source_test_target_columns_share_opened_wide_container": True,
        "source_test_target_values_used": 0,
        "source_test_target_values_published": False,
        "fixed_validation_or_test_targets_read": False,
        "structure_failure_fraction": structure_failure_fraction,
        "protected_reference_members": int(protected.member_id.nunique()),
        "outer_fold_audit_cells": int(len(folds)),
        "minimum_parents_after_purge": int(folds.parents_after_fold_purge.min()),
        "minimum_scaffolds_after_purge": int(folds.scaffolds_after_fold_purge.min()),
        "high_conflict_duplicate_parents": int(duplicates.high_conflict_duplicate.sum()),
        "per_record_provenance_available": False,
        "cohort_construction_authorized": False,
        "model_training_authorized": False,
        "next_stage_if_conditional_pass": "freeze_Gate3_B6_protocol_then_build_versioned_train_only_physchem_cohort",
    }
    figure = build_figure(folds)
    input_hashes = {str(path.resolve()): sha256(path) for path in required}
    with stage_output(args.output) as out:
        task_definitions.to_csv(out / "task_definition_registry.csv", index=False)
        records.sort_values(["task_id", "source_row"]).to_csv(out / "train_record_audit.csv", index=False)
        structures.loc[structures.standardization_error.ne("")].to_csv(out / "structure_failures.csv", index=False)
        distribution.to_csv(out / "target_distribution_audit.csv", index=False)
        duplicates.sort_values(["task_id", "parent_id"]).to_csv(out / "parent_duplicate_conflict_audit.csv", index=False)
        cross_task.to_csv(out / "cross_task_overlap_summary.csv", index=False)
        protected.to_csv(out / "protected_reference_registry.csv", index=False)
        protected_audit.to_csv(out / "protected_overlap_audit.csv", index=False)
        folds.to_csv(out / "outer_fold_purge_feasibility.csv", index=False)
        leakage.to_csv(out / "descriptor_leakage_registry.csv", index=False)
        provenance.to_csv(out / "provenance_limitations.csv", index=False)
        gate.to_csv(out / "gate_decision.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Physchem auxiliary feasibility audit v1\n\n"
            "Train-side OneADMET experimental logP/logD audit only. This directory is not an authorized training cohort. "
            "Source-test target values, fixed validation/test targets and external-reserve labels are not published or used. "
            "A conditional pass permits protocol freezing only; cohort construction and model training require new stages.\n",
            encoding="utf-8",
        )
        figure.savefig(out / "Figure_physchem_auxiliary_feasibility.png", dpi=600, bbox_inches="tight")
        plt.close(figure)
        finish_stage(
            out,
            "physchem_auxiliary_feasibility_audit",
            inputs=input_hashes,
            partial=False,
            validation_labels_read=False,
            test_labels_read=False,
            source_test_target_values_used=0,
            source_test_target_values_published=False,
            source_test_target_columns_share_opened_wide_container=True,
            decision=decision,
            cohort_construction_authorized=False,
            model_training_authorized=False,
        )
    print(f"Physchem auxiliary feasibility audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
