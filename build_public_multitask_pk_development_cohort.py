#!/usr/bin/env python3
"""Publish a versioned, leakage-guarded 24-hour public PK development cohort.

The cohort is a long table.  It deliberately keeps endpoint, species, and
experimental-system tasks separate: auxiliary animal and published OneADMET
records may improve a shared representation, but are never relabelled as human
primary supervision or formal external validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


HUMAN_PRIMARY = [
    "fu__human__plasma", "CLint__human__microsome", "Papp__human__caco2_ab",
    "F__human__absolute_oral", "CL__human__systemic_iv",
    "VDss__human__steady_state_iv", "Thalf__human__terminal_iv",
]
ENDPOINTS = {"fu", "CLint", "Papp", "F", "CL", "VDss", "Thalf"}
BASE_COLUMNS = (
    "row_id", "molecule_id", "smiles", "split", "task_id", "endpoint", "species", "system",
    "canonical_unit", "target_value", "mask", "assay_id", "doc_id", "route",
)
STRUCTURE_COLUMNS = ("source_molecule_id", "parent_id", "scaffold_group", "eligible")
POLICY_COLUMNS = ("evidence_quality_tier", "translation_role")
REQUIRED = set(BASE_COLUMNS)
STRUCTURE_FIELDS = set(STRUCTURE_COLUMNS)
POLICY_FIELDS = set(POLICY_COLUMNS)


def read_project_records(datasets: Path, split_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_required = {
        "molecule_id", "parent_id", "scaffold_group", "split", "eligible", "exclusion_reason",
    }
    if not manifest_required <= set(split_manifest.columns):
        raise ValueError(f"Project split manifest missing: {sorted(manifest_required - set(split_manifest.columns))}")
    if split_manifest.molecule_id.duplicated().any():
        raise ValueError("Project split manifest must uniquely assign source molecule IDs")
    parts = []
    for path in sorted((datasets / "tasks").glob("*.csv")):
        # Avoid opening unrelated historical tasks at all.  Besides making the
        # scope explicit, Python's parser is more robust for archived ChEMBL
        # CSV fields containing legacy quoting than the environment C parser.
        if path.stem.split("__", 1)[0] not in ENDPOINTS:
            continue
        frame = pd.read_csv(path, engine="python", dtype=str, keep_default_na=False)
        if not REQUIRED <= set(frame.columns):
            raise ValueError(f"Task schema mismatch: {path}")
        if not set(frame.endpoint) <= ENDPOINTS:
            continue
        frame["mask"] = pd.to_numeric(frame["mask"], errors="raise")
        frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
        frame = frame.loc[frame["mask"].eq(1) & frame.target_value.notna()].copy()
        frame = frame.loc[frame.split.isin(["train", "val"])].copy()
        if not np.isfinite(frame["target_value"]).all():
            raise ValueError(f"Non-finite target in {path}")
        is_human_primary = frame.task_id.isin(HUMAN_PRIMARY)
        is_animal_support = (~is_human_primary) & frame.endpoint.isin(ENDPOINTS) & ~frame.species.eq("human")
        frame = frame.loc[is_human_primary | is_animal_support].copy()
        frame["cohort_source"] = "project_curated_v15"
        frame["source_task_name"] = frame.task_id
        frame["record_role"] = np.where(
            is_human_primary.loc[frame.index] & frame.split.eq("train"), "human_primary_train",
            np.where(is_human_primary.loc[frame.index], "human_primary_validation",
                     np.where(frame.split.eq("train"), "animal_transfer_train", "animal_transfer_validation")),
        )
        frame["evidence_role"] = np.where(is_human_primary.loc[frame.index], "strict_A_curated_development", "auxiliary_C_cross_species")
        # Evidence quality and translational distance are deliberately separate.
        # Curated animal data do not become low-quality merely because they are
        # auxiliary to a human prediction target.
        frame["evidence_quality_tier"] = "R1_project_curated"
        frame["reliability_tier"] = frame["evidence_quality_tier"]
        frame["translation_role"] = np.where(is_human_primary.loc[frame.index], "human_primary_direct", "cross_species_auxiliary")
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if set(result.split) - {"train", "val"}:
        raise ValueError("Historical project test records must never enter this cohort")
    structure = split_manifest[[
        "molecule_id", "parent_id", "scaffold_group", "split", "eligible", "exclusion_reason",
    ]].rename(columns={"split": "manifest_split", "molecule_id": "source_molecule_id"})
    result = result.rename(columns={"molecule_id": "source_molecule_id"}).merge(
        structure, on="source_molecule_id", how="left", validate="many_to_one",
    )
    if result.parent_id.isna().any() or result.manifest_split.isna().any():
        raise ValueError("Project task rows are not fully covered by the frozen split manifest")
    if not result.split.eq(result.manifest_split).all():
        raise ValueError("Project task rows disagree with the frozen split manifest")
    result["eligible"] = result.eligible.astype(str).str.lower().eq("true")
    ineligible = result.loc[~result.eligible].copy()
    ineligible["quarantine_reason"] = "project_manifest_ineligible:" + ineligible.exclusion_reason.replace("", "unspecified")
    result = result.loc[result.eligible].copy()
    result["molecule_id"] = result.parent_id
    return result.drop(columns=["manifest_split", "exclusion_reason"]), ineligible


def read_oneadmet_records(splits: Path) -> pd.DataFrame:
    raw = pd.read_csv(splits / "pk_records_split.csv", dtype=str, keep_default_na=False)
    required = {"SMILES", "Set", "task_name", "target_value", "source_record_id", "parent_id", "scaffold_group",
                "strict_aux_split", "allowed_use", "target_scale"}
    if not required <= set(raw.columns):
        raise ValueError("OneADMET split schema mismatch")
    raw["target_value"] = pd.to_numeric(raw["target_value"], errors="coerce")
    raw = raw.loc[raw.allowed_use.eq("auxiliary_training") & raw.strict_aux_split.isin(["aux_train", "aux_val"])].copy()
    if raw.empty or raw.target_value.isna().any() or not np.isfinite(raw.target_value).all():
        raise ValueError("No valid OneADMET auxiliary records")
    family = np.select(
        [raw.task_name.str.startswith("Clearance-"), raw.task_name.str.startswith("Half-Life_")],
        ["CL_related", "Thalf_related"], default="other",
    )
    raw = raw.loc[family != "other"].copy()
    raw["row_id"] = raw.source_record_id.map(lambda x: f"oneadmet:{x}")
    raw["molecule_id"] = raw.parent_id
    raw["source_molecule_id"] = raw.parent_id
    raw["smiles"] = raw.SMILES
    raw["split"] = raw.strict_aux_split.map({"aux_train": "train", "aux_val": "val"})
    raw["task_id"] = "OneADMET__" + raw.task_name.str.replace(".csv", "", regex=False)
    raw["endpoint"] = family[family != "other"]
    raw["species"] = raw.task_name.str.extract(
        r"(Human|Mouse|Rat|Dog|Monkey|Rabbit|Minipig|GuineaPig)", expand=False
    ).str.lower().fillna("unknown")
    raw["system"] = "published_oneadmet_task"
    raw["canonical_unit"] = raw.target_scale
    raw["mask"] = 1
    raw["assay_id"] = ""
    raw["doc_id"] = "10.1021/acs.jmedchem.6c00049"
    raw["route"] = "as_published"
    raw["source_task_name"] = raw.task_name
    raw["cohort_source"] = "oneadmet_published_v2"
    raw["record_role"] = np.where(raw.split.eq("train"), "published_auxiliary_train", "published_auxiliary_validation")
    raw["evidence_role"] = "public_B_published_auxiliary"
    raw["evidence_quality_tier"] = "R3_published_aggregate_provenance_pending"
    raw["reliability_tier"] = raw["evidence_quality_tier"]
    raw["translation_role"] = np.where(
        raw.species.eq("human"), "published_auxiliary_same_species", "published_auxiliary_cross_species",
    )
    raw["eligible"] = True
    keep = [*BASE_COLUMNS, *STRUCTURE_COLUMNS, "source_task_name", "cohort_source", "record_role",
            "evidence_role", "reliability_tier", *POLICY_COLUMNS]
    return raw[keep]


def build_registry(records: pd.DataFrame) -> pd.DataFrame:
    keys = ["task_id", "endpoint", "species", "system", "canonical_unit", "cohort_source", "evidence_role", "reliability_tier"]
    summary = (records.groupby(keys, dropna=False)
               .agg(development_records=("row_id", "size"), development_molecules=("molecule_id", "nunique"),
                    sources=("doc_id", "nunique"))
               .reset_index())
    split_counts = records.groupby([*keys, "split"], dropna=False).size().unstack("split", fill_value=0)
    split_counts = split_counts.reindex(columns=["train", "val"], fill_value=0)
    split_counts = split_counts.rename(columns={"train": "train_records", "val": "validation_records"}).reset_index()
    summary = summary.merge(split_counts, on=keys, validate="one_to_one")
    summary["head_selection_authorized"] = (
        summary.evidence_role.eq("strict_A_curated_development")
        & summary.train_records.ge(30) & summary.validation_records.ge(10)
    )
    summary["allowed_use"] = np.where(summary.head_selection_authorized, "development_train_and_validate", "representation_or_sensitivity_only")
    return summary.sort_values(["evidence_role", "endpoint", "species", "task_id"], ignore_index=True)


def quarantine_cross_source_conflicts(project: pd.DataFrame, published: pd.DataFrame,
                                      fixed_splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep auxiliary structures out of any differently assigned project split.

    Same-split auxiliary records are useful multi-task supervision.  A public
    auxiliary label for a project validation/test structure is not discarded:
    it is retained in a quarantine artifact with its reason.
    """
    if fixed_splits.molecule_id.duplicated().any() or set(fixed_splits.split) - {"train", "val", "test"}:
        raise ValueError("Project split manifest must uniquely assign every molecule")
    if (project.groupby("molecule_id").split.nunique() > 1).any():
        raise ValueError("Project development records disagree on global split assignment")
    # Backward-compatible fallback keeps the unit-testable helper useful, while
    # production cohorts always use canonical parent and scaffold keys.
    structural = {"parent_id", "scaffold_group", "eligible"} <= set(fixed_splits.columns)
    if structural:
        eligible = fixed_splits.eligible.astype(str).str.lower().eq("true")
        protected = fixed_splits.loc[eligible, ["parent_id", "scaffold_group", "split"]].copy()
        for key in ["parent_id", "scaffold_group"]:
            if (protected.groupby(key).split.nunique() > 1).any():
                raise ValueError(f"Frozen project manifest assigns one {key} to multiple splits")
        parent_map = protected[["parent_id", "split"]].drop_duplicates().rename(columns={"split": "project_parent_split"})
        scaffold_map = protected[["scaffold_group", "split"]].drop_duplicates().rename(columns={"split": "project_scaffold_split"})
        joined = published.merge(parent_map, on="parent_id", how="left", validate="many_to_one")
        joined = joined.merge(scaffold_map, on="scaffold_group", how="left", validate="many_to_one")
    else:
        mapping = fixed_splits[["molecule_id", "split"]].rename(columns={"split": "project_parent_split"})
        joined = published.merge(mapping, on="molecule_id", how="left", validate="many_to_one")
        joined["project_scaffold_split"] = ""
    joined["quarantine_reason"] = ""
    test_overlap = joined.project_parent_split.eq("test") | joined.project_scaffold_split.eq("test")
    joined.loc[test_overlap, "quarantine_reason"] = "overlap_project_test_reserved_parent_or_scaffold"
    parent_conflict = joined.project_parent_split.isin(["train", "val"]) & joined.split.ne(joined.project_parent_split)
    scaffold_conflict = joined.project_scaffold_split.isin(["train", "val"]) & joined.split.ne(joined.project_scaffold_split)
    joined.loc[~test_overlap & parent_conflict, "quarantine_reason"] = "cross_source_parent_split_conflict"
    joined.loc[~test_overlap & ~parent_conflict & scaffold_conflict, "quarantine_reason"] = "cross_source_scaffold_split_conflict"
    quarantined = joined.loc[joined.quarantine_reason.ne("")].copy()
    kept = joined.loc[joined.quarantine_reason.eq("")].drop(
        columns=["project_parent_split", "project_scaffold_split", "quarantine_reason"]
    )
    for key in ["molecule_id", *( ["scaffold_group"] if "scaffold_group" in kept else [])]:
        combined = pd.concat([project[[key, "split"]], kept[[key, "split"]]], ignore_index=True)
        if (combined.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Combined cohort retains a {key} in more than one split")
    return kept, quarantined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--project-splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--oneadmet-splits", type=Path, default=ROOT / "data/external/oneadmet_pk_splits_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/multitask_pk_24h_v6")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check([args.datasets / "complete.json", args.project_splits / "complete.json", args.oneadmet_splits / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.datasets, "datasets")
    verify_stage(args.project_splits, "splits")
    verify_stage(args.oneadmet_splits, "oneadmet_pk_safe_splits")
    fixed_splits = pd.read_csv(args.project_splits / "split_manifest.csv", dtype=str, keep_default_na=False)
    project, project_quarantine = read_project_records(args.datasets, fixed_splits)
    published, quarantined = quarantine_cross_source_conflicts(
        project, read_oneadmet_records(args.oneadmet_splits),
        fixed_splits,
    )
    records = pd.concat([project, published], ignore_index=True)
    records = records[[*BASE_COLUMNS, *STRUCTURE_COLUMNS, "source_task_name", "cohort_source", "record_role",
                       "evidence_role", "reliability_tier", *POLICY_COLUMNS]]
    if records.row_id.duplicated().any() or set(records.split) - {"train", "val"}:
        raise ValueError("Cohort contains duplicate rows or a forbidden split")
    for key in ["molecule_id", "parent_id", "scaffold_group"]:
        if (records.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Cohort contains cross-split {key} overlap")
    if not records.molecule_id.eq(records.parent_id).all() or not records.eligible.astype(bool).all():
        raise ValueError("Cohort canonical structure identity or eligibility invariant failed")
    registry = build_registry(records)
    human_f = registry.loc[registry.task_id.eq("F__human__absolute_oral")].iloc[0]
    if human_f.validation_records != 0 or bool(human_f.head_selection_authorized):
        raise ValueError("Human F must remain blocked without a validation cohort")
    summary = {
        "records": len(records), "molecules": int(records.molecule_id.nunique()), "tasks": int(records.task_id.nunique()),
        "cross_source_auxiliary_records_quarantined": len(quarantined),
        "cross_source_auxiliary_molecules_quarantined": int(quarantined.molecule_id.nunique()),
        "project_ineligible_records_quarantined": len(project_quarantine),
        "cross_split_parent_overlap": 0, "cross_split_scaffold_overlap": 0,
        "canonical_parent_is_molecule_id": True,
        "included_splits": ["train", "val"], "historical_tests_included": False,
        "human_f_head_selection_authorized": False, "formal_external_validation_allowed": False,
        "final_model_release_authorized": False,
        "workflow": "masked long-table MTL; task/species/system-specific heads; auxiliary tasks representation-only unless explicitly authorized",
    }
    if args.check_only:
        print(f"Public multitask PK cohort valid: records={summary['records']} tasks={summary['tasks']}")
        return
    with stage_output(args.output) as out:
        records.sort_values(["cohort_source", "task_id", "split", "row_id"]).to_csv(out / "development_records_long.csv", index=False)
        registry.to_csv(out / "task_registry.csv", index=False)
        quarantined.sort_values(["quarantine_reason", "molecule_id", "task_id", "row_id"]).to_csv(
            out / "cross_source_auxiliary_quarantine.csv", index=False
        )
        project_quarantine.sort_values(["quarantine_reason", "task_id", "row_id"]).to_csv(
            out / "project_manifest_ineligible_quarantine.csv", index=False
        )
        (out / "README.md").write_text(
            "# 24-hour public multitask PK development cohort\n\n"
            "A versioned, long-form training input assembled from eligible project-curated v15 development records and published OneADMET auxiliary tasks. "
            "All historical tests, OneADMET author tests, quarantined structures, and overlap-audit-only records are excluded. "
            "Separate quarantine tables preserve ineligible project rows and published auxiliary rows whose canonical parent or scaffold conflicts with a project train, validation, or test split. "
            "Canonical parent IDs are the global molecule identity; source molecule IDs remain available for provenance. Evidence quality and human-translation role are separate fields. "
            "Task heads remain specific to endpoint/species/system. Cross-species and OneADMET records are representation or sensitivity support, never human-label substitutions. "
            "The human absolute-F head has no validation records and is explicitly blocked from model selection.\n",
            encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finish_stage(out, "public_multitask_pk_development_cohort", inputs={
            "processed_v15_datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "oneadmet_splits_complete_sha256": sha256(args.oneadmet_splits / "complete.json"),
            "project_splits_complete_sha256": sha256(args.project_splits / "complete.json"),
        }, **summary, partial=False)
    print(f"Public multitask PK development cohort: {args.output}")


if __name__ == "__main__":
    run_cli(main)
