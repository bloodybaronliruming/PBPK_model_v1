#!/usr/bin/env python3
"""Merge public-F B2 locks into a reliability-aware, triple-isolated increment view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "F__human__absolute_oral"
GATE_COLUMNS = [
    "overlap_strict_f_parent", "overlap_strict_f_scaffold", "overlap_strict_f_source",
    "overlap_boulton_parent", "overlap_boulton_scaffold", "overlap_boulton_source",
]


def normalise_dois(values: pd.Series) -> set[str]:
    return {str(value).strip().lower() for value in values.fillna("") if str(value).strip() and str(value).lower() != "nan"}


def strict_members(records: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    members = records.loc[records.task_id.eq(TASK), ["molecule_id"]].drop_duplicates()
    members = members.merge(manifest[["molecule_id", "parent_id", "scaffold_group"]], on="molecule_id", how="left", validate="many_to_one")
    if members[["parent_id", "scaffold_group"]].isna().any(axis=None):
        raise ValueError("Strict-F members lack structure groups")
    return members


def prepare_existing(records: pd.DataFrame) -> pd.DataFrame:
    required = {"request_id", "analyte", "canonical_smiles", "parent_id", "scaffold_group", "source_doi", "source_pmid",
                "source_locator", "reported_statistic", "canonical_unit", "canonical_value_fraction", "canonical_estimator_rule"}
    if not required <= set(records.columns):
        raise ValueError("Existing public-F cohort view schema mismatch")
    frame = records.copy()
    frame = frame.assign(
        record_id="pkdb:" + frame.request_id.astype(str), record_origin="PKDB_B2_lock_v1",
        reliability_tier="R1_primary_locked", evidence_role="public_B",
        condition_policy="source_locked_study_level_estimator", variant_registry_status="not_applicable",
    )
    return frame


def prepare_increment(extractions: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    required = {"doc_id", "extraction_decision", "source_doi", "source_pmid", "source_locator", "analyte", "reported_statistic",
                "canonical_unit", "canonical_value_fraction", "canonical_estimator_rule", "reliability_tier"}
    registry_required = {"doc_id", "molecule_id", "smiles", "parent_id", "scaffold_group", "public_status"}
    if not required <= set(extractions.columns) or not registry_required <= set(registry.columns):
        raise ValueError("Increment extraction or candidate registry schema mismatch")
    frame = extractions.loc[extractions.extraction_decision.eq("accepted")].copy()
    frame["doc_id"] = frame.doc_id.astype(str)
    candidates = registry.loc[registry.doc_id.astype(str).isin(frame.doc_id)].copy()
    candidates["doc_id"] = candidates.doc_id.astype(str)
    summary = candidates.groupby("doc_id", as_index=False).agg(
        candidate_records=("molecule_id", "size"), parent_id=("parent_id", "first"), scaffold_group=("scaffold_group", "first"),
        canonical_smiles=("smiles", "first"), public_statuses=("public_status", lambda values: ";".join(sorted(set(values)))),
        parent_count=("parent_id", "nunique"), scaffold_count=("scaffold_group", "nunique"), smiles_count=("smiles", "nunique"),
    )
    if not summary[["parent_count", "scaffold_count", "smiles_count"]].eq(1).all(axis=None):
        raise ValueError("A B2 source must map to exactly one parent/scaffold/structure")
    if not summary.public_statuses.eq("public_candidate_definition_audit").all():
        raise ValueError("Increment source is not a non-overlapping public candidate")
    frame = frame.merge(summary, on="doc_id", how="inner", validate="one_to_one")
    frame = frame.assign(
        record_id="chembl:" + frame.doc_id, record_origin="ChEMBL_B2_lock_v1", evidence_role="public_B",
        condition_policy=frame.extraction_note, variant_registry_status="canonical_variant_only",
    )
    return frame


def apply_reliability_decisions(combined: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    required = {"record_id", "reliability_tier", "decision_reason", "reviewer", "review_date"}
    if not required <= set(decisions.columns):
        raise ValueError("Reliability-decision schema mismatch")
    if decisions.record_id.duplicated().any() or set(decisions.record_id) != set(combined.record_id):
        raise ValueError("Reliability decisions must cover every unified B2 record exactly once")
    if not set(decisions.reliability_tier) <= {"R1_primary_locked", "R2_primary_conditional"}:
        raise ValueError("Unsupported B2 reliability tier")
    decisions = decisions.rename(columns={
        "reliability_tier": "reliability_tier_decision", "decision_reason": "reliability_decision_reason",
        "reviewer": "reliability_reviewer", "review_date": "reliability_review_date",
    })
    result = combined.merge(decisions, on="record_id", how="inner", suffixes=("_derived", "_source"), validate="one_to_one")
    if not result.reliability_tier.eq(result.reliability_tier_decision).all():
        raise ValueError("Reliability decision conflicts with source-locked reliability tier")
    return result.drop(columns=["reliability_tier_decision"])


def build_view(existing: pd.DataFrame, increment: pd.DataFrame, strict_records: pd.DataFrame, strict_manifest: pd.DataFrame,
               strict_sources: pd.DataFrame, boulton: pd.DataFrame, reliability_decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    combined = apply_reliability_decisions(pd.concat([existing, increment], ignore_index=True, sort=False), reliability_decisions)
    if combined.record_id.duplicated().any() or combined.parent_id.duplicated().any() or combined.source_doi.duplicated().any():
        raise ValueError("Unified B2 view requires unique record, parent, and source cluster")
    if not combined.canonical_unit.eq("fraction").all() or combined.canonical_value_fraction.astype(float).le(0).any():
        raise ValueError("Unified B2 view has invalid canonical F values")
    if not combined.evidence_role.eq("public_B").all() or not set(combined.reliability_tier) <= {"R1_primary_locked", "R2_primary_conditional"}:
        raise ValueError("Unified B2 view has an unsupported evidence role or reliability tier")

    strict = strict_members(strict_records, strict_manifest)
    strict_parent, strict_scaffold = set(strict.parent_id), set(strict.scaffold_group)
    strict_dois = normalise_dois(strict_sources.loc[(strict_sources.task_id == TASK) & strict_sources.status.eq("accepted"), "doi"])
    required_boulton = {"parent_id", "scaffold_group", "source_doi"}
    if not required_boulton <= set(boulton.columns):
        raise ValueError("Boulton reserve lacks triple-isolation fields")
    boulton_parent, boulton_scaffold, boulton_dois = set(boulton.parent_id), set(boulton.scaffold_group), normalise_dois(boulton.source_doi)
    combined["source_cluster_id"] = combined.source_doi.astype(str).str.strip().str.lower()
    if combined.source_cluster_id.eq("").any():
        raise ValueError("A B2 record lacks a DOI source cluster")
    combined["overlap_strict_f_parent"] = combined.parent_id.isin(strict_parent)
    combined["overlap_strict_f_scaffold"] = combined.scaffold_group.isin(strict_scaffold)
    combined["overlap_strict_f_source"] = combined.source_cluster_id.isin(strict_dois)
    combined["overlap_boulton_parent"] = combined.parent_id.isin(boulton_parent)
    combined["overlap_boulton_scaffold"] = combined.scaffold_group.isin(boulton_scaffold)
    combined["overlap_boulton_source"] = combined.source_cluster_id.isin(boulton_dois)
    if combined[GATE_COLUMNS].any(axis=None):
        conflicts = combined.loc[combined[GATE_COLUMNS].any(axis=1), ["record_id", *GATE_COLUMNS]]
        raise ValueError(f"B2 record fails triple isolation: {conflicts.to_dict(orient='records')}")
    combined["eligible_for_strict_f_external"] = False
    combined["eligible_for_validation_or_test_assignment"] = False
    combined["cohort_role"] = "public_B_B2_unassigned_pending_B3_readiness"
    columns = [
        "record_id", "record_origin", "evidence_role", "reliability_tier", "analyte", "canonical_smiles", "parent_id", "scaffold_group",
        "source_doi", "source_pmid", "source_cluster_id", "source_locator", "reported_statistic", "canonical_unit", "canonical_value_fraction",
        "canonical_estimator_rule", "condition_policy", "variant_registry_status", "reliability_decision_reason", "reliability_reviewer", "reliability_review_date", *GATE_COLUMNS,
        "eligible_for_strict_f_external", "eligible_for_validation_or_test_assignment", "cohort_role",
    ]
    result = combined[columns].sort_values("record_id", ignore_index=True)
    clusters = result.groupby("source_cluster_id", as_index=False).agg(
        records=("record_id", "size"), parents=("parent_id", "nunique"), reliability_tiers=("reliability_tier", lambda values: ";".join(sorted(set(values)))),
        origins=("record_origin", lambda values: ";".join(sorted(set(values)))), analytes=("analyte", lambda values: ";".join(sorted(values))),
    )
    summary = {
        "records": len(result), "parents": result.parent_id.nunique(), "source_clusters": result.source_cluster_id.nunique(),
        "reliability_tier_counts": result.reliability_tier.value_counts().sort_index().to_dict(),
        "triple_isolated_records": int((~result[GATE_COLUMNS].any(axis=1)).sum()),
        "strict_f_external_eligible_records": 0, "validation_or_test_assignable_records": 0,
        "public_f_training_authorized": False,
    }
    return result, clusters, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-view", type=Path, default=ROOT / "data/literature/public_f_cohort_isolation_view_v1")
    parser.add_argument("--increment-extraction", type=Path,
                        default=ROOT / "results/analysis/public_f_haloperidol_phenobarbital_extraction_lock_v1")
    parser.add_argument("--increment-registry", type=Path,
                        default=ROOT / "results/analysis/public_f_development_candidate_registry_v1")
    parser.add_argument("--reliability-decisions", type=Path,
                        default=ROOT / "data/manual_review/public_f_B2_reliability_decisions_v1.csv")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--boulton-reserve", type=Path, default=ROOT / "data/literature/absolute_f_increment_view_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/literature/public_f_B2_increment_view_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    existing_file = args.existing_view / "cohort_records.csv"
    increment_file = args.increment_extraction / "extraction_registry_locked.csv"
    registry_file = args.increment_registry / "chembl_public_candidates_blinded.csv"
    task_records, sources, manifest = args.datasets / "task_records.csv", args.datasets / "source_long.csv", args.splits / "split_manifest.csv"
    boulton_file = args.boulton_reserve / "increment_records.csv"
    required = [existing_file, args.existing_view / "complete.json", increment_file, args.increment_extraction / "complete.json",
                registry_file, args.increment_registry / "complete.json", args.reliability_decisions, task_records, sources, args.datasets / "complete.json",
                manifest, args.splits / "complete.json", boulton_file, args.boulton_reserve / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.existing_view, "public_f_cohort_isolation_view")
    verify_stage(args.increment_extraction, "public_f_unquarantined_priority_extraction_lock")
    verify_stage(args.increment_registry, "public_f_development_candidate_registration")
    verify_stage(args.datasets, "datasets")
    verify_stage(args.splits, "splits")
    verify_stage(args.boulton_reserve, "absolute_f_increment_view")
    result, clusters, summary = build_view(
        prepare_existing(pd.read_csv(existing_file, dtype=str, keep_default_na=False)),
        prepare_increment(pd.read_csv(increment_file, dtype=str, keep_default_na=False), pd.read_csv(registry_file, dtype=str, keep_default_na=False)),
        pd.read_csv(task_records, usecols=["task_id", "molecule_id"]), pd.read_csv(manifest, usecols=["molecule_id", "parent_id", "scaffold_group"]),
        pd.read_csv(sources, usecols=["task_id", "status", "doi"]), pd.read_csv(boulton_file),
        pd.read_csv(args.reliability_decisions, dtype=str, keep_default_na=False),
    )
    if args.check_only:
        print(f"Public-F B2 increment view valid: parents={summary['parents']} sources={summary['source_clusters']}")
        return
    with stage_output(args.output) as out:
        result.to_csv(out / "B2_increment_records.csv", index=False)
        clusters.to_csv(out / "source_cluster_manifest.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F reliability-aware B2 increment view\n\n"
            "This non-overwriting view combines the prior PK-DB B2 cohort with two newly source-locked ChEMBL-derived B2 records. "
            "It recomputes parent, Murcko-scaffold and DOI/source-cluster separation from frozen strict-F and the Boulton reserve. "
            "Reliability tier is explicit: R1 and R2 records stay distinct, all members remain public-B and unassigned, and no model "
            "training, validation, test assignment, or strict-F update is authorized.\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_B2_increment_view", inputs={
            "existing_view_complete_sha256": sha256(args.existing_view / "complete.json"),
            "increment_extraction_complete_sha256": sha256(args.increment_extraction / "complete.json"),
            "increment_registry_complete_sha256": sha256(args.increment_registry / "complete.json"),
            "reliability_decisions_sha256": sha256(args.reliability_decisions),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"), "splits_complete_sha256": sha256(args.splits / "complete.json"),
            "boulton_reserve_complete_sha256": sha256(args.boulton_reserve / "complete.json"),
        }, **summary, partial=False)
    print(f"Public-F B2 increment view: {args.output}")


if __name__ == "__main__":
    run_cli(main)
