#!/usr/bin/env python3
"""Publish a source- and structure-isolated human absolute-F increment view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_split_manifest import structure_groups
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "F__human__absolute_oral"


def build_view(extractions: pd.DataFrame, eligibility: pd.DataFrame, queue: pd.DataFrame,
               task_records: pd.DataFrame, manifest: pd.DataFrame, source_long: pd.DataFrame):
    accepted = extractions.loc[extractions.extraction_decision.eq("accepted")].copy()
    eligible = eligibility.loc[eligibility.eligibility_decision.eq("accepted"), ["candidate_id", "source_doi"]]
    if accepted.candidate_id.duplicated().any() or not set(accepted.candidate_id) <= set(eligible.candidate_id):
        raise ValueError("Extraction rows must have one accepted eligibility ancestor")
    queue_columns = ["candidate_id", "label", "chembl_id", "canonical_smiles", "parent_id", "scaffold_group"]
    if not set(queue_columns) <= set(queue.columns):
        raise ValueError("Candidate queue lacks structure columns")
    accepted = accepted.merge(eligible, on=["candidate_id", "source_doi"], how="inner", validate="one_to_one")
    accepted = accepted.merge(queue[queue_columns], on="candidate_id", how="inner", validate="one_to_one")
    if len(accepted) != len(extractions):
        raise ValueError("An extraction row lacks a matching queue structure or eligibility source")

    observed_groups = accepted.canonical_smiles.map(structure_groups)
    accepted["recomputed_parent_id"] = [item[0] for item in observed_groups]
    accepted["recomputed_scaffold_group"] = [item[1] for item in observed_groups]
    if not accepted.parent_id.eq(accepted.recomputed_parent_id).all() or not accepted.scaffold_group.eq(accepted.recomputed_scaffold_group).all():
        raise ValueError("Candidate structure grouping changed since queue publication")

    f_members = task_records.loc[task_records.task_id.eq(TASK), ["molecule_id", "split"]].drop_duplicates()
    f_members = f_members.merge(manifest[["molecule_id", "parent_id", "scaffold_group"]], on="molecule_id", how="left", validate="many_to_one")
    if f_members.parent_id.isna().any():
        raise ValueError("Existing F members lack split-manifest grouping")
    f_parents, f_scaffolds = set(f_members.parent_id), set(f_members.scaffold_group)
    f_docs = source_long.loc[(source_long.task_id == TASK) & source_long.status.eq("accepted"), "doi"].fillna("")
    f_dois = set(f_docs.astype(str).str.strip()) - {"", "nan"}
    accepted["overlap_existing_f_parent"] = accepted.parent_id.isin(f_parents)
    accepted["overlap_existing_f_scaffold"] = accepted.scaffold_group.isin(f_scaffolds)
    accepted["overlap_existing_f_source"] = accepted.source_doi.isin(f_dois)
    if accepted[["overlap_existing_f_parent", "overlap_existing_f_scaffold", "overlap_existing_f_source"]].any(axis=None):
        raise ValueError("F increment is not isolated from existing F parent/scaffold/source evidence")

    all_members = task_records[["task_id", "molecule_id", "split"]].drop_duplicates().merge(
        manifest[["molecule_id", "parent_id", "scaffold_group"]], on="molecule_id", how="left", validate="many_to_one")
    cross_tasks = []
    for row in accepted.itertuples(index=False):
        hit = all_members.loc[(all_members.parent_id == row.parent_id) | (all_members.scaffold_group == row.scaffold_group), "task_id"]
        cross_tasks.append(";".join(sorted(set(hit))))
    accepted["overlap_existing_any_endpoint_tasks"] = cross_tasks
    accepted["stl_f_isolated"] = True
    accepted["multitask_or_cascade_evaluation_eligible"] = accepted.overlap_existing_any_endpoint_tasks.eq("")
    accepted["reserved_role"] = "source_isolated_F_evaluation_candidate_pending_larger_source_cluster"
    accepted["source_cluster_id"] = accepted.source_doi
    if accepted.source_cluster_id.nunique() != 1:
        raise ValueError("This v1 increment view must preserve its single-source cluster")

    columns = [
        "candidate_id", "label", "chembl_id", "canonical_smiles", "parent_id", "scaffold_group", "source_doi", "source_pmid",
        "source_locator", "study_population", "oral_dose", "iv_microdose", "reported_statistic", "reported_value_percent",
        "ci_level", "ci_lower_percent", "ci_upper_percent", "canonical_unit", "canonical_value_fraction", "transformation",
        "source_pdf_sha256", "overlap_existing_f_parent", "overlap_existing_f_scaffold", "overlap_existing_f_source",
        "overlap_existing_any_endpoint_tasks", "stl_f_isolated", "multitask_or_cascade_evaluation_eligible", "reserved_role", "source_cluster_id",
    ]
    result = accepted[columns].sort_values("candidate_id", ignore_index=True)
    summary = {
        "records": len(result), "molecules": result.parent_id.nunique(), "source_clusters": result.source_cluster_id.nunique(),
        "existing_f_molecules_checked": f_members.molecule_id.nunique(), "stl_f_isolated_records": int(result.stl_f_isolated.sum()),
        "multitask_or_cascade_evaluation_eligible_records": int(result.multitask_or_cascade_evaluation_eligible.sum()),
        "role": "reserved_source_isolated_F_evaluation_candidate_pending_larger_source_cluster",
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, default=ROOT / "results/analysis/pkdb_absolute_f_extraction_lock_v1")
    parser.add_argument("--eligibility", type=Path, default=ROOT / "results/analysis/pkdb_absolute_f_eligibility_lock_v1")
    parser.add_argument("--queue", type=Path, default=ROOT / "results/analysis/pkdb_absolute_f_candidate_queue_v1")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "data/literature/absolute_f_increment_view_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    extraction_file = args.extraction / "extraction_registry_locked.csv"
    eligibility_file = args.eligibility / "eligibility_registry_locked.csv"
    queue_file = args.queue / "candidate_records_blinded.csv"
    task_records = args.datasets / "task_records.csv"
    source_long = args.datasets / "source_long.csv"
    manifest = args.splits / "split_manifest.csv"
    required = [extraction_file, args.extraction / "complete.json", eligibility_file, args.eligibility / "complete.json",
                queue_file, args.queue / "complete.json", task_records, source_long, args.datasets / "complete.json",
                manifest, args.splits / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.extraction, "pkdb_absolute_f_extraction_lock")
    verify_stage(args.eligibility, "pkdb_absolute_f_eligibility_lock")
    verify_stage(args.queue, "pkdb_absolute_f_candidate_queue")
    verify_stage(args.datasets, "datasets")
    verify_stage(args.splits, "splits")
    result, summary = build_view(pd.read_csv(extraction_file), pd.read_csv(eligibility_file), pd.read_csv(queue_file),
                                 pd.read_csv(task_records, usecols=["task_id", "molecule_id", "split"]),
                                 pd.read_csv(manifest, usecols=["molecule_id", "parent_id", "scaffold_group"]),
                                 pd.read_csv(source_long, usecols=["task_id", "status", "doi"]))
    if args.check_only:
        print(f"Absolute-F increment isolation valid: molecules={summary['molecules']} sources={summary['source_clusters']}")
        return
    with stage_output(args.output) as out:
        result.to_csv(out / "increment_records.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Structure-isolated human absolute oral-F increment view\n\n"
            "This is an independently versioned literature increment, not a modification of processed_v15. Each record has "
            "passed a primary-source eligibility and numeric extraction lock, and is checked against every existing human-F "
            "member for parent, scaffold and source-DOI overlap. Both v1 records originate from one source, so they are retained "
            "as one source cluster and reserved for a later pre-specified evaluation design; they must not be split across train, "
            "validation and test or used to make a confirmatory claim by themselves. Cross-endpoint membership is reported because "
            "it would invalidate use as an MTL/cascade evaluation member even when direct F-STL isolation holds.\n",
            encoding="utf-8")
        finish_stage(out, "absolute_f_increment_view", inputs={
            "extraction_complete_sha256": sha256(args.extraction / "complete.json"),
            "eligibility_complete_sha256": sha256(args.eligibility / "complete.json"),
            "queue_complete_sha256": sha256(args.queue / "complete.json"),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "splits_complete_sha256": sha256(args.splits / "complete.json"),
        }, **summary, partial=False)
    print(f"Absolute-F structure-isolated increment view: {args.output}")


if __name__ == "__main__":
    run_cli(main)
