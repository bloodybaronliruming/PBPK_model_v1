#!/usr/bin/env python3
"""Build a triple-isolated public-F cohort view without modifying any dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_split_manifest import structure_groups
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "F__human__absolute_oral"
STRUCTURE_COLUMNS = ["substance_sid", "label", "chembl_id", "canonical_smiles", "parent_id", "scaffold_group", "structure_status"]
GATE_COLUMNS = [
    "overlap_strict_f_parent", "overlap_strict_f_scaffold", "overlap_strict_f_source",
    "overlap_boulton_parent", "overlap_boulton_scaffold", "overlap_boulton_source",
]


def normalise_dois(values: pd.Series) -> set[str]:
    return {str(value).strip().lower() for value in values.fillna("") if str(value).strip() and str(value).lower() != "nan"}


def structural_members(records: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    members = records.loc[records.task_id.eq(TASK), ["molecule_id"]].drop_duplicates()
    members = members.merge(manifest[["molecule_id", "parent_id", "scaffold_group"]], on="molecule_id", how="left", validate="many_to_one")
    if members[["parent_id", "scaffold_group"]].isna().any(axis=None):
        raise ValueError("Existing strict-F members lack manifest structure groups")
    return members


def build_view(extractions: pd.DataFrame, registry: pd.DataFrame, mappings: pd.DataFrame, strict_records: pd.DataFrame,
               strict_manifest: pd.DataFrame, strict_sources: pd.DataFrame, boulton: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    accepted = extractions.loc[extractions.extraction_decision.eq("accepted")].copy()
    if accepted.request_id.duplicated().any():
        raise ValueError("Public-F extraction lock has duplicate request IDs")
    if not set(STRUCTURE_COLUMNS) <= set(registry.columns):
        raise ValueError("Substance registry lacks required structure fields")
    structures = registry[STRUCTURE_COLUMNS].copy()
    accepted = accepted.merge(structures, left_on="analyte", right_on="substance_sid", how="left", validate="one_to_one")
    required_mapping = {"analyte", "mapping_decision", "expected_chembl_id", "source_doi", "identity_evidence", "reviewer", "review_date", "decision_note"}
    if not required_mapping <= set(mappings.columns):
        raise ValueError("Structure mapping decisions lack required fields")
    mappings = mappings.loc[:, sorted(required_mapping)].copy()
    if mappings.analyte.duplicated().any() or set(mappings.analyte) != set(accepted.analyte):
        raise ValueError("Structure mapping decisions must cover accepted analytes exactly once")
    if not mappings.mapping_decision.eq("accepted_named_parent").all():
        raise ValueError("Only explicitly accepted named-parent mappings may enter isolation screening")
    accepted = accepted.merge(mappings, on=["analyte", "source_doi"], how="inner", validate="one_to_one")
    if accepted.canonical_smiles.isna().any() or (~accepted.chembl_id.eq(accepted.expected_chembl_id)).any():
        raise ValueError("Accepted source lacks the manually confirmed registered parent structure")

    recomputed = accepted.canonical_smiles.map(structure_groups)
    accepted["recomputed_parent_id"] = [item[0] for item in recomputed]
    accepted["recomputed_scaffold_group"] = [item[1] for item in recomputed]
    if not accepted.parent_id.eq(accepted.recomputed_parent_id).all() or not accepted.scaffold_group.eq(accepted.recomputed_scaffold_group).all():
        raise ValueError("Candidate structure grouping differs from the registered substance structure")

    strict_members = structural_members(strict_records, strict_manifest)
    strict_parent, strict_scaffold = set(strict_members.parent_id), set(strict_members.scaffold_group)
    strict_dois = normalise_dois(strict_sources.loc[(strict_sources.task_id == TASK) & strict_sources.status.eq("accepted"), "doi"])
    required_boulton = {"candidate_id", "parent_id", "scaffold_group", "source_doi", "source_cluster_id"}
    if not required_boulton <= set(boulton.columns):
        raise ValueError("Boulton reserve view lacks triple-isolation fields")
    boulton_parent, boulton_scaffold = set(boulton.parent_id), set(boulton.scaffold_group)
    boulton_dois = normalise_dois(boulton.source_doi)

    accepted["source_cluster_id"] = accepted.source_doi.str.strip().str.lower()
    if accepted.source_cluster_id.eq("").any():
        raise ValueError("Accepted public-F source lacks a DOI source cluster")
    accepted["overlap_strict_f_parent"] = accepted.parent_id.isin(strict_parent)
    accepted["overlap_strict_f_scaffold"] = accepted.scaffold_group.isin(strict_scaffold)
    accepted["overlap_strict_f_source"] = accepted.source_cluster_id.isin(strict_dois)
    accepted["overlap_boulton_parent"] = accepted.parent_id.isin(boulton_parent)
    accepted["overlap_boulton_scaffold"] = accepted.scaffold_group.isin(boulton_scaffold)
    accepted["overlap_boulton_source"] = accepted.source_cluster_id.isin(boulton_dois)
    if accepted[GATE_COLUMNS].any(axis=None):
        conflicts = accepted.loc[accepted[GATE_COLUMNS].any(axis=1), ["analyte", *GATE_COLUMNS]]
        raise ValueError(f"Public-F candidate fails triple isolation: {conflicts.to_dict(orient='records')}")

    all_members = strict_records[["task_id", "molecule_id"]].drop_duplicates().merge(
        strict_manifest[["molecule_id", "parent_id", "scaffold_group"]], on="molecule_id", how="left", validate="many_to_one")
    if all_members[["parent_id", "scaffold_group"]].isna().any(axis=None):
        raise ValueError("Existing endpoint membership lacks structural grouping")
    cross_endpoint = []
    for row in accepted.itertuples(index=False):
        tasks = all_members.loc[(all_members.parent_id == row.parent_id) | (all_members.scaffold_group == row.scaffold_group), "task_id"]
        cross_endpoint.append(";".join(sorted(set(tasks))))
    accepted["overlap_existing_any_endpoint_tasks"] = cross_endpoint
    accepted["direct_f_stl_isolated"] = True
    accepted["eligible_for_strict_f_external"] = False
    accepted["eligible_for_validation_or_test_assignment"] = False
    accepted["cohort_role"] = "public_development_candidate_pending_pre_registered_training_cohort"

    source_clusters = accepted.groupby("source_cluster_id", as_index=False).agg(
        records=("request_id", "size"), molecules=("parent_id", "nunique"), analytes=("analyte", lambda values: ";".join(sorted(values))),
        source_pmid=("source_pmid", "first"),
    )
    if not source_clusters.records.eq(1).all():
        raise ValueError("v1 public-F cohort requires one record per source cluster")
    columns = [
        "request_id", "analyte", "chembl_id", "canonical_smiles", "parent_id", "scaffold_group", "source_doi", "source_pmid",
        "source_locator", "reported_statistic", "interval_type", "canonical_unit", "canonical_value_fraction", "canonical_estimator_rule",
        "mapping_decision", "expected_chembl_id", "identity_evidence", "structure_status",
        *GATE_COLUMNS, "overlap_existing_any_endpoint_tasks", "direct_f_stl_isolated", "eligible_for_strict_f_external",
        "eligible_for_validation_or_test_assignment", "cohort_role", "source_cluster_id",
    ]
    result = accepted[columns].sort_values("analyte", ignore_index=True)
    summary = {
        "records": len(result), "molecules": result.parent_id.nunique(), "source_clusters": result.source_cluster_id.nunique(),
        "strict_f_molecules_checked": strict_members.molecule_id.nunique(), "boulton_reserved_molecules_checked": boulton.parent_id.nunique(),
        "triple_isolated_records": int((~result[GATE_COLUMNS].any(axis=1)).sum()),
        "strict_f_external_eligible_records": int(result.eligible_for_strict_f_external.sum()),
        "validation_or_test_assignable_records": int(result.eligible_for_validation_or_test_assignment.sum()),
        "role": "public_development_candidate_pending_pre_registered_training_cohort",
    }
    return result, source_clusters, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, default=ROOT / "results/analysis/public_f_pkdb_extraction_lock_v1")
    parser.add_argument("--substance-registry", type=Path, default=ROOT / "data/external/pkdb_preprocessed_v1/substance_registry.csv")
    parser.add_argument("--structure-mappings", type=Path,
                        default=ROOT / "data/manual_review/public_f_structure_mapping_decisions_v1.csv")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--boulton-reserve", type=Path, default=ROOT / "data/literature/absolute_f_increment_view_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/literature/public_f_cohort_isolation_view_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    extraction_file = args.extraction / "extraction_registry_locked.csv"
    task_records = args.datasets / "task_records.csv"
    source_long = args.datasets / "source_long.csv"
    manifest = args.splits / "split_manifest.csv"
    boulton_file = args.boulton_reserve / "increment_records.csv"
    required = [
        extraction_file, args.extraction / "complete.json", args.substance_registry, args.structure_mappings, task_records, source_long,
        args.datasets / "complete.json", manifest, args.splits / "complete.json", boulton_file, args.boulton_reserve / "complete.json",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.extraction, "public_f_pkdb_extraction_lock")
    verify_stage(args.datasets, "datasets")
    verify_stage(args.splits, "splits")
    verify_stage(args.boulton_reserve, "absolute_f_increment_view")
    result, clusters, summary = build_view(
        pd.read_csv(extraction_file), pd.read_csv(args.substance_registry), pd.read_csv(args.structure_mappings), pd.read_csv(task_records, usecols=["task_id", "molecule_id"]),
        pd.read_csv(manifest, usecols=["molecule_id", "parent_id", "scaffold_group"]),
        pd.read_csv(source_long, usecols=["task_id", "status", "doi"]), pd.read_csv(boulton_file),
    )
    if args.check_only:
        print(f"Public-F triple isolation valid: molecules={summary['molecules']} sources={summary['source_clusters']}")
        return
    with stage_output(args.output) as out:
        result.to_csv(out / "cohort_records.csv", index=False)
        clusters.to_csv(out / "source_cluster_manifest.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F triple-isolated cohort view\\n\\n"
            "This versioned view checks each accepted public-F label against both the existing strict human-F cohort and the separately "
            "reserved Boulton increment for parent, Murcko-scaffold and DOI/source-cluster overlap. It is not a data merge: all members "
            "remain unassigned to train/validation/test, cannot enter strict-F external validation, and are retained only as candidates for "
            "a later pre-registered public-development training cohort. Cross-endpoint membership is a diagnostic field and does not relax "
            "the F-specific triple-isolation gates.\\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_cohort_isolation_view", inputs={
            "extraction_complete_sha256": sha256(args.extraction / "complete.json"), "substance_registry_sha256": sha256(args.substance_registry),
            "structure_mappings_sha256": sha256(args.structure_mappings),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"), "splits_complete_sha256": sha256(args.splits / "complete.json"),
            "boulton_reserve_complete_sha256": sha256(args.boulton_reserve / "complete.json"),
        }, **summary, partial=False)
    print(f"Public-F triple-isolated cohort view: {args.output}")


if __name__ == "__main__":
    run_cli(main)
