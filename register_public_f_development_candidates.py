#!/usr/bin/env python3
"""Register label-blinded public-development candidates for human oral bioavailability."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "F__human__absolute_oral"
FORBIDDEN = {"standard_value", "canonical_value", "raw_value", "target_value", "lower_bound", "upper_bound", "value", "mean", "median", "y"}
ABSOLUTE = re.compile(r"\babsolute\s+(?:oral\s+)?bioavailability\b", re.I)
BIOAVAILABILITY = re.compile(r"\bbioavailability\b", re.I)


def public_status(parent_id, scaffold, doi, strict_parents, strict_scaffolds, strict_dois,
                  reserved_parents, reserved_scaffolds, reserved_pmids, pmid):
    if parent_id in strict_parents or scaffold in strict_scaffolds:
        return "quarantine_existing_strict_f_structure"
    if doi and doi in strict_dois:
        return "quarantine_existing_strict_f_source"
    if parent_id in reserved_parents or scaffold in reserved_scaffolds or (pmid and pmid in reserved_pmids):
        return "quarantine_reserved_strict_f_increment"
    return "public_candidate_definition_audit"


def prepare_chembl(review, manifest, strict_members, strict_dois, reserved):
    required = {"molecule_id", "smiles", "activity_id", "assay_id", "doc_id", "doi", "pubmed_id", "description",
                "standard_type", "standard_units", "standard_relation", "route", "reason", "task_id", "status"}
    if not required <= set(review.columns):
        raise ValueError("ChEMBL review schema lacks public-F registration fields")
    if FORBIDDEN & set(review.columns):
        raise ValueError("Candidate reader must not receive numeric label columns")
    frame = review.loc[(review.task_id == TASK) & review.status.eq("review")].copy()
    frame = frame.merge(manifest[["molecule_id", "parent_id", "scaffold_group"]], on="molecule_id", how="left", validate="many_to_one")
    if frame.parent_id.isna().any():
        raise ValueError("Public F candidate lacks split-manifest structure grouping")
    strict_parents, strict_scaffolds = set(strict_members.parent_id), set(strict_members.scaffold_group)
    reserved_parents, reserved_scaffolds = set(reserved.parent_id), set(reserved.scaffold_group)
    reserved_pmids = set(reserved.source_pmid.astype(str))
    frame["public_status"] = [public_status(row.parent_id, row.scaffold_group, str(row.doi).strip(),
                                               strict_parents, strict_scaffolds, strict_dois,
                                               reserved_parents, reserved_scaffolds, reserved_pmids, str(row.pubmed_id).strip())
                              for row in frame.itertuples(index=False)]
    description = frame.description.fillna("")
    frame["definition_signal"] = "oral_bioavailability_description"
    frame.loc[description.str.contains(ABSOLUTE), "definition_signal"] = "absolute_bioavailability_description"
    fields = ["activity_id", "molecule_id", "smiles", "parent_id", "scaffold_group", "assay_id", "doc_id", "doi", "pubmed_id",
              "standard_type", "standard_units", "standard_relation", "route", "definition_signal", "reason", "public_status"]
    return frame[fields].sort_values(["public_status", "doc_id", "activity_id"], ignore_index=True)


def prepare_pkdb(candidates, strict_members, strict_dois, reserved):
    needed = {"sid", "reference_pmid", "reference_title", "reference_date", "substance_sid", "label", "chembl_id",
              "canonical_smiles", "parent_id", "scaffold_group", "substance_role", "endpoint_values_hidden"}
    if not needed <= set(candidates.columns):
        raise ValueError("PK-DB candidate schema lacks public-F registration fields")
    if FORBIDDEN & set(candidates.columns):
        raise ValueError("PK-DB source unexpectedly contains numeric endpoint columns")
    frame = candidates.loc[candidates.substance_role.eq("parent_candidate_unverified") & candidates.canonical_smiles.fillna("").ne("")].copy()
    frame = frame.loc[frame.reference_title.fillna("").str.contains(BIOAVAILABILITY)].copy()
    frame["definition_signal"] = "bioavailability_title"
    frame.loc[frame.reference_title.fillna("").str.contains(ABSOLUTE), "definition_signal"] = "absolute_bioavailability_title"
    strict_parents, strict_scaffolds = set(strict_members.parent_id), set(strict_members.scaffold_group)
    reserved_parents, reserved_scaffolds = set(reserved.parent_id), set(reserved.scaffold_group)
    reserved_pmids = set(reserved.source_pmid.astype(str))
    frame["public_status"] = [public_status(row.parent_id, row.scaffold_group, "", strict_parents, strict_scaffolds, strict_dois,
                                               reserved_parents, reserved_scaffolds, reserved_pmids, str(row.reference_pmid).strip())
                              for row in frame.itertuples(index=False)]
    fields = ["sid", "reference_pmid", "reference_title", "reference_date", "substance_sid", "label", "chembl_id", "canonical_smiles",
              "parent_id", "scaffold_group", "definition_signal", "public_status", "endpoint_values_hidden"]
    result = frame[fields].drop_duplicates().sort_values(["public_status", "definition_signal", "reference_date", "sid"],
                                                          ascending=[True, True, False, True], ignore_index=True)
    if not result.endpoint_values_hidden.astype(bool).all():
        raise ValueError("PK-DB public F candidates must remain label blinded")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--pkdb", type=Path, default=ROOT / "data/external/pkdb_preprocessed_v1")
    parser.add_argument("--reserved", type=Path, default=ROOT / "data/literature/absolute_f_increment_view_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_development_candidate_registry_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    review_file = args.datasets / "review_records.csv"
    task_records = args.datasets / "task_records.csv"
    source_long = args.datasets / "source_long.csv"
    manifest = args.splits / "split_manifest.csv"
    pkdb_file = args.pkdb / "candidate_substances_blinded.csv"
    reserved_file = args.reserved / "increment_records.csv"
    required = [review_file, task_records, source_long, args.datasets / "complete.json", manifest, args.splits / "complete.json",
                pkdb_file, args.pkdb / "complete.json", reserved_file, args.reserved / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.datasets, "datasets"); verify_stage(args.splits, "splits")
    verify_stage(args.pkdb, "pkdb_candidate_preprocessing"); verify_stage(args.reserved, "absolute_f_increment_view")
    review_fields = ["molecule_id", "smiles", "activity_id", "assay_id", "doc_id", "doi", "pubmed_id", "description", "standard_type",
                     "standard_units", "standard_relation", "route", "reason", "task_id", "status"]
    review = pd.read_csv(review_file, usecols=review_fields, keep_default_na=False, dtype=str)
    strict_ids = pd.read_csv(task_records, usecols=["task_id", "molecule_id"]).query("task_id == @TASK").molecule_id.unique()
    manifest_frame = pd.read_csv(manifest, usecols=["molecule_id", "parent_id", "scaffold_group"])
    strict_members = manifest_frame.loc[manifest_frame.molecule_id.isin(strict_ids)]
    strict_source = pd.read_csv(source_long, usecols=["task_id", "status", "doi"], keep_default_na=False, dtype=str)
    strict_dois = set(strict_source.loc[(strict_source.task_id == TASK) & strict_source.status.eq("accepted"), "doi"]) - {""}
    reserved = pd.read_csv(reserved_file, usecols=["parent_id", "scaffold_group", "source_pmid"])
    chembl = prepare_chembl(review, manifest_frame, strict_members, strict_dois, reserved)
    pkdb = prepare_pkdb(pd.read_csv(pkdb_file, low_memory=False), strict_members, strict_dois, reserved)
    summary = {
        "chembl_records": len(chembl), "chembl_molecules": chembl.molecule_id.nunique(), "chembl_documents": chembl.doc_id.nunique(),
        "chembl_public_candidates": int(chembl.public_status.eq("public_candidate_definition_audit").sum()),
        "pkdb_records": len(pkdb), "pkdb_studies": pkdb.sid.nunique(),
        "pkdb_public_candidates": int(pkdb.public_status.eq("public_candidate_definition_audit").sum()), "label_blinded": True,
    }
    if args.check_only:
        print(f"Public F candidate registration valid: ChEMBL docs={summary['chembl_documents']} PK-DB studies={summary['pkdb_studies']}")
        return
    with stage_output(args.output) as out:
        chembl.to_csv(out / "chembl_public_candidates_blinded.csv", index=False)
        pkdb.to_csv(out / "pkdb_public_candidates_blinded.csv", index=False)
        chembl.groupby(["public_status", "definition_signal"], dropna=False).agg(records=("activity_id", "size"), molecules=("molecule_id", "nunique"), documents=("doc_id", "nunique")).reset_index().to_csv(out / "chembl_summary.csv", index=False)
        pkdb.groupby(["public_status", "definition_signal"], dropna=False).agg(records=("substance_sid", "size"), studies=("sid", "nunique"), molecules=("parent_id", "nunique")).reset_index().to_csv(out / "pkdb_summary.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text("# Public human oral-bioavailability candidate registration\n\nThis is a label-blinded discovery and registration layer for a future public-development task, not a training set. It deliberately does not inherit Thalf-specific PK-DB exclusions: F candidates are quarantined only for strict-F/reserved-F overlap. ChEMBL review rows remain review rows; a description signal is not proof of strict human absolute F. Numeric values are neither read nor written.\n", encoding="utf-8")
        finish_stage(out, "public_f_development_candidate_registration", inputs={"datasets_complete_sha256": sha256(args.datasets / "complete.json"), "splits_complete_sha256": sha256(args.splits / "complete.json"), "pkdb_complete_sha256": sha256(args.pkdb / "complete.json"), "reserved_complete_sha256": sha256(args.reserved / "complete.json")}, **summary, partial=False)
    print(f"Public F development candidate registration: {args.output}")


if __name__ == "__main__":
    run_cli(main)
