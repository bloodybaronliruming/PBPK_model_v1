#!/usr/bin/env python3
"""Aggregate label-blinded public-F candidates by source document and assay metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


FORBIDDEN = {"standard_value", "canonical_value", "raw_value", "target_value", "lower_bound", "upper_bound", "value", "mean", "median", "y"}
REVIEW_COLUMNS = {"activity_id", "description"}


def normalise_description(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    return value or "description_not_available"


def priority(row) -> tuple[str, str]:
    compatible = row.standard_type == "F" and row.standard_units == "%" and row.standard_relation == "="
    route = str(row.route).strip().upper()
    if compatible and route == "IV+PO":
        return "P1_metadata_IV_PO", "Retrieve source metadata first; IV+PO is a route signal, not proof of absolute F."
    if compatible and route == "PO":
        return "P2_metadata_PO", "Retrieve source metadata first; PO-only is insufficient for absolute-F admission."
    if compatible:
        return "P3_metadata_route_unknown", "Resolve route and endpoint definition before any numeric access."
    return "P4_metadata_incomplete", "Deprioritize until unit/relation/route metadata can be resolved."


def aggregate_chembl(candidates: pd.DataFrame, review: pd.DataFrame, rejected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if FORBIDDEN & set(candidates.columns) or FORBIDDEN & set(review.columns):
        raise ValueError("Label-blinded aggregation received a numeric column")
    required = {"activity_id", "molecule_id", "parent_id", "scaffold_group", "assay_id", "doc_id", "doi", "pubmed_id",
                "standard_type", "standard_units", "standard_relation", "route", "definition_signal", "reason", "public_status"}
    if not required <= set(candidates.columns) or not REVIEW_COLUMNS <= set(review.columns):
        raise ValueError("Public-F candidate/review schema mismatch")
    frame = candidates.merge(review, on="activity_id", how="left", validate="one_to_one")
    if frame.description.isna().any():
        raise ValueError("Candidate activity is missing its label-blinded endpoint description")
    frame["description_normalized"] = frame.description.map(normalise_description)
    reject_keys = set(zip(rejected.doc_id.astype(str), rejected.assay_id.astype(str)))
    frame["previous_definition_audit_status"] = ["previously_rejected" if (str(doc), str(assay)) in reject_keys else "not_previously_audited"
                                                 for doc, assay in zip(frame.doc_id, frame.assay_id)]
    quarantined = frame.loc[~frame.public_status.eq("public_candidate_definition_audit")].copy()
    if not quarantined.public_status.eq("quarantine_existing_strict_f_structure").all():
        raise ValueError("Unexpected non-public ChEMBL status; explicitly register its isolation role before aggregation")
    public = frame.loc[frame.public_status.eq("public_candidate_definition_audit")].copy()
    suppressed = public.loc[public.previous_definition_audit_status.eq("previously_rejected")].copy()
    working = public.loc[public.previous_definition_audit_status.eq("not_previously_audited")].copy()
    group_columns = ["doc_id", "assay_id", "doi", "pubmed_id", "description_normalized", "standard_type", "standard_units",
                     "standard_relation", "route", "definition_signal", "reason"]
    aggregates = working.groupby(group_columns, dropna=False).agg(
        records=("activity_id", "size"), molecules=("molecule_id", "nunique"), parents=("parent_id", "nunique"),
        scaffolds=("scaffold_group", "nunique"), activity_ids=("activity_id", lambda values: ";".join(map(str, sorted(values)))),
    ).reset_index()
    priorities = aggregates.apply(priority, axis=1, result_type="expand")
    aggregates[["priority", "next_action"]] = priorities
    aggregates = aggregates.sort_values(["priority", "parents", "records", "doc_id", "assay_id"], ascending=[True, False, False, True, True], ignore_index=True)
    document_columns = ["doc_id", "doi", "pubmed_id", "priority"]
    documents = aggregates.groupby(document_columns, dropna=False).agg(
        document_assays=("assay_id", "nunique"), records=("records", "sum"), parents=("parents", "sum"), scaffolds=("scaffolds", "sum"),
        definition_signals=("definition_signal", lambda values: ";".join(sorted(set(values)))),
        routes=("route", lambda values: ";".join(sorted(set(values)))),
        next_action=("next_action", "first"),
    ).reset_index().sort_values(["priority", "parents", "records", "doc_id"], ascending=[True, False, False, True], ignore_index=True)
    return aggregates, documents, suppressed, quarantined


def pkdb_status(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"sid", "reference_pmid", "reference_title", "reference_date", "substance_sid", "label", "chembl_id", "definition_signal", "public_status", "endpoint_values_hidden"}
    if not required <= set(candidates.columns) or not candidates.endpoint_values_hidden.astype(bool).all():
        raise ValueError("PK-DB candidate schema/label-blindness mismatch")
    current_B2_pmids = {"12369756", "8527689", "3197746"}
    stratified_pmids = {"17203292"}
    result = candidates.copy()
    result["source_review_state"] = "B0_primary_source_required"
    result.loc[result.public_status.eq("quarantine_reserved_strict_f_increment"), "source_review_state"] = "quarantine_reserved_strict_f_increment"
    result.loc[result.reference_pmid.astype(str).isin(current_B2_pmids), "source_review_state"] = "B2_numeric_locked_public_role"
    result.loc[result.reference_pmid.astype(str).isin(stratified_pmids), "source_review_state"] = "B1_eligible_stratified_aggregation_deferred"
    result.loc[result.reference_pmid.astype(str).eq("10945310"), "source_review_state"] = "B1_pending_direct_absolute_F_evidence"
    result.loc[result.substance_sid.eq("torasemide"), "source_review_state"] = "B0_pending_primary_fulltext"
    return result.sort_values(["source_review_state", "reference_date", "sid"], ascending=[True, False, True], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "results/analysis/public_f_development_candidate_registry_v1")
    parser.add_argument("--definition-audit", type=Path, default=ROOT / "results/analysis/public_f_definition_audit_v1")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/analysis/public_f_development_cohort_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_source_aggregation_v3")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    chembl_file, pkdb_file = args.registry / "chembl_public_candidates_blinded.csv", args.registry / "pkdb_public_candidates_blinded.csv"
    audit_file, review_file = args.definition_audit / "document_assay_definition_audit.csv", args.datasets / "review_records.csv"
    required = [chembl_file, pkdb_file, args.registry / "complete.json", audit_file, args.definition_audit / "complete.json",
                review_file, args.datasets / "complete.json", args.protocol / "protocol.json", args.protocol / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.registry, "public_f_development_candidate_registration")
    verify_stage(args.definition_audit, "public_f_definition_audit")
    verify_stage(args.datasets, "datasets")
    verify_stage(args.protocol, "public_f_development_cohort_protocol")
    review = pd.read_csv(review_file, usecols=list(REVIEW_COLUMNS), dtype=str, keep_default_na=False)
    audit = pd.read_csv(audit_file, usecols=["doc_id", "assay_id", "definition_decision"], dtype=str, keep_default_na=False)
    rejected = audit.loc[audit.definition_decision.str.startswith("rejected_")]
    aggregates, documents, suppressed, quarantined = aggregate_chembl(pd.read_csv(chembl_file, dtype=str, keep_default_na=False), review, rejected)
    pkdb = pkdb_status(pd.read_csv(pkdb_file, dtype=str, keep_default_na=False))
    summary = {
        "chembl_input_records": int(aggregates.records.sum()) + len(suppressed) + len(quarantined),
        "chembl_quarantined_existing_strict_f_structure_records": len(quarantined),
        "chembl_previously_rejected_document_assay_records": len(suppressed),
        "chembl_eligible_for_new_source_metadata_retrieval_records": int(aggregates.records.sum()),
        "chembl_document_assay_units": len(aggregates), "chembl_source_documents": documents.doc_id.nunique(),
        "p1_document_assay_units": int(aggregates.priority.eq("P1_metadata_IV_PO").sum()),
        "p2_document_assay_units": int(aggregates.priority.eq("P2_metadata_PO").sum()),
        "p3_document_assay_units": int(aggregates.priority.eq("P3_metadata_route_unknown").sum()),
        "p4_document_assay_units": int(aggregates.priority.eq("P4_metadata_incomplete").sum()),
        "pkdb_candidates": len(pkdb), "label_blinded": True, "numeric_values_read": False,
    }
    if summary["chembl_input_records"] != (summary["chembl_quarantined_existing_strict_f_structure_records"] +
                                            summary["chembl_previously_rejected_document_assay_records"] +
                                            summary["chembl_eligible_for_new_source_metadata_retrieval_records"]):
        raise ValueError("Public-F source aggregation record reconciliation failed")
    if args.check_only:
        print(f"Public-F source aggregation valid: documents={summary['chembl_source_documents']} units={summary['chembl_document_assay_units']}")
        return
    with stage_output(args.output) as out:
        aggregates.to_csv(out / "chembl_document_assay_aggregates_blinded.csv", index=False)
        documents.to_csv(out / "chembl_document_priority_queue_blinded.csv", index=False)
        suppressed.to_csv(out / "chembl_previously_rejected_definition_records_blinded.csv", index=False)
        quarantined.to_csv(out / "chembl_quarantined_existing_strict_f_structure_records_blinded.csv", index=False)
        pkdb.to_csv(out / "pkdb_source_status_blinded.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F source-level candidate aggregation\\n\\n"
            "This stage reads no endpoint values. It first quarantines all records whose parent/scaffold already overlaps frozen strict-F, "
            "then removes document/assay units already rejected by the previous high-signal definition audit. It aggregates only the remaining "
            "public ChEMBL discovery rows by document, assay and normalised endpoint description. P1/P2/P3 are "
            "metadata-retrieval priorities only: route, unit and relation are not proof of a compatible F definition. PK-DB rows are listed "
            "by source-review state. No output is a training cohort or a label-admission decision.\\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_source_aggregation", inputs={
            "registry_complete_sha256": sha256(args.registry / "complete.json"),
            "definition_audit_complete_sha256": sha256(args.definition_audit / "complete.json"),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
        }, **summary, partial=False)
    print(f"Public-F source aggregation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
