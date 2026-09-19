#!/usr/bin/env python3
"""Run a label-blind bibliographic prefilter across all public-F source documents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from build_public_f_source_metadata_audit_queue import (
    FORBIDDEN, database_fingerprint, normalise_identifier, read_document_metadata,
)
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PRIORITY_ORDER = {
    "P1_metadata_IV_PO": 1,
    "P2_metadata_PO": 2,
    "P3_metadata_route_unknown": 3,
    "P4_metadata_incomplete": 4,
}
SECONDARY_PATTERN = re.compile(
    r"\b(qsar|model(?:s|ing)?|prediction|predicting|estimating|database|dataset|review|perspective|"
    r"overview|artificial membrane|caco-?2|permeability)\b", re.I,
)
NONCLINICAL_PATTERN = re.compile(
    r"\b(in[ -]?vitro|cell(?:s| line)?|murine|mice|mouse|rat(?:s)?|canine|dog(?:s)?|rabbit(?:s)?|monkeys?|"
    r"horses?|equine|pigs?|porcine|ovine|sheep|bovine|cattle|hamsters?|guinea pigs?|zebrafish|"
    r"docking|synthesis|synthesi[sz]ed|in[ -]?silico|molecular dynamics)\b", re.I,
)
MEDCHEM_PATTERN = re.compile(
    r"\b(structure[- ]property relationships?|matched pairs?|matched pair analogs?|novel series|drug optimization|"
    r"drug discovery|discovery of|potential for (?:high|oral)|substructures improves|lipophilicity on potency)\b", re.I,
)
# Do not match the word "human" in a disease name such as human
# immunodeficiency virus.  A title-only rule must remain conservative.
HUMAN_PATTERN = re.compile(
    r"\b(in humans?|healthy (?:subjects|volunteers)|(?:adult |healthy )?volunteers?|patients?)\b", re.I
)
PK_PATTERN = re.compile(r"\b(pharmacokinetic(?:s)?|bioavailability|pharmacodynamic(?:s)?)\b", re.I)


def best_document_rows(documents: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest existing metadata priority once per source document."""
    required = {"doc_id", "priority", "doi", "pubmed_id", "records", "parents", "scaffolds", "document_assays"}
    if not required <= set(documents.columns):
        raise ValueError("Source aggregation document queue schema mismatch")
    if FORBIDDEN & set(documents.columns):
        raise ValueError("Bibliographic prefilter received a numeric endpoint column")
    frame = documents.copy().fillna("")
    if not set(frame.priority) <= set(PRIORITY_ORDER):
        raise ValueError("Unknown aggregation priority")
    frame["priority_rank"] = frame.priority.map(PRIORITY_ORDER)
    duplicates = frame.doc_id.value_counts()
    frame["source_has_mixed_metadata_priorities"] = frame.doc_id.map(duplicates).gt(1)
    frame = frame.sort_values(["priority_rank", "parents", "records", "doc_id"], ascending=[True, False, False, True])
    result = frame.drop_duplicates("doc_id", keep="first").copy()
    if result.doc_id.duplicated().any():
        raise ValueError("Failed to make document source rows unique")
    return result.sort_values(["priority_rank", "parents", "records", "doc_id"], ascending=[True, False, False, True], ignore_index=True)


def classify_bibliography(row: pd.Series) -> tuple[str, int, str]:
    """Return a conservative retrieval route, not an endpoint-definition decision."""
    title = str(row.get("title", ""))
    doc_type = str(row.get("doc_type", "")).upper()
    has_identifier = bool(str(row.get("doi", "")).strip() or str(row.get("pubmed_id", "")).strip())
    if doc_type in {"DATASET", "PATENT"}:
        return "defer_non_citable_dataset_or_patent", 5, "Resolve an underlying citable primary study before any definition review."
    if not has_identifier:
        return "bibliography_resolution_required", 4, "Resolve DOI/PMID or a stable primary-source citation before definition review."
    if SECONDARY_PATTERN.search(title):
        return "citation_backtracking_required", 4, "Title signals a model, compilation, review, or in-vitro surrogate; locate its experimental provenance first."
    if NONCLINICAL_PATTERN.search(title):
        return "deprioritize_nonclinical_title_signal", 4, "Title signals nonclinical discovery work; do not treat it as human F without source-specific proof."
    if MEDCHEM_PATTERN.search(title):
        return "deprioritize_medicinal_chemistry_title_signal", 4, "Title signals medicinal-chemistry optimization; retain as auxiliary provenance, not as a human-F source lead."
    if HUMAN_PATTERN.search(title) and PK_PATTERN.search(title):
        return "candidate_primary_human_pk_title_signal", 1, "Audit human/oral/parent/F semantics and primary provenance before B1 consideration."
    if PK_PATTERN.search(title):
        return "candidate_primary_pk_title_signal", 2, "Audit species, route, parent analyte, F semantics, and primary provenance before B1 consideration."
    return "candidate_publication_metadata_only", 3, "No decisive title signal; resolve study design before any endpoint or numerical review."


def merge_metadata(documents: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    source = documents.copy()
    source["doc_id"] = source.doc_id.astype(str)
    local = metadata.copy().fillna("").astype(str)
    merged = source.merge(local, on="doc_id", how="left", suffixes=("_candidate", "_local"), validate="one_to_one")
    for field in ("doi", "pubmed_id"):
        left, right = f"{field}_candidate", f"{field}_local"
        both = merged[left].map(normalise_identifier).ne("") & merged[right].map(normalise_identifier).ne("")
        conflict = both & merged[left].map(normalise_identifier).ne(merged[right].map(normalise_identifier))
        if conflict.any():
            raise ValueError(f"Candidate and local ChEMBL {field} disagree for doc_id(s): {merged.loc[conflict, 'doc_id'].tolist()}")
        merged[field] = merged[right].where(merged[right].ne(""), merged[left])
        merged = merged.drop(columns=[left, right])
    classified = merged.apply(classify_bibliography, axis=1, result_type="expand")
    merged[["bibliographic_route", "manual_review_rank", "retrieval_instruction"]] = classified
    return merged.sort_values(["manual_review_rank", "priority_rank", "parents", "records", "doc_id"],
                              ascending=[True, True, False, False, True], ignore_index=True)


def build_definition_review_queue(screened: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Select only definition-compatible metadata priorities for bounded review."""
    eligible = screened.bibliographic_route.isin({
        "candidate_primary_human_pk_title_signal", "candidate_primary_pk_title_signal",
    }) & screened.priority_rank.le(3) & screened.prior_definition_audit_status.eq("not_previously_audited")
    queue = screened.loc[eligible].head(maximum).copy()
    queue.insert(0, "definition_review_selection_rank", range(1, len(queue) + 1))
    return queue


def read_prior_audits(audits: list[Path]) -> pd.DataFrame:
    """Merge prior rejection-only audits, enforcing one source-level disposition."""
    frames = []
    expected_files = {
        "public_f_priority_source_definition_audit": ("priority_source_definition_audit_blinded.csv", "decision"),
        "public_f_prefilter_definition_audit": ("prefilter_definition_audit_blinded.csv", "decision"),
        "public_f_unquarantined_priority_source_eligibility_lock": ("source_eligibility_lock_blinded.csv", "eligibility_decision"),
    }
    for folder in audits:
        metadata = verify_stage(folder, json.loads((folder / "complete.json").read_text())["stage"])
        spec = expected_files.get(metadata["stage"])
        if spec is None:
            raise ValueError(f"Unsupported prior-audit stage: {metadata['stage']}")
        filename, disposition_column = spec
        frame = pd.read_csv(folder / filename, dtype=str, usecols=["doc_id", disposition_column], keep_default_na=False)
        frame = frame.rename(columns={disposition_column: "decision"})
        frame["audit_stage"] = metadata["stage"]
        frames.append(frame)
    prior = pd.concat(frames, ignore_index=True)
    if prior.doc_id.duplicated().any():
        duplicates = prior.loc[prior.doc_id.duplicated(keep=False), "doc_id"].tolist()
        raise ValueError(f"Prior source definition audits overlap by doc_id: {duplicates}")
    return prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation", type=Path, default=ROOT / "results/analysis/public_f_source_aggregation_v3")
    parser.add_argument("--chembl-db", type=Path, default=ROOT / "data/raw/chembl_37/chembl_37_sqlite/chembl_37.db")
    parser.add_argument("--prior-definition-audits", type=Path, nargs="+", default=[
        ROOT / "results/analysis/public_f_priority_source_definition_audit_v1",
        ROOT / "results/analysis/public_f_prefilter_definition_audit_v1",
        ROOT / "results/analysis/public_f_prefilter_definition_audit_v3",
        ROOT / "results/analysis/public_f_prefilter_definition_audit_v4",
        ROOT / "results/analysis/public_f_haloperidol_phenobarbital_eligibility_lock_v1",
    ])
    parser.add_argument("--max-definition-review-documents", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_all_source_bibliographic_prefilter_v11")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.max_definition_review_documents < 1:
        raise ValueError("--max-definition-review-documents must be positive")
    document_file = args.aggregation / "chembl_document_priority_queue_blinded.csv"
    required = [document_file, args.aggregation / "complete.json", args.chembl_db]
    for folder in args.prior_definition_audits:
        required.append(folder / "complete.json")
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.aggregation, "public_f_source_aggregation")
    input_rows = pd.read_csv(document_file, dtype=str, keep_default_na=False)
    documents = best_document_rows(input_rows)
    metadata = read_document_metadata(args.chembl_db, documents.doc_id.tolist())
    screened = merge_metadata(documents, metadata)
    prior = read_prior_audits(args.prior_definition_audits)
    screened = screened.merge(prior.rename(columns={"decision": "prior_definition_audit_decision"}),
                              on="doc_id", how="left", validate="one_to_one")
    screened["prior_definition_audit_status"] = screened.prior_definition_audit_decision.map(
        lambda value: "not_previously_audited" if pd.isna(value) else "previously_audited"
    )
    screened["prior_definition_audit_decision"] = screened.prior_definition_audit_decision.fillna("")
    definition_queue = build_definition_review_queue(screened, args.max_definition_review_documents)
    database_identity = database_fingerprint(args.chembl_db)
    summary = {
        "input_priority_document_rows": len(input_rows),
        "unique_source_documents": len(screened),
        "route_counts": screened.bibliographic_route.value_counts().sort_index().to_dict(),
        "definition_review_queue_documents": len(definition_queue),
        "previously_audited_sources_suppressed_from_definition_queue": int(
            screened.prior_definition_audit_status.eq("previously_audited").sum()),
        "numeric_values_read": False,
        "abstract_read": False,
        "selection_is_not_definition_admission": True,
        "chembl_database_fingerprint": database_identity,
    }
    if args.check_only:
        print(f"Public-F all-source bibliographic prefilter valid: sources={len(screened)} queue={len(definition_queue)}")
        return
    with stage_output(args.output) as out:
        screened.to_csv(out / "all_source_bibliographic_prefilter_blinded.csv", index=False)
        definition_queue.to_csv(out / "priority_definition_review_queue_blinded.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F all-source bibliographic prefilter\n\n"
            "This label-blind screen reads bibliographic metadata from the local ChEMBL `docs` table for every aggregated source document. "
            "It does not query abstracts, activities, or endpoint values. The routes are deliberately conservative retrieval priorities, "
            "not human-F, primary-source, B1/B2, or training-admission decisions. The bounded definition-review queue contains only titles "
            "with a human+PK or PK signal and P1/P2/P3 metadata completeness; every member still requires source-level "
            "human/route/parent/F-semantics review. Documents already resolved by a prior source-definition or B1-eligibility audit are retained "
            "with their status but suppressed from the next manual queue. P4 sources remain in the metadata-resolution branch.\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_all_source_bibliographic_prefilter", inputs={
            "aggregation_complete_sha256": sha256(args.aggregation / "complete.json"),
            "prior_definition_audit_complete_sha256": {
                str(folder.resolve()): sha256(folder / "complete.json") for folder in args.prior_definition_audits
            },
            "chembl_sqlite_fingerprint": database_identity,
        }, **summary, partial=False)
    print(f"Public-F all-source bibliographic prefilter: {args.output}")


if __name__ == "__main__":
    run_cli(main)
