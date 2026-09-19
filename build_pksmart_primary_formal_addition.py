#!/usr/bin/env python3
"""Convert locked PKSmart primary-source evidence into a formal-registry addition."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


DECISION_COLUMNS = ["candidate_id", "activity_id", "molecule_id", "source_doi", "eligibility_decision",
                    "primary_source_doi", "source_table_or_page", "route_evidence", "terminal_phase_evidence",
                    "parent_systemic_evidence", "extracted_value", "extracted_unit", "exclusion_reason",
                    "evidence_note", "reviewer", "review_date"]


def build_addition(candidates: pd.DataFrame, eligibility: pd.DataFrame, aggregate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"candidate_id", "source_row", "molecule_id", "primary_source_locator", "identity_mapping_basis"}
    if required - set(candidates.columns):
        raise ValueError("PKSmart candidate index lacks formal identity fields")
    if set(candidates.candidate_id) != set(eligibility.candidate_id) or set(candidates.candidate_id) != set(aggregate.candidate_id):
        raise ValueError("PKSmart formal addition inputs disagree on candidate membership")
    merged = candidates.merge(eligibility, on="candidate_id", suffixes=("_candidate", "_eligibility"), validate="one_to_one")
    merged = merged.merge(aggregate, on="candidate_id", suffixes=("", "_aggregate"), validate="one_to_one")
    if not merged.eligibility_decision.eq("accepted").all():
        raise ValueError("Only accepted PKSmart primary-source eligibility can enter the formal registry")
    if not merged.primary_source_locator_candidate.eq(merged.primary_source_locator_eligibility).all():
        raise ValueError("PKSmart source locator changed between candidate and eligibility lock")
    index = pd.DataFrame({
        "candidate_id": merged.candidate_id,
        "activity_id": "pksmart_source_row:" + merged.source_row_candidate.astype(str),
        "molecule_id": merged.molecule_id_candidate,
        "doi": merged.primary_source_locator_candidate,
    })
    rows = []
    for item in merged.itertuples(index=False):
        rows.append({
            "candidate_id": item.candidate_id,
            "activity_id": f"pksmart_source_row:{item.source_row_candidate}",
            "molecule_id": item.molecule_id_candidate,
            "source_doi": item.primary_source_locator_candidate,
            "eligibility_decision": "accepted",
            "primary_source_doi": item.primary_source_locator_candidate,
            "source_table_or_page": item.source_table_or_page_aggregate,
            "route_evidence": item.iv_route_evidence,
            "terminal_phase_evidence": item.terminal_phase_evidence,
            "parent_systemic_evidence": item.parent_systemic_evidence,
            "extracted_value": item.extracted_value,
            "extracted_unit": item.extracted_unit,
            "exclusion_reason": "",
            "evidence_note": (
                f"Original paper qualification locked before numerical extraction. Formal molecule label uses {item.aggregation_group}: "
                f"{item.aggregation_method} Secondary PKSmart-to-ChEMBL mapping remains {item.identity_mapping_basis_candidate}; "
                "the article itself names trilaciclib and supplies route/analyte/terminal evidence."),
            "reviewer": item.reviewer,
            "review_date": item.review_date,
        })
    decisions = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    return index, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-batch", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_review_batch_v1")
    parser.add_argument("--eligibility-lock", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_eligibility_lock_v1")
    parser.add_argument("--extraction-lock", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_extraction_lock_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_formal_addition_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.review_batch, "pksmart_primary_source_eligibility_review_batch")
    verify_stage(args.eligibility_lock, "pksmart_primary_source_eligibility_lock")
    verify_stage(args.extraction_lock, "pksmart_primary_source_extraction_lock")
    candidate_file = args.review_batch / "candidate_records_blinded.csv"
    eligibility_file = args.eligibility_lock / "eligibility_registry_locked.csv"
    aggregate_file = args.extraction_lock / "formal_aggregate_locked.csv"
    startup_self_check([candidate_file, eligibility_file, aggregate_file], output=None if args.check_only else args.output)
    index, decisions = build_addition(pd.read_csv(candidate_file, keep_default_na=False),
                                      pd.read_csv(eligibility_file, keep_default_na=False),
                                      pd.read_csv(aggregate_file, keep_default_na=False))
    if args.check_only:
        print(f"PKSmart formal addition contract valid: accepted={len(decisions)}")
        return
    with stage_output(args.output) as out:
        index.to_csv(out / "formal_candidate_index.csv", index=False)
        decisions.to_csv(out / "formal_registry_additions.csv", index=False)
        (out / "README.md").write_text(
            "# PKSmart primary-source formal registry addition\n\n"
            "This addition is generated only from the locked primary-source eligibility and numerical extraction stages. "
            "It contains one molecule-level healthy-control aggregate for trilaciclib and does not expose the PKSmart "
            "public CSV label.\n", encoding="utf-8")
        finish_stage(out, "pksmart_primary_source_formal_registry_addition", inputs={
            "review_batch_complete_sha256": sha256(args.review_batch / "complete.json"),
            "eligibility_lock_complete_sha256": sha256(args.eligibility_lock / "complete.json"),
            "extraction_lock_complete_sha256": sha256(args.extraction_lock / "complete.json"),
        }, accepted=len(decisions), partial=False)
    print(f"PKSmart formal registry addition: {args.output}")


if __name__ == "__main__":
    run_cli(main)
