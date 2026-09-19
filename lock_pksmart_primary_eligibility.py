#!/usr/bin/env python3
"""Immutably lock completed, label-free PKSmart primary-source eligibility decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_pksmart_primary_source_review_batch import ELIGIBILITY_COLUMNS
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


IDENTITY = ["candidate_id", "source_row", "molecule_id", "chembl_id", "chembl_pref_name", "identity_mapping_basis", "primary_source_locator"]


def validate_eligibility(frame: pd.DataFrame, candidates: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    if set(frame.columns) != set(ELIGIBILITY_COLUMNS):
        raise ValueError("PKSmart eligibility columns do not match the label-free template")
    forbidden = {column for column in frame.columns if any(token in column.lower() for token in ("value", "half_life", "endpoint"))}
    if forbidden:
        raise ValueError(f"PKSmart eligibility must not contain endpoint-value fields: {sorted(forbidden)}")
    work = frame[ELIGIBILITY_COLUMNS].copy().fillna("")
    if work.candidate_id.eq("").any() or work.candidate_id.duplicated().any() or set(work.eligibility_decision) != {"accepted"}:
        raise ValueError("PKSmart eligibility lock requires one completed accepted decision per candidate")
    index = candidates.set_index("candidate_id")
    if set(work.candidate_id) != set(index.index):
        raise ValueError("PKSmart eligibility must cover exactly the blinded review candidates")
    for row in work.itertuples(index=False):
        source = index.loc[row.candidate_id]
        for field in IDENTITY[1:]:
            if str(getattr(row, field)).strip().lower() != str(source[field]).strip().lower():
                raise ValueError(f"PKSmart eligibility changes candidate identity: {row.candidate_id}")
        fulltext = Path(row.fulltext_path)
        if not fulltext.is_absolute():
            fulltext = root / fulltext
        if not fulltext.is_file():
            raise FileNotFoundError(fulltext)
    for column in ["fulltext_path", "human_evidence", "iv_route_evidence", "parent_systemic_evidence",
                   "terminal_phase_evidence", "source_table_or_page", "evidence_note", "reviewer", "review_date"]:
        if work[column].eq("").any():
            raise ValueError(f"PKSmart accepted decision lacks {column}")
    if work.exclusion_reason.ne("").any():
        raise ValueError("PKSmart accepted decision cannot have an exclusion reason")
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-batch", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_review_batch_v1")
    parser.add_argument("--eligibility", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_review_batch_v1/eligibility_completed_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_eligibility_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.review_batch, "pksmart_primary_source_eligibility_review_batch")
    candidate_file = args.review_batch / "candidate_records_blinded.csv"
    startup_self_check([candidate_file, args.eligibility], output=None if args.check_only else args.output)
    locked = validate_eligibility(pd.read_csv(args.eligibility, keep_default_na=False),
                                  pd.read_csv(candidate_file, keep_default_na=False))
    if args.check_only:
        print(f"PKSmart primary-source eligibility check passed: accepted={len(locked)}")
        return
    with stage_output(args.output) as out:
        locked.to_csv(out / "eligibility_registry_locked.csv", index=False)
        finish_stage(out, "pksmart_primary_source_eligibility_lock", inputs={
            "review_batch_complete_sha256": sha256(args.review_batch / "complete.json"),
            "eligibility_input_sha256": sha256(args.eligibility),
        }, accepted=len(locked), label_blinded=True, partial=False)
    print(f"PKSmart primary-source eligibility lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
