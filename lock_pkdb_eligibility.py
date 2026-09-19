#!/usr/bin/env python3
"""Validate and immutably lock a completed label-blind PK-DB eligibility registry.

Numerical half-life values are intentionally absent from this stage.  The
resulting registry hash must be copied into the separate extraction template
before any values are entered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, configure_logging, finish_stage, sha256, stage_output


DECISIONS = {"accepted", "rejected", "pending_clinical_methods", "pending_fulltext"}
REQUIRED_COLUMNS = [
    "candidate_id", "study_sid", "substance_sid", "reference_pmid", "fulltext_path",
    "eligibility_decision", "human_evidence", "iv_route_evidence",
    "parent_systemic_evidence", "terminal_phase_evidence", "source_table_or_page",
    "exclusion_reason", "evidence_note", "reviewer", "review_date",
]
IDENTITY_COLUMNS = ["candidate_id", "study_sid", "substance_sid", "reference_pmid"]


def validate_eligibility(frame: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Validate curation evidence without reading or writing endpoint values."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(REQUIRED_COLUMNS)
    if missing or extra:
        raise ValueError(f"资格表列不一致 missing={sorted(missing)} extra={sorted(extra)}")
    forbidden = {column for column in frame.columns if any(
        token in column.lower() for token in ("value", "half_life", "half-life", "endpoint"))}
    if forbidden:
        raise ValueError(f"资格表不得含端点数值列: {sorted(forbidden)}")
    frame = frame[REQUIRED_COLUMNS].copy().fillna("")
    if frame.candidate_id.astype(str).str.strip().eq("").any() or frame.candidate_id.duplicated().any():
        raise ValueError("资格表含空或重复 candidate_id")
    if frame.eligibility_decision.astype(str).str.strip().eq("").any():
        raise ValueError("资格表必须完成所有候选的裁定后才能锁定")
    invalid = set(frame.eligibility_decision) - DECISIONS
    if invalid:
        raise ValueError(f"资格表含无效裁定: {sorted(invalid)}")

    required_candidate_columns = set(IDENTITY_COLUMNS)
    candidate_missing = required_candidate_columns - set(candidates.columns)
    if candidate_missing:
        raise ValueError(f"盲态候选表缺少列: {sorted(candidate_missing)}")
    index = candidates[IDENTITY_COLUMNS].copy().fillna("")
    if index.candidate_id.duplicated().any():
        raise ValueError("盲态候选表含重复 candidate_id")
    indexed = index.set_index("candidate_id")
    for row in frame.itertuples(index=False):
        if row.candidate_id not in indexed.index:
            raise ValueError(f"资格表候选不在盲态索引: {row.candidate_id}")
        expected = tuple(str(indexed.loc[row.candidate_id, column]).strip()
                         for column in IDENTITY_COLUMNS[1:])
        observed = tuple(str(getattr(row, column)).strip() for column in IDENTITY_COLUMNS[1:])
        if observed != expected:
            raise ValueError(f"资格表不得变更候选身份: {row.candidate_id}")

    accepted = frame.eligibility_decision.eq("accepted")
    rejected = frame.eligibility_decision.eq("rejected")
    pending = ~(accepted | rejected)
    accepted_required = ["fulltext_path", "human_evidence", "iv_route_evidence",
                         "parent_systemic_evidence", "terminal_phase_evidence",
                         "source_table_or_page", "evidence_note", "reviewer", "review_date"]
    for column in accepted_required:
        if frame.loc[accepted, column].astype(str).str.strip().eq("").any():
            raise ValueError(f"accepted 行缺少 {column}")
    if frame.loc[accepted, "exclusion_reason"].astype(str).str.strip().ne("").any():
        raise ValueError("accepted 行不得填写 exclusion_reason")
    for column in ["exclusion_reason", "evidence_note", "reviewer", "review_date"]:
        if frame.loc[rejected, column].astype(str).str.strip().eq("").any():
            raise ValueError(f"rejected 行缺少 {column}")
    for column in ["evidence_note", "reviewer", "review_date"]:
        if frame.loc[pending, column].astype(str).str.strip().eq("").any():
            raise ValueError(f"pending 行缺少 {column}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--eligibility", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_review_batch_v2/eligibility_template_blinded.csv")
    parser.add_argument("--candidate-index", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_review_batch_v2/candidate_records_blinded.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_eligibility_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_logging(args.root, "lock_pkdb_eligibility")
    for path in [args.eligibility, args.candidate_index]:
        if not path.is_file():
            raise FileNotFoundError(path)
    eligibility = validate_eligibility(
        pd.read_csv(args.eligibility, keep_default_na=False),
        pd.read_csv(args.candidate_index, keep_default_na=False),
    )
    if args.check_only:
        print(f"PK-DB eligibility check passed: rows={len(eligibility)} "
              f"decisions={eligibility.eligibility_decision.value_counts().to_dict()}")
        return
    with stage_output(args.output) as out:
        eligibility.to_csv(out / "eligibility_registry_locked.csv", index=False)
        finish_stage(out, "pkdb_external_eligibility_lock", inputs={
            "eligibility_input_sha256": sha256(args.eligibility),
            "candidate_index_sha256": sha256(args.candidate_index),
        }, records=len(eligibility),
           decisions=eligibility.eligibility_decision.value_counts().to_dict(),
           label_blinded=True, partial=False)
    print(f"Locked PK-DB eligibility registry: {args.output}")


if __name__ == "__main__":
    main()
