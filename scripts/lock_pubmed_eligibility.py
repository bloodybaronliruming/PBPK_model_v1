#!/usr/bin/env python3
"""Validate and immutably lock completed PubMed fulltext eligibility decisions without values."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, sha256, stage_output


REQUIRED_COLUMNS = [
    "candidate_id", "activity_id", "molecule_id", "pubmed_id", "doi", "fulltext_path", "eligibility_decision",
    "human_evidence", "iv_route_evidence", "parent_systemic_evidence", "terminal_phase_evidence",
    "source_table_or_page", "exclusion_reason", "evidence_note", "reviewer", "review_date",
]
IDENTITY = ["candidate_id", "activity_id", "molecule_id", "pubmed_id", "doi"]


def project_path(path: str, root: Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def validate_eligibility(frame: pd.DataFrame, candidates: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    if set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError(f"资格表列不一致 missing={sorted(set(REQUIRED_COLUMNS)-set(frame.columns))} "
                         f"extra={sorted(set(frame.columns)-set(REQUIRED_COLUMNS))}")
    forbidden = {column for column in frame.columns if any(token in column.lower() for token in ("value", "half_life", "endpoint"))}
    if forbidden:
        raise ValueError(f"资格表不得含端点数值列: {sorted(forbidden)}")
    frame = frame[REQUIRED_COLUMNS].copy().fillna("")
    if frame.candidate_id.eq("").any() or frame.candidate_id.duplicated().any():
        raise ValueError("资格表含空或重复 candidate_id")
    if set(frame.eligibility_decision) != {"accepted"}:
        raise ValueError("本独立批资格锁仅允许已完成的 accepted 候选；pending/rejected 应留在独立未完成批次")
    index = candidates.set_index("candidate_id")
    if set(frame.candidate_id) != set(index.index):
        raise ValueError("资格表必须覆盖且仅覆盖盲态独立候选索引")
    for row in frame.itertuples(index=False):
        if tuple(str(getattr(row, key)).strip().lower() for key in IDENTITY[1:]) != tuple(
                str(index.loc[row.candidate_id, key]).strip().lower() for key in IDENTITY[1:]):
            raise ValueError(f"资格表不得变更候选身份: {row.candidate_id}")
        if not project_path(row.fulltext_path, root).is_file():
            raise FileNotFoundError(f"资格表全文不存在: {row.fulltext_path}")
    for column in ["fulltext_path", "human_evidence", "iv_route_evidence", "parent_systemic_evidence",
                   "terminal_phase_evidence", "source_table_or_page", "evidence_note", "reviewer", "review_date"]:
        if frame[column].eq("").any():
            raise ValueError(f"accepted 行缺少 {column}")
    if frame.exclusion_reason.ne("").any():
        raise ValueError("accepted 行不得填写 exclusion_reason")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--eligibility", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_independent_batch_v3/eligibility_completed_v1.csv")
    parser.add_argument("--candidate-index", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_independent_batch_v3/candidate_records_blinded.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_eligibility_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for path in [args.eligibility, args.candidate_index]:
        if not path.is_file():
            raise FileNotFoundError(path)
    locked = validate_eligibility(pd.read_csv(args.eligibility, keep_default_na=False),
                                  pd.read_csv(args.candidate_index, keep_default_na=False), args.root)
    if args.check_only:
        print(f"PubMed eligibility check passed: accepted={len(locked)}")
        return
    with stage_output(args.output) as out:
        locked.to_csv(out / "eligibility_registry_locked.csv", index=False)
        finish_stage(out, "pubmed_external_eligibility_lock", inputs={
            "eligibility_input_sha256": sha256(args.eligibility),
            "candidate_index_sha256": sha256(args.candidate_index),
        }, records=len(locked), accepted=len(locked), label_blinded=True, partial=False)


if __name__ == "__main__":
    main()
