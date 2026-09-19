#!/usr/bin/env python3
"""Validate and immutably lock PubMed numerical extraction after the eligibility lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, sha256, stage_output


REQUIRED_COLUMNS = [
    "candidate_id", "activity_id", "molecule_id", "pubmed_id", "doi", "eligibility_registry_sha256",
    "extracted_value", "extracted_unit", "aggregation_group", "source_table_or_page", "extraction_note",
    "reviewer", "review_date",
]
IDENTITY = ["candidate_id", "activity_id", "molecule_id", "pubmed_id", "doi"]


def validate_extraction(frame: pd.DataFrame, eligibility: pd.DataFrame, eligibility_hash: str) -> pd.DataFrame:
    if set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError(f"提取表列不一致 missing={sorted(set(REQUIRED_COLUMNS)-set(frame.columns))} "
                         f"extra={sorted(set(frame.columns)-set(REQUIRED_COLUMNS))}")
    frame = frame[REQUIRED_COLUMNS].copy().fillna("")
    if frame.candidate_id.eq("").any() or frame.candidate_id.duplicated().any():
        raise ValueError("提取表含空或重复 candidate_id")
    locked = eligibility.set_index("candidate_id")
    if set(frame.candidate_id) != set(locked.index):
        raise ValueError("提取表必须覆盖且仅覆盖资格锁候选")
    if frame.eligibility_registry_sha256.ne(eligibility_hash).any():
        raise ValueError("提取表必须引用当前资格锁 SHA-256")
    for row in frame.itertuples(index=False):
        if tuple(str(getattr(row, key)).strip().lower() for key in IDENTITY[1:]) != tuple(
                str(locked.loc[row.candidate_id, key]).strip().lower() for key in IDENTITY[1:]):
            raise ValueError(f"提取表不得变更候选身份: {row.candidate_id}")
    numeric = pd.to_numeric(frame.extracted_value, errors="coerce")
    if numeric.isna().any() or numeric.le(0).any() or frame.extracted_unit.str.lower().ne("h").any():
        raise ValueError("提取值必须为正数小时")
    for column in ["aggregation_group", "source_table_or_page", "extraction_note", "reviewer", "review_date"]:
        if frame[column].eq("").any():
            raise ValueError(f"提取表缺少 {column}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--eligibility-lock", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_eligibility_lock_v1")
    parser.add_argument("--extraction", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_independent_batch_v3/extraction_completed_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_extraction_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    registry = args.eligibility_lock / "eligibility_registry_locked.csv"
    complete = args.eligibility_lock / "complete.json"
    for path in [registry, complete, args.extraction]:
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(complete.read_text(encoding="utf-8"))
    if meta.get("stage") != "pubmed_external_eligibility_lock":
        raise ValueError("资格锁阶段不匹配")
    lock_hash = sha256(registry)
    if meta.get("artifacts", {}).get("eligibility_registry_locked.csv") != lock_hash:
        raise ValueError("资格锁哈希不匹配")
    extracted = validate_extraction(pd.read_csv(args.extraction, keep_default_na=False),
                                    pd.read_csv(registry, keep_default_na=False), lock_hash)
    if args.check_only:
        print(f"PubMed extraction check passed: accepted={len(extracted)}")
        return
    with stage_output(args.output) as out:
        extracted.to_csv(out / "extraction_registry_locked.csv", index=False)
        finish_stage(out, "pubmed_external_extraction_lock", inputs={
            "eligibility_registry_sha256": lock_hash, "extraction_input_sha256": sha256(args.extraction),
        }, records=len(extracted), accepted=len(extracted), partial=False)


if __name__ == "__main__":
    main()
