#!/usr/bin/env python3
"""Validate and immutably lock PK-DB half-life extraction after eligibility lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, configure_logging, finish_stage, sha256, stage_output


REQUIRED_COLUMNS = [
    "candidate_id", "study_sid", "substance_sid", "reference_pmid",
    "eligibility_registry_sha256", "extracted_value", "extracted_unit",
    "aggregation_group", "source_table_or_page", "extraction_note", "reviewer", "review_date",
]
IDENTITY_COLUMNS = ["candidate_id", "study_sid", "substance_sid", "reference_pmid"]


def validate_extraction(extraction: pd.DataFrame, eligibility: pd.DataFrame,
                        eligibility_hash: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(extraction.columns)
    extra = set(extraction.columns) - set(REQUIRED_COLUMNS)
    if missing or extra:
        raise ValueError(f"提取表列不一致 missing={sorted(missing)} extra={sorted(extra)}")
    extraction = extraction[REQUIRED_COLUMNS].copy().fillna("")
    if extraction.candidate_id.astype(str).str.strip().eq("").any() or extraction.candidate_id.duplicated().any():
        raise ValueError("提取表含空或重复 candidate_id")
    locked = eligibility.set_index("candidate_id")
    if set(extraction.candidate_id) != set(locked.index):
        raise ValueError("提取表必须覆盖且仅覆盖资格锁中的候选")
    for row in extraction.itertuples(index=False):
        expected = tuple(str(locked.loc[row.candidate_id, column]).strip()
                         for column in IDENTITY_COLUMNS[1:])
        observed = tuple(str(getattr(row, column)).strip() for column in IDENTITY_COLUMNS[1:])
        if observed != expected:
            raise ValueError(f"提取表不得变更候选身份: {row.candidate_id}")
    if extraction.eligibility_registry_sha256.astype(str).str.strip().ne(eligibility_hash).any():
        raise ValueError("提取表必须引用当前锁定资格表的 SHA-256")
    accepted_ids = set(locked.index[locked.eligibility_decision.eq("accepted")])
    accepted = extraction.candidate_id.isin(accepted_ids)
    numeric = pd.to_numeric(extraction.loc[accepted, "extracted_value"], errors="coerce")
    if numeric.isna().any() or numeric.le(0).any():
        raise ValueError("accepted 候选必须有正数 extracted_value")
    for column in ["extracted_unit", "aggregation_group", "source_table_or_page",
                   "extraction_note", "reviewer", "review_date"]:
        if extraction.loc[accepted, column].astype(str).str.strip().eq("").any():
            raise ValueError(f"accepted 候选缺少 {column}")
    if extraction.loc[accepted, "extracted_unit"].str.lower().ne("h").any():
        raise ValueError("当前仅接受小时单位")
    nonaccepted_value_columns = ["extracted_value", "extracted_unit", "aggregation_group",
                                 "source_table_or_page", "extraction_note"]
    for column in nonaccepted_value_columns:
        if extraction.loc[~accepted, column].astype(str).str.strip().ne("").any():
            raise ValueError(f"非 accepted 候选不得填写 {column}")
    return extraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--eligibility-lock", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_eligibility_lock_v1")
    parser.add_argument("--extraction", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_review_batch_v2/extraction_completed_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_extraction_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_logging(args.root, "lock_pkdb_extraction")
    registry = args.eligibility_lock / "eligibility_registry_locked.csv"
    complete = args.eligibility_lock / "complete.json"
    for path in [registry, complete, args.extraction]:
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(complete.read_text(encoding="utf-8"))
    if meta.get("stage") != "pkdb_external_eligibility_lock":
        raise ValueError("资格锁阶段不匹配")
    eligibility_hash = sha256(registry)
    if meta.get("artifacts", {}).get("eligibility_registry_locked.csv") != eligibility_hash:
        raise ValueError("资格锁文件哈希不匹配")
    extraction = validate_extraction(
        pd.read_csv(args.extraction, keep_default_na=False),
        pd.read_csv(registry, keep_default_na=False), eligibility_hash)
    if args.check_only:
        print(f"PK-DB extraction check passed: accepted={int(extraction.extracted_value.astype(str).str.strip().ne('').sum())}")
        return
    with stage_output(args.output) as out:
        extraction.to_csv(out / "extraction_registry_locked.csv", index=False)
        finish_stage(out, "pkdb_external_extraction_lock", inputs={
            "eligibility_registry_sha256": eligibility_hash,
            "extraction_input_sha256": sha256(args.extraction),
        }, records=len(extraction), accepted=int(extraction.extracted_value.astype(str).str.strip().ne("").sum()),
           partial=False)
    print(f"Locked PK-DB extraction registry: {args.output}")


if __name__ == "__main__":
    main()
