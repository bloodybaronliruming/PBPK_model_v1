#!/usr/bin/env python3
"""Lock eligibility of label-blinded public-F PK-DB primary-source requests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {"request_id", "eligibility_decision", "source_doi", "source_pmid", "source_locator", "study_design",
            "parent_analyte_evidence", "absolute_f_evidence", "aggregation_requirement", "numeric_value_entered",
            "source_pdf_sha256", "reviewer", "review_date", "decision_note"}
VALID = {"accepted_primary_absolute_f", "accepted_primary_absolute_f_stratified", "pending_unreported_absolute_f", "pending_fulltext"}


def read_rows(path: Path, required: set[str]):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Schema mismatch: {path}")
        return list(reader)


def validate(decisions, queue):
    queue_ids = {row["request_id"] for row in queue}
    ids = [row["request_id"] for row in decisions]
    if len(ids) != len(set(ids)) or set(ids) != queue_ids:
        raise ValueError("Eligibility decisions must cover every request exactly once")
    for row in decisions:
        if row["eligibility_decision"] not in VALID:
            raise ValueError(f"Unsupported public-F eligibility decision: {row['eligibility_decision']}")
        if row["numeric_value_entered"].lower() != "false":
            raise ValueError("Eligibility lock cannot contain endpoint values")
        if row["eligibility_decision"].startswith("accepted"):
            for field in ["source_doi", "source_pmid", "source_locator", "study_design", "parent_analyte_evidence", "absolute_f_evidence", "source_pdf_sha256"]:
                if not row[field].strip():
                    raise ValueError(f"Accepted source lacks {field}: {row['request_id']}")
            if len(row["source_pdf_sha256"]) != 64:
                raise ValueError(f"Accepted source lacks a valid PDF hash: {row['request_id']}")
    return sorted(decisions, key=lambda row: row["request_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "results/analysis/public_f_primary_source_request_queue_v1")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/manual_review/public_f_pkdb_eligibility_decisions_v1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_pkdb_eligibility_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    queue_file = args.queue / "primary_source_request_queue_blinded.csv"
    startup_self_check([queue_file, args.queue / "complete.json", args.decisions], output=None if args.check_only else args.output)
    verify_stage(args.queue, "public_f_primary_source_request_queue")
    locked = validate(read_rows(args.decisions, REQUIRED), read_rows(queue_file, {"request_id"}))
    counts = {decision: sum(row["eligibility_decision"] == decision for row in locked) for decision in sorted(VALID)}
    if args.check_only:
        print(f"Public-F PK-DB eligibility lock valid: accepted={counts['accepted_primary_absolute_f'] + counts['accepted_primary_absolute_f_stratified']}")
        return
    with stage_output(args.output) as out:
        with (out / "eligibility_registry_locked.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(locked[0]))
            writer.writeheader(); writer.writerows(locked)
        (out / "README.md").write_text(
            "# Public-F PK-DB primary-source eligibility lock\n\n"
            "This lock records source-definition eligibility only and contains no numeric F values. Accepted sources have human oral/IV, "
            "parent-analyte and explicit absolute-F evidence. Stratified sources need an explicit aggregation rule in a later numeric lock. "
            "An AUC-derived F not directly reported by the primary study remains pending.\n", encoding="utf-8")
        finish_stage(out, "public_f_pkdb_eligibility_lock", inputs={"queue_complete_sha256": sha256(args.queue / "complete.json"), "queue_sha256": sha256(queue_file), "decisions_sha256": sha256(args.decisions)}, **counts, label_blinded=True, partial=False)
    print(f"Public-F PK-DB eligibility lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
