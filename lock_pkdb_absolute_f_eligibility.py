#!/usr/bin/env python3
"""Lock primary-source eligibility for label-blinded PK-DB absolute oral-F candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {
    "candidate_id", "eligibility_decision", "source_doi", "source_pmid", "source_locator", "study_population",
    "oral_iv_design", "parent_analyte_evidence", "absolute_f_evidence", "numeric_value_entered", "reviewer",
    "review_date", "decision_note",
}
REQUIRED_ACCEPTED = REQUIRED - {"numeric_value_entered"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED <= set(reader.fieldnames):
            raise ValueError(f"Eligibility schema mismatch: {path}")
        return list(reader)


def validate(decisions: list[dict[str, str]], queue: list[dict[str, str]]) -> list[dict[str, str]]:
    queue_ids = {row["candidate_id"] for row in queue}
    ids = [row["candidate_id"] for row in decisions]
    if len(ids) != len(set(ids)) or set(ids) != queue_ids:
        raise ValueError("Eligibility decisions must cover each queued candidate exactly once")
    for row in decisions:
        if row["eligibility_decision"] not in {"accepted", "rejected", "pending"}:
            raise ValueError(f"Unsupported eligibility decision: {row['eligibility_decision']}")
        if row["numeric_value_entered"].lower() != "false":
            raise ValueError("Eligibility lock must not contain a numeric endpoint value")
        if row["eligibility_decision"] == "accepted":
            for column in REQUIRED_ACCEPTED:
                if not row[column].strip():
                    raise ValueError(f"Accepted record lacks {column}: {row['candidate_id']}")
            if "iv" not in row["oral_iv_design"].lower() or "oral" not in row["oral_iv_design"].lower():
                raise ValueError(f"Accepted record lacks oral+IV evidence: {row['candidate_id']}")
            if "plasma" not in row["parent_analyte_evidence"].lower():
                raise ValueError(f"Accepted record lacks plasma parent-analyte evidence: {row['candidate_id']}")
    return sorted(decisions, key=lambda row: row["candidate_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "results/analysis/pkdb_absolute_f_candidate_queue_v1")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/manual_review/pkdb_absolute_f_eligibility_decisions_v1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/pkdb_absolute_f_eligibility_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    queue_file = args.queue / "candidate_records_blinded.csv"
    startup_self_check([queue_file, args.queue / "complete.json", args.decisions],
                       output=None if args.check_only else args.output)
    verify_stage(args.queue, "pkdb_absolute_f_candidate_queue")
    decisions = read_csv(args.decisions)
    with queue_file.open(newline="", encoding="utf-8") as stream:
        queue_rows = list(csv.DictReader(stream))
    locked = validate(decisions, queue_rows)
    if args.check_only:
        print(f"PK-DB absolute-F eligibility contract valid: accepted={sum(r['eligibility_decision'] == 'accepted' for r in locked)}")
        return
    with stage_output(args.output) as out:
        with (out / "eligibility_registry_locked.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(locked[0]))
            writer.writeheader()
            writer.writerows(locked)
        (out / "README.md").write_text(
            "# PK-DB absolute oral-F eligibility lock\n\n"
            "This lock records only whether each queued primary source supports the endpoint definition. It contains no F "
            "value. An accepted record confirms a human oral-plus-IV design, parent plasma analyte evidence, absolute-F "
            "definition and page/table locator. A later numeric extraction stage must cite this exact lock and must preserve "
            "the study's geometric-mean/CI statistic and microdose design provenance.\n",
            encoding="utf-8")
        finish_stage(out, "pkdb_absolute_f_eligibility_lock", inputs={
            "queue_complete_sha256": sha256(args.queue / "complete.json"),
            "queue_records_sha256": sha256(queue_file),
            "decisions_sha256": sha256(args.decisions),
        }, accepted=sum(r["eligibility_decision"] == "accepted" for r in locked),
           rejected=sum(r["eligibility_decision"] == "rejected" for r in locked),
           pending=sum(r["eligibility_decision"] == "pending" for r in locked), label_blinded=True, partial=False)
    print(f"PK-DB absolute-F eligibility lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
