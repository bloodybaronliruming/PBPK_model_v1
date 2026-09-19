#!/usr/bin/env python3
"""Lock extracted numeric human absolute oral-F labels after eligibility review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {
    "candidate_id", "extraction_decision", "source_doi", "source_pmid", "source_locator", "analyte",
    "study_population", "oral_dose", "iv_microdose", "reported_statistic", "reported_value_percent", "ci_level",
    "ci_lower_percent", "ci_upper_percent", "canonical_unit", "canonical_value_fraction", "transformation",
    "source_pdf_sha256", "reviewer", "review_date", "extraction_note",
}


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Schema mismatch: {path}")
        return list(reader)


def validate(extractions: list[dict[str, str]], eligibility: list[dict[str, str]]) -> list[dict[str, str]]:
    accepted = {row["candidate_id"]: row for row in eligibility if row["eligibility_decision"] == "accepted"}
    ids = [row["candidate_id"] for row in extractions]
    if len(ids) != len(set(ids)) or set(ids) != set(accepted):
        raise ValueError("Extraction records must cover each accepted eligibility record exactly once")
    for row in extractions:
        if row["extraction_decision"] != "accepted":
            raise ValueError(f"Unexpected extraction decision for accepted record: {row['candidate_id']}")
        locked = accepted[row["candidate_id"]]
        if row["source_doi"] != locked["source_doi"] or row["source_pmid"] != locked["source_pmid"]:
            raise ValueError(f"Extraction source differs from eligibility lock: {row['candidate_id']}")
        if row["reported_statistic"] != "geometric_mean" or row["ci_level"] != "90":
            raise ValueError(f"Pre-specified reported-statistic contract violated: {row['candidate_id']}")
        reported = float(row["reported_value_percent"])
        lower, upper = float(row["ci_lower_percent"]), float(row["ci_upper_percent"])
        canonical = float(row["canonical_value_fraction"])
        if not (0.0 < lower <= reported <= upper <= 100.0):
            raise ValueError(f"Invalid F estimate/CI: {row['candidate_id']}")
        if row["canonical_unit"] != "fraction" or row["transformation"] != "identity" or not (0.0 < canonical <= 1.0):
            raise ValueError(f"Invalid canonical F semantics: {row['candidate_id']}")
        if abs(canonical - reported / 100.0) > 1e-12:
            raise ValueError(f"Incorrect percent-to-fraction conversion: {row['candidate_id']}")
        if len(row["source_pdf_sha256"]) != 64:
            raise ValueError(f"Missing source PDF hash: {row['candidate_id']}")
    return sorted(extractions, key=lambda row: row["candidate_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility", type=Path,
                        default=ROOT / "results/analysis/pkdb_absolute_f_eligibility_lock_v1")
    parser.add_argument("--extractions", type=Path,
                        default=ROOT / "data/manual_review/pkdb_absolute_f_extraction_decisions_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pkdb_absolute_f_extraction_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    eligibility_file = args.eligibility / "eligibility_registry_locked.csv"
    startup_self_check([eligibility_file, args.eligibility / "complete.json", args.extractions],
                       output=None if args.check_only else args.output)
    verify_stage(args.eligibility, "pkdb_absolute_f_eligibility_lock")
    eligibility = read_rows(eligibility_file, {"candidate_id", "eligibility_decision", "source_doi", "source_pmid"})
    locked = validate(read_rows(args.extractions, REQUIRED), eligibility)
    if args.check_only:
        print(f"PK-DB absolute-F extraction contract valid: records={len(locked)}")
        return
    with stage_output(args.output) as out:
        with (out / "extraction_registry_locked.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(locked[0]))
            writer.writeheader()
            writer.writerows(locked)
        (out / "README.md").write_text(
            "# PK-DB absolute oral-F numeric extraction lock\n\n"
            "This lock contains two source-located human absolute oral-bioavailability labels. The canonical label is the "
            "Table 1 geometric-mean Fp.o. value converted from percent to fraction; the 90% CI, sample size, oral dose and "
            "same-study IV microdose are retained as provenance and must not be expanded into duplicate training labels. "
            "The lock is not a model evaluation set and does not change processed_v15 or any frozen model.\n",
            encoding="utf-8")
        finish_stage(out, "pkdb_absolute_f_extraction_lock", inputs={
            "eligibility_complete_sha256": sha256(args.eligibility / "complete.json"),
            "eligibility_registry_sha256": sha256(eligibility_file),
            "extractions_sha256": sha256(args.extractions),
        }, records=len(locked), molecules=len({row["candidate_id"] for row in locked}),
           statistic="geometric_mean", ci_level=90, canonical_unit="fraction", partial=False)
    print(f"PK-DB absolute-F extraction lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
