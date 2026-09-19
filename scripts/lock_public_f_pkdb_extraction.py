#!/usr/bin/env python3
"""Lock numeric public-F labels from accepted, non-stratified PK-DB primary sources."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {
    "request_id", "extraction_decision", "source_doi", "source_pmid", "source_locator", "analyte",
    "study_population", "oral_dose", "iv_dose", "reported_statistic", "reported_value_percent", "interval_type",
    "interval_lower_percent", "interval_upper_percent", "canonical_unit", "canonical_value_fraction", "transformation",
    "canonical_estimator_rule", "source_pdf_sha256", "reviewer", "review_date", "extraction_note",
}
ACCEPTED = {"accepted_primary_absolute_f"}
VALID_INTERVALS = {"90%_confidence_interval", "95%_confidence_interval", "observed_range"}


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Schema mismatch: {path}")
        return list(reader)


def validate(extractions: list[dict[str, str]], eligibility: list[dict[str, str]]) -> list[dict[str, str]]:
    accepted = {row["request_id"]: row for row in eligibility if row["eligibility_decision"] in ACCEPTED}
    ids = [row["request_id"] for row in extractions]
    if len(ids) != len(set(ids)) or set(ids) != set(accepted):
        raise ValueError("Extraction records must cover each directly accepted source exactly once; stratified sources stay excluded")
    for row in extractions:
        if row["extraction_decision"] != "accepted":
            raise ValueError(f"Unsupported extraction decision: {row['request_id']}")
        locked = accepted[row["request_id"]]
        if row["source_doi"].lower() != locked["source_doi"].lower() or row["source_pmid"] != locked["source_pmid"]:
            raise ValueError(f"Extraction source differs from eligibility lock: {row['request_id']}")
        if row["interval_type"] not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval type: {row['request_id']}")
        value = float(row["reported_value_percent"])
        lower, upper = float(row["interval_lower_percent"]), float(row["interval_upper_percent"])
        canonical = float(row["canonical_value_fraction"])
        if not (0.0 < lower <= value <= upper):
            raise ValueError(f"Invalid reported estimate/interval: {row['request_id']}")
        if row["canonical_unit"] != "fraction" or row["transformation"] != "percent_to_fraction_no_clipping":
            raise ValueError(f"Invalid canonical F semantics: {row['request_id']}")
        if not canonical > 0.0 or abs(canonical - value / 100.0) > 1e-12:
            raise ValueError(f"Incorrect percent-to-fraction conversion: {row['request_id']}")
        if len(row["source_pdf_sha256"]) != 64:
            raise ValueError(f"Missing source PDF hash: {row['request_id']}")
    return sorted(extractions, key=lambda row: row["request_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility", type=Path,
                        default=ROOT / "results/analysis/public_f_pkdb_eligibility_lock_v1")
    parser.add_argument("--extractions", type=Path,
                        default=ROOT / "data/manual_review/public_f_pkdb_extraction_decisions_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/public_f_pkdb_extraction_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    eligibility_file = args.eligibility / "eligibility_registry_locked.csv"
    startup_self_check([eligibility_file, args.eligibility / "complete.json", args.extractions],
                       output=None if args.check_only else args.output)
    verify_stage(args.eligibility, "public_f_pkdb_eligibility_lock")
    locked = validate(read_rows(args.extractions, REQUIRED), read_rows(eligibility_file, {"request_id", "eligibility_decision", "source_doi", "source_pmid"}))
    if args.check_only:
        print(f"Public-F PK-DB extraction contract valid: records={len(locked)}")
        return
    with stage_output(args.output) as out:
        with (out / "extraction_registry_locked.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(locked[0]))
            writer.writeheader()
            writer.writerows(locked)
        (out / "README.md").write_text(
            "# Public-F PK-DB numeric extraction lock\\n\\n"
            "This lock contains source-located labels from directly accepted human oral/IV primary studies. It preserves the original "
            "statistic and interval type rather than forcing a common uncertainty measure. Labels are converted from percent to fraction "
            "without clipping, so an observed estimate above 100% remains auditable. Omeprazole's genotype-stratified source remains out "
            "of this lock pending a separately registered strata policy. This is a public-development source lock, not a strict-F external "
            "set; it changes no processed dataset or frozen model.\\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_pkdb_extraction_lock", inputs={
            "eligibility_complete_sha256": sha256(args.eligibility / "complete.json"),
            "eligibility_registry_sha256": sha256(eligibility_file),
            "extractions_sha256": sha256(args.extractions),
        }, records=len(locked), molecules=len({row["analyte"] for row in locked}), canonical_unit="fraction",
           strict_f_external_set=False, omeprazole_stratified_source_deferred=True, partial=False)
    print(f"Public-F PK-DB extraction lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
