#!/usr/bin/env python3
"""Lock source-located B2 labels from non-overlapping public-F B1 sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {
    "doc_id", "extraction_decision", "source_doi", "source_pmid", "source_locator", "analyte",
    "study_population", "oral_dose", "iv_dose", "matrix", "reported_statistic", "reported_value_percent",
    "dispersion_type", "dispersion_value_percent", "canonical_unit", "canonical_value_fraction", "transformation",
    "canonical_estimator_rule", "reliability_tier", "source_pdf_sha256", "reviewer", "review_date", "extraction_note",
}
VARIANT_REQUIRED = {
    "doc_id", "variant_id", "is_canonical", "reported_statistic", "reported_value_percent", "dispersion_or_range",
    "source_locator", "use_constraint",
}
ALLOWED_TIERS = {"R1_primary_locked", "R2_primary_conditional"}


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Schema mismatch: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows: {path}")
    return rows


def validate(extractions: list[dict[str, str]], variants: list[dict[str, str]], eligibility: list[dict[str, str]]) -> list[dict[str, str]]:
    accepted = {row["doc_id"]: row for row in eligibility if row["eligibility_decision"].startswith("B1_qualified_")}
    ids = [row["doc_id"] for row in extractions]
    if len(ids) != len(set(ids)) or set(ids) != set(accepted):
        raise ValueError("Extraction records must cover each B1-qualified source exactly once")
    variants_by_doc: dict[str, list[dict[str, str]]] = {}
    for row in variants:
        variants_by_doc.setdefault(row["doc_id"], []).append(row)
    for row in extractions:
        doc_id = row["doc_id"]
        if row["extraction_decision"] != "accepted":
            raise ValueError(f"Unexpected extraction decision: {doc_id}")
        locked = accepted[doc_id]
        if row["source_doi"].lower() != locked["doi"].lower() or row["source_pmid"] != locked["pubmed_id"]:
            raise ValueError(f"Extraction source differs from B1 lock: {doc_id}")
        reported, canonical = float(row["reported_value_percent"]), float(row["canonical_value_fraction"])
        if row["canonical_unit"] != "fraction" or row["transformation"] != "percent_to_fraction_no_clipping":
            raise ValueError(f"Invalid F canonicalization: {doc_id}")
        if reported <= 0 or canonical <= 0 or abs(canonical - reported / 100.0) > 1e-12:
            raise ValueError(f"Invalid reported F or conversion: {doc_id}")
        if row["reliability_tier"] not in ALLOWED_TIERS or len(row["source_pdf_sha256"]) != 64:
            raise ValueError(f"Missing reliability tier or PDF hash: {doc_id}")
        source_variants = variants_by_doc.get(doc_id, [])
        canonical_variants = [item for item in source_variants if item["is_canonical"] == "true"]
        if len(canonical_variants) != 1 or float(canonical_variants[0]["reported_value_percent"]) != reported:
            raise ValueError(f"Exactly one matching canonical variant is required: {doc_id}")
        if any(item["is_canonical"] not in {"true", "false"} for item in source_variants):
            raise ValueError(f"Invalid canonical flag: {doc_id}")
    if set(variants_by_doc) != set(accepted):
        raise ValueError("Variant registry must cover exactly the B1-qualified sources")
    return sorted(extractions, key=lambda row: row["doc_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility", type=Path,
                        default=ROOT / "results/analysis/public_f_haloperidol_phenobarbital_eligibility_lock_v1")
    parser.add_argument("--extractions", type=Path,
                        default=ROOT / "data/manual_review/public_f_unquarantined_priority_extraction_decisions_v1.csv")
    parser.add_argument("--variants", type=Path,
                        default=ROOT / "data/manual_review/public_f_unquarantined_priority_reported_variants_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/public_f_haloperidol_phenobarbital_extraction_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    eligibility_file = args.eligibility / "source_eligibility_lock_blinded.csv"
    startup_self_check([eligibility_file, args.eligibility / "complete.json", args.extractions, args.variants],
                       output=None if args.check_only else args.output)
    verify_stage(args.eligibility, "public_f_unquarantined_priority_source_eligibility_lock")
    locked = validate(read_rows(args.extractions, REQUIRED), read_rows(args.variants, VARIANT_REQUIRED),
                      read_rows(eligibility_file, {"doc_id", "eligibility_decision", "doi", "pubmed_id"}))
    if args.check_only:
        print(f"Public-F unquarantined priority extraction lock valid: records={len(locked)}")
        return
    with stage_output(args.output) as out:
        with (out / "extraction_registry_locked.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(locked[0]))
            writer.writeheader()
            writer.writerows(locked)
        variants = read_rows(args.variants, VARIANT_REQUIRED)
        with (out / "source_reported_variants_locked.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(variants[0]))
            writer.writeheader()
            writer.writerows(variants)
        (out / "README.md").write_text(
            "# Public-F unquarantined priority numeric extraction lock\n\n"
            "Two non-overlapping B1 primary sources are locked as one study-level label each. Both are R2 primary-conditional "
            "evidence: haloperidol is a continuous-treatment patient study and phenobarbital's selected estimate is an author-reported "
            "adjusted availability for one tablet formulation. The companion variant registry preserves same-source alternatives without "
            "turning them into duplicate labels. This public-development lock changes neither strict-F nor any trained model.\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_unquarantined_priority_extraction_lock", inputs={
            "eligibility_complete_sha256": sha256(args.eligibility / "complete.json"),
            "eligibility_registry_sha256": sha256(eligibility_file),
            "extractions_sha256": sha256(args.extractions),
            "variants_sha256": sha256(args.variants),
        }, records=len(locked), molecules=len({row["analyte"] for row in locked}), canonical_unit="fraction",
           reliability_tiers=sorted({row["reliability_tier"] for row in locked}), strict_f_modified=False,
           public_f_training_authorized=False, partial=False)
    print(f"Public-F unquarantined priority extraction lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
