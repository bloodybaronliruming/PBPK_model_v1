#!/usr/bin/env python3
"""Lock definition decisions for the primary-source-review branch of public-F discovery.

The stage consumes only the label-blind source aggregation and bibliographic
queue.  It records source and endpoint semantics, never endpoint values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


FORBIDDEN = {
    "standard_value", "canonical_value", "raw_value", "target_value", "lower_bound", "upper_bound",
    "value", "mean", "median", "y",
}
REQUIRED_DECISION_COLUMNS = {
    "doc_id", "decision", "source_type", "evidence_url", "population_evidence", "route_evidence",
    "parent_analyte_evidence", "endpoint_semantics", "primary_provenance", "numeric_value_entered",
    "decision_reason", "reviewer", "review_date",
}
ALLOWED_DECISIONS = {
    "rejected_in_vitro_or_in_silico_nonhuman",
    "rejected_secondary_perspective",
    "rejected_relative_oral_bioavailability",
}


def read_decisions(path: Path) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_DECISION_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f"Definition-decision schema mismatch: {path}")
        decisions = pd.DataFrame(reader)
    if decisions.empty or decisions.doc_id.duplicated().any():
        raise ValueError("Definition decisions must be non-empty and unique by doc_id")
    if not set(decisions.decision) <= ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported source definition decision: {sorted(set(decisions.decision) - ALLOWED_DECISIONS)}")
    if not decisions.numeric_value_entered.eq("false").all():
        raise ValueError("Priority-source definition audit must not enter a numeric value")
    return decisions


def audit(queue: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    if FORBIDDEN & set(queue.columns):
        raise ValueError("Source definition audit received a numeric endpoint column")
    required = {"doc_id", "retrieval_route", "doi", "pubmed_id", "title", "priority"}
    if not required <= set(queue.columns):
        raise ValueError("Metadata queue schema mismatch")
    scope = queue.loc[queue.retrieval_route.eq("primary_source_definition_review_required")].copy()
    scope["doc_id"] = scope.doc_id.astype(str)
    if set(scope.doc_id) != set(decisions.doc_id) or scope.doc_id.duplicated().any():
        raise ValueError("Decisions must cover every primary-source-review document exactly once")
    result = scope.merge(decisions, on="doc_id", how="inner", validate="one_to_one")
    return result.sort_values("selection_rank", key=lambda values: values.astype(int), ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-queue", type=Path, default=ROOT / "results/analysis/public_f_source_metadata_audit_queue_v2")
    parser.add_argument("--aggregation", type=Path, default=ROOT / "results/analysis/public_f_source_aggregation_v2")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/manual_review/public_f_priority_source_definition_decisions_v1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_priority_source_definition_audit_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    queue_file = args.metadata_queue / "source_metadata_audit_queue_blinded.csv"
    required = [queue_file, args.metadata_queue / "complete.json", args.aggregation / "complete.json", args.decisions]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.metadata_queue, "public_f_source_metadata_audit_queue")
    verify_stage(args.aggregation, "public_f_source_aggregation")
    result = audit(pd.read_csv(queue_file, dtype=str, keep_default_na=False), read_decisions(args.decisions))
    summary = {
        "reviewed_primary_source_route_documents": len(result),
        "decision_counts": result.decision.value_counts().sort_index().to_dict(),
        "b1_definition_admitted_documents": 0,
        "b2_numeric_lock_authorized": False,
        "numeric_values_read": False,
        "strict_f_modified": False,
        "public_f_training_authorized": False,
    }
    if args.check_only:
        print(f"Public-F priority source definition audit valid: documents={len(result)} B1=0")
        return
    with stage_output(args.output) as out:
        result.to_csv(out / "priority_source_definition_audit_blinded.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F priority source definition audit\n\n"
            "This stage locks definition decisions for every source routed to primary-source review by the metadata queue. "
            "It records no endpoint values. All three reviewed sources are excluded: one is in-vitro/in-silico, one is a "
            "secondary Perspective, and one primary healthy-volunteer oral PK study reports relative oral bioavailability without "
            "an IV reference. Therefore no B1/B2 public-F admission, numerical lock, training, or strict-F change is authorized.\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_priority_source_definition_audit", inputs={
            "metadata_queue_complete_sha256": sha256(args.metadata_queue / "complete.json"),
            "aggregation_complete_sha256": sha256(args.aggregation / "complete.json"),
            "decisions_sha256": sha256(args.decisions),
        }, **summary, partial=False)
    print(f"Public-F priority source definition audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
