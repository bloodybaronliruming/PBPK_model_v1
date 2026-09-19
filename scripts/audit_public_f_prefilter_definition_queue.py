#!/usr/bin/env python3
"""Lock non-numeric source-definition decisions from a public-F prefilter queue.

This is a rejection-only stage: it documents why selected bibliographic leads
cannot enter B1 and deliberately contains no endpoint values.
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
REQUIRED = {
    "doc_id", "decision", "source_type", "evidence_url", "population_evidence", "route_evidence",
    "parent_analyte_evidence", "endpoint_semantics", "primary_provenance", "numeric_value_entered",
    "decision_reason", "reviewer", "review_date",
}
ALLOWED = {
    "rejected_oral_only_combination_PK_not_absolute_F",
    "rejected_oral_only_PK_not_absolute_F",
    "rejected_human_microdose_no_documented_oral_iv_pair",
    "rejected_metabolite_analyte_not_parent_absolute_F",
    "rejected_relative_oral_bioavailability_metabolite_not_absolute_F",
    "rejected_medchem_optimization_not_human_F_source",
    "rejected_nonhuman_prodrug_PK_not_human_F",
}


def read_decisions(path: Path) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED <= set(reader.fieldnames):
            raise ValueError(f"Definition-decision schema mismatch: {path}")
        decisions = pd.DataFrame(reader)
    if decisions.empty or decisions.doc_id.duplicated().any():
        raise ValueError("Definition decisions must be non-empty and unique by doc_id")
    if not set(decisions.decision) <= ALLOWED:
        raise ValueError(f"Unsupported source definition decision: {sorted(set(decisions.decision) - ALLOWED)}")
    if not decisions.numeric_value_entered.eq("false").all():
        raise ValueError("Prefilter definition audit must not enter a numeric value")
    return decisions


def audit(queue: pd.DataFrame, decisions: pd.DataFrame, candidate_ids: set[str]) -> pd.DataFrame:
    if FORBIDDEN & set(queue.columns):
        raise ValueError("Prefilter definition audit received a numeric endpoint column")
    required_queue = {
        "doc_id", "definition_review_selection_rank", "doi", "pubmed_id", "title", "priority",
        "bibliographic_route", "prior_definition_audit_status",
    }
    if not required_queue <= set(queue.columns):
        raise ValueError("Prefilter definition queue schema mismatch")
    scope = queue.loc[queue.doc_id.astype(str).isin(candidate_ids)].copy()
    scope["doc_id"] = scope.doc_id.astype(str)
    if set(scope.doc_id) != candidate_ids or scope.doc_id.duplicated().any():
        raise ValueError("Requested sources are absent or non-unique in the bounded definition queue")
    if not scope.prior_definition_audit_status.eq("not_previously_audited").all():
        raise ValueError("A previously audited source cannot be re-entered")
    if not scope.bibliographic_route.isin({
        "candidate_primary_human_pk_title_signal", "candidate_primary_pk_title_signal",
    }).all():
        raise ValueError("A selected source is not eligible for source-level definition review")
    if set(decisions.doc_id) != candidate_ids:
        raise ValueError("Definition decisions must cover requested documents exactly once")
    return scope.merge(decisions, on="doc_id", how="inner", validate="one_to_one").sort_values(
        "definition_review_selection_rank", key=lambda values: values.astype(int), ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefilter", type=Path,
                        default=ROOT / "results/analysis/public_f_all_source_bibliographic_prefilter_v9")
    parser.add_argument("--decisions", type=Path,
                        default=ROOT / "data/manual_review/public_f_prefilter_definition_decisions_v4.csv")
    parser.add_argument("--doc-ids", default="98940,117239,74643")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/public_f_prefilter_definition_audit_v4")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    candidate_ids = {value.strip() for value in args.doc_ids.split(",") if value.strip()}
    if not candidate_ids:
        raise ValueError("--doc-ids must contain at least one ChEMBL doc_id")
    queue_file = args.prefilter / "priority_definition_review_queue_blinded.csv"
    startup_self_check([queue_file, args.prefilter / "complete.json", args.decisions],
                       output=None if args.check_only else args.output)
    verify_stage(args.prefilter, "public_f_all_source_bibliographic_prefilter")
    result = audit(pd.read_csv(queue_file, dtype=str, keep_default_na=False), read_decisions(args.decisions), candidate_ids)
    summary = {
        "reviewed_documents": len(result),
        "decision_counts": result.decision.value_counts().sort_index().to_dict(),
        "b1_definition_admitted_documents": 0,
        "b2_numeric_lock_authorized": False,
        "numeric_values_read": False,
        "strict_f_modified": False,
        "public_f_training_authorized": False,
    }
    if args.check_only:
        print(f"Public-F prefilter definition audit valid: documents={len(result)} B1=0")
        return
    with stage_output(args.output) as out:
        result.to_csv(out / "prefilter_definition_audit_blinded.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F prefilter definition audit\n\n"
            "This rejection-only stage resolves three bounded, label-blind bibliographic leads. Each is a human PK or "
            "human-microdose source, but none documents a source-local paired oral+IV direct absolute-F design under the "
            "public-F protocol. No endpoint number was read or entered; no B1/B2 admission, training, or strict-F update is authorized.\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_prefilter_definition_audit", inputs={
            "prefilter_complete_sha256": sha256(args.prefilter / "complete.json"),
            "decisions_sha256": sha256(args.decisions),
        }, **summary, partial=False)
    print(f"Public-F prefilter definition audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
