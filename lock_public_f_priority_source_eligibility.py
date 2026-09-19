#!/usr/bin/env python3
"""Lock B1 eligibility for a selected, label-blind public-F source batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {
    "doc_id", "eligibility_decision", "evidence_url", "evidence_level", "population_evidence", "oral_iv_design",
    "parent_analyte_evidence", "absolute_f_evidence", "condition_or_matrix_policy", "b2_fulltext_requirement",
    "numeric_value_entered", "decision_reason", "reviewer", "review_date",
}
ALLOWED = {
    "B1_qualified_absolute_F_condition_policy_required",
    "B1_qualified_absolute_F_matrix_policy_required",
}


def read_decisions(path: Path) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED <= set(reader.fieldnames):
            raise ValueError(f"Eligibility-decision schema mismatch: {path}")
        frame = pd.DataFrame(reader)
    if frame.empty or frame.doc_id.duplicated().any():
        raise ValueError("Eligibility decisions must be non-empty and unique by doc_id")
    if not set(frame.eligibility_decision) <= ALLOWED:
        raise ValueError(f"Unsupported B1 eligibility decision: {sorted(set(frame.eligibility_decision) - ALLOWED)}")
    if not frame.numeric_value_entered.eq("false").all():
        raise ValueError("B1 eligibility lock must not enter numeric values")
    return frame


def lock(queue: pd.DataFrame, decisions: pd.DataFrame, candidate_ids: set[str]) -> pd.DataFrame:
    required_queue = {"doc_id", "doi", "pubmed_id", "title", "bibliographic_route", "prior_definition_audit_status"}
    if not required_queue <= set(queue.columns):
        raise ValueError("Bibliographic prefilter queue schema mismatch")
    scope = queue.loc[queue.doc_id.astype(str).isin(candidate_ids)].copy()
    scope["doc_id"] = scope.doc_id.astype(str)
    if set(scope.doc_id) != candidate_ids or scope.doc_id.duplicated().any():
        raise ValueError("Requested eligibility source is absent or non-unique in prefilter")
    if not scope.prior_definition_audit_status.eq("not_previously_audited").all():
        raise ValueError("An already-audited source cannot be re-entered")
    if not scope.bibliographic_route.isin({"candidate_primary_human_pk_title_signal", "candidate_primary_pk_title_signal"}).all():
        raise ValueError("Requested source was not eligible for source-level definition review")
    if set(decisions.doc_id) != candidate_ids:
        raise ValueError("Eligibility decisions must cover requested documents exactly once")
    return scope.merge(decisions, on="doc_id", how="inner", validate="one_to_one").sort_values("doc_id", ignore_index=True)


def strict_f_isolation_status(registry: pd.DataFrame, candidate_ids: set[str]) -> pd.DataFrame:
    required = {"doc_id", "parent_id", "scaffold_group", "public_status"}
    if not required <= set(registry.columns):
        raise ValueError("Public-F candidate registry schema mismatch")
    scope = registry.loc[registry.doc_id.astype(str).isin(candidate_ids)].copy()
    scope["doc_id"] = scope.doc_id.astype(str)
    if set(scope.doc_id) != candidate_ids:
        raise ValueError("Requested eligibility source is absent from public-F candidate registry")
    grouped = scope.groupby("doc_id", as_index=False).agg(
        candidate_records=("doc_id", "size"), parent_ids=("parent_id", lambda values: ";".join(sorted(set(values)))),
        scaffold_groups=("scaffold_group", lambda values: ";".join(sorted(set(values)))),
        public_statuses=("public_status", lambda values: ";".join(sorted(set(values)))),
    )
    if not grouped.public_statuses.eq("quarantine_existing_strict_f_structure").all():
        raise ValueError("This lock expects explicitly strict-F-quarantined source records")
    grouped["strict_f_isolation_disposition"] = "source_qualified_but_quarantine_existing_strict_f_structure"
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefilter", type=Path, default=ROOT / "results/analysis/public_f_all_source_bibliographic_prefilter_v4")
    parser.add_argument("--registry", type=Path, default=ROOT / "results/analysis/public_f_development_candidate_registry_v1")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/manual_review/public_f_cyclosporine_priority_eligibility_decisions_v1.csv")
    parser.add_argument("--doc-ids", default="98780,98938")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_cyclosporine_eligibility_lock_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    candidate_ids = {value.strip() for value in args.doc_ids.split(",") if value.strip()}
    if not candidate_ids:
        raise ValueError("--doc-ids must contain at least one ChEMBL doc_id")
    queue_file = args.prefilter / "all_source_bibliographic_prefilter_blinded.csv"
    registry_file = args.registry / "chembl_public_candidates_blinded.csv"
    startup_self_check([queue_file, args.prefilter / "complete.json", registry_file, args.registry / "complete.json", args.decisions],
                       output=None if args.check_only else args.output)
    verify_stage(args.prefilter, "public_f_all_source_bibliographic_prefilter")
    verify_stage(args.registry, "public_f_development_candidate_registration")
    result = lock(pd.read_csv(queue_file, dtype=str, keep_default_na=False), read_decisions(args.decisions), candidate_ids)
    isolation = strict_f_isolation_status(pd.read_csv(registry_file, dtype=str, keep_default_na=False), candidate_ids)
    result = result.merge(isolation, on="doc_id", how="inner", validate="one_to_one")
    summary = {
        "selected_documents": len(result),
        "source_clusters": int(result.doi.nunique()),
        "parent_molecule": "cyclosporine",
        "b1_qualified_documents": len(result),
        "b2_numeric_lock_authorized": False,
        "b2_block_reason": "quarantine_existing_strict_f_structure",
        "numeric_values_read": False,
        "strict_f_modified": False,
        "public_f_training_authorized": False,
    }
    if args.check_only:
        print(f"Public-F priority eligibility lock valid: documents={len(result)} B1={len(result)}")
        return
    with stage_output(args.output) as out:
        result.to_csv(out / "source_eligibility_lock_blinded.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F priority source eligibility lock\n\n"
            "The two selected cyclosporine publications are B1-qualified primary human oral+IV studies with parent-analyte and directly "
            "reported absolute-F evidence, but all mapped records overlap frozen strict-F structure and are quarantined. They are therefore "
            "ineligible for B2 numerical extraction or public-F training. Independently, food/fasted conditions and blood/plasma matrix choices "
            "would have required pre-registration and no pooling. This stage reads no numeric endpoint values and changes neither strict-F nor training.\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_priority_source_eligibility_lock", inputs={
            "prefilter_complete_sha256": sha256(args.prefilter / "complete.json"),
            "registry_complete_sha256": sha256(args.registry / "complete.json"),
            "decisions_sha256": sha256(args.decisions),
        }, **summary, partial=False)
    print(f"Public-F priority eligibility lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
