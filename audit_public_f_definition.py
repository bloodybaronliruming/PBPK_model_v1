#!/usr/bin/env python3
"""Audit high-signal public-F candidates at document/assay level without opening numeric labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {"doc_id", "assay_id", "doi", "pubmed_id", "definition_decision", "evidence_type", "evidence_url",
            "endpoint_semantics", "route_evidence", "decision_reason", "reviewer", "review_date"}
ADMITTED = {"accepted_public_absolute_f", "accepted_public_oral_f_broad"}


def read_decisions(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED <= set(reader.fieldnames):
            raise ValueError(f"Definition-decision schema mismatch: {path}")
        return list(reader)


def audit(candidates, decisions):
    required_candidates = {"activity_id", "doc_id", "assay_id", "doi", "pubmed_id", "definition_signal", "public_status"}
    if not required_candidates <= set(candidates.columns):
        raise ValueError("Public candidate schema mismatch")
    scope = candidates.loc[(candidates.public_status == "public_candidate_definition_audit") &
                           (candidates.definition_signal == "absolute_bioavailability_description")].copy()
    keys = set(zip(scope.doc_id.astype(str), scope.assay_id.astype(str)))
    decision_keys = [(row["doc_id"], row["assay_id"]) for row in decisions]
    if len(decision_keys) != len(set(decision_keys)) or set(decision_keys) != keys:
        raise ValueError("Definition decisions must cover every high-signal document/assay exactly once")
    decision_frame = __import__("pandas").DataFrame(decisions)
    result = scope.merge(decision_frame, on=["doc_id", "assay_id", "doi", "pubmed_id"], how="inner", validate="many_to_one")
    invalid = set(result.definition_decision) - (ADMITTED | {"rejected_endpoint_mismatch", "rejected_secondary_review", "pending_primary_source"})
    if invalid:
        raise ValueError(f"Unsupported definition decision: {sorted(invalid)}")
    return result.sort_values("activity_id", ignore_index=True)


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "results/analysis/public_f_development_candidate_registry_v1")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/manual_review/public_f_definition_decisions_v1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_definition_audit_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    candidate_file = args.registry / "chembl_public_candidates_blinded.csv"
    startup_self_check([candidate_file, args.registry / "complete.json", args.decisions], output=None if args.check_only else args.output)
    verify_stage(args.registry, "public_f_development_candidate_registration")
    result = audit(pd.read_csv(candidate_file, dtype=str), read_decisions(args.decisions))
    summary = {
        "high_signal_records": len(result), "high_signal_molecules": result.molecule_id.nunique(),
        "document_assays": result[["doc_id", "assay_id"]].drop_duplicates().shape[0],
        "admitted_records": int(result.definition_decision.isin(ADMITTED).sum()),
        "rejected_endpoint_mismatch_records": int(result.definition_decision.eq("rejected_endpoint_mismatch").sum()),
        "rejected_secondary_review_records": int(result.definition_decision.eq("rejected_secondary_review").sum()),
        "label_blinded": True,
    }
    if args.check_only:
        print(f"Public-F definition audit valid: high-signal={summary['high_signal_records']} admitted={summary['admitted_records']}")
        return
    with stage_output(args.output) as out:
        result.to_csv(out / "document_assay_definition_audit.csv", index=False)
        result.loc[result.definition_decision.isin(ADMITTED)].to_csv(out / "cohort_admission_candidates_blinded.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F high-signal definition audit\n\n"
            "The ChEMBL description signal `absolute bioavailability` is audited at document/assay level before any label can enter a public cohort. "
            "This v1 audit finds no admissible public-F candidate: one source is a human jejunal Peff compilation and the other is a review. "
            "Neither supplies a primary or clearly defined human absolute-oral-F endpoint. Numeric labels are not read or emitted.\n",
            encoding="utf-8")
        finish_stage(out, "public_f_definition_audit", inputs={"registry_complete_sha256": sha256(args.registry / "complete.json"),
            "candidate_sha256": sha256(candidate_file), "decisions_sha256": sha256(args.decisions)}, **summary, partial=False)
    print(f"Public-F definition audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
