#!/usr/bin/env python3
"""Build a label-blinded primary-source request queue for public-F candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


def priority(signal: str, pmid: str) -> tuple[int, str]:
    if signal == "absolute_bioavailability_title":
        return 1, "title explicitly states absolute bioavailability; verify primary human oral/IV design and table"
    if pmid:
        return 2, "bioavailability title with PMID; verify whether endpoint is absolute, human and source-locatable"
    return 3, "bioavailability title without PMID; first locate a citable primary source before eligibility review"


def build_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"sid", "reference_pmid", "reference_title", "reference_date", "substance_sid", "label", "chembl_id",
                "parent_id", "definition_signal", "public_status", "endpoint_values_hidden"}
    if not required <= set(candidates.columns):
        raise ValueError("Public-F PK-DB registry schema mismatch")
    frame = candidates.loc[candidates.public_status.eq("public_candidate_definition_audit")].copy()
    if not frame.endpoint_values_hidden.astype(bool).all():
        raise ValueError("Primary-source requests must remain label blinded")
    rows = []
    for (sid, pmid, title, date, signal), group in frame.groupby(
            ["sid", "reference_pmid", "reference_title", "reference_date", "definition_signal"], dropna=False, sort=True):
        rank, rationale = priority(str(signal), "" if pd.isna(pmid) else str(pmid))
        items = group.sort_values("substance_sid")
        rows.append({
            "request_id": stable_id(f"public-f-pkdb|{sid}|{pmid}|{title}"), "study_sid": sid,
            "reference_pmid": "" if pd.isna(pmid) else str(pmid), "reference_title": title,
            "reference_date": date, "definition_signal": signal, "request_priority": rank,
            "request_rationale": rationale, "candidate_count": len(items),
            "candidate_substance_sids": ";".join(items.substance_sid.astype(str)),
            "candidate_names": ";".join(items.label.astype(str)),
            "candidate_chembl_ids": ";".join(items.chembl_id.astype(str)),
            "eligibility_status": "not_reviewed", "endpoint_values_hidden": True,
        })
    return pd.DataFrame(rows).sort_values(["request_priority", "reference_date", "study_sid"], ascending=[True, False, True], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "results/analysis/public_f_development_candidate_registry_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_primary_source_request_queue_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.registry / "pkdb_public_candidates_blinded.csv"
    startup_self_check([source, args.registry / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.registry, "public_f_development_candidate_registration")
    queue = build_queue(pd.read_csv(source, keep_default_na=False))
    if args.check_only:
        print(f"Public-F primary-source request queue valid: studies={len(queue)} candidates={queue.candidate_count.sum()}")
        return
    with stage_output(args.output) as out:
        queue.to_csv(out / "primary_source_request_queue_blinded.csv", index=False)
        queue.groupby("request_priority").agg(studies=("study_sid", "size"), candidates=("candidate_count", "sum")).reset_index().to_csv(out / "queue_summary.csv", index=False)
        (out / "README.md").write_text(
            "# Public-F primary-source request queue\n\n"
            "This queue comes from the F-specific PK-DB re-screen, not the strict-F registry. It is label blinded and has no "
            "eligibility or numeric-extraction decision. A retrieved source must first establish whether its oral bioavailability is "
            "absolute, human, parent-analyte-specific and supported by an IV reference; otherwise it may only support a separately "
            "defined broad public task or be rejected.\n", encoding="utf-8")
        finish_stage(out, "public_f_primary_source_request_queue", inputs={"registry_complete_sha256": sha256(args.registry / "complete.json"), "source_sha256": sha256(source)}, studies=len(queue), candidates=int(queue.candidate_count.sum()), label_blinded=True, partial=False)
    print(f"Public-F primary-source request queue: {args.output}")


if __name__ == "__main__":
    run_cli(main)
