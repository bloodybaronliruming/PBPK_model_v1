#!/usr/bin/env python3
"""Publish a compact, reproducible composition audit for a Thalf external registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from merge_external_validation_decisions import validate_decisions
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check


def identifier_kind(identifier: str) -> str:
    value = str(identifier).strip().lower()
    if value.startswith("10."):
        return "doi"
    if value.startswith("pmid:"):
        return "pmid_fallback"
    if value:
        return "other_primary_identifier"
    return "missing"


def summarize_registry(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = validate_decisions(frame, "registry")
    accepted = registry.loc[registry.eligibility_decision.eq("accepted")].copy()
    overview = pd.DataFrame([
        {"metric": "records_total", "value": len(registry)},
        {"metric": "accepted_records", "value": len(accepted)},
        {"metric": "accepted_unique_molecules", "value": accepted.molecule_id.nunique()},
        {"metric": "accepted_unique_primary_sources", "value": accepted.primary_source_doi.nunique()},
        {"metric": "accepted_positive_hours_min", "value": accepted.extracted_value.astype(float).min() if len(accepted) else ""},
        {"metric": "accepted_positive_hours_max", "value": accepted.extracted_value.astype(float).max() if len(accepted) else ""},
    ])
    decisions = (registry.groupby("eligibility_decision", as_index=False).size()
                 .rename(columns={"size": "records"}).sort_values("eligibility_decision", ignore_index=True))
    sources = (accepted.assign(primary_source_identifier_kind=accepted.primary_source_doi.map(identifier_kind))
               .groupby(["primary_source_doi", "primary_source_identifier_kind"], as_index=False)
               .agg(accepted_records=("candidate_id", "size"), unique_molecules=("molecule_id", "nunique"))
               .sort_values(["accepted_records", "primary_source_doi"], ascending=[False, True], ignore_index=True))
    return overview, decisions, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "results/analysis/thalf_external_validation_decisions_v22.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/thalf_external_validation_composition_v22")
    args = parser.parse_args()
    startup_self_check([args.registry], output=args.output)
    overview, decisions, sources = summarize_registry(pd.read_csv(args.registry, keep_default_na=False))
    with stage_output(args.output) as out:
        overview.to_csv(out / "overview.csv", index=False)
        decisions.to_csv(out / "decision_counts.csv", index=False)
        sources.to_csv(out / "accepted_primary_source_counts.csv", index=False)
        (out / "README.md").write_text(
            "# External-validation registry composition audit\n\n"
            "This is an audit of the already locked decision registry, not a training set. It reports accepted-label "
            "coverage and source concentration so downstream model comparison can state the evidence base. "
            "A `pmid:<id>` identifier is an explicit fallback for a DOI-less primary paper, never a fabricated DOI.\n",
            encoding="utf-8")
        finish_stage(out, "thalf_external_validation_composition_audit", inputs={
            "registry_sha256": sha256(args.registry),
        }, records=int(overview.loc[overview.metric.eq("records_total"), "value"].iloc[0]), partial=False)


if __name__ == "__main__":
    run_cli(main)
