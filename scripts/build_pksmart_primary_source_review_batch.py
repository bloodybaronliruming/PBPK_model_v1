#!/usr/bin/env python3
"""Create a label-blinded eligibility-review batch for PKSmart primary sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


INDEX_COLUMNS = [
    "candidate_id", "source_row", "molecule_id", "parent_id", "scaffold_group", "chembl_id", "chembl_pref_name",
    "identity_mapping_basis", "primary_source_locator", "fulltext_path", "document_title", "endpoint_values_hidden",
]
ELIGIBILITY_COLUMNS = [
    "candidate_id", "source_row", "molecule_id", "chembl_id", "chembl_pref_name", "identity_mapping_basis",
    "primary_source_locator", "fulltext_path", "eligibility_decision", "human_evidence", "iv_route_evidence",
    "parent_systemic_evidence", "terminal_phase_evidence", "source_table_or_page", "exclusion_reason",
    "evidence_note", "reviewer", "review_date",
]


def build_batch(requests: pd.DataFrame, candidates: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame, required in [
        (requests, {"candidate_id", "chembl_id", "chembl_pref_name", "identity_mapping_basis", "primary_source_locator", "primary_source_status", "endpoint_values_hidden"}),
        (candidates, {"candidate_id", "source_row", "molecule_id", "parent_id", "scaffold_group", "endpoint_values_hidden"}),
        (manifest, {"candidate_id", "primary_source_locator", "fulltext_path", "document_title"}),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"PKSmart review input misses columns: {sorted(missing)}")
    chosen = requests.loc[requests.primary_source_status.eq("primary_article_located")].copy()
    if chosen.empty or not chosen.endpoint_values_hidden.astype(bool).all():
        raise ValueError("PKSmart primary-source review must start from label-blinded located articles")
    index = chosen.merge(candidates[["candidate_id", "source_row", "molecule_id", "parent_id", "scaffold_group"]],
                         on="candidate_id", validate="one_to_one")
    index = index.merge(manifest[["candidate_id", "primary_source_locator", "fulltext_path", "document_title"]],
                        on=["candidate_id", "primary_source_locator"], validate="one_to_one")
    if not index.endpoint_values_hidden.astype(bool).all() or index.candidate_id.duplicated().any():
        raise ValueError("PKSmart review index is non-blinded or non-unique")
    index = index[INDEX_COLUMNS].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    template = index[["candidate_id", "source_row", "molecule_id", "chembl_id", "chembl_pref_name",
                      "identity_mapping_basis", "primary_source_locator", "fulltext_path"]].copy()
    for column in ELIGIBILITY_COLUMNS:
        if column not in template:
            template[column] = ""
    return index, template[ELIGIBILITY_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_request_queue_v1")
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_candidate_queue_v1")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/primary_fulltext_manifest_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_review_batch_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.requests, "pksmart_primary_source_request_queue")
    verify_stage(args.candidates, "pksmart_external_candidate_queue")
    request_file = args.requests / "primary_source_request_queue_blinded.csv"
    candidate_file = args.candidates / "candidate_records_blinded.csv"
    startup_self_check([request_file, candidate_file, args.manifest], output=None if args.check_only else args.output)
    index, template = build_batch(pd.read_csv(request_file, keep_default_na=False),
                                  pd.read_csv(candidate_file, keep_default_na=False),
                                  pd.read_csv(args.manifest, keep_default_na=False))
    for path in index.fulltext_path:
        fulltext = ROOT / path
        if not fulltext.is_file():
            raise FileNotFoundError(fulltext)
    if args.check_only:
        print(f"PKSmart primary-source review contract valid: candidates={len(index)}; labels not loaded")
        return
    with stage_output(args.output) as out:
        index.to_csv(out / "candidate_records_blinded.csv", index=False)
        template.to_csv(out / "eligibility_template.csv", index=False)
        (out / "README.md").write_text(
            "# PKSmart primary-source eligibility review\n\n"
            "This batch is eligibility-only and label-blinded. A secondary neutralized structure mapping is only a "
            "retrieval aid; acceptance requires the original paper itself to establish compound identity, human direct "
            "IV dosing, parent systemic analyte and terminal-phase analysis. Numerical extraction is prohibited until "
            "a separate eligibility lock is published.\n", encoding="utf-8")
        finish_stage(out, "pksmart_primary_source_eligibility_review_batch", inputs={
            "request_queue_complete_sha256": sha256(args.requests / "complete.json"),
            "candidate_queue_complete_sha256": sha256(args.candidates / "complete.json"),
            "fulltext_manifest_sha256": sha256(args.manifest),
        }, candidates=len(index), label_blinded=True, eligibility_only=True, partial=False)
    print(f"PKSmart primary-source review batch: {args.output}")


if __name__ == "__main__":
    run_cli(main)
