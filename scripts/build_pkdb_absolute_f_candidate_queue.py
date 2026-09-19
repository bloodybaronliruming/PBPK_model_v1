#!/usr/bin/env python3
"""Publish a label-blinded PK-DB request queue for human absolute oral-F evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


TASK = "F__human__absolute_oral"
FORBIDDEN = {"value", "mean", "median", "min", "max", "sd", "se", "cv", "raw_value", "target_value", "y"}
ABSOLUTE_F = re.compile(r"\babsolute\s+(?:oral\s+)?bioavailability\b", re.I)


def read_f_reference_membership(records: Path) -> tuple[set[str], set[str]]:
    """Read structure membership only; never read labels, targets or source values."""
    parents, docs = set(), set()
    with records.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        needed = {"task_id", "molecule_id", "doc_id"}
        if reader.fieldnames is None or not needed <= set(reader.fieldnames):
            raise ValueError(f"Invalid task-record schema: {records}")
        for row in reader:
            if row["task_id"] != TASK:
                continue
            parents.add(row["molecule_id"])
            if row["doc_id"].strip():
                docs.add(row["doc_id"].strip())
    return parents, docs


def build_queue(candidates, f_parent_ids: set[str]):
    missing = {"sid", "reference_pmid", "reference_title", "reference_date", "substance_sid", "label",
               "chembl_id", "canonical_smiles", "parent_id", "scaffold_group", "screening_status",
               "endpoint_values_hidden"} - set(candidates.columns)
    if missing:
        raise ValueError(f"PK-DB candidate schema missing: {sorted(missing)}")
    forbidden = FORBIDDEN & {str(column).lower() for column in candidates.columns}
    if forbidden:
        raise ValueError(f"PK-DB source contains prohibited numeric endpoint columns: {sorted(forbidden)}")
    frame = candidates.copy()
    if not frame.endpoint_values_hidden.astype(bool).all():
        raise ValueError("PK-DB candidates must remain endpoint-value blinded")
    frame["title_support"] = frame.reference_title.fillna("").map(
        lambda title: "absolute_bioavailability_title" if ABSOLUTE_F.search(str(title)) else "no_absolute_bioavailability_title")
    frame["f_parent_overlap"] = frame.parent_id.fillna("").isin(f_parent_ids)
    frame["queue_status"] = "exclude_no_absolute_f_title"
    eligible = frame.screening_status.eq("primary_source_required") & frame.title_support.eq("absolute_bioavailability_title")
    frame.loc[eligible, "queue_status"] = "request_primary_source"
    frame.loc[eligible & frame.f_parent_overlap, "queue_status"] = "exclude_existing_F_parent"
    queue = frame.loc[frame.queue_status.eq("request_primary_source")].copy()
    queue["candidate_id"] = queue.apply(
        lambda row: stable_id(f"pkdb-absolute-f|{row.sid}|{row.substance_sid}|{row.parent_id}"), axis=1)
    queue["required_primary_evidence"] = (
        "Human oral and IV reference arms; parent analyte; numerical absolute F; route/dose and unit; page/table locator.")
    queue["eligibility_decision"] = "pending_primary_source_review"
    queue["endpoint_values_hidden"] = True
    output_columns = [
        "candidate_id", "sid", "reference_pmid", "reference_title", "reference_date", "substance_sid", "label",
        "chembl_id", "canonical_smiles", "parent_id", "scaffold_group", "title_support", "required_primary_evidence",
        "eligibility_decision", "endpoint_values_hidden",
    ]
    queue = queue.sort_values(["reference_date", "sid", "substance_sid"], ascending=[False, True, True], ignore_index=True)
    result = queue[output_columns]
    if not result.endpoint_values_hidden.astype(bool).all():
        raise ValueError("Queue lost its blinded-state flag")
    return result, frame.groupby("queue_status", dropna=False).size().rename("records").reset_index()


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/external/pkdb_preprocessed_v1")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/pkdb_absolute_f_candidate_queue_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source_file = args.source / "candidate_substances_blinded.csv"
    records = args.datasets / "task_records.csv"
    startup_self_check([source_file, args.source / "complete.json", records, args.datasets / "complete.json"],
                       output=None if args.check_only else args.output)
    verify_stage(args.source, "pkdb_candidate_preprocessing")
    verify_stage(args.datasets, "datasets")
    parents, _ = read_f_reference_membership(records)
    candidates = pd.read_csv(source_file, low_memory=False)
    queue, summary = build_queue(candidates, parents)
    if args.check_only:
        print(f"PK-DB absolute-F queue contract valid: candidates={len(queue)} existing_F_parents={len(parents)}")
        return
    with stage_output(args.output) as out:
        queue.to_csv(out / "candidate_records_blinded.csv", index=False)
        summary.to_csv(out / "screening_summary.csv", index=False)
        (out / "README.md").write_text(
            "# PK-DB human absolute oral-F primary-source request queue\n\n"
            "This is a source-discovery queue, not an F training or validation set. Candidates originate from the public "
            "PK-DB human+IV snapshot and have an absolute-bioavailability title signal, an eligible mapped parent, and no "
            "existing F parent overlap. The queue contains no endpoint values. A candidate may enter a later eligibility lock "
            "only after its original source confirms human oral and IV reference arms, parent analyte, numerical absolute F, "
            "route/dose/unit, and a page/table locator.\n",
            encoding="utf-8")
        finish_stage(out, "pkdb_absolute_f_candidate_queue", inputs={
            "source_complete_sha256": sha256(args.source / "complete.json"),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "task_records_sha256": sha256(records),
        }, candidates=len(queue), existing_f_parents=len(parents), label_blinded=True, partial=False)
    print(f"PK-DB absolute-F candidate queue: {args.output}")


if __name__ == "__main__":
    run_cli(main)
