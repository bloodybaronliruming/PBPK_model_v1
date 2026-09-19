#!/usr/bin/env python3
"""Publish a label-safe readiness and test-lifecycle audit for seven PK endpoints."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


ENDPOINTS = (
    {
        "task_id": "fu__human__plasma", "tier": "micro", "endpoint_label": "Human plasma fu",
        "release_state": "historical_frozen_test_consumed", "evaluated_test_molecules": 631,
        "next_gate": "Publish low-fu, applicability-domain, and model-card analyses without reselection.",
    },
    {
        "task_id": "CLint__human__microsome", "tier": "micro", "endpoint_label": "Human microsomal CLint",
        "release_state": "test_sealed_frozen_candidate", "evaluated_test_molecules": 0,
        "next_gate": "Run the frozen RF three-seed ensemble's only test evaluation; do not refit or reselect.",
    },
    {
        "task_id": "Papp__human__caco2_ab", "tier": "micro", "endpoint_label": "Human Caco-2 A-to-B Papp",
        "release_state": "historical_frozen_test_consumed", "evaluated_test_molecules": 243,
        "next_gate": "Report unit-provenance sensitivity; do not reselect on the consumed test.",
    },
    {
        "task_id": "F__human__absolute_oral", "tier": "macro", "endpoint_label": "Human absolute oral F",
        "release_state": "blocked_no_validation", "evaluated_test_molecules": 0,
        "next_gate": "Build a versioned public development cohort and a definition-audited, structure-isolated validation cohort.",
    },
    {
        "task_id": "CL__human__systemic_iv", "tier": "macro", "endpoint_label": "Human systemic IV CL",
        "release_state": "historical_frozen_test_consumed", "evaluated_test_molecules": 59,
        "next_gate": "Complete matrix/unit/source sensitivity and report the historical frozen result only.",
    },
    {
        "task_id": "VDss__human__steady_state_iv", "tier": "macro", "endpoint_label": "Human IV VDss",
        "release_state": "historical_frozen_test_consumed", "evaluated_test_molecules": 58,
        "next_gate": "Confirm Vss identity and report the historical frozen result only.",
    },
    {
        "task_id": "Thalf__human__terminal_iv", "tier": "macro", "endpoint_label": "Human terminal IV half-life",
        "release_state": "historical_frozen_test_consumed_external_pilot", "evaluated_test_molecules": 53,
        "next_gate": "Keep Strict-IV and Public-plasma separate; expand the strict external registry to at least 50 molecules.",
    },
)

SPLITS = ("train", "val", "test")
REQUIRED_TASK_COLUMNS = {"task_id", "split", "records", "molecules"}


def read_task_counts(path: Path, task_ids: set[str]) -> dict[str, dict[str, dict[str, int]]]:
    """Read only published aggregate counts, never endpoint values."""
    counts: dict[str, dict[str, dict[str, int]]] = {task: {} for task in task_ids}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_TASK_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f"Invalid task-count schema: {path}")
        for row in reader:
            task = row["task_id"]
            if task not in task_ids:
                continue
            split = row["split"]
            if split not in SPLITS:
                raise ValueError(f"Unexpected split for {task}: {split}")
            if split in counts[task]:
                raise ValueError(f"Duplicate count row for {task}/{split}")
            counts[task][split] = {"records": int(row["records"]), "molecules": int(row["molecules"])}
    for task, by_split in counts.items():
        # A missing split row is a material readiness finding (not a malformed
        # label): e.g., the current human absolute-F task has no validation
        # members. Preserve that absence explicitly as a zero-sized split.
        for split in SPLITS:
            by_split.setdefault(split, {"records": 0, "molecules": 0})
    return counts


def read_development_documents(path: Path, task_ids: set[str]) -> dict[str, int]:
    """Count accepted train/validation documents without exporting any endpoint values."""
    documents: dict[str, set[str]] = defaultdict(set)
    needed = {"task_id", "split", "status", "doc_id"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not needed <= set(reader.fieldnames):
            raise ValueError(f"Invalid source-long schema: {path}")
        for row in reader:
            task = row["task_id"]
            if task not in task_ids or row["split"] not in {"train", "val"} or row["status"] != "accepted":
                continue
            if row["doc_id"].strip():
                documents[task].add(row["doc_id"].strip())
    return {task: len(documents[task]) for task in task_ids}


def read_development_membership(path: Path, task_ids: set[str]) -> dict[str, set[str]]:
    """Read task/split/molecule membership only; endpoint values remain unused."""
    membership: dict[str, set[str]] = defaultdict(set)
    needed = {"task_id", "split", "molecule_id"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not needed <= set(reader.fieldnames):
            raise ValueError(f"Invalid task-record schema: {path}")
        for row in reader:
            task = row["task_id"]
            if task in task_ids and row["split"] in {"train", "val"}:
                membership[task].add(row["molecule_id"])
    return {task: membership[task] for task in task_ids}


def readiness_state(train_molecules: int, val_molecules: int, release_state: str) -> str:
    if val_molecules == 0:
        return "blocked_no_validation"
    if train_molecules < 30:
        return "blocked_insufficient_training"
    if release_state == "test_sealed_frozen_candidate":
        return "frozen_ready_for_one_time_test"
    if release_state == "test_sealed_candidate_freeze_required":
        return "ready_to_freeze"
    if release_state.startswith("historical_frozen_test_consumed"):
        return "historical_result_only"
    return "development_ready"


def build_tables(
    registry: dict[str, dict[str, object]],
    counts: dict[str, dict[str, dict[str, int]]],
    documents: dict[str, int],
    membership: dict[str, set[str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    endpoint_rows = []
    for spec in ENDPOINTS:
        task = spec["task_id"]
        meta = registry.get(task)
        if not meta:
            raise ValueError(f"Task registry lacks required endpoint: {task}")
        train = counts[task]["train"]
        val = counts[task]["val"]
        test = counts[task]["test"]
        endpoint_rows.append({
            "task_id": task,
            "tier": spec["tier"],
            "endpoint_label": spec["endpoint_label"],
            "canonical_unit": meta["unit"],
            "transform": meta["transform"],
            "aggregation": meta["aggregation"],
            "train_records": train["records"],
            "train_molecules": train["molecules"],
            "validation_records": val["records"],
            "validation_molecules": val["molecules"],
            "test_records": test["records"],
            "test_molecules": test["molecules"],
            "accepted_development_documents": documents[task],
            "evaluated_test_molecules": spec["evaluated_test_molecules"],
            "test_lifecycle": spec["release_state"],
            "readiness": readiness_state(train["molecules"], val["molecules"], str(spec["release_state"])),
            "next_gate": spec["next_gate"],
        })

    overlap_rows = []
    for left, right in itertools.combinations(ENDPOINTS, 2):
        left_task, right_task = str(left["task_id"]), str(right["task_id"])
        common = membership[left_task] & membership[right_task]
        denominator = min(len(membership[left_task]), len(membership[right_task]))
        overlap_rows.append({
            "left_task_id": left_task,
            "right_task_id": right_task,
            "left_development_molecules": len(membership[left_task]),
            "right_development_molecules": len(membership[right_task]),
            "shared_development_molecules": len(common),
            "shared_fraction_of_smaller_task": (len(common) / denominator) if denominator else 0.0,
        })

    summary = {
        "endpoints_total": len(endpoint_rows),
        "historical_frozen_test_consumed": sum(row["test_lifecycle"].startswith("historical_frozen_test_consumed") for row in endpoint_rows),
        "test_sealed_frozen_candidate": sum(row["test_lifecycle"] == "test_sealed_frozen_candidate" for row in endpoint_rows),
        "blocked_no_validation": sum(row["readiness"] == "blocked_no_validation" for row in endpoint_rows),
        "development_ready_or_freezeable": sum(row["readiness"] in {"development_ready", "ready_to_freeze", "frozen_ready_for_one_time_test"} for row in endpoint_rows),
        "label_safety": "No raw/target endpoint values are read or emitted; only aggregate split counts and development membership/document identifiers are audited.",
    }
    return endpoint_rows, overlap_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to publish empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/multitask_endpoint_readiness_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    task_counts = args.datasets / "task_counts.csv"
    task_registry = args.datasets / "task_registry.json"
    task_records = args.datasets / "task_records.csv"
    source_long = args.datasets / "source_long.csv"
    startup_self_check([task_counts, task_registry, task_records, source_long, args.datasets / "complete.json", args.splits / "complete.json"],
                       output=None if args.check_only else args.output)
    verify_stage(args.datasets, "datasets")
    verify_stage(args.splits, "splits")
    task_ids = {str(spec["task_id"]) for spec in ENDPOINTS}
    registry = json.loads(task_registry.read_text(encoding="utf-8"))
    counts = read_task_counts(task_counts, task_ids)
    documents = read_development_documents(source_long, task_ids)
    membership = read_development_membership(task_records, task_ids)
    endpoint_rows, overlap_rows, summary = build_tables(registry, counts, documents, membership)
    if args.check_only:
        print(f"Seven-endpoint readiness contract valid: endpoints={summary['endpoints_total']} blocked_no_validation={summary['blocked_no_validation']}")
        return
    with stage_output(args.output) as out:
        write_csv(out / "endpoint_readiness.csv", endpoint_rows)
        write_csv(out / "development_endpoint_overlap.csv", overlap_rows)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Seven-endpoint PK readiness and test-lifecycle audit\n\n"
            "This is a planning audit for human fu, microsomal CLint, Caco-2 Papp, absolute oral F, systemic IV CL, "
            "IV VDss, and terminal IV half-life. It reports aggregate split counts, development-source concentration, "
            "and test lifecycle. It neither emits endpoint values nor changes models, selections, or test access. "
            "`historical_frozen_test_consumed` means that a prior frozen result exists and may not be reused to select a new model. "
            "`test_sealed_frozen_candidate` means selection is locked and one test evaluation remains.\n",
            encoding="utf-8")
        finish_stage(out, "multitask_endpoint_readiness_audit", inputs={
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "splits_complete_sha256": sha256(args.splits / "complete.json"),
            "task_registry_sha256": sha256(task_registry),
            "task_counts_sha256": sha256(task_counts),
            "task_records_sha256": sha256(task_records),
            "source_long_sha256": sha256(source_long),
        }, partial=False, **summary)
    print(f"Seven-endpoint readiness audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
