#!/usr/bin/env python3
"""Run or verify the resumable 30-cell Gate-3 B6 formal train-CV batch."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, dump_json, run_cli, sha256, startup_self_check, verify_stage


EXPECTED_ANCESTRY = {
    "stageA_crossfit": 12,
    "stageA_outer_eval": 3,
    "auxiliary_crossfit": 10,
    "auxiliary_outer_eval": 2,
    "meta_model_fit_and_evaluation": 2,
}
EXPECTED_IMPUTATION = {
    "stageA_crossfit": 12,
    "stageA_outer_eval": 3,
    "auxiliary_candidate_cv": 20,
    "auxiliary_crossfit": 10,
    "auxiliary_outer_eval": 2,
}
ROUTES = {
    "C0_corrected_StageA",
    "C1_structure_only_control",
    "B6_physchem_candidate",
}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_bar(completed: int, total: int, width: int = 30) -> str:
    filled = width if total <= 0 else min(width, int(width * completed / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_batch_progress(completed: int, total: int, started: float,
                         estimated_cell_seconds: float, detail: str) -> None:
    elapsed = time.monotonic() - started
    remaining = max(0, total - completed)
    eta = remaining * estimated_cell_seconds
    print(
        f"BATCH_PROGRESS {progress_bar(completed, total)} {completed:02d}/{total:02d} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"estimated_cell={format_duration(estimated_cell_seconds)} detail={detail}",
        flush=True,
    )


def verify_cell(path: Path, task_id: str, outer_fold: int, runner_hash: str) -> dict:
    meta = verify_stage(path, "gate3_b6_physchem_outer_fold")
    expected = {
        "partial": False,
        "full_configuration": True,
        "engineering_preflight": False,
        "task_id": task_id,
        "outer_fold": outer_fold,
        "ancestry_rows": 29,
        "imputation_audit_rows": 47,
        "candidate_cv_imputation_audit_rows": 20,
        "maximum_all_missing_training_columns": 0,
        "maximum_parent_overlap": 0,
        "maximum_scaffold_overlap": 0,
        "route_predictions_complete": True,
        "outer_evaluation_metrics_computed": True,
        "outer_evaluation_target_values_used_before_all_routes_predicted": 0,
        "architecture_selection_authorized": False,
        "fixed_validation_authorized": False,
        "test_labels_read": False,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Cell contract mismatch {path}: {key}={meta.get(key)!r}, expected={value!r}")
    if float(meta.get("maximum_reload_difference", float("inf"))) > 1e-12:
        raise ValueError(f"Cell reload tolerance failed: {path}")
    if meta.get("code_hashes", {}).get("run_gate3_b6_physchem_outer_fold.py") != runner_hash:
        raise ValueError(f"Cell was not produced by the authorized outer-fold runner: {path}")

    ancestry = pd.read_csv(path / "nested_ancestry_audit.csv")
    if ancestry.groupby("component").size().to_dict() != EXPECTED_ANCESTRY:
        raise ValueError(f"Cell ancestry composition differs: {path}")
    if ancestry[["parent_overlap", "scaffold_overlap"]].to_numpy().max() != 0:
        raise ValueError(f"Cell ancestry overlap is nonzero: {path}")
    for row in ancestry.itertuples(index=False):
        if sha256(path / row.model_file) != row.model_sha256:
            raise ValueError(f"Cell model ancestry hash differs: {path / row.model_file}")

    imputation = pd.read_csv(path / "fold_local_imputation_audit.csv")
    if imputation.groupby("component").size().to_dict() != EXPECTED_IMPUTATION:
        raise ValueError(f"Cell imputation composition differs: {path}")
    if imputation.all_missing_columns.ne(0).any():
        raise ValueError(f"Cell contains an all-missing training feature: {path}")
    if not imputation.imputer_fit_scope.eq("current_model_training_parents_only").all():
        raise ValueError(f"Cell contains non-fold-local imputation: {path}")

    metrics = pd.read_csv(path / "outer_fold_metrics.csv")
    if len(metrics) != 3 or set(metrics.route) != ROUTES or not metrics.primary_metric.notna().all():
        raise ValueError(f"Cell route metrics are incomplete: {path}")
    predictions = pd.read_csv(path / "outer_evaluation_predictions.csv")
    if predictions.parent_id.duplicated().any() or len(predictions) != int(meta["outer_evaluation_parents"]):
        raise ValueError(f"Cell outer-evaluation predictions are incomplete: {path}")
    if not predictions[list(ROUTES)].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError(f"Cell contains nonnumeric route predictions: {path}")
    lifecycle = pd.read_csv(path / "label_lifecycle_audit.csv")
    before_score = lifecycle.loc[~lifecycle.event.eq("score_outer_fold")]
    score = lifecycle.loc[lifecycle.event.eq("score_outer_fold")]
    if before_score.outer_evaluation_target_values_used.astype(bool).any() or len(score) != 1 or not bool(score.iloc[0].outer_evaluation_target_values_used):
        raise ValueError(f"Cell target lifecycle differs: {path}")
    return {
        "task_id": task_id,
        "outer_fold": outer_fold,
        "output_directory": str(path.relative_to(ROOT)),
        "complete_sha256": sha256(path / "complete.json"),
        "outer_evaluation_parents": int(meta["outer_evaluation_parents"]),
        "maximum_reload_difference": float(meta["maximum_reload_difference"]),
        "artifacts": len(meta["artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=ROOT / "data/public_development/gate3_b6_formal_run_v4")
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts/run_gate3_b6_physchem_outer_fold.py")
    parser.add_argument("--batch-root", type=Path, default=ROOT / "results/benchmarks/gate3_b6_physchem_formal_cells_v3")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--initial-cell-seconds", type=float, default=132.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.initial_cell_seconds <= 0:
        raise ValueError("initial-cell-seconds must be positive")
    required = [
        args.authorization / "complete.json",
        args.authorization / "formal_authorization.json",
        args.authorization / "formal_cell_registry.csv",
        args.runner,
    ]
    startup_self_check(required)
    auth_meta = verify_stage(args.authorization, "gate3_b6_formal_run_freeze")
    authorization = json.loads((args.authorization / "formal_authorization.json").read_text())
    runner_hash = sha256(args.runner)
    if not auth_meta.get("formal_train_cv_authorized") or not authorization.get("formal_train_cv_authorized"):
        raise ValueError("Formal train-CV is not authorized")
    if authorization.get("outer_fold_runner_sha256") != runner_hash:
        raise ValueError("Current outer-fold runner differs from the authorized hash")
    if authorization.get("required_imputation_audit_rows_per_cell") != 47:
        raise ValueError("Formal authorization lacks the complete imputation audit contract")
    registry = pd.read_csv(args.authorization / "formal_cell_registry.csv")
    if len(registry) != 30 or registry.cell_id.duplicated().any():
        raise ValueError("Formal registry must contain exactly 30 unique cells")

    completed: list[dict] = []
    pending = []
    for row in registry.itertuples(index=False):
        output = ROOT / row.output_directory
        if (output / "complete.json").is_file():
            completed.append(verify_cell(output, str(row.task_id), int(row.outer_fold), runner_hash))
        elif output.exists():
            raise FileExistsError(f"Incomplete cell directory exists and will not be overwritten: {output}")
        else:
            pending.append(row)
    total_cells = len(registry)
    batch_started = time.monotonic()
    estimated_cell_seconds = args.initial_cell_seconds
    print(f"Gate 3 B6 batch preflight: verified_complete={len(completed)} pending={len(pending)}", flush=True)
    print_batch_progress(
        len(completed), total_cells, batch_started, estimated_cell_seconds,
        "preflight_complete",
    )
    if args.check_only:
        return

    batch_root = args.batch_root.resolve()
    registered_roots = {str((ROOT / Path(path)).parent.resolve()) for path in registry.output_directory}
    if registered_roots != {str(batch_root)}:
        raise ValueError(
            f"Batch root differs from the authorized registry: {batch_root}; "
            f"registered={sorted(registered_roots)}"
        )
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_marker = batch_root / "batch_complete.json"
    if batch_marker.exists() and pending:
        raise ValueError("Batch completion marker exists while registered cells are pending")
    durations: list[float] = []
    for number, row in enumerate(pending, start=len(completed) + 1):
        output = ROOT / row.output_directory
        command = [
            sys.executable,
            str(args.runner),
            "--task", str(row.task_id),
            "--outer-fold", str(int(row.outer_fold)),
            "--threads", str(args.threads),
            "--output", str(output),
        ]
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"starting cell {number}/30: {row.cell_id}",
            flush=True,
        )
        print_batch_progress(
            len(completed), total_cells, batch_started, estimated_cell_seconds,
            f"running={row.cell_id}",
        )
        cell_started = time.monotonic()
        subprocess.run(command, cwd=ROOT, check=True)
        durations.append(time.monotonic() - cell_started)
        completed.append(verify_cell(output, str(row.task_id), int(row.outer_fold), runner_hash))
        estimated_cell_seconds = sum(durations) / len(durations)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"completed cell {number}/30: {row.cell_id}",
            flush=True,
        )
        print_batch_progress(
            len(completed), total_cells, batch_started, estimated_cell_seconds,
            f"completed={row.cell_id}",
        )

    if len(completed) != 30:
        raise ValueError("Batch ended without all 30 verified cells")
    completed = sorted(completed, key=lambda row: (row["task_id"], row["outer_fold"]))
    summary = {
        "schema_version": 1,
        "stage": "gate3_b6_physchem_cell_batch",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_complete_sha256": sha256(args.authorization / "complete.json"),
        "outer_fold_runner_sha256": runner_hash,
        "elapsed_seconds": time.monotonic() - batch_started,
        "mean_executed_cell_seconds": sum(durations) / len(durations) if durations else 0.0,
        "cells_verified": len(completed),
        "tasks": int(registry.task_id.nunique()),
        "outer_folds": int(registry.outer_fold.nunique()),
        "all_cells_full_configuration": True,
        "all_cells_parent_scaffold_overlap_zero": True,
        "all_cells_imputation_audits_complete": True,
        "architecture_selection_authorized": False,
        "fixed_validation_authorized": False,
        "test_authorized": False,
        "cells": completed,
    }
    temporary = batch_root / ".batch_complete.json.partial"
    dump_json(temporary, summary)
    temporary.replace(batch_marker)
    print(f"Gate 3 B6 formal cell batch: {batch_marker}", flush=True)


if __name__ == "__main__":
    run_cli(main)
