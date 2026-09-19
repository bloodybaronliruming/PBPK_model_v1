#!/usr/bin/env python3
"""Compare the targeted linear follow-up with the audited RDKit2D Stage A.

Only frozen-train OOF predictions and already published audit tables are read.
Fixed validation and test labels remain inaccessible to this analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


EXTREME_INTERFACE_THRESHOLD = 10.0


def candidate_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["predicted_interface_target"] = pd.to_numeric(work.predicted_interface_target, errors="raise")
    if not np.isfinite(work.predicted_interface_target).all():
        raise ValueError("Linear follow-up contains non-finite predictions")
    work["abs_interface_prediction"] = work.predicted_interface_target.abs()
    work["extreme"] = work.abs_interface_prediction.gt(EXTREME_INTERFACE_THRESHOLD)
    result = (work.groupby(["task_id", "candidate_id", "algorithm"], as_index=False)
              .agg(max_abs_interface_prediction=("abs_interface_prediction", "max"),
                   extreme_prediction_records=("extreme", "sum")))
    result["numerical_stability_status"] = np.where(
        result.extreme_prediction_records.eq(0), "pass", "fail_extreme_abs_interface_gt_10"
    )
    return result


def compare(followup_ranking: pd.DataFrame, stability: pd.DataFrame,
            baseline_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audited = followup_ranking.merge(
        stability, on=["task_id", "candidate_id", "algorithm"], validate="one_to_one"
    )
    endpoint_rows = []
    for task, group in audited.groupby("task_id", sort=True):
        stable = group.loc[group.numerical_stability_status.eq("pass")].sort_values(
            ["primary_metric", "candidate_id"]
        )
        if stable.empty:
            raise ValueError(f"No stable targeted linear candidate for {task}")
        follow = stable.iloc[0]
        prior = baseline_audit.loc[
            baseline_audit.task_id.eq(task) & baseline_audit.numerical_stability_status.eq("pass")
        ].sort_values(["primary_metric", "candidate_id"]).iloc[0]
        endpoint_rows.append({
            "task_id": task,
            "endpoint": follow.endpoint,
            "stable_linear_candidate": follow.candidate_id,
            "stable_linear_primary_metric": follow.primary_metric,
            "stable_linear_max_abs_interface_prediction": follow.max_abs_interface_prediction,
            "baseline_best_candidate": prior.candidate_id,
            "baseline_best_primary_metric": prior.primary_metric,
            "linear_relative_change_vs_baseline_best_percent":
                100.0 * (follow.primary_metric - prior.primary_metric) / prior.primary_metric,
            "linear_stabilization_outcome": "stable_reference_available",
            "performance_outcome": "linear_not_better_than_baseline_best"
                if follow.primary_metric >= prior.primary_metric else "linear_improves_baseline_best",
            "selection_status": "stageA_train_CV_only; fixed_validation_locked",
        })
    return audited.sort_values(["task_id", "primary_metric", "candidate_id"]), pd.DataFrame(endpoint_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", type=Path,
                        default=ROOT / "results/benchmarks/stl_stageA_linear_stabilization_rdkit2d_v1")
    parser.add_argument("--baseline-audit", type=Path,
                        default=ROOT / "results/analysis/stl_stageA_first_batch_audit_v3")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/stl_stageA_linear_stabilization_audit_v1")
    args = parser.parse_args()
    startup_self_check([args.followup / "complete.json", args.baseline_audit / "complete.json"], output=args.output)
    followup_meta = verify_stage(args.followup, "stl_benchmark_screen")
    baseline_meta = verify_stage(args.baseline_audit, "stl_stageA_results_audit")
    if followup_meta.get("candidate_profile") != "linear_stabilization":
        raise ValueError("Expected the bounded linear_stabilization profile")
    if (followup_meta.get("evaluation_scope") != "train_cv" or followup_meta.get("validation_confirmation")
            or followup_meta.get("test_labels_read") or baseline_meta.get("validation_labels_read")
            or baseline_meta.get("test_labels_read")):
        raise ValueError("Analysis accepts only unblinded-free train-CV artifacts")

    predictions = pd.read_csv(args.followup / "predictions.csv")
    ranking = pd.read_csv(args.followup / "candidate_ranking.csv")
    runs = pd.read_csv(args.followup / "run_registry.csv")
    reloads = pd.read_csv(args.followup / "reload_checks.csv")
    baseline = pd.read_csv(args.baseline_audit / "candidate_audit.csv")
    if predictions.duplicated(["task_id", "candidate_id", "row_id"]).any():
        raise ValueError("Duplicate follow-up OOF prediction rows")
    if runs.groupby(["task_id", "candidate_id"]).partition.nunique().ne(5).any():
        raise ValueError("Incomplete five-fold follow-up coverage")
    if len(reloads) != ranking[["task_id", "candidate_id"]].drop_duplicates().shape[0]:
        raise ValueError("Each candidate-task requires one reload check")
    if not reloads.passed.astype(bool).all() or reloads.max_abs_difference.max() > 1e-8:
        raise ValueError("A follow-up reload check failed")

    stability = candidate_stability(predictions)
    candidates, endpoints = compare(ranking, stability, baseline)
    summary = {
        "predictions": len(predictions),
        "fits": len(runs),
        "candidate_task_evaluations": len(candidates),
        "stable_candidate_tasks": int(candidates.numerical_stability_status.eq("pass").sum()),
        "unstable_candidate_tasks": int(candidates.numerical_stability_status.ne("pass").sum()),
        "tasks_with_stable_linear_reference": int(endpoints.task_id.nunique()),
        "tasks_where_linear_beats_baseline_best": int(endpoints.performance_outcome.eq("linear_improves_baseline_best").sum()),
        "maximum_reload_difference": float(reloads.max_abs_difference.max()),
        "validation_labels_read": False,
        "test_labels_read": False,
        "fixed_validation_authorized_next": False,
    }
    with stage_output(args.output) as out:
        candidates.to_csv(out / "linear_candidate_audit.csv", index=False)
        endpoints.to_csv(out / "endpoint_comparison.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# STL targeted linear stabilization audit\n\nFrozen-train OOF comparison only. "
            "A stable linear reference is not evidence of superiority and fixed validation remains locked.\n",
            encoding="utf-8",
        )
        finish_stage(out, "stl_linear_stabilization_audit", inputs={
            "followup_complete_sha256": sha256(args.followup / "complete.json"),
            "baseline_audit_complete_sha256": sha256(args.baseline_audit / "complete.json"),
        }, **summary, partial=False)
    print(f"STL linear stabilization audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
