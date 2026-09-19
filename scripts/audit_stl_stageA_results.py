#!/usr/bin/env python3
"""Audit Stage-A STL OOF coverage, stability, source sensitivity, and shortlist.

The audit derives every result from already published train-CV predictions. It
does not refit a model and never reads fixed validation or test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import metric_row


EXTREME_INTERFACE_THRESHOLD = 10.0
MIN_SOURCE_DISJOINT_PARENTS = 30


def source_membership(records: pd.DataFrame) -> pd.DataFrame:
    train = records.loc[records.split.eq("train"), ["row_id", "task_id", "doc_id", "inner_fold_id"]].drop_duplicates()
    rows = []
    for (task, fold), group in train.groupby(["task_id", "inner_fold_id"], sort=True):
        fit_sources = set(train.loc[train.task_id.eq(task) & train.inner_fold_id.ne(fold), "doc_id"])
        part = group.copy()
        part["source_seen_in_other_training_folds"] = part.doc_id.isin(fit_sources)
        part["source_oof_subset"] = np.where(part.source_seen_in_other_training_folds,
                                               "source_seen_in_fit_folds", "source_disjoint_from_fit_folds")
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["row_id", "task_id"]).any():
        raise ValueError("Source membership is not unique per task row")
    return result


def choose_shortlist(candidate_audit: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    selected_rows = []
    for task, group in candidate_audit.groupby("task_id", sort=True):
        eligible = group.loc[group.numerical_stability_status.eq("pass")].sort_values(
            ["rank_within_task", "candidate_id"]
        )
        if eligible.empty:
            raise ValueError(f"No numerically stable candidate for {task}")
        selected = []
        used_algorithms = set()

        def add(row: pd.Series, reason: str) -> None:
            item = row.copy()
            item["shortlist_reason"] = reason
            selected.append(item)
            used_algorithms.add(row.algorithm)

        add(eligible.iloc[0], "best_overall_train_CV")
        different = eligible.loc[~eligible.algorithm.isin(used_algorithms)]
        if len(selected) < n and not different.empty:
            add(different.iloc[0], "best_overall_distinct_algorithm")
        if len(selected) < n:
            different = eligible.loc[~eligible.algorithm.isin(used_algorithms)].copy()
            if not different.empty:
                sufficiently_powered = bool(different.source_disjoint_parents.max() >= MIN_SOURCE_DISJOINT_PARENTS)
                if sufficiently_powered:
                    row = different.sort_values(["source_disjoint_primary_metric", "rank_within_task", "candidate_id"]).iloc[0]
                    add(row, "best_source_disjoint_among_distinct_algorithms")
                else:
                    add(different.iloc[0], "third_overall_distinct_algorithm; source_disjoint_descriptive_only")
        while len(selected) < n:
            remaining = eligible.loc[~eligible.candidate_id.isin([row.candidate_id for row in selected])]
            if remaining.empty:
                break
            add(remaining.iloc[0], "next_stable_overall_candidate")
        for position, row in enumerate(selected, start=1):
            row["provisional_shortlist_position"] = position
            row["shortlist_status"] = "provisional_train_CV_only; feature_views_and_linear_followup_pending"
            selected_rows.append(row)
    return pd.DataFrame(selected_rows).sort_values(["task_id", "provisional_shortlist_position"], ignore_index=True)


def audit(protocol: Path, screen: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    records = pd.read_csv(protocol / "benchmark_records.csv", dtype=str, keep_default_na=False)
    predictions = pd.read_csv(screen / "predictions.csv", dtype={"row_id": str}, keep_default_na=False)
    ranking = pd.read_csv(screen / "candidate_ranking.csv")
    runs = pd.read_csv(screen / "run_registry.csv")
    for column in ["target_value", "train_mean", "train_std_population", "interface_target",
                   "predicted_interface_target", "inner_fold_id"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    records["inner_fold_id"] = pd.to_numeric(records.inner_fold_id, errors="raise").astype(int)
    if not np.isfinite(predictions.predicted_interface_target).all():
        raise ValueError("Stage-A predictions contain non-finite values")
    if predictions.duplicated(["task_id", "candidate_id", "row_id"]).any():
        raise ValueError("Stage-A predictions contain duplicate candidate rows")
    if runs.groupby(["task_id", "candidate_id"]).partition.nunique().ne(5).any():
        raise ValueError("A candidate lacks complete five-fold coverage")
    train_counts = records.loc[records.split.eq("train")].groupby("task_id").row_id.nunique()
    observed_counts = predictions.groupby(["task_id", "candidate_id"]).row_id.nunique()
    if any(count != train_counts.loc[task] for (task, _), count in observed_counts.items()):
        raise ValueError("OOF prediction coverage does not equal the authorized training rows")

    membership = source_membership(records)
    predictions = predictions.merge(
        membership[["row_id", "task_id", "source_seen_in_other_training_folds", "source_oof_subset"]],
        on=["row_id", "task_id"], how="left", validate="many_to_one",
    )
    if predictions.source_oof_subset.isna().any() or predictions.source_oof_subset.eq("").any():
        raise ValueError("OOF predictions lack source-cluster membership")

    fold_rows = []
    for (task, candidate, partition), frame in predictions.groupby(["task_id", "candidate_id", "evaluation_partition"], sort=True):
        row = metric_row(frame, frame.predicted_interface_target.to_numpy(float), task,
                         frame.algorithm.iloc[0], frame.feature_set.iloc[0], partition)
        row["candidate_id"] = candidate
        row["evaluation_partition"] = partition
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    source_rows = []
    for (task, candidate, subset), frame in predictions.groupby(["task_id", "candidate_id", "source_oof_subset"], sort=True):
        row = metric_row(frame, frame.predicted_interface_target.to_numpy(float), task,
                         frame.algorithm.iloc[0], frame.feature_set.iloc[0], f"train_CV:{subset}")
        row["candidate_id"] = candidate
        row["source_oof_subset"] = subset
        source_rows.append(row)
    source_metrics = pd.DataFrame(source_rows)
    disjoint = source_metrics.loc[source_metrics.source_oof_subset.eq("source_disjoint_from_fit_folds"),
                                  ["task_id", "candidate_id", "records", "parents", "primary_metric", "spearman_parent_mean"]].rename(columns={
                                      "records": "source_disjoint_records", "parents": "source_disjoint_parents",
                                      "primary_metric": "source_disjoint_primary_metric",
                                      "spearman_parent_mean": "source_disjoint_spearman_parent_mean",
                                  })

    stability = (predictions.assign(
        abs_interface_prediction=predictions.predicted_interface_target.abs(),
        extreme=predictions.predicted_interface_target.abs().gt(EXTREME_INTERFACE_THRESHOLD),
    ).groupby(["task_id", "candidate_id", "algorithm"], as_index=False)
      .agg(max_abs_interface_prediction=("abs_interface_prediction", "max"),
           extreme_prediction_records=("extreme", "sum")))
    stability["numerical_stability_status"] = np.where(
        stability.extreme_prediction_records.gt(0), "fail_extreme_abs_interface_gt_10", "pass"
    )
    fold_summary = (fold_metrics.groupby(["task_id", "candidate_id"], as_index=False)
                    .agg(fold_primary_mean=("primary_metric", "mean"), fold_primary_sd=("primary_metric", "std"),
                         fold_primary_min=("primary_metric", "min"), fold_primary_max=("primary_metric", "max")))
    candidate_audit = (ranking.merge(fold_summary, on=["task_id", "candidate_id"], validate="one_to_one")
                       .merge(disjoint, on=["task_id", "candidate_id"], validate="one_to_one")
                       .merge(stability, on=["task_id", "candidate_id", "algorithm"], validate="one_to_one"))
    candidate_audit["source_disjoint_metric_ratio_vs_overall"] = (
        candidate_audit.source_disjoint_primary_metric / candidate_audit.primary_metric
    )
    candidate_audit["source_disjoint_interpretation"] = np.where(
        candidate_audit.source_disjoint_parents.ge(MIN_SOURCE_DISJOINT_PARENTS),
        "eligible_for_sensitivity_ranking_not_model_selection", "descriptive_only_underpowered",
    )
    shortlist = choose_shortlist(candidate_audit)

    coverage = (runs.groupby(["task_id", "candidate_id", "algorithm"], as_index=False)
                .agg(folds=("partition", "nunique"), min_train_parents=("train_parents", "min"),
                     max_train_parents=("train_parents", "max"), min_evaluation_parents=("evaluation_parents", "min"),
                     max_evaluation_parents=("evaluation_parents", "max")))
    actions = pd.DataFrame([
        {"priority": 1, "action": "retain_existing_OOF_and_publish_source/fold/stability audit", "reason": "no rerun required"},
        {"priority": 2, "action": "run_targeted_linear_stabilization", "reason": "Ridge produced extreme Papp/fu predictions; add alpha=1000 and ElasticNet"},
        {"priority": 3, "action": "run_ECFP4_and_combined_feature_StageA", "reason": "RDKit2D-only ranking cannot define the final STL shortlist"},
        {"priority": 4, "action": "deduplicate_effectively_identical_candidates", "reason": "CLint HistGB c00/c01 yielded identical OOF predictions"},
        {"priority": 5, "action": "defer_fixed_validation", "reason": "feature-view and linear followups remain incomplete"},
    ])
    summary = {
        "predictions": len(predictions), "candidate_task_evaluations": len(candidate_audit),
        "complete_five_fold_candidate_tasks": int(coverage.folds.eq(5).sum()),
        "nonfinite_predictions": 0,
        "numerically_unstable_candidate_tasks": int(candidate_audit.numerical_stability_status.ne("pass").sum()),
        "provisional_shortlist_entries": len(shortlist),
        "validation_labels_read": False, "test_labels_read": False,
        "rerun_existing_450_fits_required": False,
        "fixed_validation_authorized_next": False,
    }
    tables = {
        "coverage_audit.csv": coverage,
        "fold_metrics.csv": fold_metrics,
        "source_oof_membership.csv": membership,
        "source_sensitivity_metrics.csv": source_metrics,
        "candidate_audit.csv": candidate_audit.sort_values(["task_id", "rank_within_task"]),
        "provisional_shortlist.csv": shortlist,
        "followup_actions.csv": actions,
    }
    return tables, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--screen", type=Path, default=ROOT / "results/benchmarks/stl_stageA_first_batch_rdkit2d_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_stageA_first_batch_audit_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check([args.protocol / "complete.json", args.screen / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.protocol, "stl_benchmark_protocol")
    screen_meta = verify_stage(args.screen, "stl_benchmark_screen")
    if screen_meta.get("evaluation_scope") != "train_cv" or screen_meta.get("validation_confirmation") or screen_meta.get("test_labels_read"):
        raise ValueError("Audit accepts only unblinded-free Stage-A train-CV results")
    tables, summary = audit(args.protocol, args.screen)
    if args.check_only:
        print(f"STL Stage-A audit valid: candidate_tasks={summary['candidate_task_evaluations']} unstable={summary['numerically_unstable_candidate_tasks']}")
        return
    with stage_output(args.output) as out:
        for name, table in tables.items():
            table.to_csv(out / name, index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# STL Stage-A first-batch audit\n\nDerived only from frozen-train OOF predictions. Includes fold stability, source-cluster sensitivity, numerical stability, and a provisional algorithm-diverse shortlist. Validation and test labels remain unread.\n",
            encoding="utf-8")
        finish_stage(out, "stl_stageA_results_audit", inputs={
            "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
            "screen_complete_sha256": sha256(args.screen / "complete.json"),
        }, **summary, partial=False)
    print(f"STL Stage-A audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
