#!/usr/bin/env python3
"""Summarize fixed Papp/fu source-cluster holdout diagnostics reproducibly."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASKS = ("Papp__human__caco2_ab", "fu__human__plasma")
LABELS = {TASKS[0]: "Papp", TASKS[1]: "fu"}
PAPP_BINS = (0.0, 100.0, 10000.0, np.inf)
PAPP_LABELS = ("≤100", "100–10,000", ">10,000")


def primary_metric(parent: pd.DataFrame, task: str) -> float:
    if task == TASKS[0]:
        return float(np.sqrt(np.mean((parent.predicted_transformed - parent.observed_transformed) ** 2)))
    return float(np.mean(np.abs(parent.predicted_physical - parent.observed_physical)))


def aggregate_parent(frame: pd.DataFrame, include_seed: bool) -> pd.DataFrame:
    columns = ["task_id", "endpoint", "source_fold", "molecule_id"]
    if include_seed:
        columns.append("seed")
    work = frame.copy()
    work["observed_transformed"] = work.target_value.to_numpy(float)
    work["predicted_transformed"] = work.predicted_target.to_numpy(float)
    # Match metric_row: inverse-transform every record before collapsing repeated parents.
    work["observed_physical"] = np.nan
    work["predicted_physical"] = np.nan
    fu_mask = work.task_id.eq(TASKS[1])
    work.loc[fu_mask, "observed_physical"] = expit(work.loc[fu_mask, "observed_transformed"])
    work.loc[fu_mask, "predicted_physical"] = expit(work.loc[fu_mask, "predicted_transformed"])
    return work.groupby(columns, as_index=False).agg(
        observed_transformed=("observed_transformed", "mean"),
        predicted_transformed=("predicted_transformed", "mean"),
        observed_physical=("observed_physical", "mean"),
        predicted_physical=("predicted_physical", "mean"),
        max_train_ecfp4_tanimoto=("max_train_ecfp4_tanimoto", "first"),
    )


def fold_metrics(parent_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, fold, seed), frame in parent_seed.groupby(["task_id", "source_fold", "seed"], sort=True):
        rows.append({"task_id": task, "endpoint": frame.endpoint.iloc[0], "source_fold": fold, "seed": seed,
                     "parents": len(frame), "primary_metric": primary_metric(frame, task),
                     "primary_metric_name": "transformed_RMSE" if task == TASKS[0] else "physical_MAE"})
    return pd.DataFrame(rows)


def similarity_metrics(parent_ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, frame in parent_ensemble.groupby("task_id", sort=True):
        local = frame.copy()
        local["similarity_quintile"] = pd.qcut(
            local.max_train_ecfp4_tanimoto, 5,
            labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"], duplicates="drop",
        )
        for label, subset in local.groupby("similarity_quintile", observed=False):
            rows.append({"task_id": task, "endpoint": subset.endpoint.iloc[0], "similarity_quintile": str(label),
                         "parents": len(subset), "similarity_min": float(subset.max_train_ecfp4_tanimoto.min()),
                         "similarity_max": float(subset.max_train_ecfp4_tanimoto.max()),
                         "similarity_mean": float(subset.max_train_ecfp4_tanimoto.mean()),
                         "primary_metric": primary_metric(subset, task),
                         "primary_metric_name": "transformed_RMSE" if task == TASKS[0] else "physical_MAE"})
    return pd.DataFrame(rows)


def registered_papp_ranges(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep the runner's record-level range assignment and parent aggregation intact."""
    table = metrics.loc[
        metrics.task_id.eq(TASKS[0])
        & metrics.evaluation_scope.eq("source_cluster_holdout_papp_range")
    ].copy()
    if len(table) != 9:
        raise ValueError("Expected three registered Papp ranges for each of three seeds")
    table = table.rename(columns={"subset": "papp_range"})
    return table[["seed", "papp_range", "records", "parents", "transformed_RMSE"]].sort_values(["seed", "papp_range"])


def source_fold_figure(folds: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.8))
    colors = {TASKS[0]: "#4C78A8", TASKS[1]: "#C44E52"}
    for axis, task in zip(axes, TASKS, strict=True):
        table = folds.loc[folds.task_id.eq(task)].groupby("source_fold", as_index=False).agg(
            mean_metric=("primary_metric", "mean"), sd_metric=("primary_metric", "std"), parents=("parents", "first")
        )
        x = table.source_fold.to_numpy() + 1
        axis.errorbar(x, table.mean_metric, yerr=table.sd_metric, color=colors[task], marker="o", linewidth=1.5,
                      markersize=5, capsize=3)
        axis.set_title(LABELS[task], fontweight="bold")
        axis.set_xlabel("Source-cluster fold")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_ylabel("Parent-level transformed RMSE" if task == TASKS[0] else "Parent-level physical MAE")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Error variation across source-cluster holdout folds", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def similarity_figure(metrics: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    order = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    colors = {TASKS[0]: "#4C78A8", TASKS[1]: "#C44E52"}
    for axis, task in zip(axes, TASKS, strict=True):
        table = metrics.loc[metrics.task_id.eq(task)].set_index("similarity_quintile").reindex(order)
        axis.plot(range(5), table.primary_metric, color=colors[task], marker="o", linewidth=1.5, markersize=5)
        # Sample counts remain in the accompanying CSV; omitting point labels prevents text from obscuring data.
        axis.set_title(LABELS[task], fontweight="bold")
        axis.set_xticks(range(5), ["Q1", "Q2", "Q3", "Q4", "Q5"])
        axis.set_xlabel("Maximum train-set ECFP4 Tanimoto quintile")
        axis.set_ylabel("Parent-level transformed RMSE" if task == TASKS[0] else "Parent-level physical MAE")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Source-cluster holdout error by retained-training similarity", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=ROOT / "results/analysis/stl_papp_fu_source_cluster_holdout_v1")
    parser.add_argument("--learning-curve", type=Path, default=ROOT / "results/analysis/stl_papp_fu_learning_curve_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_papp_fu_source_cluster_holdout_analysis_v5")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.holdout / "complete.json", args.holdout / "predictions.csv", args.holdout / "metrics.csv",
                args.learning_curve / "complete.json", args.learning_curve / "metrics.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    holdout_meta = verify_stage(args.holdout, "stl_papp_fu_source_cluster_holdout")
    curve_meta = verify_stage(args.learning_curve, "stl_papp_fu_learning_curve")
    if holdout_meta.get("test_labels_read") or curve_meta.get("test_labels_read"):
        raise ValueError("Source analysis requires test-closed input stages")
    if args.check_only:
        print("Source-cluster holdout analysis ready")
        return
    prediction = pd.read_csv(args.holdout / "predictions.csv")
    similarity = pd.read_csv(args.holdout / "evaluation_parent_max_train_ecfp4_tanimoto.csv")
    holdout_metrics = pd.read_csv(args.holdout / "metrics.csv")
    for column in ["target_value", "train_mean", "train_std_population", "predicted_interface_target"]:
        prediction[column] = pd.to_numeric(prediction[column], errors="raise")
    prediction["predicted_target"] = prediction.predicted_interface_target * prediction.train_std_population + prediction.train_mean
    similarity_key = ["task_id", "molecule_id", "source_fold"]
    if "max_train_ecfp4_tanimoto" in prediction:
        observed_similarity = prediction[similarity_key + ["max_train_ecfp4_tanimoto"]].drop_duplicates()
        expected_similarity = similarity[similarity_key + ["max_train_ecfp4_tanimoto"]].drop_duplicates()
        check = observed_similarity.merge(expected_similarity, on=similarity_key, suffixes=("_prediction", "_artifact"), validate="one_to_one")
        if len(check) != len(expected_similarity) or not np.allclose(
            check.max_train_ecfp4_tanimoto_prediction, check.max_train_ecfp4_tanimoto_artifact, rtol=0, atol=0
        ):
            raise ValueError("Prediction-embedded similarity does not match the registered similarity artifact")
    else:
        prediction = prediction.merge(similarity, on=similarity_key, validate="many_to_one")
    parent_seed = aggregate_parent(prediction, include_seed=True)
    parent_ensemble = aggregate_parent(prediction, include_seed=False)
    folds = fold_metrics(parent_seed)
    similarity_table = similarity_metrics(parent_ensemble)
    papp_table = registered_papp_ranges(holdout_metrics)
    holdout_overall = holdout_metrics.loc[
        holdout_metrics.evaluation_scope.eq("source_cluster_holdout")
        & holdout_metrics.subset.eq("all_source_clusters")
    ].groupby(["task_id", "endpoint", "primary_metric_name"], as_index=False).agg(
        source_cluster_holdout_mean=("primary_metric", "mean"), source_cluster_holdout_sd=("primary_metric", "std")
    )
    curve = pd.read_csv(args.learning_curve / "metrics.csv")
    curve = curve.loc[curve.source_subset.eq("all_sources") & curve.fraction.eq(1.0)]
    reference = curve.groupby(["task_id", "endpoint", "primary_metric_name"], as_index=False).agg(
        scaffold_learning_curve_100pct_mean=("primary_metric", "mean"), scaffold_learning_curve_100pct_sd=("primary_metric", "std")
    )
    comparison = holdout_overall.merge(reference, on=["task_id", "endpoint", "primary_metric_name"], validate="one_to_one")
    comparison["relative_error_change_pct"] = 100 * (
        comparison.source_cluster_holdout_mean / comparison.scaffold_learning_curve_100pct_mean - 1
    )
    comparison["interpretation"] = (
        "diagnostic difference combines source holdout, additional scaffold exclusion, reduced training size, and chemical-space shift"
    )
    with stage_output(args.output) as out:
        folds.to_csv(out / "source_fold_metrics_by_seed.csv", index=False)
        similarity_table.to_csv(out / "similarity_quintile_metrics.csv", index=False)
        papp_table.to_csv(out / "papp_label_range_metrics_by_seed.csv", index=False)
        comparison.to_csv(out / "source_holdout_vs_scaffold_learning_curve_comparison.csv", index=False)
        source_plot = source_fold_figure(folds)
        source_plot.savefig(out / "Figure_source_cluster_holdout_by_fold.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(source_plot)
        similarity_plot = similarity_figure(similarity_table)
        similarity_plot.savefig(out / "Figure_source_cluster_similarity_error.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(similarity_plot)
        summary = {
            "tasks": 2, "holdout_seeds": sorted(parent_seed.seed.unique().tolist()),
            "source_cluster_holdout_model_selection_authorized": False,
            "validation_target_file_opened": False, "validation_rows_evaluated": False, "test_labels_read": False,
            "data_modified": False, "model_fitted": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Source-cluster holdout analysis\n\n"
            "For fu, every record is inverse-logit transformed before repeated-parent aggregation, matching the registered runner metric. "
            "The comparison with the 100% scaffold learning curve is diagnostic only. It combines source-cluster holdout, added scaffold exclusion, reduced training size and changed chemical-space composition, so it is not a causal estimate of any one factor.\n",
            encoding="utf-8",
        )
        finish_stage(out, "stl_papp_fu_source_cluster_holdout_analysis", inputs={
            "holdout_complete_sha256": sha256(args.holdout / "complete.json"),
            "learning_curve_complete_sha256": sha256(args.learning_curve / "complete.json"),
        }, partial=False, **summary)
    print(f"Source-cluster holdout analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
