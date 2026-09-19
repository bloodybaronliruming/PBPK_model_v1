#!/usr/bin/env python3
"""Analyze corrected P1 nested STL results against matched Stage-A controls."""
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


PAPP_TASK = "Papp__human__caco2_ab"
FU_TASK = "fu__human__plasma"


def parent_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["observed_transformed"] = work.target_value.to_numpy(float)
    work["predicted_transformed"] = (
        work.predicted_interface_target.to_numpy(float) * work.train_std_population.to_numpy(float)
        + work.train_mean.to_numpy(float)
    )
    work["observed_physical"] = expit(work.observed_transformed)
    work["predicted_physical"] = expit(work.predicted_transformed)
    return work.groupby(["task_id", "endpoint", "outer_fold", "molecule_id"], as_index=False).agg(
        observed_transformed=("observed_transformed", "mean"),
        predicted_transformed=("predicted_transformed", "mean"),
        observed_physical=("observed_physical", "mean"),
        predicted_physical=("predicted_physical", "mean"),
        source_records=("row_id", "size"),
    )


def primary_metric(frame: pd.DataFrame, task_id: str) -> float:
    if task_id == FU_TASK:
        return float(np.mean(np.abs(frame.observed_physical - frame.predicted_physical)))
    return float(np.sqrt(np.mean((frame.observed_transformed - frame.predicted_transformed) ** 2)))


def paired_bootstrap(p1: pd.DataFrame, control: pd.DataFrame, task_id: str, seed: int, repeats: int = 2000) -> tuple[float, float, float]:
    joined = p1.merge(control, on=["molecule_id", "outer_fold"], suffixes=("_p1", "_control"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        sample = joined.iloc[rng.integers(0, len(joined), size=len(joined))]
        if task_id == FU_TASK:
            p1_score = float(np.mean(np.abs(sample.observed_physical_p1 - sample.predicted_physical_p1)))
            control_score = float(np.mean(np.abs(sample.observed_physical_control - sample.predicted_physical_control)))
        else:
            p1_score = float(np.sqrt(np.mean((sample.observed_transformed_p1 - sample.predicted_transformed_p1) ** 2)))
            control_score = float(np.sqrt(np.mean((sample.observed_transformed_control - sample.predicted_transformed_control) ** 2)))
        differences.append(p1_score - control_score)
    array = np.asarray(differences)
    return float(np.mean(array)), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def comparison_figure(folds: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    tasks = folds.task_id.drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.1), squeeze=False)
    for axis, task_id in zip(axes.ravel(), tasks, strict=True):
        table = folds.loc[folds.task_id.eq(task_id)].sort_values("outer_fold")
        difference = table.primary_metric_p1.to_numpy(float) - table.primary_metric_control.to_numpy(float)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.plot(table.outer_fold + 1, difference, marker="o", color="#4C78A8", linewidth=1.3)
        axis.set_title(table.endpoint.iloc[0], fontweight="bold")
        axis.set_xlabel("Outer scaffold fold")
        axis.set_ylabel("P1 minus matched control\n(primary metric)")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Nested P1 versus matched Stage-A control", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", type=Path, default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_p1_matched_comparison_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.p1 / "complete.json", args.p1 / "predictions.csv", args.p1 / "stageA_matched_control_predictions.csv",
                args.p1 / "outer_selection_registry.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    meta = verify_stage(args.p1, "stl_p1_nested_confirmation")
    if meta.get("partial") or meta.get("test_labels_read"):
        raise ValueError("Matched comparison requires a full test-closed P1 stage")
    if args.bootstrap_repeats < 200:
        raise ValueError("Bootstrap repeats must be at least 200")
    if args.check_only:
        print("P1 matched-comparison inputs valid")
        return
    p1 = parent_predictions(pd.read_csv(args.p1 / "predictions.csv"))
    control = parent_predictions(pd.read_csv(args.p1 / "stageA_matched_control_predictions.csv"))
    selection = pd.read_csv(args.p1 / "outer_selection_registry.csv")
    rows, fold_rows, repeat_rows = [], [], []
    for index, task_id in enumerate(sorted(p1.task_id.unique())):
        p1_task = p1.loc[p1.task_id.eq(task_id)].copy()
        control_task = control.loc[control.task_id.eq(task_id)].copy()
        if set(zip(p1_task.outer_fold, p1_task.molecule_id)) != set(zip(control_task.outer_fold, control_task.molecule_id)):
            raise ValueError(f"Matched-control keys differ for {task_id}")
        for outer_fold in sorted(p1_task.outer_fold.unique()):
            p1_fold = p1_task.loc[p1_task.outer_fold.eq(outer_fold)]
            control_fold = control_task.loc[control_task.outer_fold.eq(outer_fold)]
            fold_rows.append({"task_id": task_id, "endpoint": p1_fold.endpoint.iloc[0], "outer_fold": outer_fold,
                              "parents": len(p1_fold), "primary_metric_p1": primary_metric(p1_fold, task_id),
                              "primary_metric_control": primary_metric(control_fold, task_id)})
        p1_score, control_score = primary_metric(p1_task, task_id), primary_metric(control_task, task_id)
        mean_difference, ci_low, ci_high = paired_bootstrap(p1_task, control_task, task_id, 20260916 + index, args.bootstrap_repeats)
        rows.append({"task_id": task_id, "endpoint": p1_task.endpoint.iloc[0],
                     "primary_metric_name": "physical_MAE" if task_id == FU_TASK else "transformed_RMSE",
                     "p1_primary_metric": p1_score, "matched_control_primary_metric": control_score,
                     "relative_change_pct": 100 * (p1_score / control_score - 1),
                     "paired_bootstrap_difference_mean": mean_difference,
                     "paired_bootstrap_difference_95ci_low": ci_low,
                     "paired_bootstrap_difference_95ci_high": ci_high,
                     "bootstrap_repeats": args.bootstrap_repeats})
        for group_name, mask in [("single_record_parent", p1_task.source_records.eq(1)), ("repeat_record_parent", p1_task.source_records.gt(1))]:
            if not mask.any():
                continue
            p1_subset, control_subset = p1_task.loc[mask], control_task.loc[mask]
            repeat_rows.append({"task_id": task_id, "endpoint": p1_subset.endpoint.iloc[0], "subset": group_name,
                                "parents": len(p1_subset), "p1_primary_metric": primary_metric(p1_subset, task_id),
                                "matched_control_primary_metric": primary_metric(control_subset, task_id)})
    folds = pd.DataFrame(fold_rows)
    comparison = pd.DataFrame(rows)
    repeat = pd.DataFrame(repeat_rows)
    stability = selection.loc[selection.selected_for_outer_refit.astype(bool)].groupby(
        ["task_id", "endpoint", "candidate_uid", "algorithm", "feature_set"], as_index=False
    ).size().rename(columns={"size": "outer_folds_selected"})
    with stage_output(args.output) as out:
        comparison.to_csv(out / "overall_matched_comparison.csv", index=False)
        folds.to_csv(out / "outer_fold_matched_comparison.csv", index=False)
        repeat.to_csv(out / "repeat_parent_matched_comparison.csv", index=False)
        stability.to_csv(out / "selection_stability.csv", index=False)
        figure = comparison_figure(folds)
        figure.savefig(out / "Figure_P1_vs_matched_control.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# Corrected P1 matched-control comparison\n\n"
            "P1 and the fixed Stage-A leader use identical corrected parent targets, outer scaffold folds, three estimator seeds and ensemble rule. "
            "Bootstrap intervals describe paired parent resampling in this train-only development setting; they are not external-validation confidence intervals.\n",
            encoding="utf-8")
        summary = {"tasks": int(p1.task_id.nunique()), "bootstrap_repeats": args.bootstrap_repeats,
                   "validation_target_file_opened": False, "test_labels_read": False, "model_fitted": False,
                   "data_modified": False, "model_selection_authorized": False}
        finish_stage(out, "stl_p1_matched_comparison", inputs={"p1_complete_sha256": sha256(args.p1 / "complete.json")},
                     partial=False, **summary)
    print(f"P1 matched comparison: {args.output}")


if __name__ == "__main__":
    run_cli(main)
