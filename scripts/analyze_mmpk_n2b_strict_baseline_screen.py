#!/usr/bin/env python3
"""Read-only integrity check and publication-style analysis for formal MMPK N2b results."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_mmpk_n2b")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from mmpk_nca_common import ENDPOINTS, approved_model_path
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")
SHORT = {"AUC [ng*h/mL]": "AUC", "Cmax [ng/mL]": "Cmax", "Tmax [h]": "Tmax", "t1/2 [h]": "terminal t½"}
COLORS = {"random_forest": "#4C78A8", "extra_trees": "#F58518", "xgboost": "#54A24B"}


def metrics(observed, predicted) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return {"log_RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
            "GMFE": float(10 ** np.mean(np.abs(error))), "AFE": float(10 ** np.mean(error)),
            "two_fold_accuracy": float(np.mean(np.abs(error) <= np.log10(2.0))),
            "log_R2": float(r2_score(observed, predicted))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=ROOT / "results/benchmarks/mmpk_n2b_strict_r1_baseline_screen_v1")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/mmpk_n2b_strict_r1_baseline_analysis_v1")
    args = parser.parse_args()
    startup_self_check([args.screen / "complete.json", args.n1b / "complete.json"], output=args.output)
    screen = verify_stage(args.screen, "mmpk_n2b_strict_r1_conditional_baseline_screen")
    n1b = verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    model_path = approved_model_path()
    if n1b["inputs"].get(str(model_path.resolve())) != sha256(model_path):
        raise ValueError("N1b does not bind the approved source used for read-only metric reconstruction")
    selected = pd.read_csv(args.screen / "inner_candidate_selection.csv")
    saved_metrics = pd.read_csv(args.screen / "outer_R1_metric_summary.csv")
    predictions = pd.read_csv(args.screen / "outer_R1_predictions_internal.csv")
    if len(selected) != 5 * 4 * int(screen["candidate_cells"]) or selected.groupby(["outer_test_fold", "endpoint"]).selected.sum().ne(1).any():
        raise ValueError("N2b selection registry is incomplete or contains multiple winners")
    if predictions.duplicated(["outer_test_fold", "endpoint", "model_record_id"]).any():
        raise ValueError("N2b prediction keys are duplicated")
    source = pd.read_csv(model_path, usecols=[column for _, column in ENDPOINTS])
    source.insert(0, "model_record_id", [stable_id(f"mmpk_approved_model_row:{idx}") for idx in source.index])
    labels = pd.read_csv(args.n1b / "approved_label_tier_registry_hashed.csv")
    rows = []
    for (outer, endpoint), group in predictions.groupby(["outer_test_fold", "endpoint"], sort=True):
        endpoint = str(endpoint)
        log_column = dict(ENDPOINTS)[endpoint]
        r1 = set(labels.loc[(labels.endpoint.eq(endpoint)) & labels.direct_observation_eligible.astype(bool), "model_record_id"])
        joined = group.merge(source[["model_record_id", log_column]], on="model_record_id", how="left", validate="one_to_one")
        observed = pd.to_numeric(joined[log_column], errors="coerce").to_numpy(float)
        expected = np.asarray([record in r1 for record in joined.model_record_id])
        if not expected.all() or not np.isfinite(observed).all():
            raise ValueError("Saved N2b prediction membership is not exactly R1 finite outer membership")
        result = metrics(observed, joined.predicted_transformed_value.to_numpy(float))
        saved = saved_metrics[(saved_metrics.outer_test_fold.eq(outer)) & (saved_metrics.endpoint.eq(endpoint))]
        if len(saved) != 1:
            raise ValueError("Saved N2b metric row missing")
        for name, value in result.items():
            if not np.isclose(value, float(saved.iloc[0][name]), rtol=1e-10, atol=1e-10):
                raise ValueError(f"Stored metric differs from read-only reconstruction: {outer}/{endpoint}/{name}")
        rows.append({"outer_test_fold": outer, "endpoint": endpoint, "records": len(joined), **result})
    reconstructed = pd.DataFrame(rows)
    pooled_rows = []
    for endpoint, group in predictions.groupby("endpoint", sort=False):
        log_column = dict(ENDPOINTS)[endpoint]
        joined = group.merge(source[["model_record_id", log_column]], on="model_record_id", how="left", validate="many_to_one")
        pooled_rows.append({"endpoint": endpoint, "R1_outer_records": len(joined), **metrics(pd.to_numeric(joined[log_column], errors="coerce"), joined.predicted_transformed_value)})
    pooled = pd.DataFrame(pooled_rows).set_index("endpoint").reindex(PRIMARY).reset_index()
    fold_summary = reconstructed.groupby("endpoint").agg(
        outer_folds=("outer_test_fold", "size"), log_RMSE_mean=("log_RMSE", "mean"), log_RMSE_sd=("log_RMSE", "std"),
        GMFE_mean=("GMFE", "mean"), GMFE_sd=("GMFE", "std"), AFE_mean=("AFE", "mean"),
        two_fold_accuracy_mean=("two_fold_accuracy", "mean"), log_R2_mean=("log_R2", "mean")).reindex(PRIMARY).reset_index()
    winners = selected[selected.selected].copy().sort_values(["endpoint", "outer_test_fold"])
    runner_up = (selected.sort_values(["outer_test_fold", "endpoint", "mean_inner_log_RMSE", "candidate_id"], kind="stable")
                 .groupby(["outer_test_fold", "endpoint"], as_index=False).nth(1)
                 [["outer_test_fold", "endpoint", "mean_inner_log_RMSE"]].rename(columns={"mean_inner_log_RMSE": "runner_up_inner_log_RMSE"}))
    winners = winners.merge(runner_up, on=["outer_test_fold", "endpoint"], validate="one_to_one")
    winners["inner_selection_margin_to_runner_up"] = winners.runner_up_inner_log_RMSE - winners.mean_inner_log_RMSE
    stability = winners.groupby("endpoint").agg(outer_folds=("outer_test_fold", "size"),
        distinct_selected_candidates=("candidate_id", "nunique"), distinct_algorithms=("algorithm", "nunique"),
        minimum_inner_margin=("inner_selection_margin_to_runner_up", "min"), median_inner_margin=("inner_selection_margin_to_runner_up", "median"),
        maximum_inner_margin=("inner_selection_margin_to_runner_up", "max")).reindex(PRIMARY).reset_index()
    # Figure 1: fold-level error summaries.  Figure 2: choices, not an error heatmap.
    labels_short = [SHORT[x] for x in PRIMARY]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    axis = axes[0]
    axis.bar(labels_short, fold_summary.log_RMSE_mean, yerr=fold_summary.log_RMSE_sd, color="#4C78A8", capsize=4)
    axis.set_ylabel("Log-space RMSE")
    axis.set_title("Strict outer-fold performance (R1 direct)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis = axes[1]
    axis.bar(labels_short, fold_summary.GMFE_mean, yerr=fold_summary.GMFE_sd, color="#F58518", capsize=4)
    axis.axhline(2.0, color="#666666", ls="--", lw=1, label="2-fold GMFE reference")
    axis.set_ylabel("Geometric mean fold error")
    axis.set_title("Outer-fold GMFE (R1 direct)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8)
    fig1 = "Figure_1_MMPK_N2b_strict_R1_performance.png"
    fig.savefig(Path("/tmp") / fig1, dpi=600, bbox_inches="tight")
    plt.close(fig)
    algorithms = ["random_forest", "extra_trees", "xgboost"]
    codes = {name: idx for idx, name in enumerate(algorithms)}
    matrix = np.full((len(PRIMARY), 5), np.nan)
    for row in winners.itertuples(index=False):
        matrix[list(PRIMARY).index(row.endpoint), int(row.outer_test_fold)] = codes[row.algorithm]
    cmap = matplotlib.colors.ListedColormap([COLORS[x] for x in algorithms])
    fig, axis = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
    image = axis.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=len(algorithms)-0.5, aspect="auto")
    axis.set_xticks(range(5), [f"Outer fold {i}" for i in range(5)])
    axis.set_yticks(range(len(PRIMARY)), labels_short)
    axis.set_title("Inner-CV-selected algorithm by strict outer fold")
    for yi in range(len(PRIMARY)):
        for xi in range(5):
            name = algorithms[int(matrix[yi, xi])]
            axis.text(xi, yi, {"random_forest": "RF", "extra_trees": "ET", "xgboost": "XGB"}[name], ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    axis.set_xticks(np.arange(-.5, 5, 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(PRIMARY), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.legend(handles=[Patch(color=COLORS[x], label={"random_forest": "Random forest", "extra_trees": "ExtraTrees", "xgboost": "XGBoost"}[x]) for x in algorithms],
                frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    fig2 = "Figure_2_MMPK_N2b_selected_algorithms.png"
    fig.savefig(Path("/tmp") / fig2, dpi=600, bbox_inches="tight")
    plt.close(fig)
    with stage_output(args.output) as out:
        reconstructed.to_csv(out / "outer_metrics_readonly_reconstruction.csv", index=False)
        pooled.to_csv(out / "pooled_R1_metrics.csv", index=False)
        fold_summary.to_csv(out / "outer_fold_metric_summary.csv", index=False)
        winners.to_csv(out / "selected_candidate_by_outer_fold.csv", index=False)
        stability.to_csv(out / "selection_stability_summary.csv", index=False)
        # ``/tmp`` and the project workspace can be separate filesystems.
        # shutil.move falls back to an atomic destination-side copy in that case.
        shutil.move(str(Path("/tmp") / fig1), out / fig1)
        shutil.move(str(Path("/tmp") / fig2), out / fig2)
        (out / "README.md").write_text(
            "# MMPK N2b strict R1 read-only analysis\n\n"
            "This analysis re-reads only the already-scored approved R1 membership to reproduce every stored outer metric exactly, summarizes pooled and fold-level performance, and visualizes already-frozen selection. It does not fit, calibrate, select, change, or externally score any model.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n2b_strict_r1_readonly_analysis", inputs={
            str((args.screen / "complete.json").resolve()): sha256(args.screen / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
            str(model_path.resolve()): sha256(model_path)}, read_only=True, model_fitted=False,
            model_selection_changed=False, external_labels_accessed=False, figures_dpi=600, partial=False)
    print(f"MMPK N2b strict R1 read-only analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
