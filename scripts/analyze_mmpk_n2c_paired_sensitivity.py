#!/usr/bin/env python3
"""Read-only bootstrap and publication analysis for frozen MMPK N2c results."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_mmpk_n2c")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from mmpk_nca_common import ENDPOINTS, approved_model_path
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")
ARMS = ("R1_plus_R2_train", "formulation_matched")
SHORT = {"AUC [ng*h/mL]": "AUC", "Cmax [ng/mL]": "Cmax", "Tmax [h]": "Tmax", "t1/2 [h]": "terminal t½"}
ARM_LABEL = {"R1_plus_R2_train": "R1+R2 train", "formulation_matched": "Formulation matched"}
ARM_COLOR = {"R1_plus_R2_train": "#4C78A8", "formulation_matched": "#F58518"}


def metric_values(observed: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.asarray(prediction, dtype=float) - np.asarray(observed, dtype=float)
    return {
        "log_RMSE": float(np.sqrt(mean_squared_error(observed, prediction))),
        "GMFE": float(10 ** np.mean(np.abs(error))),
        "AFE": float(10 ** np.mean(error)),
        "two_fold_accuracy": float(np.mean(np.abs(error) <= np.log10(2.0))),
        "log_R2": float(r2_score(observed, prediction)),
    }


def parent_cluster_bootstrap(frame: pd.DataFrame, replicates: int, seed: int, chunk_size: int = 250) -> np.ndarray:
    """Return paired RMSE deltas using canonical-parent cluster resampling."""
    grouped = frame.assign(
        baseline_sq=(frame.baseline_prediction - frame.observed) ** 2,
        sensitivity_sq=(frame.sensitivity_prediction - frame.observed) ** 2,
    ).groupby("parent_id", sort=True).agg(
        records=("model_record_id", "size"), baseline_sq=("baseline_sq", "sum"), sensitivity_sq=("sensitivity_sq", "sum")
    )
    if len(grouped) < 2:
        raise ValueError("Canonical-parent bootstrap requires at least two parent clusters")
    counts = grouped.records.to_numpy(float)
    baseline_sq = grouped.baseline_sq.to_numpy(float)
    sensitivity_sq = grouped.sensitivity_sq.to_numpy(float)
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=float)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        sampled = rng.integers(0, len(grouped), size=(size, len(grouped)))
        denominator = counts[sampled].sum(axis=1)
        baseline = np.sqrt(baseline_sq[sampled].sum(axis=1) / denominator)
        sensitivity = np.sqrt(sensitivity_sq[sampled].sum(axis=1) / denominator)
        result[start:start + size] = sensitivity - baseline
    return result


def load_pairs(analysis: Path, n1b: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach only already-consumed direct R1 labels to saved paired predictions."""
    predictions = pd.read_csv(analysis / "paired_predictions_internal.csv")
    formal_metrics = pd.read_csv(analysis / "paired_outer_R1_metric_summary.csv")
    required = {"outer_test_fold", "endpoint", "comparison_arm", "prediction_role", "model_record_id", "predicted_transformed_value"}
    if not required <= set(predictions) or not set(predictions.comparison_arm) <= set(ARMS):
        raise ValueError("N2c paired prediction schema is invalid")
    if predictions.duplicated(["outer_test_fold", "endpoint", "comparison_arm", "prediction_role", "model_record_id"]).any():
        raise ValueError("N2c paired predictions have duplicate keys")
    wide = predictions.pivot(index=["outer_test_fold", "endpoint", "comparison_arm", "model_record_id"],
                             columns="prediction_role", values="predicted_transformed_value").reset_index()
    if set(wide.columns) - {"outer_test_fold", "endpoint", "comparison_arm", "model_record_id", "baseline_reuse", "sensitivity"}:
        raise ValueError("N2c paired prediction roles are unexpected")
    if wide[["baseline_reuse", "sensitivity"]].isna().any().any():
        raise ValueError("N2c paired prediction roles are incomplete")
    wide = wide.rename(columns={"baseline_reuse": "baseline_prediction", "sensitivity": "sensitivity_prediction"})
    records = pd.read_csv(n1b / "approved_record_registry_hashed.csv", usecols=["model_record_id", "strict_outer_fold", "parent_id"])
    labels = pd.read_csv(n1b / "approved_label_tier_registry_hashed.csv")
    source = pd.read_csv(approved_model_path(), usecols=[column for _, column in ENDPOINTS])
    source.insert(0, "model_record_id", [stable_id(f"mmpk_approved_model_row:{idx}") for idx in source.index])
    values = []
    for endpoint, log_column in ENDPOINTS:
        if endpoint not in PRIMARY:
            continue
        allowed = labels[(labels.endpoint.eq(endpoint)) & labels.direct_observation_eligible.astype(bool)].model_record_id
        item = source[["model_record_id", log_column]].rename(columns={log_column: "observed"})
        item["endpoint"] = endpoint
        item["direct_R1"] = item.model_record_id.isin(set(allowed))
        values.append(item)
    observed = pd.concat(values, ignore_index=True)
    pairs = (wide.merge(records, on="model_record_id", how="left", validate="many_to_one")
             .merge(observed, on=["model_record_id", "endpoint"], how="left", validate="many_to_one"))
    if pairs.parent_id.isna().any() or not pairs.strict_outer_fold.eq(pairs.outer_test_fold).all():
        raise ValueError("N2c predictions do not match frozen strict outer-fold membership")
    if not pairs.direct_R1.astype(bool).all() or not np.isfinite(pairs.observed).all():
        raise ValueError("N2c predictions are not exactly finite direct-R1 membership")
    reconstructed = []
    for (outer, endpoint, arm), group in pairs.groupby(["outer_test_fold", "endpoint", "comparison_arm"], sort=True):
        baseline = metric_values(group.observed.to_numpy(float), group.baseline_prediction.to_numpy(float))
        sensitivity = metric_values(group.observed.to_numpy(float), group.sensitivity_prediction.to_numpy(float))
        saved = formal_metrics[(formal_metrics.outer_test_fold.eq(outer)) & formal_metrics.endpoint.eq(endpoint) & formal_metrics.arm.eq(arm)]
        if len(saved) != 1:
            raise ValueError("N2c formal metric row is missing")
        for prefix, result in (("baseline_", baseline), ("sensitivity_", sensitivity)):
            for metric, value in result.items():
                if not np.isclose(value, float(saved.iloc[0][prefix + metric]), rtol=1e-10, atol=1e-10):
                    raise ValueError(f"N2c stored metric mismatch: {outer}/{endpoint}/{arm}/{metric}")
        reconstructed.append({"outer_test_fold": outer, "endpoint": endpoint, "arm": arm, "paired_R1_records": int(len(group)),
                              "baseline_log_RMSE": baseline["log_RMSE"], "sensitivity_log_RMSE": sensitivity["log_RMSE"],
                              "delta_log_RMSE": sensitivity["log_RMSE"] - baseline["log_RMSE"],
                              "relative_log_RMSE_improvement": 1.0 - sensitivity["log_RMSE"] / baseline["log_RMSE"]})
    return pairs, pd.DataFrame(reconstructed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/benchmarks/mmpk_n2c_paired_sensitivity_v1")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2c_paired_sensitivity_protocol_v1")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-read-only-analysis", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    if not args.smoke and not args.confirm_read_only_analysis:
        raise ValueError("Read-only analysis opens already-consumed R1 labels; pass --confirm-read-only-analysis")
    if args.output is None:
        args.output = (ROOT / "results/analysis/mmpk_n2c_paired_sensitivity_bootstrap_smoke_v1" if args.smoke
                       else ROOT / "results/analysis/mmpk_n2c_paired_sensitivity_analysis_v1")
    required = [args.analysis / "complete.json", args.n1b / "complete.json", args.protocol / "complete.json"]
    startup_self_check(required, output=args.output)
    formal = verify_stage(args.analysis, "mmpk_n2c_paired_sensitivity")
    verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    protocol = verify_stage(args.protocol, "mmpk_n2c_paired_sensitivity_protocol")
    if not formal.get("formal_paired_sensitivity") or formal.get("external_labels_accessed"):
        raise ValueError("N2c formal stage is not eligible for read-only analysis")
    if protocol.get("bootstrap_replicates") != args.bootstrap_replicates and not args.smoke:
        raise ValueError("Formal bootstrap replicate count differs from frozen N2c protocol")
    if protocol.get("bootstrap_seed") != args.bootstrap_seed and not args.smoke:
        raise ValueError("Formal bootstrap seed differs from frozen N2c protocol")
    pairs, reconstructed = load_pairs(args.analysis, args.n1b)
    pooled_rows, bootstrap_rows, direction_rows = [], [], []
    for endpoint in PRIMARY:
        for arm in ARMS:
            subset = pairs[(pairs.endpoint.eq(endpoint)) & (pairs.comparison_arm.eq(arm))].copy()
            if subset.empty:
                raise ValueError(f"Missing N2c pair: {endpoint}/{arm}")
            baseline = metric_values(subset.observed.to_numpy(float), subset.baseline_prediction.to_numpy(float))
            sensitivity = metric_values(subset.observed.to_numpy(float), subset.sensitivity_prediction.to_numpy(float))
            delta = sensitivity["log_RMSE"] - baseline["log_RMSE"]
            direction = reconstructed[(reconstructed.endpoint.eq(endpoint)) & reconstructed.arm.eq(arm)]
            if len(direction) != 5:
                raise ValueError("N2c outer-fold direction summary is incomplete")
            deltas = parent_cluster_bootstrap(subset, args.bootstrap_replicates, args.bootstrap_seed)
            lower, upper = np.quantile(deltas, [0.025, 0.975])
            pooled_rows.append({"endpoint": endpoint, "arm": arm, "paired_R1_records": int(len(subset)),
                                "canonical_parent_clusters": int(subset.parent_id.nunique()),
                                "baseline_log_RMSE": baseline["log_RMSE"], "sensitivity_log_RMSE": sensitivity["log_RMSE"],
                                "delta_log_RMSE": delta, "relative_log_RMSE_improvement": 1.0 - sensitivity["log_RMSE"] / baseline["log_RMSE"]})
            bootstrap_rows.append({"endpoint": endpoint, "arm": arm, "bootstrap_unit": "canonical_parent",
                                   "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.bootstrap_seed,
                                   "point_delta_log_RMSE": delta, "bootstrap_delta_mean": float(np.mean(deltas)),
                                   "bootstrap_delta_ci95_lower": float(lower), "bootstrap_delta_ci95_upper": float(upper),
                                   "bootstrap_probability_delta_lt_zero": float(np.mean(deltas < 0.0))})
            direction_rows.append({"endpoint": endpoint, "arm": arm, "improved_outer_folds": int((direction.delta_log_RMSE < 0).sum()),
                                   "worse_outer_folds": int((direction.delta_log_RMSE > 0).sum()), "tied_outer_folds": int((direction.delta_log_RMSE == 0).sum())})
    pooled = pd.DataFrame(pooled_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    direction = pd.DataFrame(direction_rows)
    merged = pooled.merge(bootstrap, on=["endpoint", "arm"], validate="one_to_one").merge(direction, on=["endpoint", "arm"], validate="one_to_one")
    merged["passes_relative_2pct"] = merged.relative_log_RMSE_improvement.ge(0.02)
    merged["passes_3_of_5_outer_folds"] = merged.improved_outer_folds.ge(3)
    merged["passes_bootstrap_ci"] = merged.bootstrap_delta_ci95_upper.lt(0.0)
    merged["sensitivity_signal"] = merged[["passes_relative_2pct", "passes_3_of_5_outer_folds", "passes_bootstrap_ci"]].all(axis=1)
    labels = [f"{SHORT[e]}\n{ARM_LABEL[a]}" for e in PRIMARY for a in ARMS]
    view = merged.set_index(["endpoint", "arm"]).reindex(pd.MultiIndex.from_product([PRIMARY, ARMS], names=["endpoint", "arm"])).reset_index()
    y = np.arange(len(view))
    fig, axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    errors = np.vstack([view.point_delta_log_RMSE - view.bootstrap_delta_ci95_lower,
                        view.bootstrap_delta_ci95_upper - view.point_delta_log_RMSE])
    colors = [ARM_COLOR[arm] for arm in view.arm]
    axis.errorbar(view.point_delta_log_RMSE, y, xerr=errors, fmt="none", ecolor="#444444", elinewidth=1.2, capsize=3, zorder=1)
    axis.scatter(view.point_delta_log_RMSE, y, s=42, c=colors, edgecolors="white", linewidths=0.7, zorder=2)
    axis.axvline(0, color="#333333", lw=1, ls="--")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Paired sensitivity − baseline log-space RMSE\n(negative favors sensitivity)")
    axis.set_title("N2c canonical-parent paired bootstrap (95% CI)")
    axis.grid(axis="x", alpha=0.25)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_COLOR[a], label=ARM_LABEL[a], markersize=7) for a in ARMS]
    axis.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    fig1 = Path("/tmp/Figure_1_MMPK_N2c_paired_bootstrap.png")
    fig.savefig(fig1, dpi=600, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    for axis, arm in zip(axes, ARMS):
        sub = view[view.arm.eq(arm)]
        xpos = np.arange(len(PRIMARY))
        axis.scatter(xpos - 0.12, sub.baseline_log_RMSE, color="#7F7F7F", s=42, label="Saved baseline", zorder=3)
        axis.scatter(xpos + 0.12, sub.sensitivity_log_RMSE, color=ARM_COLOR[arm], s=42, label="Sensitivity", zorder=3)
        for row, pos in zip(sub.itertuples(index=False), xpos):
            axis.plot([pos - 0.12, pos + 0.12], [row.baseline_log_RMSE, row.sensitivity_log_RMSE], color="#B0B0B0", lw=1, zorder=1)
        axis.set_xticks(xpos, [SHORT[e] for e in PRIMARY])
        axis.set_ylabel("Pooled log-space RMSE")
        axis.set_title(ARM_LABEL[arm])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=8)
    fig2 = Path("/tmp/Figure_2_MMPK_N2c_paired_performance.png")
    fig.savefig(fig2, dpi=600, bbox_inches="tight")
    plt.close(fig)
    with stage_output(args.output) as out:
        reconstructed.to_csv(out / "outer_metrics_readonly_reconstruction.csv", index=False)
        pooled.to_csv(out / "pooled_paired_metrics.csv", index=False)
        bootstrap.to_csv(out / "canonical_parent_paired_bootstrap.csv", index=False)
        direction.to_csv(out / "outer_fold_direction_summary.csv", index=False)
        merged.to_csv(out / "sensitivity_signal_decision.csv", index=False)
        shutil.move(str(fig1), out / fig1.name)
        shutil.move(str(fig2), out / fig2.name)
        (out / "README.md").write_text(
            "# MMPK N2c paired sensitivity read-only analysis\n\n"
            "This stage attaches only already-consumed direct-R1 outer labels to frozen paired predictions, reproduces each stored formal metric, and performs the pre-registered canonical-parent paired bootstrap. It does not fit, predict, calibrate, select, access external data, or modify N2b/N2c decisions.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n2c_paired_sensitivity_readonly_analysis", inputs={
            str((args.analysis / "complete.json").resolve()): sha256(args.analysis / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
            str((args.protocol / "complete.json").resolve()): sha256(args.protocol / "complete.json")},
            read_only=True, model_fitted=False, model_selection_changed=False, calibration_changed=False,
            external_labels_accessed=False, bootstrap_unit="canonical_parent", bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed, figures_dpi=600, smoke=args.smoke, partial=bool(args.smoke))
    print(f"MMPK N2c paired sensitivity read-only analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
