#!/usr/bin/env python3
"""Read-only cluster-bootstrap uncertainty for the closed Jia 2025 R3b score."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PRIMARY = {"Fu": "paper_svr_merged", "CL": "paper_consensus_15", "VDss": "paper_consensus_15"}
PAPER = {"Fu": {"R2_log": .69, "MAE_log": .30, "RMSE_log": .41, "GMFE": 2.01, "within_2fold": .60},
         "CL": {"R2_log": .48, "MAE_log": .31, "RMSE_log": .42, "GMFE": 2.00, "within_2fold": .64},
         "VDss": {"R2_log": .60, "MAE_log": .28, "RMSE_log": .35, "GMFE": 1.88, "within_2fold": .62}}
METRICS = ["R2_log", "MAE_log", "RMSE_log", "GMFE", "within_2fold"]
LABELS = ["R² (log10)", "MAE (log10)", "RMSE (log10)", "GMFE", "Within 2-fold"]


def metrics(observed, predicted):
    error = np.abs(predicted - observed)
    return {"R2_log": float(r2_score(observed, predicted)),
            "MAE_log": float(mean_absolute_error(observed, predicted)),
            "RMSE_log": float(np.sqrt(mean_squared_error(observed, predicted))),
            "GMFE": float(10 ** error.mean()),
            "within_2fold": float(np.mean(error <= np.log10(2)))}


def cluster_bootstrap(frame, repeats, seed):
    """Resample canonical parents, keeping every record within sampled parents."""
    parents = frame.parent_hash.drop_duplicates().to_numpy()
    grouped = {p: x for p, x in frame.groupby("parent_hash", sort=False)}
    rng = np.random.default_rng(seed)
    samples = {m: np.empty(repeats, dtype=float) for m in METRICS}
    for repeat in range(repeats):
        chosen = rng.choice(parents, size=len(parents), replace=True)
        sampled = pd.concat([grouped[p] for p in chosen], ignore_index=True)
        values = metrics(sampled.observed_log10.to_numpy(float), sampled.predicted_log10.to_numpy(float))
        for name, value in values.items():
            samples[name][repeat] = value
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=ROOT / "results/final/jia2025_author_like_r3b_one_time_score_v5")
    parser.add_argument("--closeout", type=Path, default=ROOT / "results/analysis/jia2025_author_like_r3c_closeout_v2")
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/jia2025_n0_audit_v1")
    parser.add_argument("--bootstrap-input", type=Path, help="Published R3d stage whose verified CI table is reused for a graphics-only revision")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/jia2025_author_like_r3d_bootstrap_v2")
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.repeats < 1000:
        raise ValueError("At least 1000 bootstrap repeats are required")
    required = [args.final / "complete.json", args.closeout / "complete.json", args.n0 / "complete.json"]
    if args.bootstrap_input is not None:
        required.append(args.bootstrap_input / "complete.json")
    startup_self_check(required, output=args.output)
    final = verify_stage(args.final, "jia2025_author_like_r3b_one_time_descriptive_score")
    closeout = verify_stage(args.closeout, "jia2025_author_like_r3c_read_only_closeout")
    verify_stage(args.n0, "jia2025_n0_reproducibility_audit")
    if not final.get("rerun_prohibited") or not closeout.get("read_only"):
        raise ValueError("Closed one-time-score lifecycle or R3c read-only status is invalid")
    reused = args.bootstrap_input is not None
    if reused:
        previous = verify_stage(args.bootstrap_input, "jia2025_author_like_r3d_read_only_bootstrap")
        if not previous.get("read_only") or previous.get("author_test_rerun") or previous.get("model_refit"):
            raise ValueError("Bootstrap-input does not satisfy the R3d read-only lifecycle")
        uncertainty = pd.read_csv(args.bootstrap_input / "primary_metric_cluster_bootstrap_ci.csv")
        expected_cells = {(endpoint, PRIMARY[endpoint], cohort, metric)
                          for endpoint in PRIMARY for cohort in ["author_native_record", "author_parent_purged_sensitivity"]
                          for metric in METRICS}
        actual_cells = set(uncertainty[["endpoint", "model_id", "cohort", "metric"]].itertuples(index=False, name=None))
        if actual_cells != expected_cells or len(uncertainty) != len(expected_cells):
            raise ValueError("Bootstrap-input CI table has incomplete or unexpected primary cells")
        if set(uncertainty.bootstrap_repeats) != {args.repeats} or set(uncertainty.seed) != {args.seed}:
            raise ValueError("Bootstrap-input repeats or seed differs from this registered revision")
    else:
        predictions = pd.read_csv(args.final / "author_test_predictions.csv")
        saved_metrics = pd.read_csv(args.final / "author_test_metrics.csv")
        membership = pd.read_csv(args.n0 / "author_split_membership_hashed.csv")
        rows = []
        for endpoint, model_id in PRIMARY.items():
            frame = predictions[(predictions.endpoint.eq(endpoint)) & (predictions.model_id.eq(model_id))].copy()
            if frame.empty or frame.observed_log10.isna().any() or frame.predicted_log10.isna().any():
                raise ValueError(f"Invalid saved prediction frame: {endpoint}/{model_id}")
            train_parents = set(membership.loc[(membership.endpoint.eq(endpoint)) &
                                               (membership.author_split.eq("train")), "parent_id"])
            expected_purged = frame.loc[~frame.parent_hash.isin(train_parents)].copy()
            for cohort, current in (("author_native_record", frame),
                                    ("author_parent_purged_sensitivity", expected_purged)):
                point = metrics(current.observed_log10.to_numpy(float), current.predicted_log10.to_numpy(float))
                frozen = saved_metrics[(saved_metrics.endpoint.eq(endpoint)) &
                                       (saved_metrics.model_id.eq(model_id)) &
                                       (saved_metrics.cohort.eq(cohort))]
                if len(frozen) != 1 or any(not np.isclose(point[m], float(frozen.iloc[0][m]), atol=1e-12, rtol=0)
                                           for m in METRICS):
                    raise ValueError(f"Saved-prediction metric mismatch: {endpoint}/{model_id}/{cohort}")
                draws = cluster_bootstrap(current, args.repeats, args.seed + len(rows))
                for metric in METRICS:
                    low, high = np.quantile(draws[metric], [.025, .975])
                    rows.append({"endpoint": endpoint, "model_id": model_id, "cohort": cohort,
                                 "records": int(len(current)), "parents": int(current.parent_hash.nunique()),
                                 "metric": metric, "estimate": point[metric], "ci95_low": float(low),
                                 "ci95_high": float(high), "bootstrap_repeats": args.repeats,
                                 "resampling_unit": "canonical_parent_cluster", "seed": args.seed})
        uncertainty = pd.DataFrame(rows)
    with stage_output(args.output) as out:
        uncertainty.to_csv(out / "primary_metric_cluster_bootstrap_ci.csv", index=False)
        native = uncertainty[uncertainty.cohort.eq("author_native_record")].copy()
        plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
        fig, axes = plt.subplots(1, 5, figsize=(15, 4.1))
        endpoints = ["Fu", "CL", "VDss"]
        xpos = np.arange(len(endpoints))
        for ax, metric, label in zip(axes, METRICS, LABELS):
            data = native[native.metric.eq(metric)].set_index("endpoint").loc[endpoints]
            estimate = data.estimate.to_numpy(float)
            yerr = np.vstack([estimate - data.ci95_low.to_numpy(float), data.ci95_high.to_numpy(float) - estimate])
            error_handle = ax.errorbar(xpos, estimate, yerr=yerr, fmt='o', color='#F58518', capsize=3,
                                       label='B-grade author-like (95% cluster bootstrap CI)')
            point_handle = ax.scatter(xpos, [PAPER[e][metric] for e in endpoints], marker='D', color='#4C78A8',
                                      label='Jia et al. 2025 reported point estimate', zorder=3)
            ax.set_xticks(xpos, endpoints); ax.set_title(label); ax.grid(axis='y', alpha=.25)
            ax.spines[['top', 'right']].set_visible(False)
        fig.legend([point_handle, error_handle], ['Jia et al. 2025 reported point estimate',
                                                   'B-grade author-like (95% cluster bootstrap CI)'],
                   loc='upper center', bbox_to_anchor=(.5, .995), ncol=2, frameon=False, fontsize=8)
        fig.text(.5, .02, 'Read-only analysis of saved R3b predictions; parent-cluster resampling; no refitting or re-scoring.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, .80))
        fig.savefig(out / "Figure_3_Jia2025_author_like_bootstrap_CI.png", dpi=600, bbox_inches='tight')
        plt.close(fig)
        (out / "README.md").write_text(
            "# Jia 2025 R3d read-only uncertainty\n\n"
            "This package bootstraps the already saved R3b predictions at canonical-parent cluster level. "
            "It is descriptive uncertainty only: it does not rerun the scorer, refit a model, select a model, "
            "calibrate predictions, or create any new test prediction. Published Jia values are point anchors, not bootstrap inputs.\n",
            encoding="utf-8")
        finish_stage(out, "jia2025_author_like_r3d_read_only_bootstrap", inputs={
            str((args.final / "complete.json").resolve()): sha256(args.final / "complete.json"),
            str((args.closeout / "complete.json").resolve()): sha256(args.closeout / "complete.json"),
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
            **({str((args.bootstrap_input / "complete.json").resolve()): sha256(args.bootstrap_input / "complete.json")} if reused else {})},
            read_only=True, author_test_rerun=False, model_refit=False, model_selection_changed=False,
            bootstrap_repeats=args.repeats, resampling_unit="canonical_parent_cluster", seed=args.seed,
            figure_language="English", dpi=600, reused_verified_bootstrap_table=reused, partial=False)
    print(f"Jia 2025 R3d read-only bootstrap: {args.output}")


if __name__ == "__main__":
    run_cli(main)
