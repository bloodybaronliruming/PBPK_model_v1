#!/usr/bin/env python3
"""Analyze the formal shared-encoder versus fully-private multitask train-CV."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_stl_p1_matched_comparison import parent_predictions
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PRIVATE = "fully_private_task_networks"
SHARED = "shared_encoder_private_heads"
STAGEA = "corrected_stageA_STL_reference"


def score(frame: pd.DataFrame, prediction: str, metric: str) -> float:
    error = frame[prediction].to_numpy(float) - frame.observed_primary_target.to_numpy(float)
    if metric == "physical_MAE":
        return float(np.mean(np.abs(error)))
    if metric == "transformed_RMSE":
        return float(np.sqrt(np.mean(error ** 2)))
    raise ValueError(f"Unsupported metric: {metric}")


def paired_bootstrap(frame: pd.DataFrame, candidate: str, comparator: str, metric: str, seed: int, repeats: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    folds = [part.reset_index(drop=True) for _, part in frame.groupby("outer_fold", sort=True)]
    values = np.empty(repeats)
    for index in range(repeats):
        sample = pd.concat([part.iloc[rng.integers(0, len(part), len(part))] for part in folds], ignore_index=True)
        values[index] = score(sample, candidate, metric) - score(sample, comparator, metric)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def verify_formal(path: Path) -> dict:
    meta = json.loads((path / "complete.json").read_text())
    if meta.get("stage") != "multitask_shared_private_traincv" or meta.get("partial") or not meta.get("full_configuration"):
        raise ValueError("Only the full shared/private train-CV can support analysis")
    if meta.get("validation_target_file_opened") or meta.get("test_labels_read") or meta.get("evaluation_labels_used_during_fit"):
        raise ValueError("Formal run is not validation/test closed")
    if meta.get("expected_model_files") != meta.get("model_files") or not meta.get("predictions_finite"):
        raise ValueError("Formal model/prediction coverage is incomplete")
    for rel, digest in meta.get("artifacts", {}).items():
        q = path / rel
        if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or modified formal artifact: {q}")
    return meta


def build_figure(folds: pd.DataFrame, overall: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    endpoints = ["CL", "CLint", "Papp", "Thalf", "VDss", "fu"]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2), squeeze=False)
    for axis, endpoint in zip(axes.ravel(), endpoints, strict=True):
        part = folds.loc[folds.endpoint.eq(endpoint)].sort_values("outer_fold")
        row = overall.loc[overall.endpoint.eq(endpoint)].iloc[0]
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.plot(part.outer_fold + 1, part.shared_minus_private, marker="o", color="#4C78A8", linewidth=1.4)
        label = "Terminal t½" if endpoint == "Thalf" else endpoint
        axis.set_title(f"{label}\nOverall change: {row.shared_relative_change_pct:+.2f}%", fontweight="bold", fontsize=9)
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_xlabel("Outer scaffold fold")
        axis.set_ylabel("Shared minus private\n(primary metric)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Shared encoder versus fully private task networks", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traincv", type=Path, default=ROOT / "results/benchmarks/multitask_shared_private_traincv_v2")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/multitask_shared_private_protocol_v2")
    parser.add_argument("--p1", type=Path, default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/multitask_shared_private_traincv_analysis_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.traincv / "complete.json", args.traincv / "ensemble_oof_predictions.csv",
                args.protocol / "complete.json", args.protocol / "protocol.json",
                args.p1 / "complete.json", args.p1 / "stageA_matched_control_predictions.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_formal(args.traincv)
    protocol_meta = verify_stage(args.protocol, "multitask_shared_private_protocol")
    p1_meta = verify_stage(args.p1, "stl_p1_nested_confirmation")
    if protocol_meta.get("test_labels_read") or p1_meta.get("test_labels_read") or args.bootstrap_repeats < 200:
        raise ValueError("Analysis inputs or bootstrap configuration invalid")
    protocol = json.loads((args.protocol / "protocol.json").read_text())
    if args.check_only:
        print(f"Shared/private analysis ready: endpoints={len(protocol['human_evaluation_tasks'])} bootstrap={args.bootstrap_repeats}")
        return
    ensemble = pd.read_csv(args.traincv / "ensemble_oof_predictions.csv")
    if set(ensemble.route_id) != {PRIVATE, SHARED} or ensemble.duplicated(["route_id", "outer_fold", "task_id", "parent_id"]).any():
        raise ValueError("Formal ensemble route/key coverage invalid")
    metric_map = {row.endpoint: row.primary_metric for row in pd.read_csv(args.traincv / "ensemble_fold_metrics.csv").itertuples(index=False)}
    private = ensemble.loc[ensemble.route_id.eq(PRIVATE), ["outer_fold", "task_id", "endpoint", "parent_id", "observed_primary_target", "ensemble_predicted_primary_target"]].rename(
        columns={"ensemble_predicted_primary_target": "private_prediction"})
    shared = ensemble.loc[ensemble.route_id.eq(SHARED), ["outer_fold", "task_id", "parent_id", "observed_primary_target", "ensemble_predicted_primary_target"]].rename(
        columns={"observed_primary_target": "shared_observed", "ensemble_predicted_primary_target": "shared_prediction"})
    joined = private.merge(shared, on=["outer_fold", "task_id", "parent_id"], validate="one_to_one")
    if (joined.observed_primary_target - joined.shared_observed).abs().max() > 1e-12:
        raise ValueError("Shared/private observed targets differ")
    joined = joined.drop(columns="shared_observed")

    stagea_raw = parent_predictions(pd.read_csv(args.p1 / "stageA_matched_control_predictions.csv"))
    stagea_raw = stagea_raw.loc[stagea_raw.task_id.isin(protocol["human_evaluation_tasks"])].copy()
    stagea_raw["stageA_prediction"] = np.where(
        stagea_raw.endpoint.eq("fu"), stagea_raw.predicted_physical, stagea_raw.predicted_transformed)
    stagea_raw["stageA_observed"] = np.where(
        stagea_raw.endpoint.eq("fu"), stagea_raw.observed_physical, stagea_raw.observed_transformed)
    stagea = stagea_raw.rename(columns={"molecule_id": "parent_id"})[["outer_fold", "task_id", "parent_id", "stageA_prediction", "stageA_observed"]]
    joined = joined.merge(stagea, on=["outer_fold", "task_id", "parent_id"], validate="one_to_one")
    if (joined.observed_primary_target - joined.stageA_observed).abs().max() > 1e-10:
        raise ValueError("Formal and corrected Stage-A observed targets differ")

    overall_rows, fold_rows, comparison_rows = [], [], []
    for index, (task, part) in enumerate(joined.groupby("task_id", sort=True)):
        endpoint = part.endpoint.iloc[0]; metric = metric_map[endpoint]
        private_score, shared_score, stagea_score = (
            score(part, "private_prediction", metric), score(part, "shared_prediction", metric), score(part, "stageA_prediction", metric)
        )
        improvement = 100 * (private_score - shared_score) / private_score
        mean, low, high = paired_bootstrap(part, "shared_prediction", "private_prediction", metric, 20260917 + index, args.bootstrap_repeats)
        noninferior = 0
        for fold, fp in part.groupby("outer_fold", sort=True):
            ps, ss, stage = score(fp, "private_prediction", metric), score(fp, "shared_prediction", metric), score(fp, "stageA_prediction", metric)
            noninferior += int(ss <= ps)
            fold_rows.append({"task_id": task, "endpoint": endpoint, "outer_fold": fold, "parents": len(fp), "primary_metric": metric,
                              "private_score": ps, "shared_score": ss, "stageA_reference_score": stage, "shared_minus_private": ss - ps})
        endpoint_pass = bool(improvement >= 2.0 and noninferior >= 3 and high < 0)
        overall_rows.append({"task_id": task, "endpoint": endpoint, "parents": len(part), "primary_metric": metric,
                             "private_score": private_score, "shared_score": shared_score, "stageA_reference_score": stagea_score,
                             "shared_relative_change_pct": 100 * (shared_score / private_score - 1),
                             "shared_noninferior_outer_folds": noninferior})
        comparison_rows.append({"task_id": task, "endpoint": endpoint, "primary_metric": metric,
                                "shared_improvement_pct": improvement, "noninferior_outer_folds": noninferior,
                                "bootstrap_difference_mean": mean, "bootstrap_95ci_low": low, "bootstrap_95ci_high": high,
                                "endpoint_gate_passed": endpoint_pass})
    overall, folds, comparisons = pd.DataFrame(overall_rows), pd.DataFrame(fold_rows), pd.DataFrame(comparison_rows)
    gate = protocol["advance_gate"]
    endpoints_passing = int(comparisons.endpoint_gate_passed.sum())
    maximum_harm = float(overall.shared_relative_change_pct.max())
    global_pass = bool(endpoints_passing >= int(gate["minimum_endpoints_passing"]) and maximum_harm <= float(gate["no_endpoint_relative_harm_over_pct"]))
    decision = {
        "endpoints_passing": endpoints_passing, "minimum_endpoints_passing": int(gate["minimum_endpoints_passing"]),
        "maximum_endpoint_relative_harm_pct": maximum_harm,
        "maximum_allowed_endpoint_relative_harm_pct": float(gate["no_endpoint_relative_harm_over_pct"]),
        "shared_architecture_advanced": global_pass,
        "next_action": "advance_shared_encoder_to_next_finite_ablation" if global_pass else "retain_shared_encoder_as_non_advanced_ablation",
    }
    with stage_output(args.output) as out:
        overall.to_csv(out / "endpoint_overall_metrics.csv", index=False)
        folds.to_csv(out / "endpoint_outer_fold_metrics.csv", index=False)
        comparisons.to_csv(out / "paired_bootstrap_comparison.csv", index=False)
        joined.to_csv(out / "matched_parent_predictions.csv", index=False)
        (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        fig = build_figure(folds, overall)
        fig.savefig(out / "Figure_shared_vs_private_trainCV.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        (out / "README.md").write_text(
            "# Shared/private formal train-CV analysis\n\nComparisons use identical human outer-fold parents. "
            "Stage-A is a strong reference on the same evaluation parents but does not use the globally purged multitask training resource, so it is not a compute/data-matched neural control. "
            "Bootstrap uses 2,000 fold-stratified paired parent resamples. Fixed validation/test remain closed.\n")
        finish_stage(out, "multitask_shared_private_traincv_analysis",
                     inputs={"traincv_complete_sha256": sha256(args.traincv / "complete.json"),
                             "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                             "p1_complete_sha256": sha256(args.p1 / "complete.json")},
                     endpoints=6, bootstrap_repeats=args.bootstrap_repeats, shared_architecture_advanced=global_pass,
                     validation_target_file_opened=False, test_labels_read=False, model_fitted=False, partial=False)
    print(f"Shared/private train-CV analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
