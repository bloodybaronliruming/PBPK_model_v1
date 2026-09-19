#!/usr/bin/env python3
"""Compare formal two-group OOF predictions with locked shared/private and Stage-A controls."""
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

from analyze_multitask_shared_private_traincv import paired_bootstrap, score
from analyze_stl_p1_matched_comparison import parent_predictions
from multitask_two_group_common import ROUTE_ID
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PRIVATE = "fully_private_task_networks"
SHARED = "shared_encoder_private_heads"
ENDPOINT_ORDER = ["CL", "CLint", "Papp", "Thalf", "VDss", "fu"]


def verify_formal(path: Path, stage: str) -> dict:
    meta = verify_stage(path, stage)
    if not meta.get("full_configuration") or meta.get("partial"):
        raise ValueError(f"Only a full formal run can support analysis: {path}")
    if meta.get("validation_target_file_opened") or meta.get("test_labels_read"):
        raise ValueError(f"Formal input opened validation/test labels: {path}")
    if meta.get("evaluation_labels_used_during_fit"):
        raise ValueError(f"Formal input used outer-evaluation labels during fit: {path}")
    if meta.get("expected_model_files") != meta.get("model_files"):
        raise ValueError(f"Formal model coverage is incomplete: {path}")
    if not meta.get("predictions_finite"):
        raise ValueError(f"Formal predictions are non-finite: {path}")
    return meta


def build_figure(folds: pd.DataFrame, overall: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.3), squeeze=False)
    for axis, endpoint in zip(axes.ravel(), ENDPOINT_ORDER, strict=True):
        part = folds.loc[folds.endpoint.eq(endpoint)].sort_values("outer_fold")
        row = overall.loc[overall.endpoint.eq(endpoint)].iloc[0]
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.plot(
            part.outer_fold + 1, part.grouped_minus_shared, marker="o",
            color="#3B7A57" if endpoint == "fu" else "#4C78A8", linewidth=1.4,
        )
        label = "Terminal t½" if endpoint == "Thalf" else endpoint
        axis.set_title(
            f"{label}\nOverall change vs shared: {row.grouped_relative_change_vs_shared_pct:+.2f}%",
            fontweight="bold", fontsize=9,
        )
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_xlabel("Outer scaffold fold")
        axis.set_ylabel("Two-group minus shared\n(primary metric)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Capacity-matched two-group encoder versus full sharing", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traincv", type=Path, default=ROOT / "results/benchmarks/multitask_two_group_traincv_v1")
    parser.add_argument("--comparators", type=Path, default=ROOT / "results/benchmarks/multitask_shared_private_traincv_v2")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/multitask_two_group_protocol_v2")
    parser.add_argument("--p1", type=Path, default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/multitask_two_group_traincv_analysis_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.traincv / "complete.json", args.traincv / "ensemble_oof_predictions.csv",
        args.traincv / "ensemble_fold_metrics.csv",
        args.comparators / "complete.json", args.comparators / "ensemble_oof_predictions.csv",
        args.protocol / "complete.json", args.protocol / "protocol.json",
        args.p1 / "complete.json", args.p1 / "stageA_matched_control_predictions.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_formal(args.traincv, "multitask_two_group_traincv")
    comparator_meta = verify_formal(args.comparators, "multitask_shared_private_traincv")
    protocol_meta = verify_stage(args.protocol, "multitask_two_group_protocol")
    p1_meta = verify_stage(args.p1, "stl_p1_nested_confirmation")
    if any(meta.get("test_labels_read", False) for meta in [protocol_meta, p1_meta]):
        raise ValueError("Protocol or Stage-A comparator is not test-closed")
    if args.bootstrap_repeats < 200:
        raise ValueError("At least 200 paired bootstrap repeats are required")
    protocol = json.loads((args.protocol / "protocol.json").read_text(encoding="utf-8"))
    if args.check_only:
        print(
            f"Two-group analysis ready: endpoints={len(protocol['human_evaluation_tasks'])} "
            f"bootstrap={args.bootstrap_repeats} locked_comparators=3"
        )
        return

    grouped_raw = pd.read_csv(args.traincv / "ensemble_oof_predictions.csv")
    comparator_raw = pd.read_csv(args.comparators / "ensemble_oof_predictions.csv")
    key = ["outer_fold", "task_id", "parent_id"]
    if set(grouped_raw.route_id) != {ROUTE_ID} or grouped_raw.duplicated(["route_id", *key]).any():
        raise ValueError("Two-group prediction route/key coverage is invalid")
    if set(comparator_raw.route_id) != {PRIVATE, SHARED} or comparator_raw.duplicated(["route_id", *key]).any():
        raise ValueError("Locked comparator route/key coverage is invalid")
    columns = [*key, "endpoint", "observed_primary_target", "ensemble_predicted_primary_target"]
    grouped = grouped_raw[columns].rename(columns={
        "observed_primary_target": "grouped_observed",
        "ensemble_predicted_primary_target": "grouped_prediction",
    })
    private = comparator_raw.loc[comparator_raw.route_id.eq(PRIVATE), columns].rename(columns={
        "observed_primary_target": "private_observed",
        "ensemble_predicted_primary_target": "private_prediction",
    })
    shared = comparator_raw.loc[comparator_raw.route_id.eq(SHARED), [*key, "observed_primary_target", "ensemble_predicted_primary_target"]].rename(columns={
        "observed_primary_target": "shared_observed",
        "ensemble_predicted_primary_target": "shared_prediction",
    })
    joined = grouped.merge(private, on=[*key, "endpoint"], validate="one_to_one").merge(
        shared, on=key, validate="one_to_one",
    )
    if len(joined) != len(grouped) or len(joined) != len(private) or len(joined) != len(shared):
        raise ValueError("Two-group and locked comparator evaluation parents differ")
    if max(
        float((joined.grouped_observed - joined.private_observed).abs().max()),
        float((joined.grouped_observed - joined.shared_observed).abs().max()),
    ) > 1e-12:
        raise ValueError("Observed targets differ between formal routes")
    joined = joined.rename(columns={"grouped_observed": "observed_primary_target"}).drop(
        columns=["private_observed", "shared_observed"],
    )

    stagea_raw = parent_predictions(pd.read_csv(args.p1 / "stageA_matched_control_predictions.csv"))
    stagea_raw = stagea_raw.loc[stagea_raw.task_id.isin(protocol["human_evaluation_tasks"])].copy()
    stagea_raw["stageA_prediction"] = np.where(
        stagea_raw.endpoint.eq("fu"), stagea_raw.predicted_physical, stagea_raw.predicted_transformed,
    )
    stagea_raw["stageA_observed"] = np.where(
        stagea_raw.endpoint.eq("fu"), stagea_raw.observed_physical, stagea_raw.observed_transformed,
    )
    stagea = stagea_raw.rename(columns={"molecule_id": "parent_id"})[
        ["outer_fold", "task_id", "parent_id", "stageA_prediction", "stageA_observed"]
    ]
    joined = joined.merge(stagea, on=key, validate="one_to_one")
    if (joined.observed_primary_target - joined.stageA_observed).abs().max() > 1e-10:
        raise ValueError("Formal and corrected Stage-A observed targets differ")

    metric_map = pd.read_csv(args.traincv / "ensemble_fold_metrics.csv").set_index("task_id").primary_metric.to_dict()
    overall_rows: list[dict] = []
    fold_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for index, (task, part) in enumerate(joined.groupby("task_id", sort=True)):
        endpoint = str(part.endpoint.iloc[0])
        metric = str(metric_map[task])
        scores = {
            "grouped": score(part, "grouped_prediction", metric),
            "shared": score(part, "shared_prediction", metric),
            "private": score(part, "private_prediction", metric),
            "stageA": score(part, "stageA_prediction", metric),
        }
        fold_noninferior = {"shared": 0, "private": 0, "stageA": 0}
        for fold, fold_part in part.groupby("outer_fold", sort=True):
            fold_scores = {
                "grouped": score(fold_part, "grouped_prediction", metric),
                "shared": score(fold_part, "shared_prediction", metric),
                "private": score(fold_part, "private_prediction", metric),
                "stageA": score(fold_part, "stageA_prediction", metric),
            }
            for comparator in fold_noninferior:
                fold_noninferior[comparator] += int(fold_scores["grouped"] <= fold_scores[comparator])
            fold_rows.append({
                "task_id": task, "endpoint": endpoint, "outer_fold": fold,
                "parents": len(fold_part), "primary_metric": metric,
                "grouped_score": fold_scores["grouped"], "shared_score": fold_scores["shared"],
                "private_score": fold_scores["private"], "stageA_reference_score": fold_scores["stageA"],
                "grouped_minus_shared": fold_scores["grouped"] - fold_scores["shared"],
                "grouped_minus_private": fold_scores["grouped"] - fold_scores["private"],
                "grouped_minus_stageA": fold_scores["grouped"] - fold_scores["stageA"],
            })
        overall_rows.append({
            "task_id": task, "endpoint": endpoint, "parents": len(part), "primary_metric": metric,
            "grouped_score": scores["grouped"], "shared_score": scores["shared"],
            "private_score": scores["private"], "stageA_reference_score": scores["stageA"],
            "grouped_relative_change_vs_shared_pct": 100 * (scores["grouped"] / scores["shared"] - 1),
            "grouped_relative_change_vs_private_pct": 100 * (scores["grouped"] / scores["private"] - 1),
            "grouped_relative_change_vs_stageA_pct": 100 * (scores["grouped"] / scores["stageA"] - 1),
            "noninferior_folds_vs_shared": fold_noninferior["shared"],
            "noninferior_folds_vs_private": fold_noninferior["private"],
            "noninferior_folds_vs_stageA": fold_noninferior["stageA"],
        })
        for offset, (name, prediction) in enumerate([
            ("full_shared", "shared_prediction"),
            ("fully_private", "private_prediction"),
            ("corrected_stageA", "stageA_prediction"),
        ]):
            mean, low, high = paired_bootstrap(
                part, "grouped_prediction", prediction, metric,
                20260917 + index * 10 + offset, args.bootstrap_repeats,
            )
            bootstrap_rows.append({
                "task_id": task, "endpoint": endpoint, "primary_metric": metric,
                "comparator": name, "difference_definition": "two_group_metric_minus_comparator_metric",
                "bootstrap_difference_mean": mean, "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high, "bootstrap_repeats": args.bootstrap_repeats,
            })

    overall = pd.DataFrame(overall_rows)
    folds = pd.DataFrame(fold_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    fu = overall.loc[overall.endpoint.eq("fu")].iloc[0]
    fu_boot = bootstrap.loc[
        bootstrap.endpoint.eq("fu") & bootstrap.comparator.eq("full_shared")
    ].iloc[0]
    mechanism = protocol["mechanistic_gate"]
    mechanism_checks = [
        {
            "criterion": "fu_improvement_vs_full_shared_pct",
            "observed": -float(fu.grouped_relative_change_vs_shared_pct),
            "required": float(mechanism["fu_minimum_improvement_vs_full_shared_pct"]),
            "operator": ">=", "passed": -float(fu.grouped_relative_change_vs_shared_pct) >= float(mechanism["fu_minimum_improvement_vs_full_shared_pct"]),
        },
        {
            "criterion": "fu_noninferior_outer_folds_vs_full_shared",
            "observed": int(fu.noninferior_folds_vs_shared),
            "required": int(mechanism["fu_minimum_noninferior_outer_folds"]),
            "operator": ">=", "passed": int(fu.noninferior_folds_vs_shared) >= int(mechanism["fu_minimum_noninferior_outer_folds"]),
        },
        {
            "criterion": "fu_grouped_minus_shared_bootstrap_95ci_high",
            "observed": float(fu_boot.bootstrap_95ci_high), "required": 0.0,
            "operator": "<", "passed": float(fu_boot.bootstrap_95ci_high) < 0,
        },
        {
            "criterion": "fu_harm_vs_fully_private_pct",
            "observed": float(fu.grouped_relative_change_vs_private_pct),
            "required": float(mechanism["fu_maximum_harm_vs_fully_private_pct"]),
            "operator": "<=", "passed": float(fu.grouped_relative_change_vs_private_pct) <= float(mechanism["fu_maximum_harm_vs_fully_private_pct"]),
        },
        {
            "criterion": "maximum_other_endpoint_harm_vs_full_shared_pct",
            "observed": float(overall.loc[overall.endpoint.ne("fu"), "grouped_relative_change_vs_shared_pct"].max()),
            "required": float(mechanism["maximum_other_endpoint_harm_vs_full_shared_pct"]),
            "operator": "<=", "passed": float(overall.loc[overall.endpoint.ne("fu"), "grouped_relative_change_vs_shared_pct"].max()) <= float(mechanism["maximum_other_endpoint_harm_vs_full_shared_pct"]),
        },
    ]
    mechanism_frame = pd.DataFrame(mechanism_checks)
    mechanism_passed = bool(mechanism_frame.passed.all())

    candidate_gate = protocol["model_candidacy_gate"]
    candidacy_rows = []
    for row in overall.itertuples(index=False):
        boot = bootstrap.loc[
            bootstrap.task_id.eq(row.task_id) & bootstrap.comparator.eq("corrected_stageA")
        ].iloc[0]
        improvement = -float(row.grouped_relative_change_vs_stageA_pct)
        statistical_gate = bool(
            improvement >= float(candidate_gate["minimum_improvement_vs_corrected_stageA_pct"])
            and int(row.noninferior_folds_vs_stageA) >= int(candidate_gate["minimum_noninferior_outer_folds"])
            and float(boot.bootstrap_95ci_high) < 0
        )
        candidacy_rows.append({
            "task_id": row.task_id, "endpoint": row.endpoint,
            "improvement_vs_corrected_stageA_pct": improvement,
            "minimum_required_improvement_pct": float(candidate_gate["minimum_improvement_vs_corrected_stageA_pct"]),
            "noninferior_outer_folds_vs_stageA": int(row.noninferior_folds_vs_stageA),
            "minimum_required_noninferior_folds": int(candidate_gate["minimum_noninferior_outer_folds"]),
            "bootstrap_95ci_high_grouped_minus_stageA": float(boot.bootstrap_95ci_high),
            "statistical_candidacy_gate_passed": statistical_gate,
            "source_sensitivity_required": bool(candidate_gate["source_sensitivity_required_after_gate"]),
            "final_model_candidacy_confirmed": False,
            "status": "source_sensitivity_required" if statistical_gate else "statistical_gate_not_passed",
        })
    candidacy = pd.DataFrame(candidacy_rows)
    pending = candidacy.loc[candidacy.statistical_candidacy_gate_passed, "endpoint"].astype(str).tolist()
    decision = {
        "mechanistic_gate_passed": mechanism_passed,
        "mechanistic_checks_passed": int(mechanism_frame.passed.sum()),
        "mechanistic_checks_total": len(mechanism_frame),
        "statistical_model_candidacy_endpoints": pending,
        "final_model_candidacy_endpoints": [],
        "source_sensitivity_required_for_statistical_candidates": bool(pending),
        "fixed_validation_opened": False,
        "test_opened": False,
        "next_action": (
            "run_source_cluster_sensitivity_for_statistical_candidates"
            if pending else "retain_two_group_as_mechanistic_ablation_and_do_not_open_fixed_validation"
        ),
    }
    with stage_output(args.output) as out:
        overall.to_csv(out / "endpoint_overall_metrics.csv", index=False)
        folds.to_csv(out / "endpoint_outer_fold_metrics.csv", index=False)
        bootstrap.to_csv(out / "paired_bootstrap_comparisons.csv", index=False)
        mechanism_frame.to_csv(out / "mechanistic_gate.csv", index=False)
        candidacy.to_csv(out / "model_candidacy_gate.csv", index=False)
        joined.to_csv(out / "matched_parent_predictions.csv", index=False)
        (out / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        figure = build_figure(folds, overall)
        figure.savefig(out / "Figure_two_group_vs_full_shared_trainCV.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# Formal two-group multitask analysis\n\n"
            "The single preregistered, capacity-matched two-group route is compared on identical outer-fold parents "
            "with locked full-shared, fully-private and corrected Stage-A predictions. Bootstrap resampling is paired "
            "by parent and stratified by outer fold. Statistical Stage-A gates remain provisional until the required "
            "source-cluster sensitivity analysis passes. Fixed validation and test remain closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "multitask_two_group_traincv_analysis",
            inputs={
                "traincv_complete_sha256": sha256(args.traincv / "complete.json"),
                "comparator_complete_sha256": sha256(args.comparators / "complete.json"),
                "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                "p1_complete_sha256": sha256(args.p1 / "complete.json"),
            },
            endpoints=6, bootstrap_repeats=args.bootstrap_repeats,
            mechanistic_gate_passed=mechanism_passed,
            statistical_model_candidacy_endpoints=pending,
            final_model_candidacy_endpoints=[], source_sensitivity_pending=bool(pending),
            validation_target_file_opened=False, test_labels_read=False,
            model_fitted=False, partial=False,
        )
    print(f"Two-group multitask train-CV analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
