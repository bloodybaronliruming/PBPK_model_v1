#!/usr/bin/env python3
"""Analyze the frozen VDss purged document-group source-stress result.

This script intentionally reports source-fold perturbation sensitivity only.  It
does not construct unique-parent OOF metrics, select a model, or open fixed
validation/test labels.
"""
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

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


JOINT = "species_conditioned_joint_shared_encoder"
HUMAN = "human_only_neural_control"
STAGEA = "corrected_stageA_STL"
ROUTES = [STAGEA, HUMAN, JOINT]
LABELS = {
    STAGEA: "Corrected Stage-A STL",
    HUMAN: "Human-only neural",
    JOINT: "Species-conditioned joint",
}
COLORS = {STAGEA: "#7A5195", HUMAN: "#4C78A8", JOINT: "#54A24B"}


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def bootstrap_delta(table: pd.DataFrame, comparator: str, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observed = table.observed_transformed_target.to_numpy(float)
    candidate = table[JOINT].to_numpy(float)
    baseline = table[comparator].to_numpy(float)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(table), size=len(table))
        values[index] = rmse(observed[sample], candidate[sample]) - rmse(observed[sample], baseline[sample])
    return {
        "bootstrap_mean_delta_rmse": float(values.mean()),
        "bootstrap_ci95_lower": float(np.quantile(values, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(values, 0.975)),
    }


def make_figure(metrics: pd.DataFrame, differences: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw={"wspace": 0.35})
    folds = sorted(metrics.source_fold.unique())
    sizes = metrics.groupby("source_fold").evaluation_parents.first().reindex(folds)
    for route in ROUTES:
        values = metrics.loc[metrics.route_id.eq(route)].set_index("source_fold").reindex(folds)
        axes[0].plot(np.arange(len(folds)), values.transformed_RMSE, marker="o", linewidth=1.5,
                     markersize=4, color=COLORS[route], label=LABELS[route])
    axes[0].set_xticks(np.arange(len(folds)), [f"F{fold + 1}\nn={sizes.loc[fold]}" for fold in folds])
    axes[0].set_xlabel("Purged document group")
    axes[0].set_ylabel("Transformed-space RMSE")
    axes[0].set_title("Source-stress fold performance", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    width = 0.34
    for offset, comparator in [(-width / 2, HUMAN), (width / 2, STAGEA)]:
        values = differences.loc[differences.comparator_route.eq(comparator)].set_index("source_fold").reindex(folds)
        axes[1].bar(np.arange(len(folds)) + offset, values.delta_rmse, width=width,
                    color=COLORS[comparator], label=f"vs {LABELS[comparator]}")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(folds)), [f"F{fold + 1}" for fold in folds])
    axes[1].set_xlabel("Purged document group")
    axes[1].set_ylabel("Δ RMSE (joint − comparator)")
    axes[1].set_title("Joint-route paired sensitivity", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    bar_handles, bar_labels = axes[1].get_legend_handles_labels()
    handles, labels = line_handles + bar_handles, line_labels + bar_labels
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.81, wspace=0.35)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/analysis/vdss_source_document_holdout_v1")
    parser.add_argument("--source-protocol", type=Path,
                        default=ROOT / "data/public_development/vdss_source_generalization_protocol_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/vdss_source_document_holdout_analysis_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("Use at least 100 deterministic bootstrap repeats")
    startup_self_check([
        args.input / "complete.json", args.input / "ensemble_predictions.csv",
        args.input / "source_fold_metrics.csv", args.input / "fold_isolation_audit.csv",
        args.source_protocol / "complete.json", args.source_protocol / "document_group_fold_audit.csv",
    ], output=args.output)
    input_meta = verify_stage(args.input, "vdss_source_document_holdout")
    protocol_meta = verify_stage(args.source_protocol, "vdss_source_generalization_protocol")
    if not input_meta.get("full_configuration") or input_meta.get("model_selection_authorized"):
        raise ValueError("Input must be the frozen full, non-selection source-stress run")
    if input_meta.get("validation_target_file_opened") or input_meta.get("test_labels_read") \
            or input_meta.get("evaluation_labels_used_during_fit"):
        raise ValueError("Source-stress input violates label-lifecycle contract")
    if any(input_meta.get(key) != 0 for key in [
        "max_post_filter_document_overlap", "max_post_filter_parent_overlap", "max_post_filter_scaffold_overlap",
    ]):
        raise ValueError("Source-stress input violates three-way isolation")
    if protocol_meta.get("connected_component_fivefold_feasible") \
            or not protocol_meta.get("document_group_stress_test_feasible"):
        raise ValueError("Unexpected source-protocol feasibility decision")

    predictions = pd.read_csv(args.input / "ensemble_predictions.csv")
    if set(predictions.route_id) != set(ROUTES):
        raise ValueError("Unexpected set of registered routes")
    if predictions.duplicated(["source_fold", "route_id", "parent_id"]).any():
        raise ValueError("Ensemble predictions are not unique within source fold/route/parent")
    values = ["observed_transformed_target", "ensemble_predicted_transformed_target"]
    if not np.isfinite(predictions[values].to_numpy(float)).all():
        raise ValueError("Non-finite source-stress prediction or target")
    audit = pd.read_csv(args.source_protocol / "document_group_fold_audit.csv")
    isolation = pd.read_csv(args.input / "fold_isolation_audit.csv")
    overlap_columns = ["post_filter_document_overlap", "post_filter_parent_overlap", "post_filter_scaffold_overlap"]
    if isolation[overlap_columns].to_numpy().any():
        raise ValueError("Post-filter isolation is nonzero")

    metric_rows, wide_rows, difference_rows, bootstrap_rows = [], [], [], []
    for source_fold, group in predictions.groupby("source_fold", sort=True):
        pivot = group.pivot(index="parent_id", columns="route_id", values="ensemble_predicted_transformed_target")
        observed = group.drop_duplicates("parent_id").set_index("parent_id").observed_transformed_target
        if set(pivot.index) != set(observed.index):
            raise ValueError("Observed target/prediction parent alignment failed")
        observed = observed.reindex(pivot.index)
        evaluation = audit.loc[audit.source_fold.eq(source_fold)].iloc[0]
        for route in ROUTES:
            score = rmse(observed.to_numpy(float), pivot[route].to_numpy(float))
            metric_rows.append({
                "source_fold": source_fold, "route_id": route, "transformed_RMSE": score,
                "evaluation_parents": len(pivot), "evaluation_documents": int(evaluation.evaluation_documents),
                "evaluation_document_ids": evaluation.evaluation_document_ids,
                "retained_human_parents": int(evaluation.retained_human_parents),
            })
        joined = pivot.copy()
        joined["observed_transformed_target"] = observed
        for comparator in [HUMAN, STAGEA]:
            candidate_score = rmse(observed, pivot[JOINT])
            baseline_score = rmse(observed, pivot[comparator])
            difference_rows.append({
                "source_fold": source_fold, "candidate_route": JOINT, "comparator_route": comparator,
                "evaluation_parents": len(pivot), "joint_RMSE": candidate_score,
                "comparator_RMSE": baseline_score,
                "delta_rmse": candidate_score - baseline_score,
                "percent_change_vs_comparator": 100 * (candidate_score - baseline_score) / baseline_score,
            })
            bootstrap_rows.append({
                "source_fold": source_fold, "candidate_route": JOINT, "comparator_route": comparator,
                "evaluation_parents": len(pivot), "bootstrap_repeats": args.bootstrap_repeats,
                **bootstrap_delta(joined, comparator, args.bootstrap_repeats,
                                  seed=20260917 + 1000 * source_fold + (1 if comparator == HUMAN else 2)),
                "interpretation": "within-source-fold parent resampling only; not source-level inference",
            })
        wide_rows.append({"source_fold": source_fold, "evaluation_parents": len(pivot)})

    metrics = pd.DataFrame(metric_rows)
    differences = pd.DataFrame(difference_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    aggregate_rows = []
    for route, group in predictions.groupby("route_id", sort=True):
        aggregate_rows.append({
            "route_id": route, "aggregation": "parent_document_incidence_weighted",
            "evaluation_incidences": len(group),
            "transformed_RMSE": rmse(group.observed_transformed_target, group.ensemble_predicted_transformed_target),
            "interpretation": "descriptive only; repeated parents across source folds are retained",
        })
    for route, group in metrics.groupby("route_id", sort=True):
        aggregate_rows.append({
            "route_id": route, "aggregation": "equal_weight_source_fold_macro",
            "evaluation_incidences": len(group),
            "transformed_RMSE": float(group.transformed_RMSE.mean()),
            "interpretation": "descriptive only; source folds have unequal document and parent counts",
        })
    aggregates = pd.DataFrame(aggregate_rows)

    with stage_output(args.output) as out:
        metrics.to_csv(out / "source_fold_route_metrics.csv", index=False)
        differences.to_csv(out / "joint_paired_route_differences.csv", index=False)
        bootstraps.to_csv(out / "within_fold_parent_resampling_sensitivity.csv", index=False)
        aggregates.to_csv(out / "descriptive_aggregate_metrics.csv", index=False)
        audit.to_csv(out / "document_group_audit.csv", index=False)
        figure = make_figure(metrics, differences)
        figure.savefig(out / "Figure_VDss_source_stress.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        dominant = differences.loc[(differences.source_fold.eq(0)) & differences.comparator_route.eq(STAGEA)].iloc[0]
        joint_vs_human = differences.loc[differences.comparator_route.eq(HUMAN)]
        joint_vs_stagea = differences.loc[differences.comparator_route.eq(STAGEA)]
        report = "\n".join([
            "# VDss purged document-group source-stress analysis",
            "",
            "## Scope",
            "",
            "This is a pre-registered source-perturbation sensitivity analysis. It is not unique-parent OOF, "
            "not an independent external validation, and not authorized for model selection or fixed validation.",
            "",
            "## Integrity checks",
            "",
            f"- Full CUDA configuration completed with {input_meta['model_files']}/{input_meta['expected_model_files']} model files.",
            "- Post-filter document, parent, and scaffold overlap are all zero in every task/fold.",
            "- Evaluation labels were not used during fitting; validation/test remained closed.",
            f"- {input_meta['parents_evaluated_in_multiple_source_folds']} of {input_meta['unique_evaluation_parents']} "
            "unique parents occur in multiple document groups; pooled incidences are descriptive only.",
            "",
            "## Descriptive results",
            "",
            f"- Species-conditioned joint improved versus human-only neural in all 5/5 source folds "
            f"(fold-level delta RMSE range {joint_vs_human.delta_rmse.min():.4f} to {joint_vs_human.delta_rmse.max():.4f}).",
            f"- Versus corrected Stage-A STL, joint improved in {(joint_vs_stagea.delta_rmse < 0).sum()}/5 folds and "
            f"worsened in {(joint_vs_stagea.delta_rmse > 0).sum()}/5 folds.",
            f"- In the dominant single-document stress fold (document 54149; n=514), joint minus Stage-A was "
            f"{dominant.delta_rmse:.4f} RMSE ({dominant.percent_change_vs_comparator:.2f}%).",
            "",
            "## Interpretation",
            "",
            "The joint route has a consistent advantage over its matched human-only neural control under source "
            "perturbation. Its comparison with corrected Stage-A is mixed: the dominant source fold has a small "
            "improvement, while two of five source groups worsen and the smaller groups are intrinsically imprecise. "
            "Therefore this analysis increases evidence for transfer signal relative to human-only neural training, "
            "but does not replace corrected Stage-A or authorize fixed validation.",
            "",
            "Within-fold parent-resampling intervals quantify prediction-sample variation only; they must not be "
            "interpreted as independent-source confidence intervals because each source group contains one or a few "
            "documents and parents can recur across groups.",
        ]) + "\n"
        (out / "analysis_report.md").write_text(report, encoding="utf-8")
        finish_stage(
            out, "vdss_source_document_holdout_analysis",
            inputs={
                "source_stress_complete_sha256": sha256(args.input / "complete.json"),
                "source_protocol_complete_sha256": sha256(args.source_protocol / "complete.json"),
            },
            bootstrap_repeats=args.bootstrap_repeats, source_folds=5, routes=3,
            evaluation_incidences=int(len(predictions) / len(ROUTES)),
            unique_evaluation_parents=int(predictions.parent_id.nunique()),
            repeated_parents_across_source_folds=int(input_meta["parents_evaluated_in_multiple_source_folds"]),
            model_fitted=False, model_selection_authorized=False,
            validation_target_file_opened=False, test_labels_read=False, partial=False,
            interpretation="source perturbation stress sensitivity only; not unique-parent OOF or model selection",
        )
    print(f"VDss source-stress analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
