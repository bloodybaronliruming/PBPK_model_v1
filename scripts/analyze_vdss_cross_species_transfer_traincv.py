#!/usr/bin/env python3
"""Analyze frozen VDss cross-species train-CV against matched controls."""
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

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


HUMAN_TASK = "VDss__human__steady_state_iv"
HUMAN_ROUTE = "human_only_neural_control"
TRANSFER_ROUTES = [
    "cross_species_pretrain_human_finetune",
    "species_conditioned_joint_shared_encoder",
]


def verify_traincv(path: Path, allow_partial: bool) -> dict:
    meta = json.loads((path / "complete.json").read_text())
    if meta.get("stage") != "vdss_cross_species_transfer_traincv" or meta.get("schema_version") != 1:
        raise ValueError("Unexpected VDss train-CV stage")
    if meta.get("partial", False) and not allow_partial:
        raise ValueError("Partial train-CV is engineering-only; pass --allow-partial only for analysis smoke")
    for name, digest in meta.get("artifacts", {}).items():
        artifact = path / name
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or modified train-CV artifact: {artifact}")
    if meta.get("validation_target_file_opened") or meta.get("test_labels_read"):
        raise ValueError("Analysis requires validation/test-closed train-CV")
    if meta.get("evaluation_labels_used_during_fit"):
        raise ValueError("Outer-evaluation labels were used during fitting")
    return meta


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def stagea_parent_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame.task_id.eq(HUMAN_TASK)].copy()
    frame["observed_transformed_target"] = frame.target_value.astype(float)
    frame["stageA_predicted_transformed_target"] = (
        frame.predicted_interface_target.astype(float) * frame.train_std_population.astype(float)
        + frame.train_mean.astype(float)
    )
    parent = frame.groupby(["outer_fold", "molecule_id"], as_index=False).agg(
        observed_transformed_target=("observed_transformed_target", "mean"),
        stageA_predicted_transformed_target=("stageA_predicted_transformed_target", "mean"),
    ).rename(columns={"molecule_id": "parent_id"})
    if parent.duplicated(["outer_fold", "parent_id"]).any():
        raise ValueError("Stage-A parent predictions are not unique")
    return parent


def stratified_paired_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    comparator: str,
    seed: int,
    repeats: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = []
    folds = [group.reset_index(drop=True) for _, group in frame.groupby("outer_fold", sort=True)]
    for _ in range(repeats):
        sample = pd.concat([
            fold.iloc[rng.integers(0, len(fold), size=len(fold))]
            for fold in folds
        ], ignore_index=True)
        differences.append(rmse(sample.observed_transformed_target, sample[candidate])
                           - rmse(sample.observed_transformed_target, sample[comparator]))
    values = np.asarray(differences)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route, group in frame.groupby("route_id", sort=True):
        rows.append({"route_id": route, "subset": "overall", "parents": len(group),
                     "transformed_RMSE": rmse(group.observed_transformed_target,
                                               group.ensemble_predicted_transformed_target)})
        for fold, fold_frame in group.groupby("outer_fold", sort=True):
            rows.append({"route_id": route, "subset": f"outer_fold_{fold}", "parents": len(fold_frame),
                         "transformed_RMSE": rmse(fold_frame.observed_transformed_target,
                                                   fold_frame.ensemble_predicted_transformed_target)})
    return pd.DataFrame(rows)


def comparison_figure(fold_metrics: pd.DataFrame) -> plt.Figure:
    colors = {
        HUMAN_ROUTE: "#4C78A8",
        "cross_species_pretrain_human_finetune": "#F58518",
        "species_conditioned_joint_shared_encoder": "#54A24B",
        "corrected_stageA_STL": "#7A5195",
    }
    labels = {
        HUMAN_ROUTE: "Human-only neural",
        "cross_species_pretrain_human_finetune": "Animal pretrain → human fine-tune",
        "species_conditioned_joint_shared_encoder": "Species-conditioned joint",
        "corrected_stageA_STL": "Corrected Stage-A STL",
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for route, table in fold_metrics.groupby("route_id", sort=False):
        table = table.sort_values("outer_fold")
        axis.plot(table.outer_fold + 1, table.transformed_RMSE, marker="o", linewidth=1.5,
                  color=colors[route], label=labels[route])
    axis.set_xlabel("Outer scaffold fold")
    axis.set_ylabel("Transformed RMSE")
    axis.set_xticks(sorted(fold_metrics.outer_fold.unique()) + np.ones(fold_metrics.outer_fold.nunique()))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="best")
    axis.set_title("VDss controlled cross-species transfer in train-CV", fontweight="bold")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traincv", type=Path,
                        default=ROOT / "results/benchmarks/vdss_cross_species_transfer_traincv_v1")
    parser.add_argument("--p1", type=Path,
                        default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "data/public_development/vdss_cross_species_transfer_protocol_v4")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/vdss_cross_species_transfer_traincv_analysis_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.traincv / "complete.json", args.traincv / "ensemble_oof_predictions.csv",
                args.p1 / "complete.json", args.p1 / "stageA_matched_control_predictions.csv",
                args.protocol / "complete.json", args.protocol / "protocol.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_traincv(args.traincv, args.allow_partial)
    p1_meta = verify_stage(args.p1, "stl_p1_nested_confirmation")
    protocol_meta = verify_stage(args.protocol, "vdss_cross_species_transfer_protocol")
    if p1_meta.get("test_labels_read") or protocol_meta.get("test_labels_read"):
        raise ValueError("Comparison inputs must remain test-closed")
    contract = json.loads((args.protocol / "protocol.json").read_text())
    if contract.get("schema_version") != 4:
        raise ValueError("Analysis requires decision-complete protocol v4")
    registered_repeats = int(contract["advance_gate"]["paired_parent_bootstrap_repeats"])
    if not train_meta.get("partial") and args.bootstrap_repeats != registered_repeats:
        raise ValueError("Formal analysis bootstrap repeats differ from protocol v4")
    if args.bootstrap_repeats < 100:
        raise ValueError("Bootstrap repeats are too small")
    if args.check_only:
        print(f"VDss transfer analysis ready: partial_input={train_meta.get('partial', False)}")
        return

    ensemble = pd.read_csv(args.traincv / "ensemble_oof_predictions.csv")
    ensemble["source_seen_in_human_train"] = ensemble.source_seen_in_human_train.astype(str).str.lower().isin(
        ["true", "1"]
    )
    if set(ensemble.route_id) != {HUMAN_ROUTE, *TRANSFER_ROUTES}:
        raise ValueError("Train-CV route set is incomplete")
    stagea = stagea_parent_predictions(args.p1 / "stageA_matched_control_predictions.csv")
    joined = ensemble.merge(stagea, on=["outer_fold", "parent_id"], how="left",
                            suffixes=("", "_stageA"), validate="many_to_one")
    if joined.stageA_predicted_transformed_target.isna().any():
        raise ValueError("Stage-A does not cover every VDss evaluation parent")
    observed_difference = np.max(np.abs(
        joined.observed_transformed_target - joined.observed_transformed_target_stageA
    ))
    if observed_difference > 1e-10:
        raise ValueError(f"Observed parent targets differ from Stage-A: {observed_difference:.3g}")
    joined = joined.drop(columns="observed_transformed_target_stageA")

    human = joined.loc[joined.route_id.eq(HUMAN_ROUTE), [
        "outer_fold", "parent_id", "ensemble_predicted_transformed_target"
    ]].rename(columns={"ensemble_predicted_transformed_target": "human_only_prediction"})
    joined = joined.merge(human, on=["outer_fold", "parent_id"], validate="many_to_one")
    joined["fold_similarity_quantile"] = joined.groupby(["route_id", "outer_fold"])[
        "max_human_train_ecfp4_tanimoto"
    ].rank(method="average", pct=True)
    joined["low_similarity_bottom_quintile"] = joined.fold_similarity_quantile.le(0.20)

    overall = metric_rows(joined)
    fold_rows = []
    for route, table in joined.groupby("route_id", sort=True):
        for fold, group in table.groupby("outer_fold", sort=True):
            fold_rows.append({"route_id": route, "outer_fold": fold, "parents": len(group),
                              "transformed_RMSE": rmse(group.observed_transformed_target,
                                                        group.ensemble_predicted_transformed_target)})
    stagea_unique = joined.drop_duplicates(["outer_fold", "parent_id"])
    for fold, group in stagea_unique.groupby("outer_fold", sort=True):
        fold_rows.append({"route_id": "corrected_stageA_STL", "outer_fold": fold, "parents": len(group),
                          "transformed_RMSE": rmse(group.observed_transformed_target,
                                                    group.stageA_predicted_transformed_target)})
    fold_metrics = pd.DataFrame(fold_rows)

    comparison_rows, bootstrap_rows, sensitivity_rows, decision_rows = [], [], [], []
    minimum_improvement = float(contract["advance_gate"]["minimum_mean_improvement_vs_human_only_pct"])
    minimum_folds = int(contract["advance_gate"]["minimum_noninferior_outer_folds"])
    maximum_harm = float(contract["advance_gate"]["low_similarity_maximum_relative_harm_pct"])
    source_minimum = int(contract["advance_gate"]["source_unseen_minimum_parents_for_gate"])
    for route_index, route in enumerate(TRANSFER_ROUTES):
        route_table = joined.loc[joined.route_id.eq(route)].copy()
        for comparator_name, comparator_column in [
            (HUMAN_ROUTE, "human_only_prediction"),
            ("corrected_stageA_STL", "stageA_predicted_transformed_target"),
        ]:
            candidate_column = "ensemble_predicted_transformed_target"
            candidate_score = rmse(route_table.observed_transformed_target, route_table[candidate_column])
            comparator_score = rmse(route_table.observed_transformed_target, route_table[comparator_column])
            fold_noninferior = 0
            for _, fold in route_table.groupby("outer_fold", sort=True):
                if rmse(fold.observed_transformed_target, fold[candidate_column]) <= \
                        rmse(fold.observed_transformed_target, fold[comparator_column]):
                    fold_noninferior += 1
            mean_difference, ci_low, ci_high = stratified_paired_bootstrap(
                route_table, candidate_column, comparator_column,
                20260917 + 100 * route_index + (0 if comparator_name == HUMAN_ROUTE else 1),
                args.bootstrap_repeats,
            )
            improvement = 100 * (1 - candidate_score / comparator_score)
            comparison_rows.append({
                "route_id": route, "comparator": comparator_name, "parents": len(route_table),
                "candidate_transformed_RMSE": candidate_score,
                "comparator_transformed_RMSE": comparator_score,
                "improvement_pct": improvement, "noninferior_outer_folds": fold_noninferior,
                "outer_folds": route_table.outer_fold.nunique(),
                "paired_bootstrap_difference_mean": mean_difference,
                "paired_bootstrap_difference_95ci_low": ci_low,
                "paired_bootstrap_difference_95ci_high": ci_high,
            })
            bootstrap_rows.append({"route_id": route, "comparator": comparator_name,
                                   "bootstrap_repeats": args.bootstrap_repeats,
                                   "difference_definition": "candidate_RMSE_minus_comparator_RMSE",
                                   "difference_mean": mean_difference, "ci_low": ci_low, "ci_high": ci_high})
            subset_results = {}
            for subset, mask in [
                ("low_similarity_bottom_quintile", route_table.low_similarity_bottom_quintile),
                ("source_unseen", ~route_table.source_seen_in_human_train),
                ("source_seen", route_table.source_seen_in_human_train),
            ]:
                part = route_table.loc[mask]
                if part.empty:
                    candidate_subset = comparator_subset = relative_harm = np.nan
                else:
                    candidate_subset = rmse(part.observed_transformed_target, part[candidate_column])
                    comparator_subset = rmse(part.observed_transformed_target, part[comparator_column])
                    relative_harm = 100 * (candidate_subset / comparator_subset - 1)
                sensitivity_rows.append({
                    "route_id": route, "comparator": comparator_name, "subset": subset,
                    "parents": len(part), "candidate_transformed_RMSE": candidate_subset,
                    "comparator_transformed_RMSE": comparator_subset, "relative_harm_pct": relative_harm,
                })
                subset_results[subset] = (len(part), relative_harm)
            low_similarity_ok = bool(
                np.isfinite(subset_results["low_similarity_bottom_quintile"][1])
                and subset_results["low_similarity_bottom_quintile"][1] <= maximum_harm
            )
            source_n, source_harm = subset_results["source_unseen"]
            source_gate_estimable = source_n >= source_minimum
            source_ok = bool(source_gate_estimable and np.isfinite(source_harm) and source_harm <= maximum_harm)
            core_gate = bool(improvement >= minimum_improvement and fold_noninferior >= minimum_folds
                             and ci_high < 0 and low_similarity_ok)
            decision_rows.append({
                "route_id": route, "comparator": comparator_name,
                "improvement_gate": improvement >= minimum_improvement,
                "fold_gate": fold_noninferior >= minimum_folds,
                "bootstrap_gate": ci_high < 0,
                "low_similarity_gate": low_similarity_ok,
                "source_unseen_parents": source_n,
                "source_gate_estimable": source_gate_estimable,
                "source_gate": source_ok if source_gate_estimable else "not_estimable",
                "core_traincv_gate": core_gate,
                "fixed_validation_ready": bool(core_gate and source_ok),
                "next_action": (
                    "advance_to_source_cluster_sensitivity" if core_gate and not source_gate_estimable
                    else "eligible_for_fixed_validation_review" if core_gate and source_ok
                    else "do_not_advance"
                ),
            })

    comparison = pd.DataFrame(comparison_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    decisions = pd.DataFrame(decision_rows)
    partial = bool(train_meta.get("partial", False))
    if partial:
        decisions["next_action"] = "engineering_only_no_decision"
        decisions["fixed_validation_ready"] = False
    with stage_output(args.output) as out:
        overall.to_csv(out / "route_metrics.csv", index=False)
        fold_metrics.to_csv(out / "outer_fold_metrics.csv", index=False)
        comparison.to_csv(out / "matched_comparison.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(out / "paired_bootstrap_summary.csv", index=False)
        sensitivity.to_csv(out / "sensitivity_comparison.csv", index=False)
        decisions.to_csv(out / "decision_registry.csv", index=False)
        figure = comparison_figure(fold_metrics)
        figure.savefig(out / "Figure_VDss_transfer_trainCV.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# VDss controlled-transfer train-CV analysis\n\n"
            "Primary predictions are three-seed arithmetic ensembles in transformed target space. Comparisons are "
            "paired on identical human outer-fold parents. Bootstrap resampling is paired and stratified by outer fold. "
            "The bottom fold-specific ECFP4-similarity quintile is a mandatory sensitivity. Ordinary scaffold-CV has "
            "too few source-unseen parents for a reliable gate; a core-positive route must pass a separate source-cluster "
            "sensitivity before fixed-validation review.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "vdss_cross_species_transfer_traincv_analysis",
            inputs={
                "traincv_complete_sha256": sha256(args.traincv / "complete.json"),
                "p1_complete_sha256": sha256(args.p1 / "complete.json"),
                "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
            },
            routes=3, transfer_routes=2, bootstrap_repeats=args.bootstrap_repeats,
            train_cv_route_decision_authorized=not partial,
            fixed_validation_selection_authorized=False,
            validation_target_file_opened=False, test_labels_read=False,
            model_fitted=False, partial=partial,
        )
    print(f"VDss transfer train-CV analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
