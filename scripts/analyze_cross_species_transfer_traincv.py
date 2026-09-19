#!/usr/bin/env python3
"""Analyze a formal endpoint-configured cross-species transfer train-CV run.

The analysis is deliberately train-CV-only.  It verifies immutable input
artifacts, aligns all routes with the frozen Stage-A parent predictions, and
reports paired, fold-stratified bootstrap comparisons.  It never opens a
fixed validation target or test label.
"""
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
from scipy.special import expit

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


HUMAN_ROUTE = "human_only_neural_control"
TRANSFER_ROUTES = [
    "cross_species_pretrain_human_finetune",
    "species_conditioned_joint_shared_encoder",
]
STAGEA_ROUTE = "corrected_stageA_STL"


def rmse(observed: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((observed_array - predicted_array) ** 2)))


def mae(observed: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(observed_array - predicted_array)))


def primary_score(frame: pd.DataFrame, prediction: str, metric_name: str) -> float:
    if metric_name == "physical_MAE":
        return mae(frame.observed_target, frame[prediction])
    if metric_name == "transformed_RMSE":
        return rmse(frame.observed_target, frame[prediction])
    raise ValueError(f"Unsupported primary metric: {metric_name}")


def verify_traincv(path: Path) -> dict:
    """Verify a full, closed train-CV artifact before deriving a decision."""
    meta = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or meta.get("stage") != "cross_species_transfer_traincv":
        raise ValueError("Unexpected cross-species train-CV stage")
    if meta.get("partial") or not meta.get("full_configuration"):
        raise ValueError("Only a full formal train-CV run may support a route decision")
    if meta.get("validation_target_file_opened") or meta.get("test_labels_read"):
        raise ValueError("Analysis requires validation- and test-closed train-CV")
    if meta.get("evaluation_labels_used_during_fit"):
        raise ValueError("Outer evaluation labels were used during fitting")
    if not meta.get("evaluation_labels_read_after_all_fold_models_fit"):
        raise ValueError("Evaluation-label lifecycle audit is incomplete")
    if not meta.get("train_cv_route_comparison_authorized"):
        raise ValueError("Train-CV route comparison was not authorized")
    if meta.get("fixed_validation_selection_authorized"):
        raise ValueError("This stage must not select a fixed-validation candidate")
    if meta.get("expected_model_files") != meta.get("model_files"):
        raise ValueError("Formal model-file count is incomplete")
    if not meta.get("predictions_finite") or float(meta.get("max_reload_abs_difference", np.inf)) != 0.0:
        raise ValueError("Prediction finiteness or model reload audit failed")
    if not all(bool(meta.get(key)) for key in [
        "encoder_updates_nonzero", "human_head_updates_nonzero", "training_phase_task_coverage_nonzero",
    ]):
        raise ValueError("Training coverage/update audit failed")
    for relative, digest in meta.get("artifacts", {}).items():
        artifact = path / relative
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or modified train-CV artifact: {artifact}")
    return meta


def stagea_parent_predictions(path: Path, human_task: str, metric_name: str) -> pd.DataFrame:
    """Put the frozen P1 Stage-A output on the registered primary target scale."""
    table = pd.read_csv(path)
    table = table.loc[table.task_id.eq(human_task)].copy()
    if table.empty:
        raise ValueError(f"Stage-A matched-control predictions lack {human_task}")
    table["observed_transformed"] = table.target_value.astype(float)
    table["stageA_predicted_transformed"] = (
        table.predicted_interface_target.astype(float) * table.train_std_population.astype(float)
        + table.train_mean.astype(float)
    )
    if metric_name == "physical_MAE":
        table["observed_target"] = expit(table.observed_transformed.to_numpy(float))
        table["stageA_predicted_target"] = expit(table.stageA_predicted_transformed.to_numpy(float))
    elif metric_name == "transformed_RMSE":
        table["observed_target"] = table.observed_transformed
        table["stageA_predicted_target"] = table.stageA_predicted_transformed
    else:
        raise ValueError(f"Unsupported Stage-A primary metric: {metric_name}")
    result = table.groupby(["outer_fold", "molecule_id"], as_index=False).agg(
        observed_target=("observed_target", "mean"),
        stageA_predicted_target=("stageA_predicted_target", "mean"),
        stageA_source_records=("row_id", "size"),
    ).rename(columns={"molecule_id": "parent_id"})
    if result.duplicated(["outer_fold", "parent_id"]).any():
        raise ValueError("Stage-A parent predictions are not unique after aggregation")
    return result


def paired_bootstrap(
    frame: pd.DataFrame, candidate: str, comparator: str, *, metric_name: str, seed: int, repeats: int
) -> tuple[float, float, float]:
    """Resample parents within each outer fold, retaining the paired routes."""
    rng = np.random.default_rng(seed)
    fold_tables = [part.reset_index(drop=True) for _, part in frame.groupby("outer_fold", sort=True)]
    differences = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = pd.concat(
            [part.iloc[rng.integers(0, len(part), size=len(part))] for part in fold_tables],
            ignore_index=True,
        )
        differences[index] = primary_score(sampled, candidate, metric_name) - primary_score(
            sampled, comparator, metric_name
        )
    return float(differences.mean()), float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def route_metrics(frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    rows = []
    for route, part in frame.groupby("route_id", sort=True):
        rows.append({"route_id": route, "subset": "overall", "parents": len(part),
                     metric_name: primary_score(part, "ensemble_predicted_target", metric_name)})
        for fold, fold_part in part.groupby("outer_fold", sort=True):
            rows.append({"route_id": route, "subset": f"outer_fold_{fold}", "parents": len(fold_part),
                         metric_name: primary_score(fold_part, "ensemble_predicted_target", metric_name)})
    return pd.DataFrame(rows)


def build_figure(
    fold_metrics: pd.DataFrame, overall_metrics: pd.DataFrame, endpoint: str, metric_name: str
) -> plt.Figure:
    colors = {
        HUMAN_ROUTE: "#4C78A8",
        "cross_species_pretrain_human_finetune": "#F58518",
        "species_conditioned_joint_shared_encoder": "#54A24B",
        STAGEA_ROUTE: "#7A5195",
    }
    labels = {
        HUMAN_ROUTE: "Human-only neural",
        "cross_species_pretrain_human_finetune": "Animal pretrain → human fine-tune",
        "species_conditioned_joint_shared_encoder": "Species-conditioned joint encoder",
        STAGEA_ROUTE: "Frozen Stage-A STL",
    }
    order = [STAGEA_ROUTE, HUMAN_ROUTE, *TRANSFER_ROUTES]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    figure, (overall_axis, fold_axis) = plt.subplots(1, 2, figsize=(8.4, 3.7), gridspec_kw={"width_ratios": [0.9, 1.6]})
    overall = overall_metrics.loc[overall_metrics.subset.eq("overall"), ["route_id", metric_name]]
    if set(overall.route_id) != set(order) or len(overall) != len(order):
        raise ValueError("Overall figure metrics do not uniquely cover every registered route")
    overall = overall.set_index("route_id").reindex(order).reset_index()
    overall_axis.barh(np.arange(len(order)), overall[metric_name], color=[colors[item] for item in order])
    overall_axis.set_yticks(np.arange(len(order)), [labels[item] for item in order])
    overall_axis.invert_yaxis()
    overall_axis.set_xlabel("Overall parent-level MAE" if metric_name == "physical_MAE" else "Overall parent-level RMSE")
    overall_axis.set_title("All human OOF parents", fontweight="bold", fontsize=9)
    for route in order:
        part = fold_metrics.loc[fold_metrics.route_id.eq(route)].sort_values("outer_fold")
        fold_axis.plot(part.outer_fold.to_numpy() + 1, part[metric_name], marker="o", markersize=4,
                       linewidth=1.5, color=colors[route], label=labels[route])
    fold_axis.set_xticks([1, 2, 3, 4, 5])
    fold_axis.set_xlabel("Outer scaffold fold")
    fold_axis.set_ylabel("Physical MAE" if metric_name == "physical_MAE" else "Transformed RMSE")
    fold_axis.set_title("Identical human outer-fold parents", fontweight="bold", fontsize=9)
    for axis in (overall_axis, fold_axis):
        axis.grid(axis="x" if axis is overall_axis else "y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fold_axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7)
    figure.suptitle(f"{endpoint} controlled cross-species transfer: formal train-CV", fontweight="bold", y=1.02)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traincv", type=Path, default=ROOT / "results/benchmarks/cl_cross_species_transfer_traincv_v1")
    parser.add_argument("--p1", type=Path, default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/cl_cross_species_transfer_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/cl_cross_species_transfer_traincv_analysis_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.traincv / "complete.json", args.traincv / "ensemble_oof_predictions.csv",
        args.p1 / "complete.json", args.p1 / "stageA_matched_control_predictions.csv",
        args.protocol / "complete.json", args.protocol / "protocol.json",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_traincv(args.traincv)
    p1_meta = verify_stage(args.p1, "stl_p1_nested_confirmation")
    protocol_meta = verify_stage(args.protocol, "cross_species_transfer_protocol")
    if p1_meta.get("test_labels_read") or protocol_meta.get("test_labels_read"):
        raise ValueError("Comparison inputs must remain test-closed")
    protocol = json.loads((args.protocol / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("endpoint") != train_meta.get("endpoint"):
        raise ValueError("Endpoint/protocol version mismatch")
    if args.bootstrap_repeats != int(protocol["advance_gate"]["paired_parent_bootstrap_repeats"]):
        raise ValueError("Formal bootstrap repeat count differs from the registered protocol")
    if args.bootstrap_repeats < 100:
        raise ValueError("Bootstrap repeats are too small")
    metric_name = str(protocol["human_primary_metric"])
    if metric_name == "physical_MAE":
        observed_input = "observed_physical_target"
        prediction_input = "ensemble_predicted_physical_target"
        target_space = "physical fraction space"
    elif metric_name == "transformed_RMSE":
        observed_input = "observed_transformed_target"
        prediction_input = "ensemble_predicted_transformed_target"
        target_space = "transformed target space"
    else:
        raise ValueError(f"Unsupported registered primary metric: {metric_name}")
    if args.check_only:
        print(f"{protocol['endpoint']} formal transfer analysis ready: bootstrap_repeats={args.bootstrap_repeats}")
        return

    ensemble = pd.read_csv(args.traincv / "ensemble_oof_predictions.csv")
    required_columns = {
        "route_id", "outer_fold", "parent_id", observed_input, prediction_input,
        "max_human_train_ecfp4_tanimoto", "source_seen_in_human_train",
    }
    if not required_columns.issubset(ensemble.columns):
        raise ValueError(f"Missing ensemble columns: {sorted(required_columns - set(ensemble.columns))}")
    if set(ensemble.route_id) != {HUMAN_ROUTE, *TRANSFER_ROUTES}:
        raise ValueError("Train-CV route set is incomplete")
    if ensemble.duplicated(["route_id", "outer_fold", "parent_id"]).any():
        raise ValueError("Ensemble predictions are not unique by route/fold/parent")
    ensemble = ensemble.rename(columns={observed_input: "observed_target", prediction_input: "ensemble_predicted_target"})
    if metric_name == "physical_MAE" and not (
        ensemble.observed_target.between(0.0, 1.0).all()
        and ensemble.ensemble_predicted_target.between(0.0, 1.0).all()
    ):
        raise ValueError("Physical fu observations or predictions are outside [0,1]")
    ensemble["source_seen_in_human_train"] = ensemble.source_seen_in_human_train.astype(str).str.lower().isin(["true", "1"])
    stagea = stagea_parent_predictions(
        args.p1 / "stageA_matched_control_predictions.csv", protocol["human_task"], metric_name
    )
    joined = ensemble.merge(
        stagea, on=["outer_fold", "parent_id"], how="left", validate="many_to_one", suffixes=("", "_stageA")
    )
    if joined.stageA_predicted_target.isna().any():
        raise ValueError("Stage-A does not cover every transfer evaluation parent")
    difference = (joined.observed_target - joined.observed_target_stageA).abs().max()
    if difference > 1e-10:
        raise ValueError(f"Observed targets differ from frozen Stage-A: {difference:.3g}")
    joined = joined.drop(columns="observed_target_stageA")
    human = joined.loc[joined.route_id.eq(HUMAN_ROUTE), ["outer_fold", "parent_id", "ensemble_predicted_target"]]
    human = human.rename(columns={"ensemble_predicted_target": "human_only_prediction"})
    joined = joined.merge(human, on=["outer_fold", "parent_id"], validate="many_to_one")
    joined["fold_similarity_quantile"] = joined.groupby(["route_id", "outer_fold"])["max_human_train_ecfp4_tanimoto"].rank(method="average", pct=True)
    joined["low_similarity_bottom_quintile"] = joined.fold_similarity_quantile.le(0.20)

    metrics = route_metrics(joined, metric_name)
    stagea_unique = joined.drop_duplicates(["outer_fold", "parent_id"])
    stagea_rows = [{"route_id": STAGEA_ROUTE, "subset": "overall", "parents": len(stagea_unique),
                    metric_name: primary_score(stagea_unique, "stageA_predicted_target", metric_name)}]
    for fold, part in stagea_unique.groupby("outer_fold", sort=True):
        stagea_rows.append({"route_id": STAGEA_ROUTE, "subset": f"outer_fold_{fold}", "parents": len(part),
                            metric_name: primary_score(part, "stageA_predicted_target", metric_name)})
    metrics = pd.concat([metrics, pd.DataFrame(stagea_rows)], ignore_index=True)
    fold_metrics = metrics.loc[metrics.subset.str.startswith("outer_fold_")].copy()
    fold_metrics["outer_fold"] = fold_metrics.subset.str.removeprefix("outer_fold_").astype(int)

    gate = protocol["advance_gate"]
    stagea_gate = protocol["stageA_replacement_gate"]
    comparison_rows, bootstrap_rows, sensitivity_rows, decision_rows = [], [], [], []
    for route_number, route in enumerate(TRANSFER_ROUTES):
        part = joined.loc[joined.route_id.eq(route)].copy()
        for comparator_name, comparator_column, comparator_gate, comparator_role in [
            (HUMAN_ROUTE, "human_only_prediction", gate, "transfer_attribution"),
            (STAGEA_ROUTE, "stageA_predicted_target", stagea_gate, "stageA_replacement"),
        ]:
            candidate_score = primary_score(part, "ensemble_predicted_target", metric_name)
            comparator_score = primary_score(part, comparator_column, metric_name)
            noninferior_folds = sum(
                primary_score(fold, "ensemble_predicted_target", metric_name)
                <= primary_score(fold, comparator_column, metric_name)
                for _, fold in part.groupby("outer_fold", sort=True)
            )
            mean_difference, ci_low, ci_high = paired_bootstrap(
                part, "ensemble_predicted_target", comparator_column, metric_name=metric_name,
                seed=20260917 + route_number * 100 + (0 if comparator_name == HUMAN_ROUTE else 1),
                repeats=args.bootstrap_repeats,
            )
            improvement = 100.0 * (1.0 - candidate_score / comparator_score)
            comparison_rows.append({
                "route_id": route, "comparator": comparator_name, "comparison_role": comparator_role,
                "parents": len(part), f"candidate_{metric_name}": candidate_score,
                f"comparator_{metric_name}": comparator_score, "improvement_pct": improvement,
                "noninferior_outer_folds": noninferior_folds, "outer_folds": part.outer_fold.nunique(),
                "paired_bootstrap_difference_mean": mean_difference,
                "paired_bootstrap_difference_95ci_low": ci_low,
                "paired_bootstrap_difference_95ci_high": ci_high,
            })
            bootstrap_rows.append({"route_id": route, "comparator": comparator_name,
                                   "bootstrap_repeats": args.bootstrap_repeats,
                                   "difference_definition": f"candidate_{metric_name}_minus_comparator_{metric_name}",
                                   "difference_mean": mean_difference, "ci_low": ci_low, "ci_high": ci_high})
            subset_values: dict[str, tuple[int, float]] = {}
            for subset, selector in [
                ("low_similarity_bottom_quintile", part.low_similarity_bottom_quintile),
                ("source_seen_diagnostic", part.source_seen_in_human_train),
                ("source_unseen_diagnostic", ~part.source_seen_in_human_train),
            ]:
                subset_part = part.loc[selector]
                if subset_part.empty:
                    candidate_subset = comparator_subset = relative_harm = np.nan
                else:
                    candidate_subset = primary_score(subset_part, "ensemble_predicted_target", metric_name)
                    comparator_subset = primary_score(subset_part, comparator_column, metric_name)
                    relative_harm = 100.0 * (candidate_subset / comparator_subset - 1.0)
                sensitivity_rows.append({"route_id": route, "comparator": comparator_name, "subset": subset,
                                         "parents": len(subset_part), f"candidate_{metric_name}": candidate_subset,
                                         f"comparator_{metric_name}": comparator_subset,
                                         "relative_harm_pct": relative_harm})
                subset_values[subset] = (len(subset_part), relative_harm)
            low_similarity_harm = subset_values["low_similarity_bottom_quintile"][1]
            low_similarity_ok = bool(np.isfinite(low_similarity_harm) and low_similarity_harm <= float(comparator_gate["low_similarity_maximum_relative_harm_pct"]))
            core_gate = bool(
                improvement >= float(comparator_gate["minimum_mean_improvement_vs_human_only_pct"] if comparator_name == HUMAN_ROUTE else comparator_gate["minimum_improvement_pct"])
                and noninferior_folds >= int(comparator_gate["minimum_noninferior_outer_folds"])
                and ci_high < 0.0 and low_similarity_ok
            )
            if comparator_name == HUMAN_ROUTE:
                next_action = "transfer_signal_retained_pending_stageA_comparison" if core_gate else "transfer_signal_not_sufficient"
            else:
                next_action = "source_pressure_test_required_before_fixed_validation_review" if core_gate else "retain_as_neural_ablation_not_stageA_replacement"
            decision_rows.append({
                "route_id": route, "comparator": comparator_name, "comparison_role": comparator_role,
                "improvement_gate": improvement >= float(comparator_gate["minimum_mean_improvement_vs_human_only_pct"] if comparator_name == HUMAN_ROUTE else comparator_gate["minimum_improvement_pct"]),
                "fold_gate": noninferior_folds >= int(comparator_gate["minimum_noninferior_outer_folds"]),
                "bootstrap_gate": ci_high < 0.0, "low_similarity_gate": low_similarity_ok,
                "source_graph_balanced_fivefold_feasible": bool(protocol["source_graph"]["balanced_component_fivefold_feasible"]),
                "core_traincv_gate": core_gate,
                "fixed_validation_ready": False,
                "next_action": next_action,
            })

    comparison = pd.DataFrame(comparison_rows)
    decisions = pd.DataFrame(decision_rows)
    source_fivefold_feasible = bool(protocol["source_graph"]["balanced_component_fivefold_feasible"])
    stagea_source_test_triggered = bool(
        decisions.loc[decisions.comparator.eq(STAGEA_ROUTE), "core_traincv_gate"].any()
    )
    if not stagea_source_test_triggered:
        source_next_step = "not_triggered_stageA_replacement_gate_failed"
    elif source_fivefold_feasible:
        source_next_step = "register_balanced_source_component_sensitivity"
    else:
        source_next_step = "register_unbalanced_source_pressure_test"
    source_registry = pd.DataFrame([{
        "human_parents": protocol["source_graph"]["human_parents"],
        "human_documents": protocol["source_graph"]["human_documents"],
        "connected_components": protocol["source_graph"]["connected_components"],
        "largest_component_parents": protocol["source_graph"]["largest_component_parents"],
        "largest_component_fraction": protocol["source_graph"]["largest_component_fraction"],
        "balanced_component_fivefold_feasible": source_fivefold_feasible,
        "stageA_replacement_gate_triggered": stagea_source_test_triggered,
        "required_next_step": source_next_step,
    }])
    with stage_output(args.output) as out:
        metrics.to_csv(out / "route_metrics.csv", index=False)
        fold_metrics.to_csv(out / "outer_fold_metrics.csv", index=False)
        comparison.to_csv(out / "matched_comparison.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(out / "paired_bootstrap_summary.csv", index=False)
        pd.DataFrame(sensitivity_rows).to_csv(out / "sensitivity_comparison.csv", index=False)
        decisions.to_csv(out / "decision_registry.csv", index=False)
        source_registry.to_csv(out / "source_generalization_registry.csv", index=False)
        figure = build_figure(fold_metrics, metrics, protocol["endpoint"], metric_name)
        figure.savefig(out / f"Figure_{protocol['endpoint']}_transfer_trainCV.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            f"# {protocol['endpoint']} controlled cross-species transfer train-CV analysis\\n\\n"
            f"Primary predictions are arithmetic three-seed ensembles in {target_space}; the registered metric is "
            f"{metric_name}. Every comparison is "
            "paired on identical human outer-fold parents and uses parent-level, fold-stratified bootstrap resampling. "
            "The low-similarity sensitivity is the bottom within-fold ECFP4-similarity quintile. Source seen/unseen "
            f"subsets are diagnostic only. Balanced source-component fivefold feasibility is {source_fivefold_feasible}; "
            f"the registered next action is {source_next_step}. "
            "No fixed-validation target or test label is opened or selected by this stage.\\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "cross_species_transfer_traincv_analysis",
            inputs={
                "traincv_complete_sha256": sha256(args.traincv / "complete.json"),
                "p1_complete_sha256": sha256(args.p1 / "complete.json"),
                "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
            },
            endpoint=protocol["endpoint"], primary_metric=metric_name, target_space=target_space,
            routes=3, transfer_routes=2, bootstrap_repeats=args.bootstrap_repeats,
            train_cv_route_decision_authorized=True, fixed_validation_selection_authorized=False,
            validation_target_file_opened=False, test_labels_read=False, model_fitted=False, partial=False,
        )
    print(f"{protocol['endpoint']} transfer train-CV analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
