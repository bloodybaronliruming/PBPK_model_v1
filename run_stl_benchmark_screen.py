#!/usr/bin/env python3
"""Run a bounded, protocol-governed strong-STL screening experiment.

The default scope is frozen five-fold training CV.  Fixed validation requires
an explicit flag and is intended only after a shortlist has been registered.
Any row-limited run is an engineering smoke and cannot select a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from stl_benchmark_common import IMPLEMENTED_ALGORITHMS, feature_view, make_estimator, parameter_candidates


def limit_records(records: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if not maximum:
        return records.copy()
    keys = ["task_id", "split", "inner_fold_id"]
    parents = (records[keys + ["molecule_id"]].drop_duplicates()
               .sort_values([*keys, "molecule_id"]).groupby(keys, group_keys=False).head(maximum))
    return records.merge(parents.assign(_keep=True), on=[*keys, "molecule_id"], how="inner", validate="many_to_one").drop(columns="_keep")


def collapse_training_parents(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["molecule_id", "feature_index"]
    result = (frame.groupby(columns, as_index=False)
              .agg(interface_target=("interface_target", "mean"), source_records=("row_id", "size")))
    return result


def reporting_values(frame: pd.DataFrame, interface_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = interface_prediction * frame.train_std_population.to_numpy(float) + frame.train_mean.to_numpy(float)
    inverse = frame.reporting_inverse.iloc[0]
    if inverse == "power10":
        reporting = np.power(10.0, np.clip(transformed, -300.0, 300.0))
    elif inverse == "expit":
        reporting = expit(transformed)
    elif inverse == "identity":
        reporting = transformed
    else:
        raise ValueError(f"Unknown reporting inverse: {inverse}")
    return transformed, reporting


def metric_row(frame: pd.DataFrame, prediction: np.ndarray, task: str, algorithm: str,
               feature_set: str, scope: str) -> dict:
    transformed, reporting = reporting_values(frame, prediction)
    observed_transformed = frame.target_value.to_numpy(float)
    observed_reporting = reporting_values(frame, frame.interface_target.to_numpy(float))[1]
    parent = pd.DataFrame({
        "molecule_id": frame.molecule_id.to_numpy(), "observed_transformed": observed_transformed,
        "predicted_transformed": transformed, "observed_reporting": observed_reporting,
        "predicted_reporting": reporting,
    }).groupby("molecule_id", as_index=False).mean(numeric_only=True)
    transformed_rmse = float(np.sqrt(mean_squared_error(parent.observed_transformed, parent.predicted_transformed)))
    physical_mae = float(mean_absolute_error(parent.observed_reporting, parent.predicted_reporting))
    rank = (
        spearmanr(parent.observed_transformed, parent.predicted_transformed).statistic
        if len(parent) > 1 and parent.observed_transformed.nunique() > 1
        and parent.predicted_transformed.nunique() > 1 else np.nan
    )
    endpoint = frame.endpoint.iloc[0]
    return {
        "task_id": task, "endpoint": endpoint, "algorithm": algorithm, "feature_set": feature_set,
        "evaluation_scope": scope, "records": len(frame), "parents": len(parent),
        "primary_metric_name": "physical_MAE" if endpoint == "fu" else "transformed_RMSE",
        "primary_metric": physical_mae if endpoint == "fu" else transformed_rmse,
        "transformed_RMSE": transformed_rmse, "physical_MAE": physical_mae,
        "spearman_parent_mean": float(rank) if np.isfinite(rank) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/stl_broad_screen_v1")
    parser.add_argument("--algorithms", default="dummy_mean,ridge,extra_trees,hist_gradient_boosting")
    parser.add_argument("--feature-set", choices=["ecfp4", "rdkit2d", "ecfp4_rdkit2d"], default="rdkit2d")
    parser.add_argument("--scope", choices=["train_cv", "validation"], default="train_cv")
    parser.add_argument("--confirm-validation", action="store_true")
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Smoke only; limits each task/fold")
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--allow-expensive-kernel", action="store_true",
                        help="Explicitly allow kernel methods above their safe parent-count guard")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trials-per-algorithm", type=int, default=2,
                        help="Maximum deterministic parameter candidates per algorithm")
    parser.add_argument("--candidate-profile", choices=["stage_a", "linear_stabilization"], default="stage_a",
                        help="Bounded registered candidate subset; stabilization is Ridge/ElasticNet only")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    algorithms = list(dict.fromkeys(filter(None, args.algorithms.split(","))))
    if not algorithms or not set(algorithms) <= IMPLEMENTED_ALGORITHMS or args.trials_per_algorithm < 1:
        raise ValueError("Unknown or unimplemented algorithm adapter")
    if args.candidate_profile == "linear_stabilization" and not set(algorithms) <= {"ridge", "elasticnet"}:
        raise ValueError("linear_stabilization accepts only ridge and elasticnet")
    if args.scope == "validation" and not args.confirm_validation:
        raise ValueError("Fixed validation requires --confirm-validation")
    startup_self_check([args.protocol / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.protocol, "stl_benchmark_protocol")
    records = pd.read_csv(args.protocol / "benchmark_records.csv", dtype=str, keep_default_na=False)
    numeric = ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]
    for column in numeric:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records["feature_index"] = records.feature_index.astype(int)
    records["inner_fold_id"] = records.inner_fold_id.astype(int)
    records["interface_target"] = (records.target_value - records.train_mean) / records.train_std_population
    records = limit_records(records, args.max_parents_per_task_fold)
    registry = pd.read_csv(args.protocol / "algorithm_registry.csv")
    unavailable = [a for a in algorithms if not bool(registry.loc[registry.algorithm.eq(a), "runtime_available"].iloc[0])]
    if unavailable:
        raise RuntimeError(f"Requested optional algorithms unavailable: {unavailable}")
    incompatible = []
    for algorithm in algorithms:
        allowed = str(registry.loc[registry.algorithm.eq(algorithm), "allowed_feature_views"].iloc[0]).split(",")
        if "all" not in allowed and args.feature_set not in allowed:
            incompatible.append(f"{algorithm}:{args.feature_set}")
    if incompatible:
        raise ValueError(f"Algorithm/feature combinations are outside the registered protocol: {incompatible}")
    max_train_parents = int(records.loc[records.split.eq("train")].groupby("task_id").molecule_id.nunique().max())
    if not args.allow_expensive_kernel:
        if "kernel_ridge_rbf" in algorithms and max_train_parents > 2000:
            raise ValueError("Kernel ridge is limited to <=2000 training parents unless --allow-expensive-kernel is explicit")
        if "svr_rbf" in algorithms and max_train_parents > 6000:
            raise ValueError("RBF SVR is limited to <=6000 training parents unless --allow-expensive-kernel is explicit")
    if args.check_only:
        print(f"STL screen input valid: records={len(records)} tasks={records.task_id.nunique()} algorithms={len(algorithms)}")
        return
    cache_file = np.load(args.protocol / "canonical_parent_features_float32.npz")
    x = feature_view({key: cache_file[key] for key in cache_file.files}, args.feature_set)
    with stage_output(args.output) as out:
        prediction_rows, metric_rows, run_rows, reload_rows = [], [], [], []
        for task, task_frame in records.groupby("task_id", sort=True):
            if args.scope == "train_cv":
                evaluation_parts = [(f"fold_{fold}", task_frame.split.eq("train") & task_frame.inner_fold_id.eq(fold),
                                     task_frame.split.eq("train") & task_frame.inner_fold_id.ne(fold))
                                    for fold in sorted(task_frame.loc[task_frame.split.eq("train"), "inner_fold_id"].unique())]
            else:
                evaluation_parts = [("fixed_validation", task_frame.split.eq("val"), task_frame.split.eq("train"))]
            for algorithm in algorithms:
                candidates = parameter_candidates(
                    algorithm, args.trials_per_algorithm, args.seed, profile=args.candidate_profile
                )
                for candidate_index, parameters in enumerate(candidates):
                    profile_tag = "" if args.candidate_profile == "stage_a" else f"__{args.candidate_profile}"
                    candidate_id = f"{algorithm}{profile_tag}__c{candidate_index:02d}"
                    all_eval = []
                    for label, eval_mask, train_mask in evaluation_parts:
                        train = collapse_training_parents(task_frame.loc[train_mask])
                        evaluation = task_frame.loc[eval_mask].copy()
                        if train.empty or evaluation.empty:
                            raise ValueError(f"Empty training/evaluation partition: {task} {label}")
                        estimator = make_estimator(algorithm, args.seed, args.threads, small_budget=True, parameters=parameters)
                        estimator.fit(x[train.feature_index.to_numpy(int)], train.interface_target.to_numpy(float))
                        prediction = np.asarray(estimator.predict(x[evaluation.feature_index.to_numpy(int)])).reshape(-1)
                        table = evaluation[["row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id",
                                            "sensitivity_subset", "target_value", "train_mean", "train_std_population",
                                            "reporting_inverse", "interface_target"]].copy()
                        table["predicted_interface_target"] = prediction
                        table["algorithm"] = algorithm
                        table["candidate_id"] = candidate_id
                        table["feature_set"] = args.feature_set
                        table["evaluation_partition"] = label
                        prediction_rows.append(table)
                        all_eval.append(table)
                        run_rows.append({"task_id": task, "algorithm": algorithm, "feature_set": args.feature_set,
                                         "candidate_id": candidate_id, "partition": label, "train_parents": len(train),
                                         "evaluation_parents": evaluation.molecule_id.nunique(),
                                         "parameters": json.dumps(parameters, sort_keys=True, default=list)})
                        if args.verify_reload and label == evaluation_parts[0][0]:
                            folder = out / "reload_checks"; folder.mkdir(exist_ok=True)
                            path = folder / f"{task}__{candidate_id}.joblib"
                            joblib.dump(estimator, path, compress=3)
                            after = joblib.load(path).predict(x[evaluation.feature_index.to_numpy(int)])
                            delta = float(np.max(np.abs(prediction - np.asarray(after).reshape(-1))))
                            if delta > 1e-8:
                                raise ValueError(f"Reload prediction mismatch: {task} {algorithm}")
                            reload_rows.append({"task_id": task, "algorithm": algorithm, "candidate_id": candidate_id,
                                                "max_abs_difference": delta, "passed": True})
                    joined = pd.concat(all_eval, ignore_index=True)
                    overall = metric_row(joined, joined.predicted_interface_target.to_numpy(), task, algorithm,
                                         args.feature_set, args.scope)
                    overall["candidate_id"] = candidate_id
                    overall["parameters"] = json.dumps(parameters, sort_keys=True, default=list)
                    metric_rows.append(overall)
                    if args.scope == "validation":
                        for subset, subset_frame in joined.groupby("sensitivity_subset"):
                            if subset and subset != "train_not_applicable":
                                row = metric_row(subset_frame, subset_frame.predicted_interface_target.to_numpy(), task,
                                                 algorithm, args.feature_set, f"validation:{subset}")
                                row["candidate_id"] = candidate_id
                                row["parameters"] = json.dumps(parameters, sort_keys=True, default=list)
                                metric_rows.append(row)
        pd.concat(prediction_rows, ignore_index=True).to_csv(out / "predictions.csv", index=False)
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(out / "metrics.csv", index=False)
        overall = metrics.loc[metrics.evaluation_scope.eq(args.scope)].copy()
        overall["rank_within_task"] = overall.groupby("task_id").primary_metric.rank(method="min", ascending=True).astype(int)
        overall.sort_values(["task_id", "rank_within_task", "candidate_id"]).to_csv(out / "candidate_ranking.csv", index=False)
        pd.DataFrame(run_rows).to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(reload_rows).to_csv(out / "reload_checks.csv", index=False)
        limited = bool(args.max_parents_per_task_fold)
        (out / "README.md").write_text(
            "# Strong STL bounded screen\n\nProtocol-governed train-CV or explicitly confirmed validation run. Row-limited runs are engineering smoke tests and never authorize selection.\n",
            encoding="utf-8")
        finish_stage(out, "stl_benchmark_screen", inputs={"protocol_complete_sha256": sha256(args.protocol / "complete.json")},
                     algorithms=algorithms, feature_set=args.feature_set, evaluation_scope=args.scope,
                     candidate_profile=args.candidate_profile,
                     trials_per_algorithm=args.trials_per_algorithm,
                     candidates_evaluated=int(overall.candidate_id.nunique()),
                     candidate_task_evaluations=len(overall), seed=args.seed, limited_smoke=limited,
                     model_selection_authorized=not limited and args.scope == "train_cv",
                     selection_scope="stageA_train_CV_shortlist_only" if args.scope == "train_cv" else "stageB_fixed_validation_confirmation",
                     final_model_selection_authorized=False,
                     validation_confirmation=bool(args.scope == "validation"), test_labels_read=False, partial=limited)
    print(f"STL benchmark screen: {args.output}")


if __name__ == "__main__":
    run_cli(main)
