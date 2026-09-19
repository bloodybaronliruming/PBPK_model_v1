#!/usr/bin/env python3
"""Run the pre-registered P1 strong-STL nested scaffold confirmation.

Parameter selection is confined to four grouped scaffold folds inside each
frozen outer training fold. The selected configuration is then refit on the
complete outer training parents and evaluated only on that outer fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records, metric_row
from stl_benchmark_common import feature_view, make_estimator


VIEW_SCREEN = {
    "rdkit2d": "stl_stageA_first_batch_rdkit2d_v1",
    "ecfp4": "stl_stageA_ecfp4_v1",
    "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1",
}


def parents(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["molecule_id", "feature_index", "scaffold_group"]
    work = frame.copy()
    if work.task_id.nunique() != 1:
        raise ValueError("Parent collapsing requires one task at a time")
    if work.task_id.iloc[0] == "fu__human__plasma":
        # The canonical fu parent label is mean physical fu, expressed back in
        # logit space for a model that emits transformed targets.
        work["physical_target"] = expit(work.target_value.to_numpy(float))
        result = work.groupby(groups, as_index=False).agg(
            parent_physical_target=("physical_target", "mean"), source_records=("row_id", "size")
        )
        result["parent_target"] = logit(np.clip(result.parent_physical_target.to_numpy(float), 1e-12, 1 - 1e-12))
    else:
        result = work.groupby(groups, as_index=False).agg(
            parent_target=("target_value", "mean"), source_records=("row_id", "size")
        )
    if result.molecule_id.duplicated().any():
        raise ValueError("A parent maps to multiple feature or scaffold values")
    return result


def metric_from_raw(task_id: str, observed: np.ndarray, predicted: np.ndarray) -> float:
    if task_id == "fu__human__plasma":
        return float(np.mean(np.abs(expit(observed) - expit(predicted))))
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def validate_partition(train: pd.DataFrame, evaluation: pd.DataFrame, task: str, fold: int) -> None:
    if train.empty or evaluation.empty:
        raise ValueError(f"Empty outer partition: {task} fold={fold}")
    parent_overlap = set(train.molecule_id) & set(evaluation.molecule_id)
    scaffold_overlap = set(train.scaffold_group.astype(str)) & set(evaluation.scaffold_group.astype(str))
    if parent_overlap or scaffold_overlap:
        raise ValueError(f"Outer leakage: {task} fold={fold}")


def stagea_leaders(stagea_audit: Path, benchmarks: Path) -> dict[str, dict]:
    decisions = pd.read_csv(stagea_audit / "endpoint_decisions.csv")
    registries = {view: pd.read_csv(benchmarks / name / "run_registry.csv") for view, name in VIEW_SCREEN.items()}
    leaders = {}
    for decision in decisions.itertuples(index=False):
        view = str(decision.stageA_leading_feature_view)
        rows = registries[view].loc[
            (registries[view].task_id.eq(decision.task_id))
            & (registries[view].candidate_id.eq(decision.stageA_leading_candidate))
        ]
        if len(rows) != 5 or rows.algorithm.nunique() != 1 or rows.parameters.nunique() != 1:
            raise ValueError(f"Stage-A leader registry inconsistent for {decision.task_id}")
        leaders[str(decision.task_id)] = {
            "candidate_id": str(decision.stageA_leading_candidate), "algorithm": str(rows.algorithm.iloc[0]),
            "feature_set": view, "parameters": json.loads(rows.parameters.iloc[0]),
        }
    return leaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--candidate-registry", type=Path, default=ROOT / "data/public_development/stl_p1_candidate_registry_v1")
    parser.add_argument("--stagea-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confirmation-seeds", nargs="+", type=int, default=[20260920, 20260921, 20260922])
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Engineering smoke only")
    parser.add_argument("--smoke", action="store_true", help="Run only outer fold 0; cannot select models")
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or not args.confirmation_seeds or len(set(args.confirmation_seeds)) != len(args.confirmation_seeds):
        raise ValueError("Threads and confirmation seeds must be valid")
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
                args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
                args.candidate_registry / "complete.json", args.candidate_registry / "candidate_registry.csv",
                args.stagea_audit / "complete.json", args.stagea_audit / "endpoint_decisions.csv"]
    required.extend(args.benchmarks / name / "complete.json" for name in VIEW_SCREEN.values())
    required.extend(args.benchmarks / name / "run_registry.csv" for name in VIEW_SCREEN.values())
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    protocol_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    registry_meta = verify_stage(args.candidate_registry, "stl_p1_candidate_registry")
    audit_meta = verify_stage(args.stagea_audit, "stl_stageA_three_view_audit")
    screen_meta = [verify_stage(args.benchmarks / name, "stl_benchmark_screen") for name in VIEW_SCREEN.values()]
    if train_meta.get("fixed_validation_targets_published") or any(meta.get("test_labels_read") for meta in [train_meta, protocol_meta, registry_meta, audit_meta, *screen_meta]):
        raise ValueError("P1 nested confirmation accepts only test-closed train-only inputs")
    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records.feature_index = records.feature_index.astype(int)
    records.inner_fold_id = records.inner_fold_id.astype(int)
    if not records.split.eq("train").all() or records.task_id.nunique() != 6:
        raise ValueError("P1 requires exactly six train-only endpoint tables")
    records = limit_records(records, args.max_parents_per_task_fold)
    candidate = pd.read_csv(args.candidate_registry / "candidate_registry.csv", dtype=str, keep_default_na=False)
    if set(candidate.task_id) != set(records.task_id) or candidate.candidate_uid.duplicated().any():
        raise ValueError("Candidate registry does not uniquely cover train-only tasks")
    candidate["parameters"] = candidate.parameters.map(json.loads)
    cache_file = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    cache = {key: cache_file[key] for key in cache_file.files}
    leaders = stagea_leaders(args.stagea_audit, args.benchmarks)
    if set(leaders) != set(records.task_id):
        raise ValueError("Stage-A leaders do not cover P1 tasks")
    outer_folds = [0] if args.smoke else [0, 1, 2, 3, 4]
    confirmation_seeds = args.confirmation_seeds[:1] if args.smoke else args.confirmation_seeds
    expected_inner_fits = len(candidate) * len(outer_folds) * 4
    expected_final_fits = records.task_id.nunique() * len(outer_folds) * len(confirmation_seeds) * 2
    if args.check_only:
        print(f"P1 nested confirmation ready: tasks=6 candidates={len(candidate)} inner_fits={expected_inner_fits} final_fits={expected_final_fits} smoke={args.smoke}")
        return
    partial = bool(args.smoke or args.max_parents_per_task_fold)
    with stage_output(args.output) as out:
        inner_rows, selection_rows, prediction_rows, seed_rows, metric_rows, isolation_rows, reload_rows = [], [], [], [], [], [], []
        control_prediction_rows, control_seed_rows = [], []
        for task_id, task_records in records.groupby("task_id", sort=True):
            task_candidates = candidate.loc[candidate.task_id.eq(task_id)].copy()
            if task_candidates.empty:
                raise ValueError(f"No registered P1 candidates for {task_id}")
            endpoint = task_records.endpoint.iloc[0]
            for outer_fold in outer_folds:
                outer_train_records = task_records.loc[task_records.inner_fold_id.ne(outer_fold)].copy()
                outer_eval_records = task_records.loc[task_records.inner_fold_id.eq(outer_fold)].copy()
                outer_train, outer_eval = parents(outer_train_records), parents(outer_eval_records)
                validate_partition(outer_train, outer_eval, task_id, outer_fold)
                if outer_train.scaffold_group.nunique() < 4:
                    raise ValueError(f"Insufficient outer-train scaffold groups: {task_id} fold={outer_fold}")
                candidate_scores = []
                for row in task_candidates.itertuples(index=False):
                    matrix = feature_view(cache, row.feature_set)
                    group_split = GroupKFold(n_splits=4)
                    fold_scores = []
                    for inner_fold, (inner_train_index, inner_eval_index) in enumerate(group_split.split(
                        outer_train, groups=outer_train.scaffold_group.astype(str)
                    )):
                        inner_train, inner_eval = outer_train.iloc[inner_train_index], outer_train.iloc[inner_eval_index]
                        if set(inner_train.scaffold_group.astype(str)) & set(inner_eval.scaffold_group.astype(str)):
                            raise ValueError(f"Inner scaffold leakage: {task_id} outer={outer_fold} candidate={row.candidate_uid}")
                        mean = float(inner_train.parent_target.mean())
                        std = float(inner_train.parent_target.std(ddof=0))
                        if not np.isfinite(std) or std <= 0:
                            raise ValueError("Invalid inner target scale")
                        seed = stable_seed("p1-inner", task_id, outer_fold, row.candidate_uid, inner_fold)
                        estimator = make_estimator(row.algorithm, seed, args.threads, small_budget=True, parameters=row.parameters)
                        estimator.fit(matrix[inner_train.feature_index.to_numpy(int)], (inner_train.parent_target.to_numpy(float) - mean) / std)
                        predicted = np.asarray(estimator.predict(matrix[inner_eval.feature_index.to_numpy(int)])).reshape(-1) * std + mean
                        score = metric_from_raw(task_id, inner_eval.parent_target.to_numpy(float), predicted)
                        fold_scores.append(score)
                        inner_rows.append({"task_id": task_id, "endpoint": endpoint, "outer_fold": outer_fold, "inner_fold": inner_fold,
                                           "candidate_uid": row.candidate_uid, "algorithm": row.algorithm, "feature_set": row.feature_set,
                                           "parameters": json.dumps(row.parameters, sort_keys=True), "train_parents": len(inner_train),
                                           "evaluation_parents": len(inner_eval), "inner_primary_metric": score,
                                           "primary_metric_name": row.primary_metric_name, "inner_scaffold_overlap": 0})
                    candidate_scores.append((float(np.mean(fold_scores)), str(row.candidate_uid), row))
                candidate_scores.sort(key=lambda item: (item[0], item[1]))
                selected_score, _, selected = candidate_scores[0]
                for rank, (score, _, spec) in enumerate(candidate_scores, start=1):
                    selection_rows.append({"task_id": task_id, "endpoint": endpoint, "outer_fold": outer_fold,
                                           "candidate_uid": spec.candidate_uid, "algorithm": spec.algorithm, "feature_set": spec.feature_set,
                                           "parameters": json.dumps(spec.parameters, sort_keys=True), "mean_inner_primary_metric": score,
                                           "inner_rank": rank, "selected_for_outer_refit": rank == 1,
                                           "primary_metric_name": spec.primary_metric_name})
                matrix = feature_view(cache, selected.feature_set)
                mean_outer = float(outer_train.parent_target.mean())
                std_outer = float(outer_train.parent_target.std(ddof=0))
                if not np.isfinite(std_outer) or std_outer <= 0:
                    raise ValueError("Invalid outer target scale")
                seed_predictions = []
                for seed in confirmation_seeds:
                    estimator_seed = stable_seed("p1-final", task_id, outer_fold, selected.candidate_uid, seed)
                    estimator = make_estimator(selected.algorithm, estimator_seed, args.threads, small_budget=True, parameters=selected.parameters)
                    estimator.fit(matrix[outer_train.feature_index.to_numpy(int)], (outer_train.parent_target.to_numpy(float) - mean_outer) / std_outer)
                    raw_prediction = np.asarray(estimator.predict(matrix[outer_eval.feature_index.to_numpy(int)])).reshape(-1) * std_outer + mean_outer
                    seed_predictions.append(raw_prediction)
                    seed_rows.append(pd.DataFrame({"task_id": task_id, "endpoint": endpoint, "outer_fold": outer_fold,
                                                   "seed": seed, "molecule_id": outer_eval.molecule_id, "candidate_uid": selected.candidate_uid,
                                                   "predicted_target_value": raw_prediction}))
                    if args.verify_reload and outer_fold == outer_folds[0] and seed == confirmation_seeds[0]:
                        with tempfile.TemporaryDirectory(dir=out) as temporary:
                            path = Path(temporary) / "model.joblib"
                            joblib.dump(estimator, path, compress=3)
                            reloaded = np.asarray(joblib.load(path).predict(matrix[outer_eval.feature_index.to_numpy(int)])).reshape(-1) * std_outer + mean_outer
                        difference = float(np.max(np.abs(raw_prediction - reloaded)))
                        if difference > 1e-8:
                            raise ValueError(f"Reload mismatch: {task_id} fold={outer_fold}")
                        reload_rows.append({"task_id": task_id, "outer_fold": outer_fold, "candidate_uid": selected.candidate_uid,
                                            "max_abs_difference": difference})
                ensemble_prediction = np.mean(np.vstack(seed_predictions), axis=0)
                global_mean, global_std = float(task_records.train_mean.iloc[0]), float(task_records.train_std_population.iloc[0])
                parent_prediction = pd.DataFrame({"molecule_id": outer_eval.molecule_id,
                                                  "predicted_interface_target": (ensemble_prediction - global_mean) / global_std})
                table = outer_eval_records[["row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id", "target_value",
                                           "train_mean", "train_std_population", "reporting_inverse"]].merge(parent_prediction, on="molecule_id", validate="many_to_one")
                table["interface_target"] = (table.target_value - table.train_mean) / table.train_std_population
                table["outer_fold"], table["candidate_uid"], table["algorithm"], table["feature_set"] = outer_fold, selected.candidate_uid, selected.algorithm, selected.feature_set
                prediction_rows.append(table)
                control = leaders[task_id]
                control_matrix = feature_view(cache, control["feature_set"])
                control_seed_predictions = []
                for seed in confirmation_seeds:
                    control_seed = stable_seed("p1-stagea-control", task_id, outer_fold, control["candidate_id"], seed)
                    control_estimator = make_estimator(control["algorithm"], control_seed, args.threads, small_budget=True,
                                                       parameters=control["parameters"])
                    control_estimator.fit(control_matrix[outer_train.feature_index.to_numpy(int)],
                                          (outer_train.parent_target.to_numpy(float) - mean_outer) / std_outer)
                    control_raw = np.asarray(control_estimator.predict(control_matrix[outer_eval.feature_index.to_numpy(int)])).reshape(-1) * std_outer + mean_outer
                    control_seed_predictions.append(control_raw)
                    control_seed_rows.append(pd.DataFrame({"task_id": task_id, "endpoint": endpoint, "outer_fold": outer_fold,
                                                           "seed": seed, "molecule_id": outer_eval.molecule_id,
                                                           "candidate_uid": control["candidate_id"], "algorithm": control["algorithm"],
                                                           "feature_set": control["feature_set"], "predicted_target_value": control_raw}))
                    if args.verify_reload and outer_fold == outer_folds[0] and seed == confirmation_seeds[0]:
                        with tempfile.TemporaryDirectory(dir=out) as temporary:
                            path = Path(temporary) / "stagea_control_model.joblib"
                            joblib.dump(control_estimator, path, compress=3)
                            reloaded = np.asarray(joblib.load(path).predict(control_matrix[outer_eval.feature_index.to_numpy(int)])).reshape(-1) * std_outer + mean_outer
                        difference = float(np.max(np.abs(control_raw - reloaded)))
                        if difference > 1e-8:
                            raise ValueError(f"Matched-control reload mismatch: {task_id} fold={outer_fold}")
                        reload_rows.append({"task_id": task_id, "outer_fold": outer_fold, "candidate_uid": control["candidate_id"],
                                            "model_role": "stageA_matched_control", "max_abs_difference": difference})
                control_ensemble = np.mean(np.vstack(control_seed_predictions), axis=0)
                control_parent_prediction = pd.DataFrame({"molecule_id": outer_eval.molecule_id,
                                                          "predicted_interface_target": (control_ensemble - global_mean) / global_std})
                control_table = outer_eval_records[["row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id", "target_value",
                                                   "train_mean", "train_std_population", "reporting_inverse"]].merge(control_parent_prediction, on="molecule_id", validate="many_to_one")
                control_table["interface_target"] = (control_table.target_value - control_table.train_mean) / control_table.train_std_population
                control_table["outer_fold"], control_table["candidate_uid"], control_table["algorithm"], control_table["feature_set"] = outer_fold, control["candidate_id"], control["algorithm"], control["feature_set"]
                control_prediction_rows.append(control_table)
                isolation_rows.append({"task_id": task_id, "outer_fold": outer_fold, "candidate_uid": selected.candidate_uid,
                                       "outer_train_parents": len(outer_train), "outer_evaluation_parents": len(outer_eval),
                                       "outer_parent_overlap": 0, "outer_scaffold_overlap": 0,
                                       "inner_scaffold_overlap_max": 0, "evaluation_targets_used_in_inner_selection": False,
                                       "selected_mean_inner_primary_metric": selected_score})
        predictions = pd.concat(prediction_rows, ignore_index=True)
        control_predictions = pd.concat(control_prediction_rows, ignore_index=True)
        if predictions.duplicated(["task_id", "row_id"]).any() or predictions.predicted_interface_target.isna().any():
            raise ValueError("OOF predictions are duplicate or incomplete")
        if control_predictions.duplicated(["task_id", "row_id"]).any() or control_predictions.predicted_interface_target.isna().any():
            raise ValueError("Matched-control OOF predictions are duplicate or incomplete")
        for task_id, frame in predictions.groupby("task_id", sort=True):
            row = metric_row(frame, frame.predicted_interface_target.to_numpy(), task_id,
                             "nested_p1_selected", "outer_selected", "nested_train_cv")
            row["outer_folds"] = len(outer_folds)
            row["confirmation_seeds"] = json.dumps(confirmation_seeds)
            metric_rows.append(row)
            control_frame = control_predictions.loc[control_predictions.task_id.eq(task_id)]
            control_row = metric_row(control_frame, control_frame.predicted_interface_target.to_numpy(), task_id,
                                     "stageA_matched_control", "registered_stageA_leader", "nested_train_cv_matched_control")
            control_row["outer_folds"] = len(outer_folds)
            control_row["confirmation_seeds"] = json.dumps(confirmation_seeds)
            metric_rows.append(control_row)
            if task_id == "Papp__human__caco2_ab":
                local = frame.copy()
                local["papp_value"] = np.power(10.0, local.target_value.to_numpy(float))
                local["papp_range"] = pd.cut(local.papp_value, [0.0, 100.0, 10000.0, np.inf], labels=["≤100", "100–10,000", ">10,000"], include_lowest=True)
                for label, subset in local.groupby("papp_range", observed=False):
                    if subset.empty:
                        continue
                    subset_row = metric_row(subset, subset.predicted_interface_target.to_numpy(), task_id,
                                            "nested_p1_selected", "outer_selected", f"nested_train_cv:{label}")
                    metric_rows.append(subset_row)
        metrics = pd.DataFrame(metric_rows)
        predictions.to_csv(out / "predictions.csv", index=False)
        control_predictions.to_csv(out / "stageA_matched_control_predictions.csv", index=False)
        metrics.to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(inner_rows).to_csv(out / "inner_selection_metrics.csv", index=False)
        selections = pd.DataFrame(selection_rows)
        selections.to_csv(out / "outer_selection_registry.csv", index=False)
        pd.concat(seed_rows, ignore_index=True).to_csv(out / "confirmation_seed_predictions.csv", index=False)
        pd.concat(control_seed_rows, ignore_index=True).to_csv(out / "stageA_matched_control_seed_predictions.csv", index=False)
        pd.DataFrame(isolation_rows).to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(reload_rows).to_csv(out / "reload_checks.csv", index=False)
        stability = selections.loc[selections.selected_for_outer_refit].groupby(
            ["task_id", "endpoint", "candidate_uid", "algorithm", "feature_set"], as_index=False
        ).size().rename(columns={"size": "outer_folds_selected"})
        stability.to_csv(out / "selection_stability.csv", index=False)
        summary = {"tasks": int(records.task_id.nunique()), "candidates": int(len(candidate)), "outer_folds": outer_folds,
                   "inner_grouped_scaffold_folds": 4, "confirmation_seeds": confirmation_seeds,
                   "inner_fits": int(len(inner_rows)), "final_fits": int(len(seed_rows) + len(control_seed_rows)), "limited_smoke": partial,
                   "selection_method": "four_grouped_scaffold_inner_folds_then_outer_refit", "validation_target_file_opened": False,
                   "validation_rows_evaluated": False, "test_labels_read": False,
                   "model_selection_authorized": not partial, "final_model_selection_authorized": False}
        (out / "README.md").write_text(
            "# P1 nested strong-STL confirmation\n\n"
            "Each outer scaffold fold selects from the frozen P1 registry using four grouped scaffold folds inside outer training. "
            "The selected configuration is refit on complete outer training and evaluated on its untouched outer fold. "
            "fu parent targets are mean physical fu mapped to logit space. A fixed Stage-A leader is rerun with the same outer folds, seeds and aggregation as a matched control. "
            "Source-cluster and nearest-similarity diagnostics are reserved for finalists after this nested train-CV stage.\n",
            encoding="utf-8")
        finish_stage(out, "stl_p1_nested_confirmation", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "candidate_registry_complete_sha256": sha256(args.candidate_registry / "complete.json"),
            "stagea_audit_complete_sha256": sha256(args.stagea_audit / "complete.json"),
        }, partial=partial, **summary)
    print(f"P1 nested STL confirmation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
