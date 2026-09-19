#!/usr/bin/env python3
"""Test the RDKit Ipc descriptor with fixed, train-only Stage-A leader heads.

For every endpoint, the exact Stage-A ExtraTrees leader is evaluated under
three pre-registered views: Ipc as registered, Ipc dropped, and Ipc log1p.
No labels outside the frozen training folds are opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records, metric_row
from stl_benchmark_common import feature_view, make_estimator


VIEW_SCREEN = {"rdkit2d": "stl_stageA_first_batch_rdkit2d_v1", "ecfp4": "stl_stageA_ecfp4_v1", "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1"}
IPC_NAME = "Ipc"


def parents(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["molecule_id", "feature_index"], as_index=False).agg(
        parent_target=("target_value", "mean"), source_records=("row_id", "size"))


def ipc_column(view: str, descriptor_index: int) -> int:
    if view == "rdkit2d":
        return descriptor_index
    if view == "ecfp4_rdkit2d":
        return 2048 + descriptor_index
    raise ValueError(f"Ipc is unavailable in feature view {view}")


def alter_ipc(matrix: np.ndarray, column: int, treatment: str) -> np.ndarray:
    if treatment == "as_registered":
        return matrix
    if treatment == "dropped":
        return np.delete(matrix, column, axis=1)
    if treatment == "log1p":
        result = matrix.copy()
        value = result[:, column]
        if np.any(value < 0) or not np.isfinite(value).all():
            raise ValueError("Ipc must be finite and non-negative for log1p treatment")
        result[:, column] = np.log1p(value)
        return result
    raise ValueError(f"Unknown Ipc treatment: {treatment}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/stl_ipc_feature_ablation_v1")
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Engineering smoke only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    screen_paths = {view: args.benchmarks / name for view, name in VIEW_SCREEN.items()}
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
                args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
                args.stl_protocol / "feature_registry.json", args.stageA_audit / "complete.json",
                args.stageA_audit / "endpoint_decisions.csv"] + [path / "complete.json" for path in screen_paths.values()]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    audit_meta = verify_stage(args.stageA_audit, "stl_stageA_three_view_audit")
    screen_meta = {view: verify_stage(path, "stl_benchmark_screen") for view, path in screen_paths.items()}
    if not train_meta.get("fold_local_target_standardization_required") or train_meta.get("fixed_validation_targets_published"):
        raise ValueError("Train-only/fold-local preprocessing contract is not active")
    if any(meta.get("test_labels_read") for meta in [train_meta, stl_meta, audit_meta, *screen_meta.values()]):
        raise ValueError("An upstream stage reports test-label access")
    if audit_meta.get("validation_labels_read"):
        raise ValueError("Fixed validation must remain closed")
    if args.threads < 1:
        raise ValueError("threads must be positive")

    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records.feature_index = records.feature_index.astype(int)
    records.inner_fold_id = records.inner_fold_id.astype(int)
    records = limit_records(records, args.max_parents_per_task_fold)
    registry = json.loads((args.stl_protocol / "feature_registry.json").read_text(encoding="utf-8"))
    descriptor_names = registry["rdkit2d"]["descriptor_names"]
    if IPC_NAME not in descriptor_names:
        raise ValueError("Feature registry does not contain Ipc")
    descriptor_index = descriptor_names.index(IPC_NAME)
    cache_file = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    cache = {name: cache_file[name] for name in cache_file.files}
    run_tables = {view: pd.read_csv(path / "run_registry.csv") for view, path in screen_paths.items()}
    leaders = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv")
    leader_specs = {}
    for leader in leaders.itertuples(index=False):
        view = str(leader.stageA_leading_feature_view)
        row = run_tables[view].loc[(run_tables[view].task_id.eq(leader.task_id)) &
                                   (run_tables[view].candidate_id.eq(leader.stageA_leading_candidate))].iloc[0]
        if row.algorithm != "extra_trees" or view not in {"rdkit2d", "ecfp4_rdkit2d"}:
            raise ValueError(f"Ipc ablation requires Ipc-containing ExtraTrees leader: {leader.task_id}")
        leader_specs[leader.task_id] = {"endpoint": leader.endpoint, "view": view,
                                        "parameters": json.loads(row.parameters), "candidate": leader.stageA_leading_candidate}
    if set(leader_specs) != set(records.task_id.unique()):
        raise ValueError("Stage-A leaders do not cover the train-only tasks")
    treatments = ("as_registered", "dropped", "log1p")
    expected_fits = len(leader_specs) * len(treatments) * 5
    if args.check_only:
        print(f"Ipc ablation ready: tasks={len(leader_specs)} fits={expected_fits} ipc_descriptor_index={descriptor_index} limited_smoke={bool(args.max_parents_per_task_fold)}")
        return

    limited = bool(args.max_parents_per_task_fold)
    with stage_output(args.output) as out:
        predictions, metrics, runs, reloads, isolations, registry_rows = [], [], [], [], [], []
        for task, task_frame in records.groupby("task_id", sort=True):
            spec = leader_specs[task]
            original = feature_view(cache, spec["view"])
            index = ipc_column(spec["view"], descriptor_index)
            ipc_values = original[:, index]
            for treatment in treatments:
                matrix = alter_ipc(original, index, treatment)
                candidate = f"ipc_{treatment}__stageA_leader"
                registry_rows.append({"task_id": task, "endpoint": spec["endpoint"], "candidate_id": candidate,
                                      "treatment": treatment, "stageA_feature_view": spec["view"],
                                      "stageA_candidate": spec["candidate"], "algorithm": "extra_trees",
                                      "parameters": json.dumps(spec["parameters"], sort_keys=True),
                                      "original_dimensions": original.shape[1], "treated_dimensions": matrix.shape[1],
                                      "ipc_descriptor_index_rdkit2d": descriptor_index, "ipc_feature_column_original": index})
                all_eval = []
                for fold in range(5):
                    train_rows = task_frame.loc[task_frame.inner_fold_id.ne(fold)].copy()
                    eval_rows = task_frame.loc[task_frame.inner_fold_id.eq(fold)].copy()
                    train_parent, eval_parent = parents(train_rows), parents(eval_rows)
                    parent_overlap = len(set(train_parent.molecule_id) & set(eval_parent.molecule_id))
                    scaffold_overlap = len(set(train_rows.scaffold_group) & set(eval_rows.scaffold_group))
                    if parent_overlap or scaffold_overlap:
                        raise ValueError(f"Fold leakage: {task} {treatment} fold={fold}")
                    mean, std = float(train_parent.parent_target.mean()), float(train_parent.parent_target.std(ddof=0))
                    if not np.isfinite(std) or std <= 0:
                        raise ValueError(f"Invalid fold target scale: {task} {fold}")
                    estimator = make_estimator("extra_trees", int(args.seed + fold), args.threads, small_budget=True,
                                               parameters=spec["parameters"])
                    x_train = matrix[train_parent.feature_index.to_numpy(int)]
                    x_eval = matrix[eval_parent.feature_index.to_numpy(int)]
                    estimator.fit(x_train, (train_parent.parent_target.to_numpy(float) - mean) / std)
                    prediction_z = np.asarray(estimator.predict(x_eval)).reshape(-1)
                    prediction_raw = prediction_z * std + mean
                    global_mean, global_std = float(task_frame.train_mean.iloc[0]), float(task_frame.train_std_population.iloc[0])
                    parent_prediction = pd.DataFrame({"molecule_id": eval_parent.molecule_id,
                                                      "predicted_interface_target": (prediction_raw - global_mean) / global_std,
                                                      "ipc_extreme_parent": np.abs(ipc_values[eval_parent.feature_index.to_numpy(int)]) > 1e15})
                    table = eval_rows[["row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id", "sensitivity_subset",
                                      "target_value", "train_mean", "train_std_population", "reporting_inverse"]].merge(parent_prediction, on="molecule_id", validate="many_to_one")
                    table["interface_target"] = (table.target_value - table.train_mean) / table.train_std_population
                    table["candidate_id"], table["treatment"], table["algorithm"], table["feature_set"] = candidate, treatment, "extra_trees", spec["view"]
                    table["evaluation_partition"] = f"fold_{fold}"
                    predictions.append(table); all_eval.append(table)
                    runs.append({"task_id": task, "candidate_id": candidate, "treatment": treatment, "outer_fold": fold,
                                 "train_parents": len(train_parent), "evaluation_parents": len(eval_parent),
                                 "parameters": json.dumps(spec["parameters"], sort_keys=True), "fold_target_mean": mean,
                                 "fold_target_std_population": std})
                    isolations.append({"task_id": task, "candidate_id": candidate, "outer_fold": fold,
                                       "parent_overlap": parent_overlap, "scaffold_overlap": scaffold_overlap,
                                       "evaluation_targets_used_in_preprocessing": False})
                    if fold == 0:
                        with tempfile.TemporaryDirectory(dir=out) as temp:
                            path = Path(temp) / "model.joblib"
                            joblib.dump(estimator, path, compress=3)
                            after = np.asarray(joblib.load(path).predict(x_eval)).reshape(-1)
                        difference = float(np.max(np.abs(prediction_z - after)))
                        if difference > 1e-8:
                            raise ValueError(f"Reload mismatch: {task} {treatment}")
                        reloads.append({"task_id": task, "candidate_id": candidate, "max_abs_difference": difference})
                oof = pd.concat(all_eval, ignore_index=True)
                metric = metric_row(oof, oof.predicted_interface_target.to_numpy(), task, "extra_trees", spec["view"], "train_cv")
                metric["candidate_id"], metric["treatment"] = candidate, treatment
                metrics.append(metric)
                extreme = oof.loc[oof.ipc_extreme_parent].copy()
                if not extreme.empty:
                    subset = metric_row(extreme, extreme.predicted_interface_target.to_numpy(), task, "extra_trees", spec["view"], "train_cv_ipc_extreme")
                    subset["candidate_id"], subset["treatment"], subset["subset"] = candidate, treatment, "ipc_extreme_parent"
                    metrics.append(subset)
        metrics_frame = pd.DataFrame(metrics)
        metrics_frame["rank_within_task_scope"] = metrics_frame.groupby(["task_id", "evaluation_scope"]).primary_metric.rank(method="min").astype(int)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        if prediction_frame.duplicated(["task_id", "candidate_id", "row_id"]).any():
            raise ValueError("Duplicate Ipc OOF prediction")
        expected_rows = records.groupby("task_id").row_id.nunique()
        coverage = prediction_frame.groupby(["task_id", "candidate_id"]).row_id.nunique()
        if any(count != expected_rows[task] for (task, _), count in coverage.items()):
            raise ValueError("Incomplete Ipc OOF coverage")
        prediction_frame.to_csv(out / "predictions.csv", index=False)
        metrics_frame.to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(runs).to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(reloads).to_csv(out / "reload_checks.csv", index=False)
        pd.DataFrame(isolations).to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(registry_rows).to_csv(out / "candidate_registry.csv", index=False)
        summary = {"tasks": len(leader_specs), "treatments": len(treatments), "fits": len(runs),
                   "ipc_descriptor_index_rdkit2d": descriptor_index, "limited_smoke": limited,
                   "validation_target_file_opened": False, "validation_rows_evaluated": False,
                   "test_labels_read": False, "model_selection_authorized": not limited,
                   "selection_scope": "train_CV_Ipc_feature_ablation_only" if not limited else "engineering_only"}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Ipc feature ablation\n\nExact Stage-A ExtraTrees leader parameters are compared with Ipc as registered, removed, or transformed by log1p. "
            "This is a train-CV feature sensitivity only; it does not revise Papp labels or open validation/test.\n", encoding="utf-8")
        finish_stage(out, "stl_ipc_feature_ablation", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json"),
        }, partial=limited, **summary)
    print(f"STL Ipc feature ablation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
