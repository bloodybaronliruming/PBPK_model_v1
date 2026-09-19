#!/usr/bin/env python3
"""Run bounded train-CV probes on the frozen MoLFormer parent cache.

Only frozen training folds are evaluated.  Ridge, ExtraTrees and LightGBM use
two preregistered configurations each.  A row-limited execution is an
engineering smoke and is permanently ineligible for model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records, metric_row
from stl_benchmark_common import make_estimator


ALGORITHMS = ("ridge", "extra_trees", "lightgbm")
PARAMETER_SPACE = {
    "ridge": [
        {"alpha": 10.0},
        {"alpha": 1000.0},
    ],
    "extra_trees": [
        {"n_estimators": 180, "max_features": 0.5, "min_samples_leaf": 2},
        {"n_estimators": 180, "max_features": 0.7, "min_samples_leaf": 5},
    ],
    "lightgbm": [
        {"n_estimators": 180, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 20},
        {"n_estimators": 180, "num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 50},
    ],
}


def collapse_training_parents(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.groupby(["molecule_id", "embedding_row"], as_index=False)
            .agg(interface_target=("interface_target", "mean"), source_records=("row_id", "size")))


def partition_audit(task: str, fold: int, train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    train_parents = set(train.molecule_id)
    eval_parents = set(evaluation.molecule_id)
    train_scaffolds = set(train.scaffold_group)
    eval_scaffolds = set(evaluation.scaffold_group)
    row = {
        "task_id": task,
        "inner_fold_id": fold,
        "train_parents": len(train_parents),
        "evaluation_parents": len(eval_parents),
        "parent_overlap": len(train_parents & eval_parents),
        "train_scaffolds": len(train_scaffolds),
        "evaluation_scaffolds": len(eval_scaffolds),
        "scaffold_overlap": len(train_scaffolds & eval_scaffolds),
    }
    if row["parent_overlap"] or row["scaffold_overlap"]:
        raise ValueError(f"Frozen fold leakage: {task} fold={fold} {row}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--provider-protocol", type=Path, default=ROOT / "data/public_development/algorithm_landscape_smiles_provider_v1")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/public_development/molformer_parent_embedding_cache_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/gate1b_frozen_smiles_probe_v1")
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Engineering smoke only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.stl_protocol / "complete.json", args.stl_protocol / "benchmark_records.csv",
        args.provider_protocol / "complete.json", args.provider_protocol / "smiles_ablation_plan.csv",
        args.embedding_cache / "complete.json", args.embedding_cache / "embeddings_float32.npy",
        args.embedding_cache / "parent_embedding_index.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    stl = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    provider = verify_stage(args.provider_protocol, "algorithm_landscape_smiles_provider_protocol")
    cache = verify_stage(args.embedding_cache, "molformer_parent_embedding_cache")
    if any(meta.get("test_labels_read") for meta in (stl, provider, cache)):
        raise ValueError("An upstream stage reports protected test-label access")
    if provider.get("fixed_validation_authorized") or provider.get("validation_labels_read") or cache.get("validation_labels_read"):
        raise ValueError("Frozen-SMILES train-CV probes require fixed validation to remain closed")
    if int(cache.get("parents", 0)) != 7939 or int(cache.get("embedding_dimension", 0)) != 768:
        raise ValueError("Unexpected frozen embedding-cache dimensions")
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    records = pd.read_csv(args.stl_protocol / "benchmark_records.csv", dtype=str, keep_default_na=False)
    # No validation row is carried beyond this boundary or evaluated by this program.
    records = records.loc[records.split.eq("train")].copy()
    for column in ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records["feature_index"] = records.feature_index.astype(int)
    records["inner_fold_id"] = records.inner_fold_id.astype(int)
    records["interface_target"] = (records.target_value - records.train_mean) / records.train_std_population
    records = limit_records(records, args.max_parents_per_task_fold)
    if records.task_id.nunique() != 6 or not records.inner_fold_id.between(0, 4).all():
        raise ValueError("Expected six authorized tasks and frozen folds 0--4")

    index = pd.read_csv(args.embedding_cache / "parent_embedding_index.csv", dtype={"molecule_id": str})
    if len(index) != 7939 or index.molecule_id.duplicated().any() or not index.valid_embedding.all():
        raise ValueError("Embedding index is incomplete, duplicated or contains invalid rows")
    joined = records.merge(
        index[["molecule_id", "embedding_row", "feature_index"]].rename(columns={"feature_index": "cache_feature_index"}),
        on="molecule_id", how="left", validate="many_to_one",
    )
    if joined.embedding_row.isna().any() or not joined.feature_index.eq(joined.cache_feature_index).all():
        raise ValueError("STL protocol and embedding cache do not have identical parent mappings")
    joined["embedding_row"] = joined.embedding_row.astype(int)
    matrix = np.load(args.embedding_cache / "embeddings_float32.npy", mmap_mode="r")
    if matrix.shape != (7939, 768) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix shape, dtype or finite-value check failed")

    if args.check_only:
        limited = bool(args.max_parents_per_task_fold)
        print(
            f"Frozen-SMILES probe ready: records={len(joined)} parents={joined.molecule_id.nunique()} "
            f"tasks=6 candidates=6 fits={6 * 6 * 5} limited_smoke={limited}"
        )
        return

    limited = bool(args.max_parents_per_task_fold)
    with stage_output(args.output) as out:
        predictions, metrics, runs, reloads, audits = [], [], [], [], []
        for task, task_frame in joined.groupby("task_id", sort=True):
            folds = sorted(task_frame.inner_fold_id.unique())
            if folds != [0, 1, 2, 3, 4]:
                raise ValueError(f"Task lacks the five frozen folds: {task} {folds}")
            evaluation_parts = []
            for fold in folds:
                evaluation = task_frame.loc[task_frame.inner_fold_id.eq(fold)].copy()
                training_rows = task_frame.loc[task_frame.inner_fold_id.ne(fold)].copy()
                audits.append(partition_audit(task, fold, training_rows, evaluation))
                evaluation_parts.append((fold, training_rows, evaluation))
            for algorithm in ALGORITHMS:
                for candidate_index, parameters in enumerate(PARAMETER_SPACE[algorithm]):
                    candidate_id = f"{algorithm}__molformer__c{candidate_index:02d}"
                    all_evaluation = []
                    for fold, training_rows, evaluation in evaluation_parts:
                        training = collapse_training_parents(training_rows)
                        estimator = make_estimator(
                            algorithm, args.seed, args.threads, small_budget=True, parameters=parameters
                        )
                        estimator.fit(
                            matrix[training.embedding_row.to_numpy(int)],
                            training.interface_target.to_numpy(float),
                        )
                        prediction = np.asarray(
                            estimator.predict(matrix[evaluation.embedding_row.to_numpy(int)])
                        ).reshape(-1)
                        if prediction.shape != (len(evaluation),) or not np.isfinite(prediction).all():
                            raise ValueError(f"Non-finite or misaligned predictions: {task} {candidate_id} fold={fold}")
                        table = evaluation[[
                            "row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id",
                            "sensitivity_subset", "target_value", "train_mean", "train_std_population",
                            "reporting_inverse", "interface_target",
                        ]].copy()
                        table["predicted_interface_target"] = prediction
                        table["algorithm"] = algorithm
                        table["candidate_id"] = candidate_id
                        table["feature_set"] = "frozen_molformer"
                        table["evaluation_partition"] = f"fold_{fold}"
                        predictions.append(table)
                        all_evaluation.append(table)
                        runs.append({
                            "task_id": task, "algorithm": algorithm, "candidate_id": candidate_id,
                            "feature_set": "frozen_molformer", "partition": f"fold_{fold}",
                            "train_parents": len(training), "evaluation_parents": evaluation.molecule_id.nunique(),
                            "parameters": json.dumps(parameters, sort_keys=True),
                        })
                        if args.verify_reload and fold == folds[0]:
                            folder = out / "reload_checks"
                            folder.mkdir(exist_ok=True)
                            path = folder / f"{task}__{candidate_id}.joblib"
                            joblib.dump(estimator, path, compress=3)
                            after = np.asarray(joblib.load(path).predict(
                                matrix[evaluation.embedding_row.to_numpy(int)]
                            )).reshape(-1)
                            difference = float(np.max(np.abs(prediction - after)))
                            if difference > 1e-8:
                                raise ValueError(f"Reload mismatch: {task} {candidate_id}")
                            reloads.append({
                                "task_id": task, "algorithm": algorithm, "candidate_id": candidate_id,
                                "max_abs_difference": difference, "passed": True,
                            })
                    oof = pd.concat(all_evaluation, ignore_index=True)
                    row = metric_row(
                        oof, oof.predicted_interface_target.to_numpy(), task, algorithm,
                        "frozen_molformer", "train_cv",
                    )
                    row["candidate_id"] = candidate_id
                    row["parameters"] = json.dumps(parameters, sort_keys=True)
                    metrics.append(row)

        prediction_frame = pd.concat(predictions, ignore_index=True)
        expected_rows = joined.groupby("task_id").row_id.nunique()
        observed = prediction_frame.groupby(["task_id", "candidate_id"]).row_id.nunique()
        for (task, _), count in observed.items():
            if count != expected_rows[task]:
                raise ValueError(f"Incomplete OOF coverage for {task}: {count} != {expected_rows[task]}")
        if prediction_frame.duplicated(["task_id", "candidate_id", "row_id"]).any():
            raise ValueError("Duplicate OOF row within a task/candidate")
        run_frame = pd.DataFrame(runs)
        if run_frame.groupby(["task_id", "candidate_id"]).partition.nunique().ne(5).any():
            raise ValueError("A task/candidate does not have five completed folds")
        if args.verify_reload and len(reloads) != 6 * len(ALGORITHMS) * 2:
            raise ValueError("Each task/candidate requires one reload check")

        metric_frame = pd.DataFrame(metrics)
        metric_frame["rank_within_task"] = metric_frame.groupby("task_id").primary_metric.rank(
            method="min", ascending=True
        ).astype(int)
        prediction_frame.to_csv(out / "predictions.csv", index=False)
        metric_frame.to_csv(out / "metrics.csv", index=False)
        metric_frame.sort_values(["task_id", "rank_within_task", "candidate_id"]).to_csv(
            out / "candidate_ranking.csv", index=False
        )
        run_frame.to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(reloads).to_csv(out / "reload_checks.csv", index=False)
        pd.DataFrame(audits).drop_duplicates(["task_id", "inner_fold_id"]).to_csv(
            out / "fold_isolation_audit.csv", index=False
        )
        pd.DataFrame([
            {"algorithm": algorithm, "candidate_id": f"{algorithm}__molformer__c{index:02d}",
             "parameters": json.dumps(parameters, sort_keys=True)}
            for algorithm in ALGORITHMS for index, parameters in enumerate(PARAMETER_SPACE[algorithm])
        ]).to_csv(out / "preregistered_parameter_space.csv", index=False)
        (out / "README.md").write_text(
            "# Gate 1B frozen-MoLFormer bounded probes\n\n"
            "Ridge, ExtraTrees and LightGBM are evaluated on the immutable parent cache with the same six tasks, five frozen folds and endpoint metrics as Stage A. "
            "A limited run is an engineering smoke and cannot select a model. Fixed validation and test are not evaluated.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "gate1b_frozen_smiles_screen",
            inputs={
                "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
                "provider_protocol_complete_sha256": sha256(args.provider_protocol / "complete.json"),
                "embedding_cache_complete_sha256": sha256(args.embedding_cache / "complete.json"),
            },
            algorithms=list(ALGORITHMS), candidates_per_algorithm=2,
            tasks=6, completed_fits=len(run_frame), candidate_task_evaluations=len(metric_frame),
            evaluation_scope="train_cv", feature_set="frozen_molformer",
            seed=args.seed, limited_smoke=limited,
            model_selection_authorized=not limited,
            selection_scope="Gate1B_train_CV_incremental_gate_only" if not limited else "engineering_only",
            validation_labels_read=False, test_labels_read=False, partial=limited,
        )
    print(f"Gate 1B frozen-SMILES probe: {args.output}")


if __name__ == "__main__":
    run_cli(main)
