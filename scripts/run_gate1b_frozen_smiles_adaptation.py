#!/usr/bin/env python3
"""Run corrected, train-only frozen-SMILES adaptation probes.

Each held-out fold fits target centering/scaling from collapsed training parents.
The bounded comparison contains an exact Stage-A structure control, the same
estimator/parameters on MoLFormer, two PCA-Ridge heads and two small MLP heads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records, metric_row
from stl_benchmark_common import feature_view, make_estimator


PCA_CONFIGS = [
    {"max_components": 32, "ridge_alpha": 10.0},
    {"max_components": 128, "ridge_alpha": 100.0},
]
MLP_CONFIGS = [
    {"hidden_layer_sizes": (128,), "alpha": 1e-3, "learning_rate_init": 3e-4,
     "max_iter": 300, "early_stopping": True, "n_iter_no_change": 20},
    {"hidden_layer_sizes": (256, 64), "alpha": 1e-3, "learning_rate_init": 3e-4,
     "max_iter": 300, "early_stopping": True, "n_iter_no_change": 20},
]
VIEW_SCREEN = {
    "rdkit2d": "stl_stageA_first_batch_rdkit2d_v1",
    "ecfp4": "stl_stageA_ecfp4_v1",
    "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1",
}


def parent_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.groupby(["molecule_id", "feature_index", "embedding_row"], as_index=False)
            .agg(parent_target=("target_value", "mean"), source_records=("row_id", "size")))


def make_pca_ridge(config: dict, n_train: int, seed: int) -> tuple[Pipeline, int]:
    components = min(int(config["max_components"]), n_train - 1, 768)
    if components < 2:
        raise ValueError("PCA-Ridge requires at least three training parents")
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, svd_solver="randomized", random_state=seed)),
        ("model", Ridge(alpha=float(config["ridge_alpha"]), solver="lsqr", tol=1e-6)),
    ])
    return estimator, components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/public_development/molformer_parent_embedding_cache_v1")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/gate1b_frozen_smiles_adaptation_v1")
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Engineering smoke only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    screen_paths = {view: args.benchmarks / name for view, name in VIEW_SCREEN.items()}
    required = [
        args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
        args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
        args.embedding_cache / "complete.json", args.embedding_cache / "embeddings_float32.npy",
        args.embedding_cache / "parent_embedding_index.csv",
        args.stageA_audit / "complete.json", args.stageA_audit / "endpoint_decisions.csv",
    ] + [path / "complete.json" for path in screen_paths.values()]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    cache_meta = verify_stage(args.embedding_cache, "molformer_parent_embedding_cache")
    audit_meta = verify_stage(args.stageA_audit, "stl_stageA_three_view_audit")
    screen_meta = {view: verify_stage(path, "stl_benchmark_screen") for view, path in screen_paths.items()}
    if not train_meta.get("fold_local_target_standardization_required") or train_meta.get("fixed_validation_targets_published"):
        raise ValueError("Train-only/fold-local preprocessing contract is not active")
    if any(meta.get("test_labels_read") for meta in [train_meta, stl_meta, cache_meta, audit_meta, *screen_meta.values()]):
        raise ValueError("An upstream stage reports test-label access")
    if audit_meta.get("validation_labels_read") or cache_meta.get("validation_labels_read"):
        raise ValueError("Adaptation screen requires fixed validation to remain closed")

    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records["feature_index"] = records.feature_index.astype(int)
    records["inner_fold_id"] = records.inner_fold_id.astype(int)
    records["interface_target"] = (records.target_value - records.train_mean) / records.train_std_population
    records = limit_records(records, args.max_parents_per_task_fold)
    if records.task_id.nunique() != 6 or not records.inner_fold_id.between(0, 4).all():
        raise ValueError("Train-only input does not contain six tasks and folds 0--4")

    index = pd.read_csv(args.embedding_cache / "parent_embedding_index.csv", dtype={"molecule_id": str})
    records = records.merge(
        index[["molecule_id", "embedding_row", "feature_index"]].rename(columns={"feature_index": "cache_feature_index"}),
        on="molecule_id", how="left", validate="many_to_one",
    )
    if records.embedding_row.isna().any() or not records.feature_index.eq(records.cache_feature_index).all():
        raise ValueError("Train protocol and frozen embedding cache parent mappings differ")
    records["embedding_row"] = records.embedding_row.astype(int)
    embedding = np.load(args.embedding_cache / "embeddings_float32.npy", mmap_mode="r")
    structure_npz = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    structure_cache = {key: structure_npz[key] for key in structure_npz.files}
    if embedding.shape != (7939, 768) or not np.isfinite(embedding).all():
        raise ValueError("Frozen embedding matrix is invalid")

    leaders = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv")
    run_tables = {view: pd.read_csv(path / "run_registry.csv") for view, path in screen_paths.items()}
    leader_specs = {}
    for leader in leaders.itertuples(index=False):
        rows = run_tables[leader.stageA_leading_feature_view]
        row = rows.loc[
            rows.task_id.eq(leader.task_id) & rows.candidate_id.eq(leader.stageA_leading_candidate)
        ].iloc[0]
        leader_specs[leader.task_id] = {
            "endpoint": leader.endpoint, "feature_view": leader.stageA_leading_feature_view,
            "algorithm": row.algorithm, "parameters": json.loads(row.parameters),
            "historical_candidate_id": leader.stageA_leading_candidate,
            "historical_primary_metric": float(leader.stageA_leading_primary_metric),
        }
    if set(leader_specs) != set(records.task_id.unique()):
        raise ValueError("Stage-A leader specs do not cover all train-only tasks")

    candidates_per_task = 6
    expected_fits = 6 * candidates_per_task * 5
    if args.check_only:
        print(
            f"Frozen-SMILES adaptation ready: records={len(records)} parents={records.molecule_id.nunique()} "
            f"tasks=6 candidates_per_task={candidates_per_task} fits={expected_fits} "
            f"limited_smoke={bool(args.max_parents_per_task_fold)}"
        )
        return

    limited = bool(args.max_parents_per_task_fold)
    with stage_output(args.output) as out:
        predictions, metrics, runs, reloads, fold_contracts, isolation_rows, candidate_registry = [], [], [], [], [], [], []
        for task, task_frame in records.groupby("task_id", sort=True):
            spec = leader_specs[task]
            structure_x = feature_view(structure_cache, spec["feature_view"])
            candidate_specs = [
                {"candidate_id": "structure_control__stageA_leader", "family": "structure_control",
                 "input": "structure", "algorithm": spec["algorithm"], "parameters": spec["parameters"]},
                {"candidate_id": "molformer_matched__stageA_leader", "family": "molformer_matched",
                 "input": "embedding", "algorithm": spec["algorithm"], "parameters": spec["parameters"]},
                *[
                    {"candidate_id": f"pca_ridge__c{i:02d}", "family": "pca_ridge", "input": "embedding",
                     "algorithm": "pca_ridge", "parameters": config}
                    for i, config in enumerate(PCA_CONFIGS)
                ],
                *[
                    {"candidate_id": f"mlp__c{i:02d}", "family": "mlp", "input": "embedding",
                     "algorithm": "mlp", "parameters": config}
                    for i, config in enumerate(MLP_CONFIGS)
                ],
            ]
            for candidate in candidate_specs:
                candidate_registry.append({
                    "task_id": task, "endpoint": spec["endpoint"], **candidate,
                    "parameters": json.dumps(candidate["parameters"], sort_keys=True, default=list),
                    "stageA_feature_view": spec["feature_view"],
                    "historical_stageA_candidate_id": spec["historical_candidate_id"],
                    "historical_stageA_primary_metric": spec["historical_primary_metric"],
                })
                all_eval = []
                for fold in range(5):
                    train_rows = task_frame.loc[task_frame.inner_fold_id.ne(fold)].copy()
                    eval_rows = task_frame.loc[task_frame.inner_fold_id.eq(fold)].copy()
                    train_parent = parent_table(train_rows)
                    eval_parent = parent_table(eval_rows)
                    parent_overlap = len(set(train_parent.molecule_id) & set(eval_parent.molecule_id))
                    scaffold_overlap = len(set(train_rows.scaffold_group) & set(eval_rows.scaffold_group))
                    if parent_overlap or scaffold_overlap:
                        raise ValueError(f"Fold leakage: {task} fold={fold}")
                    fold_mean = float(train_parent.parent_target.mean())
                    fold_std = float(train_parent.parent_target.std(ddof=0))
                    if not np.isfinite(fold_std) or fold_std <= 0:
                        raise ValueError(f"Invalid fold-local target SD: {task} fold={fold}")
                    y_train = (train_parent.parent_target.to_numpy(float) - fold_mean) / fold_std
                    effective_components = pd.NA
                    seed = int(args.seed + fold)
                    if candidate["family"] == "pca_ridge":
                        estimator, effective_components = make_pca_ridge(candidate["parameters"], len(train_parent), seed)
                    else:
                        estimator = make_estimator(
                            candidate["algorithm"], seed, args.threads, small_budget=True,
                            parameters=candidate["parameters"],
                        )
                    if candidate["input"] == "structure":
                        x_train = structure_x[train_parent.feature_index.to_numpy(int)]
                        x_eval = structure_x[eval_parent.feature_index.to_numpy(int)]
                    else:
                        x_train = embedding[train_parent.embedding_row.to_numpy(int)]
                        x_eval = embedding[eval_parent.embedding_row.to_numpy(int)]
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator.fit(x_train, y_train)
                    prediction_z = np.asarray(estimator.predict(x_eval)).reshape(-1)
                    transformed_prediction = prediction_z * fold_std + fold_mean
                    global_mean = float(task_frame.train_mean.iloc[0])
                    global_std = float(task_frame.train_std_population.iloc[0])
                    global_interface_prediction = (transformed_prediction - global_mean) / global_std
                    if not np.isfinite(global_interface_prediction).all():
                        raise ValueError(f"Non-finite adaptation prediction: {task} {candidate['candidate_id']} fold={fold}")
                    parent_prediction = pd.DataFrame({
                        "molecule_id": eval_parent.molecule_id,
                        "predicted_interface_target": global_interface_prediction,
                    })
                    table = eval_rows[[
                        "row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id",
                        "sensitivity_subset", "target_value", "train_mean", "train_std_population",
                        "reporting_inverse", "interface_target",
                    ]].merge(parent_prediction, on="molecule_id", validate="many_to_one")
                    table["algorithm"] = candidate["family"]
                    table["candidate_id"] = candidate["candidate_id"]
                    table["feature_set"] = "stageA_structure" if candidate["input"] == "structure" else "frozen_molformer"
                    table["evaluation_partition"] = f"fold_{fold}"
                    table["fold_target_mean"] = fold_mean
                    table["fold_target_std_population"] = fold_std
                    predictions.append(table)
                    all_eval.append(table)
                    warning_text = " | ".join(sorted({str(item.message) for item in caught}))
                    runs.append({
                        "task_id": task, "candidate_id": candidate["candidate_id"], "family": candidate["family"],
                        "partition": f"fold_{fold}", "train_parents": len(train_parent),
                        "evaluation_parents": len(eval_parent), "fold_target_mean": fold_mean,
                        "fold_target_std_population": fold_std,
                        "effective_pca_components": effective_components,
                        "warnings": warning_text, "warning_count": len(caught),
                        "parameters": json.dumps(candidate["parameters"], sort_keys=True, default=list),
                    })
                    fold_contracts.append({
                        "task_id": task, "candidate_id": candidate["candidate_id"], "inner_fold_id": fold,
                        "target_fit_unit": "collapsed_parent", "target_fit_parents": len(train_parent),
                        "fold_target_mean": fold_mean, "fold_target_std_population": fold_std,
                        "evaluation_targets_used_in_preprocessing": False,
                    })
                    isolation_rows.append({
                        "task_id": task, "candidate_id": candidate["candidate_id"], "inner_fold_id": fold,
                        "parent_overlap": parent_overlap, "scaffold_overlap": scaffold_overlap,
                    })
                    if args.verify_reload and fold == 0:
                        with tempfile.TemporaryDirectory(dir=out) as temporary:
                            path = Path(temporary) / "model.joblib"
                            joblib.dump(estimator, path, compress=3)
                            after_z = np.asarray(joblib.load(path).predict(x_eval)).reshape(-1)
                        difference = float(np.max(np.abs(prediction_z - after_z)))
                        if difference > 1e-8:
                            raise ValueError(f"Reload mismatch: {task} {candidate['candidate_id']}")
                        reloads.append({
                            "task_id": task, "candidate_id": candidate["candidate_id"],
                            "max_abs_difference": difference, "passed": True,
                        })
                oof = pd.concat(all_eval, ignore_index=True)
                row = metric_row(oof, oof.predicted_interface_target.to_numpy(), task, candidate["family"],
                                 "stageA_structure" if candidate["input"] == "structure" else "frozen_molformer",
                                 "train_cv")
                row["candidate_id"] = candidate["candidate_id"]
                row["parameters"] = json.dumps(candidate["parameters"], sort_keys=True, default=list)
                metrics.append(row)

        prediction_frame = pd.concat(predictions, ignore_index=True)
        metric_frame = pd.DataFrame(metrics)
        run_frame = pd.DataFrame(runs)
        if prediction_frame.duplicated(["task_id", "candidate_id", "row_id"]).any():
            raise ValueError("Duplicate OOF adaptation prediction")
        expected = records.groupby("task_id").row_id.nunique()
        coverage = prediction_frame.groupby(["task_id", "candidate_id"]).row_id.nunique()
        for (task, _), count in coverage.items():
            if count != expected[task]:
                raise ValueError(f"Incomplete OOF adaptation coverage: {task} {count} != {expected[task]}")
        if len(run_frame) != expected_fits or run_frame.groupby(["task_id", "candidate_id"]).partition.nunique().ne(5).any():
            raise ValueError("Adaptation run registry is incomplete")
        if args.verify_reload and len(reloads) != 6 * candidates_per_task:
            raise ValueError("Each task/candidate requires one reload check")
        metric_frame["rank_within_task"] = metric_frame.groupby("task_id").primary_metric.rank(method="min").astype(int)
        prediction_frame.to_csv(out / "predictions.csv", index=False)
        metric_frame.to_csv(out / "metrics.csv", index=False)
        metric_frame.sort_values(["task_id", "rank_within_task", "candidate_id"]).to_csv(out / "candidate_ranking.csv", index=False)
        run_frame.to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(reloads).to_csv(out / "reload_checks.csv", index=False)
        pd.DataFrame(fold_contracts).to_csv(out / "fold_target_preprocessing.csv", index=False)
        pd.DataFrame(isolation_rows).to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(candidate_registry).to_csv(out / "candidate_registry.csv", index=False)
        summary = {
            "tasks": 6, "candidates_per_task": candidates_per_task, "fits": len(run_frame),
            "oof_predictions": len(prediction_frame), "limited_smoke": limited,
            "fold_local_target_standardization": True,
            "validation_target_file_opened": False, "validation_rows_evaluated": False,
            "test_labels_read": False, "model_selection_authorized": not limited,
            "selection_scope": "Gate1B_train_CV_adaptation_only" if not limited else "engineering_only",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Corrected frozen-MoLFormer adaptation screen\n\nTrain-only input with parent-level fold-local target standardization. "
            "The endpoint-specific structure control and matched MoLFormer head use identical algorithms and parameters; PCA-Ridge and small MLP test representation adaptation. "
            "Limited runs are engineering smoke and cannot select models.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate1b_frozen_smiles_adaptation", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "embedding_cache_complete_sha256": sha256(args.embedding_cache / "complete.json"),
            "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json"),
            **{f"{view}_screen_complete_sha256": sha256(path / "complete.json") for view, path in screen_paths.items()},
        }, partial=limited, **summary)
    print(f"Gate 1B frozen-SMILES adaptation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
