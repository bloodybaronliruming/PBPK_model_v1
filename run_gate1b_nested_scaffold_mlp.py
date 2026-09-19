#!/usr/bin/env python3
"""Re-run small structure and MoLFormer MLPs with nested scaffold early stopping.

This diagnostic isolates the former sklearn random internal-validation choice.
It excludes fu, which has a separate physical-label/bounded-output experiment.
All external evaluation remains five-fold train CV; fixed validation/test stay
closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from gate1b_torch_common import fit_mlp, fit_mlp_fixed_epochs, inner_scaffold_masks, predict_checkpoint
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records, metric_row
from stl_benchmark_common import feature_view

EXCLUDED_TASK = "fu__human__plasma"
VIEW_SCREEN = {"rdkit2d": "stl_stageA_first_batch_rdkit2d_v1", "ecfp4": "stl_stageA_ecfp4_v1", "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1"}
SEEDS = (17, 29, 43)


def parents(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["molecule_id", "feature_index", "embedding_row"], as_index=False).agg(
        parent_target=("target_value", "mean"), source_records=("row_id", "size"), scaffold_group=("scaffold_group", "first"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/public_development/molformer_parent_embedding_cache_v1")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/gate1b_nested_scaffold_mlp_v1")
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Engineering smoke only")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    screen_paths = {view: args.benchmarks / name for view, name in VIEW_SCREEN.items()}
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv", args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz", args.embedding_cache / "complete.json", args.embedding_cache / "embeddings_float32.npy", args.embedding_cache / "parent_embedding_index.csv", args.stageA_audit / "complete.json", args.stageA_audit / "endpoint_decisions.csv"] + [path / "complete.json" for path in screen_paths.values()]
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
        raise ValueError("Fixed validation must remain closed")

    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records.feature_index = records.feature_index.astype(int)
    records.inner_fold_id = records.inner_fold_id.astype(int)
    records = limit_records(records.loc[~records.task_id.eq(EXCLUDED_TASK)].copy(), args.max_parents_per_task_fold)
    if records.task_id.nunique() != 5 or not records.inner_fold_id.between(0, 4).all():
        raise ValueError("Expected five non-fu tasks across frozen folds")
    index = pd.read_csv(args.embedding_cache / "parent_embedding_index.csv", dtype={"molecule_id": str})
    records = records.merge(index[["molecule_id", "embedding_row", "feature_index"]].rename(columns={"feature_index": "cache_feature_index"}), on="molecule_id", how="left", validate="many_to_one")
    if records.embedding_row.isna().any() or not records.feature_index.eq(records.cache_feature_index).all():
        raise ValueError("Train protocol and frozen embedding cache mappings differ")
    records.embedding_row = records.embedding_row.astype(int)
    embedding = np.load(args.embedding_cache / "embeddings_float32.npy", mmap_mode="r")
    structure_npz = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    structure_cache = {key: structure_npz[key] for key in structure_npz.files}
    leaders = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv").set_index("task_id")
    if not set(records.task_id).issubset(set(leaders.index)):
        raise ValueError("Stage-A leaders do not cover nested MLP tasks")
    expected_selection_fits = 5 * 2 * 5 * len(SEEDS)
    if args.check_only:
        print(f"Nested-scaffold MLP ready: records={len(records)} tasks=5 selection_fits={expected_selection_fits} total_neural_fits={2*expected_selection_fits} device={'cuda' if torch.cuda.is_available() else 'cpu'}")
        return

    limited = bool(args.max_parents_per_task_fold)
    with stage_output(args.output) as out:
        predictions, seed_predictions, metrics, runs, curves, isolation, reloads = [], [], [], [], [], [], []
        for task, task_frame in records.groupby("task_id", sort=True):
            structure_view = str(leaders.loc[task, "stageA_leading_feature_view"])
            views = {"structure_nested_mlp": feature_view(structure_cache, structure_view), "molformer_nested_mlp": embedding}
            for candidate, matrix in views.items():
                candidate_eval = []
                for fold in range(5):
                    outer_train_rows = task_frame.loc[task_frame.inner_fold_id.ne(fold)].copy()
                    outer_eval_rows = task_frame.loc[task_frame.inner_fold_id.eq(fold)].copy()
                    train_parent, eval_parent = parents(outer_train_rows), parents(outer_eval_rows)
                    parent_overlap = len(set(train_parent.molecule_id) & set(eval_parent.molecule_id))
                    scaffold_overlap = len(set(train_parent.scaffold_group) & set(eval_parent.scaffold_group))
                    if parent_overlap or scaffold_overlap:
                        raise ValueError(f"Outer fold leakage: {task} {candidate} fold={fold}")
                    inner_train_mask, inner_valid_mask = inner_scaffold_masks(train_parent.scaffold_group, int(args.seed + 100 * fold))
                    x_train = matrix[train_parent.feature_index.to_numpy(int)] if candidate.startswith("structure") else matrix[train_parent.embedding_row.to_numpy(int)]
                    x_eval = matrix[eval_parent.feature_index.to_numpy(int)] if candidate.startswith("structure") else matrix[eval_parent.embedding_row.to_numpy(int)]
                    y_outer = train_parent.parent_target.to_numpy(float)
                    mean_inner, std_inner = float(y_outer[inner_train_mask].mean()), float(y_outer[inner_train_mask].std(ddof=0))
                    mean_outer, std_outer = float(y_outer.mean()), float(y_outer.std(ddof=0))
                    if min(std_inner, std_outer) <= 0 or not np.isfinite([std_inner, std_outer]).all():
                        raise ValueError(f"Invalid target scale: {task} {candidate} fold={fold}")
                    seed_outputs = []
                    for offset in SEEDS:
                        seed = int(args.seed + 10000 * (list(sorted(records.task_id.unique())).index(task) + 1) + 100 * fold + offset)
                        selected, curve = fit_mlp(x_train[inner_train_mask], (y_outer[inner_train_mask] - mean_inner) / std_inner, x_train[inner_valid_mask], (y_outer[inner_valid_mask] - mean_inner) / std_inner, seed=seed, bounded_output=False, loss_name="mse", monitor_name="rmse", max_epochs=args.max_epochs, patience=args.patience)
                        final = fit_mlp_fixed_epochs(x_train, (y_outer - mean_outer) / std_outer, seed=seed, bounded_output=False, loss_name="mse", epochs=selected["best_epoch"])
                        raw_prediction = predict_checkpoint(final, x_eval) * std_outer + mean_outer
                        seed_outputs.append(raw_prediction)
                        curves.extend([{**row, "task_id": task, "candidate_id": candidate, "outer_fold": fold, "seed": seed, "best_epoch": selected["best_epoch"]} for row in curve])
                        runs.append({"task_id": task, "candidate_id": candidate, "outer_fold": fold, "seed": seed, "outer_train_parents": len(train_parent), "inner_train_parents": int(inner_train_mask.sum()), "inner_validation_parents": int(inner_valid_mask.sum()), "evaluation_parents": len(eval_parent), "best_epoch": selected["best_epoch"], "epochs_ran": selected["epochs_ran"], "best_internal_rmse": selected["best_internal_metric"], "final_refit_on_complete_outer_train": True, "device_used": final["device_used"]})
                        if fold == 0:
                            with tempfile.TemporaryDirectory(dir=out) as temp:
                                model_path = Path(temp) / f"{task}_{candidate}_{seed}.pt"
                                torch.save(final, model_path)
                                loaded = torch.load(model_path, map_location="cpu", weights_only=False)
                                difference = float(np.max(np.abs(raw_prediction - (predict_checkpoint(loaded, x_eval) * std_outer + mean_outer))))
                            if difference > 1e-7:
                                raise ValueError(f"Reload mismatch: {task} {candidate} {seed}")
                            reloads.append({"task_id": task, "candidate_id": candidate, "seed": seed, "max_abs_difference": difference})
                        seed_predictions.append(pd.DataFrame({"task_id": task, "candidate_id": candidate, "outer_fold": fold, "seed": seed, "molecule_id": eval_parent.molecule_id, "predicted_target_value": raw_prediction}))
                    averaged = np.mean(np.vstack(seed_outputs), axis=0)
                    global_mean, global_std = float(task_frame.train_mean.iloc[0]), float(task_frame.train_std_population.iloc[0])
                    parent_prediction = pd.DataFrame({"molecule_id": eval_parent.molecule_id, "predicted_interface_target": (averaged - global_mean) / global_std})
                    table = outer_eval_rows[["row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id", "sensitivity_subset", "target_value", "train_mean", "train_std_population", "reporting_inverse"]].merge(parent_prediction, on="molecule_id", validate="many_to_one")
                    table["interface_target"] = (table.target_value - table.train_mean) / table.train_std_population
                    table["algorithm"], table["candidate_id"], table["feature_set"], table["evaluation_partition"] = "nested_scaffold_mlp", candidate, "stageA_structure" if candidate.startswith("structure") else "frozen_molformer", f"fold_{fold}"
                    predictions.append(table); candidate_eval.append(table)
                    isolation.append({"task_id": task, "candidate_id": candidate, "outer_fold": fold, "outer_parent_overlap": parent_overlap, "outer_scaffold_overlap": scaffold_overlap, "inner_scaffold_overlap": 0, "evaluation_targets_used_in_inner_split": False})
                oof = pd.concat(candidate_eval, ignore_index=True)
                row = metric_row(oof, oof.predicted_interface_target.to_numpy(), task, "nested_scaffold_mlp", "stageA_structure" if candidate.startswith("structure") else "frozen_molformer", "train_cv")
                row["candidate_id"] = candidate; metrics.append(row)
        prediction_frame, metric_frame = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
        metric_frame["rank_within_task"] = metric_frame.groupby("task_id").primary_metric.rank(method="min").astype(int)
        prediction_frame.to_csv(out / "predictions.csv", index=False); metric_frame.to_csv(out / "metrics.csv", index=False)
        pd.concat(seed_predictions, ignore_index=True).to_csv(out / "neural_seed_predictions.csv", index=False)
        pd.DataFrame(runs).to_csv(out / "run_registry.csv", index=False); pd.DataFrame(curves).to_csv(out / "neural_learning_curves.csv", index=False)
        pd.DataFrame(isolation).to_csv(out / "fold_and_inner_scaffold_audit.csv", index=False); pd.DataFrame(reloads).to_csv(out / "reload_checks.csv", index=False)
        summary = {"tasks": 5, "representations": 2, "outer_folds": 5, "neural_seeds": len(SEEDS), "selection_fits": expected_selection_fits, "total_neural_fits": len(runs) * 2, "limited_smoke": limited, "internal_early_stopping": "deterministic_whole_scaffold_holdout_within_outer_train", "final_refit_on_complete_outer_train": True, "validation_target_file_opened": False, "validation_rows_evaluated": False, "test_labels_read": False, "model_selection_authorized": not limited}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text("# Nested-scaffold MLP diagnostic\n\nFive non-fu tasks compare structure and frozen-MoLFormer MLPs. Whole-scaffold internal validation selects the epoch, then a final model refits on all outer-training parents. This is train-CV-only and does not alter prior model decisions.\n", encoding="utf-8")
        finish_stage(out, "gate1b_nested_scaffold_mlp", inputs={"train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"), "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"), "embedding_cache_complete_sha256": sha256(args.embedding_cache / "complete.json"), "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json")}, partial=limited, **summary)
    print(f"Gate 1B nested-scaffold MLP: {args.output}")


if __name__ == "__main__":
    run_cli(main)
