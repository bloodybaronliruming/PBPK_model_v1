#!/usr/bin/env python3
"""Evaluate fu with physical parent labels and bounded, scaffold-contained neural heads.

This is an explicitly labelled diagnostic cohort, not a replacement for the
locked transformed-label Stage-A result.  All scoring is parent-level physical
MAE.  The runner reads only the train-only protocol and never opens validation
or test target files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch

from gate1b_torch_common import fit_mlp, fit_mlp_fixed_epochs, inner_scaffold_masks, predict_checkpoint
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records
from stl_benchmark_common import feature_view, make_estimator

TASK = "fu__human__plasma"
VIEW_SCREEN = {
    "rdkit2d": "stl_stageA_first_batch_rdkit2d_v1",
    "ecfp4": "stl_stageA_ecfp4_v1",
    "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1",
}
SEEDS = (17, 29, 43)


def physical_parent_table(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    local["physical_fu"] = 1.0 / (1.0 + np.exp(-local.target_value.to_numpy(float)))
    grouped = local.groupby(["molecule_id", "feature_index"], as_index=False)
    result = grouped.agg(
        parent_physical_fu=("physical_fu", "mean"), parent_logit_fu=("target_value", "mean"),
        source_records=("row_id", "size"), scaffold_group=("scaffold_group", "first"),
    )
    spread = grouped["physical_fu"].agg(["min", "max", "std"]).reset_index(drop=True)
    result["physical_min"] = spread["min"].to_numpy(float)
    result["physical_max"] = spread["max"].to_numpy(float)
    result["physical_std"] = spread["std"].fillna(0.0).to_numpy(float)
    result["logit_mean_backtransformed"] = 1.0 / (1.0 + np.exp(-result.parent_logit_fu.to_numpy(float)))
    result["aggregation_gap_abs"] = np.abs(result.parent_physical_fu - result.logit_mean_backtransformed)
    return result


def metric_row(prediction: pd.DataFrame, candidate_id: str) -> dict:
    error = prediction.predicted_physical_fu.to_numpy(float) - prediction.parent_physical_fu.to_numpy(float)
    return {
        "task_id": TASK, "candidate_id": candidate_id, "parents": len(prediction),
        "primary_metric": float(np.mean(np.abs(error))), "physical_mae": float(np.mean(np.abs(error))),
        "physical_rmse": float(np.sqrt(np.mean(error ** 2))),
        "spearman": float(prediction[["parent_physical_fu", "predicted_physical_fu"]].corr(method="spearman").iloc[0, 1]),
        "prediction_outside_unit_interval": int(((prediction.predicted_physical_fu < 0) | (prediction.predicted_physical_fu > 1)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/gate1b_fu_physical_alignment_v1")
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Engineering smoke only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    screen_paths = {view: args.benchmarks / name for view, name in VIEW_SCREEN.items()}
    required = [
        args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
        args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
        args.stageA_audit / "complete.json", args.stageA_audit / "endpoint_decisions.csv",
    ] + [path / "complete.json" for path in screen_paths.values()]
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

    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records["feature_index"] = records.feature_index.astype(int)
    records["inner_fold_id"] = records.inner_fold_id.astype(int)
    records = records.loc[records.task_id.eq(TASK)].copy()
    if records.empty or not records.inner_fold_id.between(0, 4).all():
        raise ValueError("Expected fu train records across five fixed folds")
    records = limit_records(records, args.max_parents_per_task_fold)

    leaders = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv")
    leader = leaders.loc[leaders.task_id.eq(TASK)].iloc[0]
    view = str(leader.stageA_leading_feature_view)
    run_registry = pd.read_csv(screen_paths[view] / "run_registry.csv")
    leader_row = run_registry.loc[
        run_registry.task_id.eq(TASK) & run_registry.candidate_id.eq(leader.stageA_leading_candidate)
    ].iloc[0]
    if leader_row.algorithm != "extra_trees":
        raise ValueError("Physical fu alignment v1 is pre-registered for the Stage-A ExtraTrees leader")
    leader_parameters = json.loads(leader_row.parameters)
    structure = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    x_all = feature_view({key: structure[key] for key in structure.files}, view)
    candidates = ["structure_logit_control", "structure_physical_extra_trees", "structure_bounded_huber_mlp", "structure_bounded_mae_mlp"]
    expected_fits = 5 * (2 + 2 * len(SEEDS))
    if args.check_only:
        print(f"fu physical alignment ready: records={len(records)} parents={records.molecule_id.nunique()} fits={expected_fits} device={'cuda' if torch.cuda.is_available() else 'cpu'}")
        return

    limited = bool(args.max_parents_per_task_fold)
    with stage_output(args.output) as out:
        all_predictions, seed_predictions, metrics, runs, curves, aggregation, inner_audit, reloads = [], [], [], [], [], [], [], []
        for fold in range(5):
            outer_train_rows = records.loc[records.inner_fold_id.ne(fold)].copy()
            outer_eval_rows = records.loc[records.inner_fold_id.eq(fold)].copy()
            train_parent = physical_parent_table(outer_train_rows)
            eval_parent = physical_parent_table(outer_eval_rows)
            aggregation.extend(pd.concat([train_parent.assign(outer_partition="train", outer_fold=fold), eval_parent.assign(outer_partition="evaluation", outer_fold=fold)]).to_dict("records"))
            parent_overlap = len(set(train_parent.molecule_id) & set(eval_parent.molecule_id))
            scaffold_overlap = len(set(train_parent.scaffold_group) & set(eval_parent.scaffold_group))
            if parent_overlap or scaffold_overlap:
                raise ValueError(f"Outer fold leakage: fu fold={fold}")
            x_train = x_all[train_parent.feature_index.to_numpy(int)]
            x_eval = x_all[eval_parent.feature_index.to_numpy(int)]
            y_logit = train_parent.parent_logit_fu.to_numpy(float)
            y_physical = train_parent.parent_physical_fu.to_numpy(float)
            target_eval = eval_parent[["molecule_id", "parent_physical_fu", "source_records", "scaffold_group"]].copy()

            # Same fixed Stage-A tree configuration, scored fairly on the new physical parent label.
            for candidate, target in [("structure_logit_control", y_logit), ("structure_physical_extra_trees", y_physical)]:
                estimator = make_estimator("extra_trees", int(args.seed + fold), args.threads, small_budget=True, parameters=leader_parameters)
                estimator.fit(x_train, target)
                raw = np.asarray(estimator.predict(x_eval), dtype=float)
                prediction = 1.0 / (1.0 + np.exp(-raw)) if candidate == "structure_logit_control" else np.clip(raw, 0.0, 1.0)
                table = target_eval.assign(candidate_id=candidate, outer_fold=fold, predicted_physical_fu=prediction)
                all_predictions.append(table)
                runs.append({"candidate_id": candidate, "outer_fold": fold, "seed": pd.NA, "train_parents": len(train_parent), "evaluation_parents": len(eval_parent), "fit_target": "logit_parent_mean" if candidate.endswith("control") else "physical_parent_mean", "inner_validation_used": False, "algorithm": "extra_trees", "parameters": json.dumps(leader_parameters, sort_keys=True)})
                if fold == 0:
                    with tempfile.TemporaryDirectory(dir=out) as temp:
                        model_path = Path(temp) / f"{candidate}.joblib"
                        joblib.dump(estimator, model_path, compress=3)
                        again = np.asarray(joblib.load(model_path).predict(x_eval), dtype=float)
                    difference = float(np.max(np.abs(raw - again)))
                    if difference > 1e-8:
                        raise ValueError(f"Reload mismatch: {candidate}")
                    reloads.append({"candidate_id": candidate, "outer_fold": fold, "seed": pd.NA, "max_abs_difference": difference})

            inner_train_mask, inner_valid_mask = inner_scaffold_masks(train_parent.scaffold_group, int(args.seed + fold))
            if set(train_parent.loc[inner_train_mask, "scaffold_group"]) & set(train_parent.loc[inner_valid_mask, "scaffold_group"]):
                raise ValueError("Internal scaffold leakage")
            inner_audit.append({"outer_fold": fold, "outer_train_parents": len(train_parent), "outer_evaluation_parents": len(eval_parent), "outer_parent_overlap": parent_overlap, "outer_scaffold_overlap": scaffold_overlap, "inner_train_parents": int(inner_train_mask.sum()), "inner_validation_parents": int(inner_valid_mask.sum()), "inner_scaffold_overlap": 0, "evaluation_targets_used_in_inner_split": False})
            neural_results = {}
            for candidate, loss_name in [("structure_bounded_huber_mlp", "huber"), ("structure_bounded_mae_mlp", "mae")]:
                per_seed = []
                for offset in SEEDS:
                    seed = int(args.seed + 1000 * fold + offset)
                    checkpoint, curve = fit_mlp(
                        x_train[inner_train_mask], y_physical[inner_train_mask],
                        x_train[inner_valid_mask], y_physical[inner_valid_mask],
                        seed=seed, bounded_output=True, loss_name=loss_name, monitor_name="mae",
                        max_epochs=args.max_epochs, patience=args.patience,
                    )
                    final_checkpoint = fit_mlp_fixed_epochs(
                        x_train, y_physical, seed=seed, bounded_output=True, loss_name=loss_name,
                        epochs=checkpoint["best_epoch"],
                    )
                    prediction = predict_checkpoint(final_checkpoint, x_eval)
                    per_seed.append(prediction)
                    seed_table = target_eval.assign(candidate_id=candidate, outer_fold=fold, seed=seed, predicted_physical_fu=prediction)
                    seed_predictions.append(seed_table)
                    curves.extend([{**item, "candidate_id": candidate, "outer_fold": fold, "seed": seed, "best_epoch": checkpoint["best_epoch"]} for item in curve])
                    runs.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "train_parents": len(train_parent), "early_stop_train_parents": int(inner_train_mask.sum()), "evaluation_parents": len(eval_parent), "fit_target": "physical_parent_mean", "inner_validation_used": True, "final_refit_on_complete_outer_train": True, "algorithm": "torch_mlp_bounded", "loss": loss_name, "best_epoch": checkpoint["best_epoch"], "epochs_ran": checkpoint["epochs_ran"], "best_internal_metric": checkpoint["best_internal_metric"], "device_used": final_checkpoint["device_used"]})
                    if fold == 0:
                        with tempfile.TemporaryDirectory(dir=out) as temp:
                            model_path = Path(temp) / f"{candidate}_{seed}.pt"
                            torch.save(final_checkpoint, model_path)
                            # This checkpoint was written in the immediately preceding line into a
                            # private temporary directory; it contains NumPy scaler arrays as well
                            # as tensor weights, which PyTorch's restrictive weights-only loader
                            # intentionally rejects.
                            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
                            again = predict_checkpoint(loaded, x_eval)
                        difference = float(np.max(np.abs(prediction - again)))
                        if difference > 1e-7:
                            raise ValueError(f"Reload mismatch: {candidate} seed={seed}")
                        reloads.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "max_abs_difference": difference})
                neural_results[candidate] = np.mean(np.vstack(per_seed), axis=0)
            for candidate, prediction in neural_results.items():
                all_predictions.append(target_eval.assign(candidate_id=candidate, outer_fold=fold, predicted_physical_fu=prediction))

        prediction_frame = pd.concat(all_predictions, ignore_index=True)
        seed_frame = pd.concat(seed_predictions, ignore_index=True)
        if prediction_frame.duplicated(["candidate_id", "molecule_id"]).any():
            raise ValueError("Duplicate parent OOF prediction")
        for candidate, table in prediction_frame.groupby("candidate_id", sort=True):
            metrics.append(metric_row(table, candidate))
        metric_frame = pd.DataFrame(metrics).sort_values("primary_metric").reset_index(drop=True)
        metric_frame["rank"] = np.arange(1, len(metric_frame) + 1)
        base = float(metric_frame.loc[metric_frame.candidate_id.eq("structure_logit_control"), "primary_metric"].iloc[0])
        metric_frame["relative_to_logit_control_pct"] = 100.0 * (metric_frame.primary_metric / base - 1.0)
        seed_metric_rows = []
        for (candidate, seed), table in seed_frame.groupby(["candidate_id", "seed"], sort=True):
            seed_metric_rows.append({**metric_row(table, candidate), "seed": seed})
        prediction_frame.to_csv(out / "parent_oof_predictions.csv", index=False)
        seed_frame.to_csv(out / "neural_seed_predictions.csv", index=False)
        metric_frame.to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(seed_metric_rows).to_csv(out / "neural_seed_metrics.csv", index=False)
        pd.DataFrame(runs).to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(curves).to_csv(out / "neural_learning_curves.csv", index=False)
        pd.DataFrame(aggregation).drop_duplicates(["outer_fold", "outer_partition", "molecule_id"]).to_csv(out / "physical_parent_label_audit.csv", index=False)
        pd.DataFrame(inner_audit).to_csv(out / "fold_and_inner_scaffold_audit.csv", index=False)
        pd.DataFrame(reloads).to_csv(out / "reload_checks.csv", index=False)
        summary = {"task_id": TASK, "parents": int(records.molecule_id.nunique()), "candidates": len(candidates), "outer_folds": 5, "neural_seeds": len(SEEDS), "fits": len(runs), "limited_smoke": limited, "label_definition": "arithmetic_mean_of_record_level_expit_transformed_fu_per_parent", "scoring": "parent_level_physical_MAE", "bounded_neural_output": True, "internal_early_stopping": "deterministic_whole_scaffold_holdout_within_outer_train", "validation_target_file_opened": False, "validation_rows_evaluated": False, "test_labels_read": False, "model_selection_authorized": not limited}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text("# fu physical-label alignment diagnostic\n\nThis train-only diagnostic retains the locked Stage-A logit-tree control, evaluates all candidates on arithmetic-mean physical parent fu labels, and tests bounded sigmoid-output MLP heads with whole-scaffold internal early stopping. It cannot overwrite or select a frozen validation/test result.\n", encoding="utf-8")
        finish_stage(out, "gate1b_fu_physical_alignment", inputs={"train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"), "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"), "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json")}, partial=limited, **summary)
    print(f"Gate 1B fu physical alignment: {args.output}")


if __name__ == "__main__":
    run_cli(main)
