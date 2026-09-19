#!/usr/bin/env python3
"""Run the frozen capacity-matched fu/non-fu two-group multitask train-CV."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmpk_toolkit import featurize
from multitask_two_group_common import ROUTE_ID, TwoGroupSharedPrivateNet, cpu_predict
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_multitask_shared_private_traincv import (
    META_COLUMNS, aggregate_parents, apply_eval_scaler, scaled_features, train_phases,
)
from run_vdss_cross_species_transfer_traincv import load_labels_only, parse_int_list, seed_everything, stable_limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/multitask_two_group_protocol_v2")
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/multitask_two_group_traincv_v1")
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="20260917,20260918,20260919")
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-parents-per-task", type=int, default=0)
    parser.add_argument("--max-eval-parents-per-task", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    folds, seeds = parse_int_list(args.outer_folds), parse_int_list(args.seeds)
    if not folds or not seeds or not set(folds) <= set(range(5)):
        raise ValueError("Invalid outer folds or seeds")
    if min(args.phase1_epochs, args.phase2_epochs, args.batch_size, args.threads) < 1:
        raise ValueError("Epochs, batch size and threads must be positive")
    if min(args.max_parents_per_task, args.max_eval_parents_per_task) < 0:
        raise ValueError("Parent limits cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)")

    records_path = args.interface / "training_records.csv"
    required = [
        args.protocol / "complete.json", args.protocol / "protocol.json",
        args.protocol / "route_registry.csv", args.protocol / "task_group_registry.csv",
        args.protocol / "task_sampling_registry.csv", args.protocol / "task_target_registry.csv",
        args.interface / "complete.json", records_path,
        args.stl / "complete.json", args.stl / "benchmark_train_records.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_meta = verify_stage(args.protocol, "multitask_two_group_protocol")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(meta.get("test_labels_read", False) for meta in [protocol_meta, interface_meta, stl_meta]):
        raise ValueError("Formal two-group inputs must remain test-closed")

    protocol = json.loads((args.protocol / "protocol.json").read_text(encoding="utf-8"))
    task_order = list(map(str, protocol["tasks"]))
    human_tasks = list(map(str, protocol["human_evaluation_tasks"]))
    task_index = {task: index for index, task in enumerate(task_order)}
    registry = pd.read_csv(args.protocol / "task_target_registry.csv")
    sampling = pd.read_csv(args.protocol / "task_sampling_registry.csv")
    group_registry = pd.read_csv(args.protocol / "task_group_registry.csv")
    route = pd.read_csv(args.protocol / "route_registry.csv").iloc[0]
    if len(task_order) != 45 or len(human_tasks) != 6 or set(registry.task_id) != set(task_order):
        raise ValueError("Frozen task registry is incomplete")
    if sampling.task_id.duplicated().any() or set(sampling.task_id) != set(task_order):
        raise ValueError("Sampling registry does not uniquely cover all tasks")
    phase2_tasks = sampling.loc[sampling.phase2_human_refinement.astype(bool), "task_id"].astype(str).tolist()
    if set(phase2_tasks) != set(human_tasks) or "F__human__absolute_oral" in phase2_tasks:
        raise ValueError("Phase-2 authorization differs from the frozen six-task policy")
    if not np.isclose(sampling.phase2_task_weight.sum(), 1.0):
        raise ValueError("Frozen phase-2 task weights do not sum to one")
    groups = group_registry.set_index("task_id").loc[task_order]
    task_groups = groups.encoder_group.astype(str).tolist()
    if task_groups.count("fu_family") != 9 or task_groups.count("non_fu_family") != 36:
        raise ValueError("Frozen task grouping is incomplete")
    if route.route_id != ROUTE_ID or int(route.encoder_width) != 64 or int(route.latent) != 64:
        raise ValueError("Runner architecture differs from the frozen route")
    if route.architecture != "two_MLP_64_64_GELU_layernorm_encoders":
        raise ValueError("Frozen architecture identifier is unexpected")
    registered_seeds = [int(value) for value in str(route.formal_seeds).split(";")]
    full_configuration = bool(
        folds == list(range(5)) and seeds == registered_seeds
        and args.phase1_epochs == int(route.phase1_epochs)
        and args.phase2_epochs == int(route.phase2_epochs)
        and args.batch_size == int(route.batch_size)
        and np.isclose(args.learning_rate, float(route.learning_rate))
        and np.isclose(args.weight_decay, float(route.weight_decay))
        and args.max_parents_per_task == 0 and args.max_eval_parents_per_task == 0
    )
    bounded = registry.set_index("task_id").loc[task_order].model_output.eq("bounded_sigmoid_fraction").tolist()
    phase1_probabilities = (
        sampling.set_index("task_id").tier_loss_weight / sampling.set_index("task_id").tasks_in_tier
    ).to_dict()

    metadata = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=META_COLUMNS)
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(task_order)].copy()
    stl = pd.read_csv(
        args.stl / "benchmark_train_records.csv", dtype=str, keep_default_na=False,
        usecols=["row_id", "task_id", "parent_id", "scaffold_group", "inner_fold_id"],
    )
    stl = stl.loc[stl.task_id.isin(human_tasks)].copy()
    stl["inner_fold_id"] = pd.to_numeric(stl.inner_fold_id, errors="raise").astype(int)
    if set(stl.task_id) != set(human_tasks):
        raise ValueError("Human outer-fold membership is incomplete")
    if args.check_only:
        print(
            f"Two-group train-CV ready: folds={folds} seeds={seeds} models={len(folds) * len(seeds)} "
            f"tasks={len(task_order)} human_heads={len(human_tasks)} device={args.device} "
            f"full_configuration={full_configuration}"
        )
        return

    prediction_rows: list[pd.DataFrame] = []
    isolation_rows: list[dict] = []
    target_rows: list[dict] = []
    coverage_rows: list[dict] = []
    loss_rows: list[dict] = []
    check_rows: list[dict] = []
    with stage_output(args.output) as out:
        model_dir = out / "models"
        model_dir.mkdir()
        for fold in folds:
            eval_membership = stl.loc[stl.inner_fold_id.eq(fold)].copy()
            eval_union_parents = set(eval_membership.parent_id)
            eval_union_scaffolds = set(eval_membership.scaffold_group)
            fold_meta = metadata.copy()
            fold_meta["parent_conflict"] = fold_meta.parent_id.isin(eval_union_parents)
            fold_meta["scaffold_conflict"] = fold_meta.scaffold_group.isin(eval_union_scaffolds)
            retained = fold_meta.loc[~fold_meta.parent_conflict & ~fold_meta.scaffold_conflict].copy()
            if set(retained.task_id) != set(task_order):
                raise ValueError(f"Fold {fold} lost a registered task")
            for task, before in fold_meta.groupby("task_id", sort=True):
                after = retained.loc[retained.task_id.eq(task)]
                isolation_rows.append({
                    "outer_fold": fold, "task_id": task, "records_before": len(before),
                    "records_after": len(after), "parents_after": after.parent_id.nunique(),
                    "scaffolds_after": after.scaffold_group.nunique(),
                    "post_filter_parent_overlap": int(after.parent_id.isin(eval_union_parents).sum()),
                    "post_filter_scaffold_overlap": int(after.scaffold_group.isin(eval_union_scaffolds).sum()),
                })

            labels = load_labels_only(records_path, set(retained.row_id))
            retained = retained.merge(labels, on="row_id", validate="one_to_one")
            retained["target_value"] = pd.to_numeric(retained.target_value, errors="raise")
            parents = aggregate_parents(retained, registry)
            fitting = pd.concat([
                stable_limit(part, args.max_parents_per_task, 20260917 + fold)
                for _, part in parents.groupby("task_id", sort=True)
            ], ignore_index=True)
            stats = fitting.groupby("task_id").target_value.agg(["mean", lambda values: values.std(ddof=0)]).reset_index()
            stats.columns = ["task_id", "target_mean", "target_std_population"]
            stats = stats.merge(registry[["task_id", "target_standardization"]], on="task_id", validate="one_to_one")
            transformed = stats.target_standardization.ne("none_physical_fraction")
            if (stats.loc[transformed, "target_std_population"] <= 0).any():
                raise ValueError("Invalid fold-local target standardization")
            stats["outer_fold"] = fold
            target_rows.extend(stats.to_dict("records"))
            fitting = fitting.merge(
                stats.drop(columns="outer_fold"), on=["task_id", "target_standardization"], validate="many_to_one",
            )
            fitting["model_target"] = np.where(
                fitting.target_standardization.eq("none_physical_fraction"), fitting.target_value,
                (fitting.target_value - fitting.target_mean) / fitting.target_std_population,
            )
            fitting["task_index"] = fitting.task_id.map(task_index).astype(int)

            eval_meta = fold_meta.loc[
                fold_meta.row_id.isin(set(eval_membership.row_id)) & fold_meta.task_id.isin(human_tasks)
            ].copy()
            eval_parent = eval_meta.groupby(["task_id", "parent_id"], as_index=False).agg(
                endpoint=("endpoint", "first"), smiles=("smiles", "first"),
                scaffold_group=("scaffold_group", "first"), source_records=("row_id", "size"),
            )
            if args.max_eval_parents_per_task > 0:
                eval_parent = pd.concat([
                    stable_limit(part, args.max_eval_parents_per_task, 20261917 + fold)
                    for _, part in eval_parent.groupby("task_id", sort=True)
                ], ignore_index=True)
            selected_keys = eval_parent.set_index(["task_id", "parent_id"]).index
            selected_eval = eval_meta.loc[
                eval_meta.set_index(["task_id", "parent_id"]).index.isin(selected_keys)
            ].copy()
            eval_parent["task_index"] = eval_parent.task_id.map(task_index).astype(int)
            unique_smiles = list(dict.fromkeys([*fitting.smiles.astype(str), *eval_parent.smiles.astype(str)]))
            feature_all, _ = featurize(unique_smiles, feature_set="rdkit2d", progress=False)
            lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
            x_fit_raw = feature_all[np.asarray([lookup[value] for value in fitting.smiles])]
            x_eval_raw = feature_all[np.asarray([lookup[value] for value in eval_parent.smiles])]
            x_fit, scaler = scaled_features(x_fit_raw, fitting, ROUTE_ID, task_order)
            x_eval = apply_eval_scaler(x_eval_raw, eval_parent.task_id.tolist(), scaler)

            fold_predictions = []
            for seed in seeds:
                init_seed = seed + fold * 100000
                seed_everything(init_seed, args.threads)
                model = TwoGroupSharedPrivateNet(
                    x_fit.shape[1], len(task_order), 64, 64, bounded, task_groups,
                ).to(args.device)
                initial = torch.cat([value.detach().cpu().reshape(-1) for value in model.parameters()])
                coverage, losses = train_phases(
                    model, x_fit, fitting, task_order, human_tasks, phase1_probabilities,
                    phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, seed=init_seed + 101, device=args.device,
                )
                final = torch.cat([value.detach().cpu().reshape(-1) for value in model.parameters()])
                parameter_change = float(torch.max(torch.abs(final - initial)))
                if parameter_change <= 0:
                    raise ValueError("Two-group model parameters did not update")
                checkpoint = {
                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "route_id": ROUTE_ID, "task_order": task_order, "task_groups": task_groups,
                    "bounded": bounded, "n_features": x_fit.shape[1], "width": 64, "latent": 64,
                    "scaler": scaler, "seed": seed, "outer_fold": fold,
                }
                model_path = model_dir / f"fold{fold}__seed{seed}__{ROUTE_ID}.pt"
                torch.save(checkpoint, model_path)
                first = cpu_predict(checkpoint, x_eval, eval_parent.task_index.to_numpy(int))
                loaded = torch.load(model_path, map_location="cpu", weights_only=False)
                second = cpu_predict(loaded, x_eval, eval_parent.task_index.to_numpy(int))
                difference = float(np.max(np.abs(first - second)))
                if difference > 1e-7 or not np.isfinite(first).all():
                    raise ValueError("Prediction reload/finite audit failed")
                prediction = eval_parent[
                    ["task_id", "endpoint", "parent_id", "scaffold_group", "source_records"]
                ].copy()
                prediction["outer_fold"] = fold
                prediction["seed"] = seed
                prediction["route_id"] = ROUTE_ID
                prediction["predicted_model_target"] = first
                fold_predictions.append(prediction)
                coverage_rows.extend({"outer_fold": fold, "seed": seed, "route_id": ROUTE_ID, **row} for row in coverage)
                loss_rows.extend({"outer_fold": fold, "seed": seed, "route_id": ROUTE_ID, **row} for row in losses)
                check_rows.append({
                    "outer_fold": fold, "seed": seed, "route_id": ROUTE_ID,
                    "max_parameter_change": parameter_change, "finite_predictions": True,
                    "reload_max_abs_difference": difference,
                    "evaluation_targets_read_during_fit": False,
                })

            # The outer-fold target is opened only after every seed for this fold has fit and predicted.
            eval_labels = load_labels_only(records_path, set(selected_eval.row_id))
            selected_eval = selected_eval.merge(eval_labels, on="row_id", validate="one_to_one")
            selected_eval["target_value"] = pd.to_numeric(selected_eval.target_value, errors="raise")
            observed = aggregate_parents(selected_eval, registry)[
                ["task_id", "parent_id", "target_value"]
            ].rename(columns={"target_value": "observed_primary_target"})
            fold_prediction = pd.concat(fold_predictions, ignore_index=True).merge(
                observed, on=["task_id", "parent_id"], validate="many_to_one",
            )
            fold_prediction = fold_prediction.merge(
                stats[["task_id", "target_mean", "target_std_population", "target_standardization"]],
                on="task_id", validate="many_to_one",
            )
            fold_prediction["predicted_primary_target"] = np.where(
                fold_prediction.target_standardization.eq("none_physical_fraction"),
                fold_prediction.predicted_model_target,
                fold_prediction.predicted_model_target * fold_prediction.target_std_population
                + fold_prediction.target_mean,
            )
            prediction_rows.append(fold_prediction)

        predictions = pd.concat(prediction_rows, ignore_index=True)
        key = ["outer_fold", "seed", "route_id", "task_id", "parent_id"]
        if predictions.duplicated(key).any():
            raise ValueError("Prediction keys are not unique")
        ensemble = predictions.groupby(
            ["route_id", "outer_fold", "task_id", "endpoint", "parent_id", "scaffold_group"], as_index=False,
        ).agg(
            observed_primary_target=("observed_primary_target", "first"),
            ensemble_predicted_primary_target=("predicted_primary_target", "mean"),
        )
        metric_rows = []
        metric_lookup = registry.set_index("task_id").primary_metric.to_dict()
        for keys, part in ensemble.groupby(["route_id", "outer_fold", "task_id", "endpoint"], sort=True):
            route_id, fold, task, endpoint = keys
            metric = metric_lookup[task]
            error = part.ensemble_predicted_primary_target - part.observed_primary_target
            score = float(np.mean(np.abs(error))) if metric == "physical_MAE" else float(np.sqrt(np.mean(error ** 2)))
            metric_rows.append({
                "route_id": route_id, "outer_fold": fold, "task_id": task, "endpoint": endpoint,
                "parents": len(part), "primary_metric": metric, "primary_score": score,
            })
        isolation = pd.DataFrame(isolation_rows)
        checks = pd.DataFrame(check_rows)
        coverage = pd.DataFrame(coverage_rows)
        bounded_tasks = set(registry.loc[registry.model_output.eq("bounded_sigmoid_fraction"), "task_id"])
        bounded_predictions = predictions.loc[
            predictions.task_id.isin(bounded_tasks), "predicted_primary_target"
        ]
        predictions.to_csv(out / "seed_level_oof_predictions.csv", index=False)
        ensemble.to_csv(out / "ensemble_oof_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out / "ensemble_fold_metrics.csv", index=False)
        isolation.to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(target_rows).to_csv(out / "fold_local_target_statistics.csv", index=False)
        coverage.to_csv(out / "training_coverage.csv", index=False)
        pd.DataFrame(loss_rows).to_csv(out / "training_losses.csv", index=False)
        checks.to_csv(out / "route_checks.csv", index=False)
        (out / "README.md").write_text(
            "# Two-group multitask train-CV\n\n"
            "This run evaluates the single frozen capacity-matched fu/non-fu encoder candidate. "
            "Every outer fold globally purges the union of the six human evaluation parent/scaffold sets. "
            "Outer labels are opened only after all requested seeds for that fold fit and predict; fixed validation/test remain closed.\n",
            encoding="utf-8",
        )
        expected_models = len(folds) * len(seeds)
        finish_stage(
            out, "multitask_two_group_traincv",
            inputs={
                "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_complete_sha256": sha256(args.stl / "complete.json"),
            },
            route_id=ROUTE_ID, routes=1, tasks=len(task_order), human_evaluation_heads=len(human_tasks),
            outer_folds=folds, seeds=seeds, expected_model_files=expected_models,
            model_files=len(list(model_dir.glob("*.pt"))),
            max_post_filter_parent_overlap=int(isolation.post_filter_parent_overlap.max()),
            max_post_filter_scaffold_overlap=int(isolation.post_filter_scaffold_overlap.max()),
            predictions_finite=bool(np.isfinite(predictions.predicted_primary_target).all()),
            bounded_predictions_within_unit_interval=bool(bounded_predictions.between(0, 1).all()),
            max_reload_abs_difference=float(checks.reload_max_abs_difference.max()),
            training_phase_task_coverage_nonzero=bool(coverage.updates.gt(0).all()),
            evaluation_labels_read_after_all_fold_models_fit=True,
            evaluation_labels_used_during_fit=False, validation_target_file_opened=False,
            test_labels_read=False, architecture_selection_authorized=full_configuration,
            fixed_validation_selection_authorized=False, model_fitted=True,
            partial=not full_configuration, full_configuration=full_configuration,
            device=args.device,
            cuda_device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "not_used",
        )
    print(f"Two-group multitask train-CV: {args.output}")


if __name__ == "__main__":
    run_cli(main)
