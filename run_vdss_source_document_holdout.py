#!/usr/bin/env python3
"""Run the purged VDss document-group source-generalization stress test."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from dmpk_toolkit import featurize
from gate1b_torch_common import fit_feature_scaler, scale_features
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_vdss_cross_species_transfer_traincv import (
    HUMAN_TASK, TASKS, SPECIES_ORDER, SpeciesPrivateNet, cpu_predict, load_labels_only,
    maximum_tanimoto, parameter_vector, parent_table, parse_int_list, seed_everything,
    stable_limit, train_task_balanced,
)
from stl_benchmark_common import make_estimator


NEURAL_ROUTES = ["human_only_neural_control", "species_conditioned_joint_shared_encoder"]
ROUTES = ["corrected_stageA_STL", *NEURAL_ROUTES]
MEMBERSHIP_COLUMNS = [
    "row_id", "task_id", "split", "endpoint", "species", "system", "canonical_unit",
    "smiles", "parent_id", "scaffold_group", "doc_id",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path,
                        default=ROOT / "data/public_development/vdss_source_generalization_protocol_v1")
    parser.add_argument("--transfer-protocol", type=Path,
                        default=ROOT / "data/public_development/vdss_cross_species_transfer_protocol_v4")
    parser.add_argument("--interface", type=Path,
                        default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path,
                        default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/vdss_source_document_holdout_v1")
    parser.add_argument("--source-folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="20260917,20260918,20260919")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--latent-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-train-parents-per-task", type=int, default=0)
    parser.add_argument("--max-evaluation-parents", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    folds, seeds = parse_int_list(args.source_folds), parse_int_list(args.seeds)
    if not set(folds) <= set(range(5)):
        raise ValueError("Source folds must be drawn from 0..4")
    if min(args.epochs, args.batch_size, args.encoder_width, args.latent_width, args.threads) < 1:
        raise ValueError("Invalid source-stress configuration")
    if min(args.max_train_parents_per_task, args.max_evaluation_parents) < 0:
        raise ValueError("Parent caps cannot be negative")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError(
            "CUDA source stress requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) "
            "for deterministic CuBLAS execution"
        )

    records_path = args.interface / "training_records.csv"
    stl_path = args.stl / "benchmark_train_records.csv"
    assignment_path = args.source_protocol / "document_group_assignment.csv"
    required = [args.source_protocol / "complete.json", args.source_protocol / "protocol.json", assignment_path,
                args.transfer_protocol / "complete.json", args.transfer_protocol / "corrected_stageA_reference.csv",
                args.interface / "complete.json", records_path, args.stl / "complete.json", stl_path]
    startup_self_check(required, output=None if args.check_only else args.output)
    source_meta = verify_stage(args.source_protocol, "vdss_source_generalization_protocol")
    transfer_meta = verify_stage(args.transfer_protocol, "vdss_cross_species_transfer_protocol")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(meta.get("test_labels_read", False) for meta in [source_meta, transfer_meta, interface_meta, stl_meta]):
        raise ValueError("Source stress inputs must remain test-closed")
    if source_meta.get("connected_component_fivefold_feasible") or not source_meta.get("document_group_stress_test_feasible"):
        raise ValueError("Unexpected source-generalization decision")
    source_contract = json.loads((args.source_protocol / "protocol.json").read_text())
    if source_contract.get("interpretation") != \
            "source perturbation stress test only; no model selection or fixed-validation authorization":
        raise ValueError("Source diagnostic interpretation is not frozen")
    baseline = pd.read_csv(args.transfer_protocol / "corrected_stageA_reference.csv")
    if len(baseline) != 1 or baseline.stageA_algorithm.iloc[0] != "extra_trees" \
            or baseline.stageA_feature_set.iloc[0] != "rdkit2d":
        raise ValueError("Unexpected corrected Stage-A VDss reference")
    stagea_parameters = json.loads(baseline.stageA_parameters.iloc[0])

    assignment = pd.read_csv(assignment_path, dtype={"doc_id": str})
    metadata = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=MEMBERSHIP_COLUMNS)
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(TASKS)].copy()
    human_source = pd.read_csv(
        stl_path, dtype=str, keep_default_na=False,
        usecols=["row_id", "task_id", "molecule_id", "doc_id", "scaffold_group"],
    )
    human_source = human_source.loc[human_source.task_id.eq(HUMAN_TASK)].rename(
        columns={"molecule_id": "parent_id"}
    )
    unique_smiles = list(dict.fromkeys(metadata.smiles.astype(str)))
    feature_all, descriptor_names = featurize(unique_smiles, feature_set="ecfp4_rdkit2d", progress=False)
    ecfp_all, rdkit_all = feature_all[:, :2048], feature_all[:, 2048:]
    feature_lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
    full_configuration = bool(
        folds == list(range(5)) and seeds == [20260917, 20260918, 20260919]
        and args.epochs == 30 and args.batch_size == 128
        and args.encoder_width == 128 and args.latent_width == 64
        and args.max_train_parents_per_task == 0 and args.max_evaluation_parents == 0
        and np.isclose(args.learning_rate, 0.0003) and np.isclose(args.weight_decay, 0.001)
    )
    # Exercise the strict JSON boundary before any model is fitted.  The final
    # np.isclose result is numpy.bool_ unless the complete-status flag is cast.
    json.dumps({"full_configuration": full_configuration}, allow_nan=False)
    if args.check_only:
        print(
            f"VDss source stress ready: folds={folds} seeds={seeds} routes=3 structures={len(unique_smiles)} "
            f"device={args.device} full_configuration={full_configuration}"
        )
        return

    species_map = {species: index for index, species in enumerate(SPECIES_ORDER)}
    human_index = species_map["human"]
    prediction_rows, isolation_rows, target_rows = [], [], []
    coverage_rows, loss_rows, check_rows = [], [], []
    with stage_output(args.output) as out:
        model_dir = out / "models"
        model_dir.mkdir()
        for source_fold in folds:
            evaluation_documents = set(assignment.loc[assignment.source_fold.eq(source_fold), "doc_id"].astype(str))
            evaluation_source_rows = human_source.loc[human_source.doc_id.isin(evaluation_documents)].copy()
            evaluation_rows = set(evaluation_source_rows.row_id)
            evaluation_parents = set(evaluation_source_rows.parent_id)
            evaluation_scaffolds = set(evaluation_source_rows.scaffold_group)
            retained = metadata.loc[
                ~metadata.doc_id.isin(evaluation_documents)
                & ~metadata.parent_id.isin(evaluation_parents)
                & ~metadata.scaffold_group.isin(evaluation_scaffolds)
            ].copy()
            if set(retained.task_id) != set(TASKS):
                raise ValueError(f"Source fold {source_fold} removed a registered task")
            for task, before in metadata.groupby("task_id", sort=True):
                after = retained.loc[retained.task_id.eq(task)]
                isolation_rows.append({
                    "source_fold": source_fold, "task_id": task,
                    "evaluation_documents": len(evaluation_documents),
                    "evaluation_parents": len(evaluation_parents),
                    "evaluation_scaffolds": len(evaluation_scaffolds),
                    "records_before": len(before), "records_after": len(after),
                    "parents_after": after.parent_id.nunique(), "scaffolds_after": after.scaffold_group.nunique(),
                    "post_filter_document_overlap": len(set(after.doc_id) & evaluation_documents),
                    "post_filter_parent_overlap": len(set(after.parent_id) & evaluation_parents),
                    "post_filter_scaffold_overlap": len(set(after.scaffold_group) & evaluation_scaffolds),
                })
            fit_labels = load_labels_only(records_path, set(retained.row_id))
            retained = retained.merge(fit_labels, on="row_id", validate="one_to_one")
            retained["target_value"] = pd.to_numeric(retained.target_value, errors="raise")
            parents = parent_table(retained)
            fitting = pd.concat([
                stable_limit(group, args.max_train_parents_per_task, 20260917 + source_fold)
                for _, group in parents.groupby("task_id", sort=True)
            ], ignore_index=True)
            stats = fitting.groupby("task_id").target_value.agg(["mean", lambda values: values.std(ddof=0)]).reset_index()
            stats.columns = ["task_id", "target_mean", "target_std_population"]
            if (stats.target_std_population <= 0).any():
                raise ValueError(f"Invalid target scale in source fold {source_fold}")
            stats["source_fold"] = source_fold
            target_rows.extend(stats.to_dict("records"))
            fitting = fitting.merge(stats.drop(columns="source_fold"), on="task_id", validate="many_to_one")
            fitting["standardized_target"] = (
                fitting.target_value - fitting.target_mean
            ) / fitting.target_std_population
            fitting["species_index"] = fitting.species.map(species_map).astype(int)
            x_fit_raw = rdkit_all[np.asarray([feature_lookup[value] for value in fitting.smiles])]
            human_fit = fitting.loc[fitting.task_id.eq(HUMAN_TASK)]
            human_reference_steps = math.ceil(len(human_fit) / args.batch_size)

            evaluation_meta = metadata.loc[
                metadata.row_id.isin(evaluation_rows) & metadata.task_id.eq(HUMAN_TASK)
            ].copy()
            evaluation_parent = evaluation_meta.groupby("parent_id", as_index=False).agg(
                smiles=("smiles", "first"), scaffold_group=("scaffold_group", "first"),
                source_records=("row_id", "size"),
                document_ids=("doc_id", lambda values: ";".join(sorted(set(values.astype(str))))),
            )
            evaluation_parent = stable_limit(
                evaluation_parent, args.max_evaluation_parents, 20261917 + source_fold
            )
            selected_evaluation_parents = set(evaluation_parent.parent_id)
            selected_evaluation_rows = set(evaluation_meta.loc[
                evaluation_meta.parent_id.isin(selected_evaluation_parents), "row_id"
            ])
            x_eval_raw = rdkit_all[np.asarray([feature_lookup[value] for value in evaluation_parent.smiles])]
            eval_ecfp = ecfp_all[np.asarray([feature_lookup[value] for value in evaluation_parent.smiles])]
            human_fit_ecfp = ecfp_all[np.asarray([feature_lookup[value] for value in human_fit.smiles])]
            evaluation_parent["max_human_train_ecfp4_tanimoto"] = maximum_tanimoto(eval_ecfp, human_fit_ecfp)
            fold_predictions = []
            for seed in seeds:
                initialization_seed = seed + 100000 * source_fold
                # Exact corrected Stage-A family/config on the same purged human training parents.
                stagea = make_estimator("extra_trees", initialization_seed, args.threads,
                                        small_budget=True, parameters=stagea_parameters)
                human_mean = float(human_fit.target_value.mean())
                human_std = float(human_fit.target_value.std(ddof=0))
                stagea.fit(x_fit_raw[human_fit.index.to_numpy(int)],
                           (human_fit.target_value.to_numpy(float) - human_mean) / human_std)
                stagea_prediction_z = np.asarray(stagea.predict(x_eval_raw)).reshape(-1)
                stagea_prediction = stagea_prediction_z * human_std + human_mean
                stagea_path = model_dir / f"fold{source_fold}__seed{seed}__corrected_stageA_STL.joblib"
                joblib.dump(stagea, stagea_path, compress=3)
                reloaded_stagea = np.asarray(joblib.load(stagea_path).predict(x_eval_raw)).reshape(-1)
                stagea_difference = float(np.max(np.abs(stagea_prediction_z - reloaded_stagea)))
                fold_predictions.append(("corrected_stageA_STL", seed, stagea_prediction))
                check_rows.append({
                    "source_fold": source_fold, "seed": seed, "route_id": "corrected_stageA_STL",
                    "encoder_max_parameter_change": np.nan, "human_head_max_parameter_change": np.nan,
                    "finite_predictions": bool(np.isfinite(stagea_prediction).all()),
                    "reload_max_abs_difference": stagea_difference,
                    "evaluation_parents": len(evaluation_parent),
                    "evaluation_targets_read_during_fit": False,
                })
                for route in NEURAL_ROUTES:
                    route_tasks = [HUMAN_TASK] if route == "human_only_neural_control" else TASKS
                    scaler_rows = fitting.task_id.isin(route_tasks).to_numpy()
                    x_mean, x_scale = fit_feature_scaler(x_fit_raw[scaler_rows])
                    x_fit = scale_features(x_fit_raw, x_mean, x_scale)
                    seed_everything(initialization_seed, args.threads)
                    model = SpeciesPrivateNet(
                        x_fit.shape[1], len(SPECIES_ORDER), args.encoder_width, args.latent_width
                    ).to(args.device)
                    encoder_initial = parameter_vector(model, "encoder")
                    human_head_initial = parameter_vector(model, f"heads.{human_index}")
                    phase = "human_only" if route == "human_only_neural_control" else "joint_task_balanced"
                    phase_seed = initialization_seed + (101 if route == "human_only_neural_control" else 301)
                    coverage, losses = train_task_balanced(
                        model, x_fit, fitting, route_tasks, epochs=args.epochs,
                        batch_size=args.batch_size, learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay, seed=phase_seed, device=args.device,
                        steps_per_task=human_reference_steps,
                    )
                    for task, values in coverage.items():
                        coverage_rows.append({"source_fold": source_fold, "seed": seed, "route_id": route,
                                              "phase": phase, "task_id": task, **values})
                    loss_rows.extend([{"source_fold": source_fold, "seed": seed, "route_id": route,
                                       "phase": phase, **row} for row in losses])
                    encoder_delta = float(torch.max(torch.abs(parameter_vector(model, "encoder") - encoder_initial)))
                    human_head_delta = float(torch.max(torch.abs(
                        parameter_vector(model, f"heads.{human_index}") - human_head_initial
                    )))
                    checkpoint = {
                        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                        "n_features": int(x_fit.shape[1]), "species_order": SPECIES_ORDER,
                        "encoder_width": args.encoder_width, "latent_width": args.latent_width,
                        "x_mean": x_mean, "x_scale": x_scale, "route_id": route,
                        "seed": seed, "source_fold": source_fold,
                    }
                    model_path = model_dir / f"fold{source_fold}__seed{seed}__{route}.pt"
                    torch.save(checkpoint, model_path)
                    first = cpu_predict(checkpoint, x_eval_raw, np.full(len(x_eval_raw), human_index))
                    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
                    second = cpu_predict(loaded, x_eval_raw, np.full(len(x_eval_raw), human_index))
                    difference = float(np.max(np.abs(first - second)))
                    human_stats = stats.loc[stats.task_id.eq(HUMAN_TASK)].iloc[0]
                    transformed = first * float(human_stats.target_std_population) + float(human_stats.target_mean)
                    if not np.isfinite(transformed).all() or difference > 1e-7:
                        raise ValueError(f"Prediction check failed: fold={source_fold} seed={seed} route={route}")
                    fold_predictions.append((route, seed, transformed))
                    check_rows.append({
                        "source_fold": source_fold, "seed": seed, "route_id": route,
                        "encoder_max_parameter_change": encoder_delta,
                        "human_head_max_parameter_change": human_head_delta,
                        "finite_predictions": bool(np.isfinite(transformed).all()),
                        "reload_max_abs_difference": difference,
                        "evaluation_parents": len(evaluation_parent),
                        "evaluation_targets_read_during_fit": False,
                    })

            # Read held-out document targets only after every registered model has predicted.
            evaluation_labels = load_labels_only(records_path, selected_evaluation_rows)
            labeled = evaluation_meta.loc[evaluation_meta.row_id.isin(selected_evaluation_rows)].merge(
                evaluation_labels, on="row_id", validate="one_to_one"
            )
            labeled["target_value"] = pd.to_numeric(labeled.target_value, errors="raise")
            observed = labeled.groupby("parent_id", as_index=False).agg(
                observed_transformed_target=("target_value", "mean")
            )
            base = evaluation_parent[[
                "parent_id", "scaffold_group", "source_records", "document_ids",
                "max_human_train_ecfp4_tanimoto",
            ]].merge(observed, on="parent_id", validate="one_to_one")
            for route, seed, prediction in fold_predictions:
                table = base.copy()
                table["source_fold"], table["seed"], table["route_id"] = source_fold, seed, route
                table["predicted_transformed_target"] = prediction
                prediction_rows.append(table)

        predictions = pd.concat(prediction_rows, ignore_index=True)
        if predictions.duplicated(["source_fold", "seed", "route_id", "parent_id"]).any():
            raise ValueError("Duplicate source-stress predictions")
        ensemble = predictions.groupby([
            "source_fold", "route_id", "parent_id", "scaffold_group", "source_records", "document_ids",
            "max_human_train_ecfp4_tanimoto", "observed_transformed_target",
        ], as_index=False).predicted_transformed_target.mean().rename(
            columns={"predicted_transformed_target": "ensemble_predicted_transformed_target"}
        )
        metrics = ensemble.groupby(["route_id", "source_fold"], as_index=False).apply(
            lambda frame: pd.Series({
                "parents": len(frame),
                "transformed_RMSE": float(np.sqrt(np.mean((
                    frame.observed_transformed_target - frame.ensemble_predicted_transformed_target
                ) ** 2))),
            }), include_groups=False
        ).reset_index(drop=True)
        pd.DataFrame(isolation_rows).to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(target_rows).to_csv(out / "fold_local_target_statistics.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(out / "training_coverage.csv", index=False)
        pd.DataFrame(loss_rows).to_csv(out / "training_losses.csv", index=False)
        pd.DataFrame(check_rows).to_csv(out / "route_checks.csv", index=False)
        predictions.to_csv(out / "seed_level_predictions.csv", index=False)
        ensemble.to_csv(out / "ensemble_predictions.csv", index=False)
        metrics.to_csv(out / "source_fold_metrics.csv", index=False)
        isolation = pd.DataFrame(isolation_rows)
        checks = pd.DataFrame(check_rows)
        neural_checks = checks.loc[checks.route_id.isin(NEURAL_ROUTES)]
        parent_fold_counts = ensemble[["source_fold", "parent_id"]].drop_duplicates().groupby(
            "parent_id"
        ).source_fold.nunique()
        expected_models = len(folds) * len(seeds) * len(ROUTES)
        model_files = len(list(model_dir.iterdir()))
        if model_files != expected_models:
            raise ValueError(f"Expected {expected_models} model files, found {model_files}")
        if not checks.finite_predictions.all() or checks.reload_max_abs_difference.max() > 1e-7:
            raise ValueError("Registered route prediction/reload audit failed")
        if not (neural_checks.encoder_max_parameter_change.gt(0).all()
                and neural_checks.human_head_max_parameter_change.gt(0).all()):
            raise ValueError("A registered neural route did not update the expected parameters")
        overlap_columns = [
            "post_filter_document_overlap", "post_filter_parent_overlap", "post_filter_scaffold_overlap",
        ]
        if isolation[overlap_columns].to_numpy().any():
            raise ValueError("Purged source-fold isolation audit failed")
        (out / "README.md").write_text(
            "# VDss purged document-group stress test\n\n"
            "Evaluation documents and every associated parent/scaffold are removed from all five species training "
            "tasks. Evaluation labels are read only after all models for a source fold have predicted. Because the "
            "source graph is dominated by one component/document, this is a deliberately unbalanced sensitivity "
            "diagnostic and cannot authorize model selection or fixed validation.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "vdss_source_document_holdout",
            inputs={
                "source_protocol_complete_sha256": sha256(args.source_protocol / "complete.json"),
                "transfer_protocol_complete_sha256": sha256(args.transfer_protocol / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_train_only_complete_sha256": sha256(args.stl / "complete.json"),
            },
            routes=3, tasks=5, source_folds=folds, seeds=seeds, epochs=args.epochs,
            batch_size=args.batch_size, device=args.device,
            cuda_device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "not_used",
            diagnostic_scope="purged_document_group_source_stress_only",
            connected_component_fivefold_feasible=False,
            evaluation_labels_used_during_fit=False,
            validation_target_file_opened=False, test_labels_read=False,
            model_selection_authorized=False, fixed_validation_selection_authorized=False,
            model_fitted=True, partial=not full_configuration, full_configuration=full_configuration,
            expected_model_files=expected_models, model_files=model_files,
            seed_level_predictions=len(predictions), ensemble_predictions=len(ensemble),
            unique_evaluation_parents=int(ensemble.parent_id.nunique()),
            parents_evaluated_in_multiple_source_folds=int(parent_fold_counts.gt(1).sum()),
            max_source_folds_per_parent=int(parent_fold_counts.max()),
            max_post_filter_document_overlap=int(isolation.post_filter_document_overlap.max()),
            max_post_filter_parent_overlap=int(isolation.post_filter_parent_overlap.max()),
            max_post_filter_scaffold_overlap=int(isolation.post_filter_scaffold_overlap.max()),
            predictions_finite=bool(checks.finite_predictions.all()),
            max_reload_abs_difference=float(checks.reload_max_abs_difference.max()),
            neural_encoder_updates_nonzero=bool(neural_checks.encoder_max_parameter_change.gt(0).all()),
            neural_human_head_updates_nonzero=bool(neural_checks.human_head_max_parameter_change.gt(0).all()),
            interpretation="source perturbation stress test only; not unique-parent OOF",
        )
    print(f"VDss source document holdout: {args.output}")


if __name__ == "__main__":
    run_cli(main)
