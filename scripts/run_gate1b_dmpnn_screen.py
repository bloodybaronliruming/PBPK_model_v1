#!/usr/bin/env python3
"""Run Gate 1B graph+RDKit2D D-MPNN on frozen training folds only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch

from gate1b_graph_common import GraphRDKitRegressor, candidates
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records, metric_row


def parent_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.groupby(["molecule_id", "feature_index", "graph_smiles"], as_index=False)
            .agg(interface_target=("interface_target", "mean"), source_records=("row_id", "size")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--multimodal-protocol", type=Path, default=ROOT / "data/public_development/gate1b_multimodal_protocol_v2")
    parser.add_argument("--graph-manifest", type=Path, default=ROOT / "data/public_development/gate1b_graph_manifest_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/gate1b_dmpnn_stage1_v1")
    parser.add_argument("--profile", choices=["smoke", "stage1"], default="stage1")
    parser.add_argument("--max-parents-per-task-fold", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.stl_protocol / "complete.json", args.multimodal_protocol / "complete.json",
                args.graph_manifest / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    multi_meta = verify_stage(args.multimodal_protocol, "gate1b_multimodal_input_protocol")
    graph_meta = verify_stage(args.graph_manifest, "gate1b_graph_manifest")
    if stl_meta.get("test_labels_read") or multi_meta.get("test_labels_read") or graph_meta.get("test_labels_read"):
        raise ValueError("D-MPNN screen requires test-blind inputs")
    if multi_meta.get("fixed_validation_authorized_now"):
        raise ValueError("This runner is train-CV-only; validation must remain closed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; Gate 1B forbids silent CPU fallback")
    records = pd.read_csv(args.stl_protocol / "benchmark_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records["feature_index"] = records.feature_index.astype(int)
    records["inner_fold_id"] = records.inner_fold_id.astype(int)
    records["interface_target"] = (records.target_value - records.train_mean) / records.train_std_population
    records = records.loc[records.split.eq("train")].copy()
    records = limit_records(records, args.max_parents_per_task_fold)
    graphs = pd.read_csv(args.graph_manifest / "graph_manifest.csv")
    records = records.merge(graphs[["molecule_id", "feature_index", "graph_smiles"]],
                            on=["molecule_id", "feature_index"], validate="many_to_one")
    if records.graph_smiles.isna().any() or records.task_id.nunique() != 6:
        raise ValueError("D-MPNN records lack full six-task graph coverage")
    cache = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    x = cache["rdkit2d"]
    configs = candidates(args.profile)
    if args.check_only:
        device = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
        print(f"Gate 1B D-MPNN input valid: records={len(records)} tasks=6 configs={len(configs)} device={device}")
        return
    with stage_output(args.output) as out:
        prediction_rows, metric_rows, run_rows, reload_rows = [], [], [], []
        for task, task_frame in records.groupby("task_id", sort=True):
            folds = sorted(task_frame.inner_fold_id.unique())
            if folds != [0, 1, 2, 3, 4]:
                raise ValueError(f"Task lacks frozen five-fold coverage: {task} {folds}")
            for candidate_index, parameters in enumerate(configs):
                candidate_id = f"dmpnn_rdkit2d__c{candidate_index:02d}"
                all_eval = []
                for fold in folds:
                    train_frame = task_frame.loc[task_frame.inner_fold_id.ne(fold)]
                    eval_frame = task_frame.loc[task_frame.inner_fold_id.eq(fold)].copy()
                    train = parent_table(train_frame)
                    evaluation_parents = parent_table(eval_frame)
                    train_scaffolds = set(train_frame.scaffold_group)
                    eval_scaffolds = set(eval_frame.scaffold_group)
                    if set(train.molecule_id) & set(evaluation_parents.molecule_id) or train_scaffolds & eval_scaffolds:
                        raise ValueError(f"Parent/scaffold leakage in {task} fold {fold}")
                    estimator = GraphRDKitRegressor(
                        **parameters, seed=int(args.seed + int(fold)), device=args.device, threads=args.threads
                    )
                    estimator.fit(
                        x[train.feature_index.to_numpy(int)], train.interface_target.to_numpy(float),
                        train.graph_smiles.to_numpy(), np.ones(len(train), dtype=np.float32),
                    )
                    parent_prediction = estimator.predict(
                        x[evaluation_parents.feature_index.to_numpy(int)],
                        evaluation_parents.graph_smiles.to_numpy(),
                    )
                    if not np.isfinite(parent_prediction).all():
                        raise FloatingPointError(f"Non-finite D-MPNN prediction: {task} fold {fold}")
                    predicted = evaluation_parents[["molecule_id"]].copy()
                    predicted["predicted_interface_target"] = parent_prediction
                    table = eval_frame[["row_id", "molecule_id", "task_id", "endpoint", "split", "inner_fold_id",
                                        "sensitivity_subset", "target_value", "train_mean", "train_std_population",
                                        "reporting_inverse", "interface_target"]].merge(predicted, on="molecule_id", validate="many_to_one")
                    table["algorithm"] = "dmpnn"
                    table["candidate_id"] = candidate_id
                    table["feature_set"] = "graph_rdkit2d"
                    table["evaluation_partition"] = f"fold_{fold}"
                    prediction_rows.append(table)
                    all_eval.append(table)
                    run_rows.append({
                        "task_id": task, "algorithm": "dmpnn", "feature_set": "graph_rdkit2d",
                        "candidate_id": candidate_id, "partition": f"fold_{fold}",
                        "train_parents": len(train), "evaluation_parents": len(evaluation_parents),
                        "train_scaffolds": len(train_scaffolds), "evaluation_scaffolds": len(eval_scaffolds),
                        "parent_overlap": 0, "scaffold_overlap": 0,
                        "device_used": estimator.device_used_, "cuda_device_name": estimator.cuda_device_name_,
                        "parameters": json.dumps(parameters, sort_keys=True),
                    })
                    if args.verify_reload and fold == folds[0]:
                        with tempfile.TemporaryDirectory(dir=out) as temp:
                            path = Path(temp) / "model.joblib"
                            joblib.dump(estimator, path, compress=3)
                            restored = joblib.load(path)
                            after = restored.predict(
                                x[evaluation_parents.feature_index.to_numpy(int)],
                                evaluation_parents.graph_smiles.to_numpy(),
                            )
                        delta = float(np.max(np.abs(parent_prediction - after)))
                        if delta > 1e-7:
                            raise ValueError(f"D-MPNN reload mismatch: {task} {candidate_id} {delta}")
                        reload_rows.append({"task_id": task, "candidate_id": candidate_id,
                                            "max_abs_difference": delta, "passed": True})
                    del estimator
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                joined = pd.concat(all_eval, ignore_index=True)
                row = metric_row(joined, joined.predicted_interface_target.to_numpy(), task, "dmpnn",
                                 "graph_rdkit2d", "train_cv")
                row["candidate_id"] = candidate_id
                row["parameters"] = json.dumps(parameters, sort_keys=True)
                metric_rows.append(row)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        if predictions.duplicated(["candidate_id", "row_id"]).any():
            raise ValueError("Duplicate D-MPNN OOF rows within a candidate")
        coverage = predictions.groupby("candidate_id").row_id.nunique()
        if not coverage.eq(records.row_id.nunique()).all():
            raise ValueError(f"Incomplete D-MPNN OOF coverage: {coverage.to_dict()}")
        predictions.to_csv(out / "predictions.csv", index=False)
        metrics = pd.DataFrame(metric_rows)
        metrics["rank_within_task"] = metrics.groupby("task_id").primary_metric.rank(method="min").astype(int)
        metrics.to_csv(out / "metrics.csv", index=False)
        metrics.sort_values(["task_id", "rank_within_task", "candidate_id"]).to_csv(out / "candidate_ranking.csv", index=False)
        pd.DataFrame(run_rows).to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(reload_rows).to_csv(out / "reload_checks.csv", index=False)
        limited = bool(args.max_parents_per_task_fold) or args.profile == "smoke"
        summary = {
            "tasks": 6, "folds_per_task": 5, "candidate_configs": len(configs),
            "fits": len(run_rows), "oof_rows": len(predictions), "profile": args.profile,
            "device_requested": args.device,
            "cuda_verified": bool(args.device == "cuda" and all(x["device_used"] == "cuda" for x in run_rows)),
            "reload_checks": len(reload_rows),
            "parent_overlap": 0, "scaffold_overlap": 0,
            "validation_labels_read": False, "test_labels_read": False,
            "model_selection_authorized": not limited,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Gate 1B D-MPNN screen\n\nGraph+RDKit2D on frozen train folds only. Limited/profile-smoke runs are engineering checks and never authorize selection.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate1b_dmpnn_screen", inputs={
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "multimodal_protocol_complete_sha256": sha256(args.multimodal_protocol / "complete.json"),
            "graph_manifest_complete_sha256": sha256(args.graph_manifest / "complete.json"),
        }, **summary, partial=limited)
    print(f"Gate 1B D-MPNN screen: {args.output}")


if __name__ == "__main__":
    run_cli(main)
