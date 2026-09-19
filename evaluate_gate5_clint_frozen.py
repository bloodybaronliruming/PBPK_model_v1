#!/usr/bin/env python3
"""One-time, confirmed evaluation of the Gate 5 CLint candidate; never refits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress

from dmpk_toolkit import load_task, metrics, prediction_table
from gate5_clint_common import (ROOT, TASK, load_feature_matrix, prediction_from_bundle,
    required_model_files, validate_frozen_bundle)
from pipeline_common import dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def molecule_bootstrap(frame: pd.DataFrame, prediction: np.ndarray, repeats: int, seed: int) -> dict:
    work = pd.DataFrame({"molecule_id": frame.molecule_id, "observed": np.log10(frame.raw_value.to_numpy(float)),
                         "predicted": np.log10(np.asarray(prediction, dtype=float))}).groupby("molecule_id", as_index=False).mean()
    squared = (work.predicted.to_numpy() - work.observed.to_numpy()) ** 2
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(work), size=(repeats, len(work)))
    values = np.sqrt(squared[sampled].mean(axis=1))
    return {"unit": "molecule", "molecules": len(work), "repeats": repeats, "seed": seed,
            "log10_rmse_95_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def test_similarity(train_bits: np.ndarray, test_bits: np.ndarray) -> np.ndarray:
    inter = np.asarray(test_bits, dtype=np.int32) @ np.asarray(train_bits, dtype=np.int32).T
    left, right = test_bits.sum(axis=1), train_bits.sum(axis=1)
    union = left[:, None] + right[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0).max(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final/CLint_gate5_stageA_et_c01_test_v1")
    parser.add_argument("--confirm-one-time-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check(required_model_files(args.frozen) + [args.stl_protocol / "complete.json"],
                       output=None if args.check_only else args.output)
    registry, contract = validate_frozen_bundle(args.frozen)
    protocol = json.loads((args.frozen / "evaluation_protocol.json").read_text(encoding="utf-8"))
    if protocol["bootstrap"]["repeats"] < 1000 or protocol["primary_metric"] != "molecule_mean_log10_RMSE":
        raise ValueError("Frozen Gate 5 evaluator protocol is invalid")
    if protocol.get("evaluator_code_sha256") != sha256(Path(__file__)):
        raise ValueError("Evaluator code differs from the pre-test frozen Gate 5 protocol")
    if args.check_only:
        print("Gate 5 CLint evaluator contract valid; test labels were not loaded.")
        return
    if not args.confirm_one_time_test:
        raise ValueError("One-time CLint test evaluation requires --confirm-one-time-test")
    startup_self_check([args.datasets / "complete.json", args.splits / "complete.json"], output=args.output)
    data_meta = verify_stage(args.datasets, "datasets")
    split_meta = verify_stage(args.splits, "splits")
    frame, spec, _ = load_task(ROOT, args.datasets, args.splits, TASK)
    test = frame.loc[frame.split.eq("test")].reset_index(drop=True)
    if test.empty or test.molecule_id.nunique() != 28:
        raise ValueError(f"Expected the registered 28-molecule CLint test, found {test.molecule_id.nunique()}")
    membership = pd.read_csv(args.frozen / "train_membership.csv")
    if set(test.scaffold_group) & set(membership.scaffold_group):
        raise ValueError("Frozen Gate 5 training scaffolds overlap the test set")
    feature_manifest = pd.read_csv(args.stl_protocol / "feature_row_manifest.csv")
    test_features = test[["molecule_id"]].merge(feature_manifest[["molecule_id", "feature_index"]], on="molecule_id", how="left", validate="many_to_one")
    if test_features.feature_index.isna().any():
        raise ValueError("A test molecule is absent from the frozen deterministic feature cache")
    matrix = load_feature_matrix(args.stl_protocol)
    bundle = joblib.load(args.frozen / "model.joblib")
    transformed, prediction = prediction_from_bundle(bundle, matrix[test_features.feature_index.to_numpy(int)])
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Gate 5 CLint test prediction is invalid")
    train_bits = matrix[membership.feature_index.to_numpy(int), :2048]
    similarity = test_similarity(train_bits, matrix[test_features.feature_index.to_numpy(int), :2048])
    result = metrics(test, prediction, spec)
    bootstrap = molecule_bootstrap(test, prediction, protocol["bootstrap"]["repeats"], protocol["bootstrap"]["seed"])
    observed_log, predicted_log = np.log10(test.raw_value.to_numpy(float)), np.log10(prediction)
    if np.unique(predicted_log).size < 2 or np.unique(observed_log).size < 2:
        calibration_payload = {"slope": None, "intercept": None, "rvalue": None,
                               "status": "not_estimable_constant_prediction_or_observation"}
    else:
        calibration = linregress(predicted_log, observed_log)
        calibration_payload = {"slope": float(calibration.slope), "intercept": float(calibration.intercept),
                               "rvalue": float(calibration.rvalue), "status": "estimated_descriptive_only"}
    table = prediction_table(test, prediction, spec, "gate5_frozen_one_time_test", membership.scaffold_group.tolist())
    table["max_train_tanimoto"] = similarity
    table["inside_train_only_AD"] = similarity >= float(protocol["applicability_domain"]["train_only_floor"])
    document_summary = (test.assign(max_train_tanimoto=similarity).groupby("doc_id", as_index=False)
                        .agg(records=("row_id", "size"), molecules=("molecule_id", "nunique"),
                             mean_max_train_tanimoto=("max_train_tanimoto", "mean")))
    document_summary["descriptive_metric_eligible"] = document_summary.molecules.ge(3)
    failures = {
        "invalid_or_unmapped_test_feature": 0,
        "train_scaffold_overlap": 0,
        "nonfinite_or_nonpositive_prediction": 0,
        "below_train_only_AD_floor": int((~table.inside_train_only_AD).sum()),
        "ad_floor": float(protocol["applicability_domain"]["train_only_floor"]),
        "test_records": len(test), "test_molecules": int(test.molecule_id.nunique()),
    }
    payload = {"task_id": TASK, "model": registry["candidate_id"], "full_test": result,
               "bootstrap": bootstrap, "calibration_descriptive": calibration_payload,
               "interpretation": "One-time internal fixed-split test of a Gate 4 pre-test-frozen candidate. Diagnostics are descriptive only; no refit, calibration, threshold update, or reselection is permitted.",
               "test_data_complete_sha256": sha256(args.datasets / "complete.json"), "test_splits_complete_sha256": sha256(args.splits / "complete.json")}
    with stage_output(args.output) as out:
        table.to_csv(out / "test_predictions.csv", index=False)
        document_summary.to_csv(out / "test_document_sensitivity.csv", index=False)
        dump_json(out / "metrics.json", payload)
        dump_json(out / "failure_report.json", failures)
        pd.DataFrame([{"task_id": TASK, "records": result["records"], "molecules": result["molecules"],
                         "log10_rmse": result["primary"], "log10_rmse_ci_lower": bootstrap["log10_rmse_95_ci"][0],
                         "log10_rmse_ci_upper": bootstrap["log10_rmse_95_ci"][1], "spearman_molecule_mean": result["spearman_molecule_mean"],
                         "gmfe": result["gmfe_positive_subset"], "within_2fold": result["within_2fold_positive_subset"],
                         "within_3fold": result["within_3fold_positive_subset"], "ad_covered_molecules": int(table.inside_train_only_AD.sum())}]).to_csv(out / "test_summary.csv", index=False, float_format="%.6f")
        (out / "analysis_report.md").write_text(
            "# Gate 5 CLint one-time test evaluation\n\n"
            f"The pre-test-frozen ExtraTrees c01 candidate was evaluated once on {result['molecules']} molecules. "
            f"Molecule-mean log10 RMSE was {result['primary']:.4f} with a pre-registered 95% bootstrap CI "
            f"[{bootstrap['log10_rmse_95_ci'][0]:.4f}, {bootstrap['log10_rmse_95_ci'][1]:.4f}]. "
            "This is an internal fixed-split result, not an independent external validation; diagnostics did not alter the candidate.\n",
            encoding="utf-8")
        finish_stage(out, "gate5_clint_one_time_test_evaluation", inputs={
            "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "splits_complete_sha256": sha256(args.splits / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
        }, task_id=TASK, test_evaluated=True, refit=False, calibration=False, selection_changed=False,
        bootstrap_replicates=protocol["bootstrap"]["repeats"], partial=False)
    print(f"Gate 5 CLint one-time test evaluation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
