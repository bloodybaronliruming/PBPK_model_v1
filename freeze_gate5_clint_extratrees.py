#!/usr/bin/env python3
"""Fit and package the Gate 4-approved CLint ExtraTrees candidate without test access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gate5_clint_common import (CANDIDATE_ID, PARAMETERS, ROOT, SEED, TASK, load_feature_matrix,
    load_gate4_candidate, read_train_records, train_nearest_similarity)
from pipeline_common import dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from stl_benchmark_common import make_estimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate4", type=Path, default=ROOT / "data/public_development/cross_gate_candidate_freeze_v1")
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--stagea-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    startup_self_check([args.gate4 / "complete.json", args.train_protocol / "complete.json",
                        args.stl_protocol / "complete.json", args.stagea_audit / "complete.json"],
                       output=None if args.check_only else args.output)
    gate4_meta, gate4_candidate = load_gate4_candidate(args.gate4)
    train = read_train_records(args.train_protocol)
    matrix = load_feature_matrix(args.stl_protocol)
    if int(gate4_candidate["train_records"]) != len(train) or int(gate4_candidate["train_parents"]) != train.molecule_id.nunique():
        raise ValueError("Gate 4 CLint sample count no longer matches the train-only protocol")
    if train.feature_index.min() < 0 or train.feature_index.max() >= len(matrix):
        raise ValueError("CLint train feature index is outside the frozen cache")
    upstream = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    stagea_audit = verify_stage(args.stagea_audit, "stl_stageA_three_view_audit")
    if gate4_meta["inputs"]["stagea_complete_sha256"] != sha256(args.stagea_audit / "complete.json"):
        raise ValueError("Gate 4 does not point to the published Stage-A audit")
    if stagea_audit["inputs"].get("combined_complete_sha256") != sha256(ROOT / "results/benchmarks/stl_stageA_ecfp4_rdkit2d_v1/complete.json"):
        raise ValueError("Stage-A audit no longer points to the published combined benchmark")
    if verify_stage(args.train_protocol, "stl_train_only_protocol")["inputs"]["upstream_complete_sha256"] != sha256(args.stl_protocol / "complete.json"):
        raise ValueError("Train-only protocol does not originate from the frozen STL feature protocol")
    target = train.target_value.to_numpy(float)
    mean, std = float(target.mean()), float(target.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("CLint full-train target standard deviation is invalid")
    x_train = matrix[train.feature_index.to_numpy(int)]
    if not np.isfinite(x_train).all(axis=None, where=~np.isnan(x_train)):
        raise ValueError("CLint feature input contains an invalid nonmissing value")
    nearest, ad_floor = train_nearest_similarity(x_train[:, :2048])
    if args.check_only:
        print("Gate 5 CLint preflight valid: train-only labels, exact ExtraTrees c01, unread test.")
        return
    estimator = make_estimator("extra_trees", SEED, args.threads, small_budget=True, parameters=PARAMETERS)
    estimator.fit(x_train, (target - mean) / std)
    probe = x_train[: min(16, len(x_train))]
    before = np.asarray(estimator.predict(probe), dtype=float)
    with stage_output(args.output) as out:
        model_path = out / "model.joblib"
        joblib.dump({
            "format_version": 1, "task_id": TASK, "candidate_id": CANDIDATE_ID,
            "estimator": estimator, "target_mean": mean, "target_standard_deviation": std,
            "feature_set": "ecfp4_rdkit2d", "feature_dimensions": int(x_train.shape[1]),
            "train_scaffold_groups": sorted(train.scaffold_group.unique().tolist()),
            "train_molecule_ids": sorted(train.molecule_id.tolist()),
        }, model_path, compress=3)
        restored = joblib.load(model_path)
        after = np.asarray(restored["estimator"].predict(probe), dtype=float)
        delta = float(np.max(np.abs(before - after)))
        if delta > 1e-12:
            raise ValueError(f"Gate 5 CLint reload mismatch: {delta:.3g}")
        train.loc[:, ["molecule_id", "scaffold_group", "feature_index"]].to_csv(out / "train_membership.csv", index=False)
        dump_json(out / "candidate_registry.json", {
            "registry_version": 1, "task_id": TASK, "candidate_id": CANDIDATE_ID,
            "gate4_candidate_registry_sha256": sha256(args.gate4 / "endpoint_candidate_freeze_registry.csv"),
            "gate4_complete_sha256": sha256(args.gate4 / "complete.json"),
            "stagea_audit_complete_sha256": sha256(args.stagea_audit / "complete.json"),
            "stagea_complete_sha256": sha256(ROOT / "results/benchmarks/stl_stageA_ecfp4_rdkit2d_v1/complete.json"),
            "algorithm": "extra_trees", "parameters": PARAMETERS, "seed": SEED,
            "model_sha256": sha256(model_path), "train_records": len(train),
            "train_parents": int(train.molecule_id.nunique()), "test_labels_evaluated": False,
            "evaluator_code_sha256": sha256(ROOT / "scripts/evaluate_gate5_clint_frozen.py"),
            "selection_statement": "Candidate fixed by Gate 4 train-CV evidence before test access; no validation/test metric, calibration, or reselection used.",
            "post_test_policy": "No refit, feature change, calibration, threshold change, candidate swap, or hyperparameter change is permitted after the one-time test evaluation.",
        })
        dump_json(out / "model_contract.json", {
            "task_id": TASK, "endpoint": "CLint", "canonical_unit": "uL/min/mg_protein",
            "algorithm": "extra_trees", "parameters": PARAMETERS, "seed": SEED,
            "feature_set": "ecfp4_rdkit2d", "feature_dimensions": int(x_train.shape[1]),
            "target_transform": "log10", "target_standardization": "full_train_parent_mean_population_sd",
            "target_mean": mean, "target_standard_deviation": std, "imputation": "median_fit_on_full_train_only",
            "reload_probe_rows": len(probe), "reload_max_abs_difference": delta,
        })
        dump_json(out / "evaluation_protocol.json", {
            "protocol_version": 1, "test_access": "single confirmed read after frozen model hash verification",
            "evaluator_code_sha256": sha256(ROOT / "scripts/evaluate_gate5_clint_frozen.py"),
            "primary_metric": "molecule_mean_log10_RMSE", "bootstrap": {"unit": "molecule", "repeats": 10000, "seed": 20260918, "confidence_interval": "percentile_95"},
            "secondary_metrics": ["physical_MAE", "physical_RMSE", "physical_R2", "Spearman_molecule_mean", "GMFE", "within_2fold", "within_3fold"],
            "calibration": "post-test descriptive log10 observed_on_predicted slope/intercept only; no calibration refit",
            "source_sensitivity": "descriptive test-document summaries only when a document has >=3 unique parents; never used for candidate selection",
            "applicability_domain": {"representation": "ECFP4", "score": "max_train_Tanimoto", "train_only_floor": ad_floor,
                                      "floor_definition": "5th percentile of leave-one-out max train Tanimoto", "train_nearest_summary": {"min": float(nearest.min()), "median": float(np.median(nearest)), "max": float(nearest.max())}},
            "failure_report": ["invalid_or_unmapped_test_feature", "train_scaffold_overlap", "nonfinite_or_nonpositive_prediction", "below_train_only_AD_floor"],
            "prohibitions": ["test-driven_refit", "test-driven_calibration", "test-driven_candidate_change", "test-driven_threshold_change"],
        })
        (out / "README.md").write_text(
            "# Gate 5 CLint frozen candidate\n\n"
            "Full-train ECFP4+RDKit2D ExtraTrees c01, selected by Gate 4 before test access. "
            "The package contains no validation/test labels and fixes the one-time evaluator contract.\n",
            encoding="utf-8")
        finish_stage(out, "gate5_clint_stagea_et_candidate", inputs={
            "gate4_complete_sha256": sha256(args.gate4 / "complete.json"),
            "gate4_registry_sha256": sha256(args.gate4 / "endpoint_candidate_freeze_registry.csv"),
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "stagea_audit_complete_sha256": sha256(args.stagea_audit / "complete.json"),
        }, task_id=TASK, candidate_id=CANDIDATE_ID, algorithm="extra_trees", seed=SEED,
        train_records=len(train), train_parents=int(train.molecule_id.nunique()),
        validation_labels_read=False, test_labels_read=False, test_evaluated=False, model_fitted=True, partial=False)
    print(f"Gate 5 CLint candidate frozen: {args.output}; validation/test labels not read.")


if __name__ == "__main__":
    run_cli(main)
