#!/usr/bin/env python3
"""Train a public human-plasma half-life model without opening source-test labels.

The model is explicitly distinct from the strict direct-IV terminal-half-life
model.  Selection is by nested scaffold OOF within the public training split;
the public validation split is reported once and the blind source-test members
are never opened by this training program.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dmpk_toolkit import (ALGORITHMS, FEATURE_SETS, OPTIONAL_MODULES, candidates, featurize, fit_model,
                          metrics, prediction_table, primary_score, tune)
from pipeline_common import ROOT, dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TRAIN = "public_train"
VALIDATION = "public_validation"
TEST_MEMBERSHIP = "public_source_test_scaffold_isolated"
TASK_ID = "Thalf__human__plasma_public"
SPEC = {"endpoint": "Thalf", "transform": "log10", "unit": "h"}
REQUIRED_DEVELOPMENT = {"source_record_id", "SMILES", "parent_id", "scaffold_group", "fold_id",
                        "target_value", "raw_value_h", "cohort_usage"}


def load_development(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_DEVELOPMENT - set(frame.columns)
    if missing:
        raise ValueError(f"Public development table misses columns: {sorted(missing)}")
    if not set(frame.cohort_usage) <= {TRAIN, VALIDATION} or not {TRAIN, VALIDATION} <= set(frame.cohort_usage):
        raise ValueError("Development table must contain only public train and validation partitions")
    if frame.source_record_id.duplicated().any() or frame.SMILES.duplicated().any():
        raise ValueError("Public development table has duplicate source records or SMILES")
    frame = frame.rename(columns={"source_record_id": "row_id", "SMILES": "smiles", "parent_id": "molecule_id",
                                  "raw_value_h": "raw_value"}).copy()
    frame["split"] = np.where(frame.cohort_usage.eq(TRAIN), "train", "val")
    frame["task_id"] = TASK_ID
    if not np.isfinite(frame.raw_value.to_numpy(float)).all() or (frame.raw_value.to_numpy(float) <= 0).any():
        raise ValueError("Public development half-life must be positive finite hours")
    if not np.isclose(np.log10(frame.raw_value.to_numpy(float)), frame.target_value.to_numpy(float), rtol=1e-7, atol=1e-7).all():
        raise ValueError("Public development raw and log10 values disagree")
    train = frame.loc[frame.split.eq("train")]
    folds = sorted(train.fold_id.unique())
    if folds != list(range(5)) or (train.groupby("scaffold_group").fold_id.nunique() > 1).any():
        raise ValueError("Public development training folds are not the locked five scaffold folds")
    if set(train.scaffold_group) & set(frame.loc[frame.split.eq("val"), "scaffold_group"]):
        raise ValueError("Public development train and validation scaffold sets overlap")
    return frame.reset_index(drop=True)


def train_algorithms(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, list[dict], dict]:
    algorithms = list(dict.fromkeys(["dummy", *args.algorithms.split(",")]))
    if not set(algorithms) <= set(ALGORITHMS):
        raise ValueError(f"Unsupported public-model algorithms: {algorithms}")
    if args.device != "cpu":
        raise ValueError("This baseline program is CPU-only; GPU candidates require a separately pre-registered stage")
    for algorithm in algorithms:
        if algorithm in OPTIONAL_MODULES:
            importlib.import_module(OPTIONAL_MODULES[algorithm])
    train_idx = np.flatnonzero(frame.split.eq("train"))
    tuning: list[dict] = []
    summary: dict = {}
    bundles: dict = {}
    X, names = featurize(frame.smiles.tolist(), feature_set=args.feature_set)
    for algorithm in algorithms:
        options = candidates(algorithm, args.trials, args.seed)
        oof = np.full(len(frame), np.nan)
        fold_predictions = []
        for fold in range(5):
            fit_idx = train_idx[frame.iloc[train_idx].fold_id.to_numpy() != fold]
            held_idx = train_idx[frame.iloc[train_idx].fold_id.to_numpy() == fold]
            parameters = tune(frame, X, fit_idx, algorithm, options, SPEC, names, args, tuning, f"outer_{fold}")
            bundle = fit_model(frame, X, fit_idx, algorithm, parameters, SPEC, names, args)
            if set(bundle.train_groups) & set(frame.iloc[held_idx].scaffold_group):
                raise ValueError("Public-model OOF scaffold leakage")
            values = bundle.predict_features(X[held_idx], frame.iloc[held_idx].smiles.to_numpy())
            oof[held_idx] = values
            fold_predictions.append(prediction_table(frame.iloc[held_idx], values, SPEC, "nested_scaffold_oof", bundle.train_groups))
        if not np.isfinite(oof[train_idx]).all():
            raise ValueError("Public-model OOF predictions are incomplete")
        final_parameters = tune(frame, X, train_idx, algorithm, options, SPEC, names, args, tuning, "final_training_cv")
        final_bundle = fit_model(frame, X, train_idx, algorithm, final_parameters, SPEC, names, args)
        val_idx = np.flatnonzero(frame.split.eq("val"))
        if set(final_bundle.train_groups) & set(frame.iloc[val_idx].scaffold_group):
            raise ValueError("Public-model validation scaffold leakage")
        val_values = final_bundle.predict_features(X[val_idx], frame.iloc[val_idx].smiles.to_numpy())
        summary[algorithm] = {
            "nested_oof": metrics(frame.iloc[train_idx], oof[train_idx], SPEC),
            "public_validation": metrics(frame.iloc[val_idx], val_values, SPEC),
            "parameters": final_parameters,
            "oof_predictions": pd.concat(fold_predictions, ignore_index=True),
            "validation_predictions": prediction_table(frame.iloc[val_idx], val_values, SPEC, "public_validation", final_bundle.train_groups),
        }
        bundles[algorithm] = final_bundle
    selected = min(algorithms, key=lambda name: summary[name]["nested_oof"]["primary"])
    return summary, tuning, {"selected": selected, "bundles": bundles, "feature_names": names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path,
                        default=ROOT / "data/public_benchmarks/oneadmet_human_plasma_thalf_v2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models/public_thalf/oneadmet_human_plasma_v1")
    parser.add_argument("--algorithms", default="ridge,extratrees")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="rdkit2d")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        raise ValueError("trials and threads must be positive")
    meta = verify_stage(args.cohort, "oneadmet_public_thalf_cohort")
    development = args.cohort / "development_records.csv"
    source_test = args.cohort / "source_test_membership_blinded.csv"
    startup_self_check([development, source_test], output=None if args.check_only else args.output)
    # Header-only validation proves the trainer cannot receive source-test labels.
    test_header = pd.read_csv(source_test, nrows=0)
    if {"target_value", "raw_value_h"} & set(test_header.columns) or "cohort_usage" not in test_header.columns:
        raise ValueError("Public source-test membership is not label-blinded")
    frame = load_development(development)
    if args.check_only:
        print(f"Public Thalf development contract valid: train={int(frame.split.eq('train').sum())}; "
              f"validation={int(frame.split.eq('val').sum())}; source-test labels not loaded")
        return
    with threadpool_limits(limits=args.threads), stage_output(args.output) as out:
        summary, tuning, state = train_algorithms(frame, args)
        serializable = {}
        for algorithm, result in summary.items():
            folder = out / algorithm
            folder.mkdir()
            result["oof_predictions"].to_csv(folder / "oof_predictions.csv", index=False)
            result["validation_predictions"].to_csv(folder / "validation_predictions.csv", index=False)
            joblib.dump(state["bundles"][algorithm], folder / "model.joblib", compress=3)
            serializable[algorithm] = {key: value for key, value in result.items()
                                       if key not in {"oof_predictions", "validation_predictions"}}
        dump_json(out / "metrics.json", serializable)
        dump_json(out / "tuning_history.json", tuning)
        dump_json(out / "selection.json", {
            "algorithm": state["selected"], "criterion": "nested_scaffold_oof_log10_rmse",
            "caveat": "Public validation is reported once; source-test labels were not opened.",
            "endpoint": "public_human_plasma_half_life_hours",
            "not_a_strict_iv_terminal_model": True,
        })
        dump_json(out / "feature_schema.json", {"feature_set": args.feature_set, "descriptor_names": state["feature_names"]})
        (out / "README.md").write_text(
            "# OneADMET public human-plasma half-life training\n\n"
            "This model is trained only on the protected public development split. It is not a strict direct-IV "
            "terminal-half-life model. Source-test membership remained label-blinded during training; do not score it "
            "until a candidate has been frozen in a separate stage.\n",
            encoding="utf-8")
        finish_stage(out, "oneadmet_public_thalf_training", inputs={
            "cohort_complete_sha256": sha256(args.cohort / "complete.json"),
            "development_records_sha256": sha256(development),
            "source_test_membership_sha256": sha256(source_test),
        }, task_id=TASK_ID, endpoint="public_human_plasma_half_life_hours", algorithms=args.algorithms,
           feature_set=args.feature_set, seed=args.seed, source_test_labels_evaluated=False,
           strict_iv_terminal=False, partial=False)
    print(f"OneADMET public Thalf training stage: {args.output}")


if __name__ == "__main__":
    run_cli(main)
