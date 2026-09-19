#!/usr/bin/env python3
"""Freeze the selected OneADMET public human-plasma candidate before source-test scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


EXPECTED_STAGE = "oneadmet_public_thalf_training"
TASK_ID = "Thalf__human__plasma_public"


def candidate_contract(training: Path, cohort: Path) -> dict:
    meta = verify_stage(training, EXPECTED_STAGE)
    cohort_meta = verify_stage(cohort, "oneadmet_public_thalf_cohort")
    if meta.get("task_id") != TASK_ID or meta.get("strict_iv_terminal") or meta.get("source_test_labels_evaluated"):
        raise ValueError("Training stage is not an unscored public human-plasma candidate")
    expected_inputs = {
        "cohort_complete_sha256": sha256(cohort / "complete.json"),
        "development_records_sha256": sha256(cohort / "development_records.csv"),
        "source_test_membership_sha256": sha256(cohort / "source_test_membership_blinded.csv"),
    }
    if meta.get("inputs") != expected_inputs:
        raise ValueError("Training stage is not linked to the current protected public cohort")
    selection = json.loads((training / "selection.json").read_text(encoding="utf-8"))
    algorithm = selection.get("algorithm")
    if algorithm != "extratrees" or selection.get("criterion") != "nested_scaffold_oof_log10_rmse":
        raise ValueError("Only the preselected public ExtraTrees candidate can be frozen")
    model_path = training / algorithm / "model.joblib"
    validation_path = training / algorithm / "validation_predictions.csv"
    bundle = joblib.load(model_path)
    validation = pd.read_csv(validation_path)
    predicted = bundle.predict_smiles(validation.smiles.tolist())
    if not np.allclose(predicted, validation.predicted_physical.to_numpy(float), rtol=2e-6, atol=2e-8):
        raise ValueError("Frozen public candidate cannot reproduce its validation predictions")
    test_header = pd.read_csv(cohort / "source_test_membership_blinded.csv", nrows=0)
    if {"target_value", "raw_value_h"} & set(test_header.columns):
        raise ValueError("Source-test membership contains labels and cannot support a frozen evaluation")
    return {
        "task_id": TASK_ID,
        "endpoint": "public_human_plasma_half_life_hours",
        "strict_iv_terminal": False,
        "model_kind": "stl_public_candidate",
        "algorithm": algorithm,
        "feature_set": meta["feature_set"],
        "seed": meta["seed"],
        "training_stage": str(training.resolve()),
        "training_complete_sha256": sha256(training / "complete.json"),
        "model_sha256": sha256(model_path),
        "validation_predictions_sha256": sha256(validation_path),
        "cohort_complete_sha256": sha256(cohort / "complete.json"),
        "source_test_membership_sha256": sha256(cohort / "source_test_membership_blinded.csv"),
        "selection": selection,
        "source_test_labels_evaluated": False,
        "policy": (
            "No algorithm, feature set, hyperparameter, calibration, cohort member, or data source may change after "
            "this registry. Source-test labels may be opened only by a separate evaluator. This public model is not a "
            "strict human direct-IV terminal-half-life model and cannot be merged with the formal external registry."),
        "cohort_protected_formal_external_molecules": cohort_meta.get("protected_formal_external_molecules"),
        "cohort_protected_pksmart_records": cohort_meta.get("protected_pksmart_records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path,
                        default=ROOT / "models/public_thalf/oneadmet_human_plasma_v1")
    parser.add_argument("--cohort", type=Path,
                        default=ROOT / "data/public_benchmarks/oneadmet_human_plasma_thalf_v2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models/frozen/Thalf__human__plasma_public/oneadmet_v1_et_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check([args.training / "complete.json", args.training / "selection.json",
                        args.training / "extratrees/model.joblib", args.training / "extratrees/validation_predictions.csv",
                        args.cohort / "complete.json", args.cohort / "development_records.csv",
                        args.cohort / "source_test_membership_blinded.csv"], output=None if args.check_only else args.output)
    contract = candidate_contract(args.training, args.cohort)
    if args.check_only:
        print("OneADMET public candidate contract valid; source-test labels remain unopened")
        return
    with stage_output(args.output) as out:
        (out / "candidate_registry.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "README.md").write_text(
            "# Frozen OneADMET public human-plasma half-life candidate\n\n"
            "This registry freezes the selected ExtraTrees public human-plasma model before source-test evaluation. "
            "It is not a strict IV-terminal model. A separate evaluator must verify this registry and open the "
            "source test once without retraining, reselection, or calibration.\n",
            encoding="utf-8")
        finish_stage(out, "oneadmet_public_thalf_frozen_candidate", inputs={
            "training_complete_sha256": contract["training_complete_sha256"],
            "cohort_complete_sha256": contract["cohort_complete_sha256"],
        }, task_id=TASK_ID, source_test_labels_evaluated=False, selection_changed=False,
           strict_iv_terminal=False, partial=False)
    print(f"Frozen OneADMET public Thalf candidate: {args.output}")


if __name__ == "__main__":
    run_cli(main)
