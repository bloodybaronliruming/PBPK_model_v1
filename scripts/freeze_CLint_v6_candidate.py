#!/usr/bin/env python3
"""Freeze the pre-test CLint v6 RF three-seed ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline_common import ROOT, dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "CLint__human__microsome"
SEEDS = (2026, 2027, 2028)
RUNS = (
    "models/stl/CLint__human__microsome/v6_hlm_evidence_baseline",
    "models/stl/CLint__human__microsome/v6_hlm_evidence_seed2027",
    "models/stl/CLint__human__microsome/v6_hlm_evidence_seed2028",
)


def validate_member(root: Path, relative: str, seed: int):
    run = root / relative
    meta = verify_stage(run, "stl_training")
    if meta.get("task_id") != TASK or meta.get("seed") != seed or meta.get("test_evaluated"):
        raise ValueError(f"Ineligible CLint training member: {run}")
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if selection != {
        "algorithm": "rf",
        "criterion": "training_nested_oof_primary",
        "caveat": "selected OOF score is selection-biased; assess generalization on heldout validation/test",
        "cascade": "use explicit algorithm and protected outer scopes; do not reuse full-train predictions as training priors",
    }:
        raise ValueError(f"CLint selection contract changed: {run}")
    model = run / "rf/model.joblib"
    startup_self_check([model, run / "rf/val_predictions.csv"])
    bundle = joblib.load(model)
    if bundle.algorithm != "rf" or bundle.task_spec.get("endpoint") != "CLint":
        raise ValueError(f"CLint RF bundle contract mismatch: {run}")
    saved = pd.read_csv(run / "rf/val_predictions.csv")
    actual = bundle.predict_smiles(saved.smiles.tolist())
    if not np.allclose(actual, saved.predicted_physical.to_numpy(float), rtol=2e-6, atol=2e-8):
        raise ValueError(f"CLint RF reload mismatch: {run}")
    return run, meta, bundle


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/v6_rf3_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    runs = [ROOT / relative for relative in RUNS]
    startup_self_check([run / "complete.json" for run in runs], output=None if args.check_only else args.output)
    members = [validate_member(ROOT, relative, seed) for relative, seed in zip(RUNS, SEEDS)]
    common_inputs = members[0][1]["inputs"]
    expected_inputs = {
        "datasets_complete_sha256": sha256(ROOT / "data/processed_v6/datasets/complete.json"),
        "splits_complete_sha256": sha256(ROOT / "data/processed_v6/splits/complete.json"),
    }
    if common_inputs != expected_inputs or any(meta["inputs"] != common_inputs for _, meta, _ in members[1:]):
        raise ValueError("CLint ensemble members do not share the locked processed_v6 inputs")
    train_groups = members[0][2].train_groups
    if any(bundle.train_groups != train_groups for _, _, bundle in members[1:]):
        raise ValueError("CLint ensemble members have different training scaffold groups")
    registry = {
        "registry_version": 1,
        "task_id": TASK,
        "data_version": "processed_v6",
        "model_kind": "stl_seed_ensemble",
        "algorithm": "rf",
        "aggregation": "arithmetic_mean_log10_prediction",
        "seeds": list(SEEDS),
        "selection_rule": "training_nested_oof_primary; RF selected in all three seeds",
        "selection_caveat": (
            "ExtraTrees had lower validation RMSE in all three seeds, while RF had lower nested OOF RMSE in all three. "
            "The already saved per-seed selection rule gives precedence to nested OOF; this registry does not use test labels."),
        "members": [{
            "seed": seed,
            "run_dir": str(run.resolve()),
            "run_complete_sha256": sha256(run / "complete.json"),
            "model_sha256": sha256(run / "rf/model.joblib"),
        } for seed, (run, _, _) in zip(SEEDS, members)],
        "test_labels_evaluated": False,
        "policy": "No member, feature, hyperparameter, aggregation, or selection rule may change after test evaluation.",
    }
    if args.check_only:
        print("CLint v6 freeze contract valid; test labels were not loaded.")
        return
    with stage_output(args.output) as out:
        dump_json(out / "candidate_registry.json", registry)
        (out / "selection_report.md").write_text(
            "# Frozen CLint v6 candidate\n\n"
            "The candidate is the geometric mean on the physical CLint scale of three RF models (seeds 2026--2028), "
            "equivalent to the arithmetic mean in log10 space. In every seed, RF was selected by the pre-existing "
            "nested-OOF primary-RMSE rule. ExtraTrees had lower validation RMSE in every seed; this disagreement is "
            "retained as a limitation and was not resolved with test labels. The test labels remain unread.\n",
            encoding="utf-8")
        finish_stage(out, "clint_v6_frozen_candidate", inputs={
            str(run.resolve()): sha256(run / "complete.json") for run, _, _ in members
        }, task_id=TASK, algorithm="rf", seeds=list(SEEDS), test_labels_evaluated=False, partial=False)
    print(f"Frozen CLint v6 candidate: {args.output}; test labels not evaluated.")


if __name__ == "__main__":
    run_cli(main)
