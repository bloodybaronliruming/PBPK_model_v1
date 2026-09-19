#!/usr/bin/env python3
"""Evaluate the frozen CLint v6 RF ensemble once, without refitting or reselection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dmpk_toolkit import load_task, metrics, prediction_table
from pipeline_common import ROOT, dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "CLint__human__microsome"


def geometric_ensemble(member_predictions):
    values = np.asarray(member_predictions, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("CLint ensemble members must provide at least two positive finite prediction vectors")
    return np.power(10.0, np.log10(values).mean(axis=0))


def bootstrap_log10_rmse(frame: pd.DataFrame, prediction: np.ndarray, repeats: int, seed: int) -> dict:
    work = pd.DataFrame({
        "molecule_id": frame.molecule_id.to_numpy(),
        "observed": np.log10(frame.raw_value.to_numpy(float)),
        "predicted": np.log10(np.asarray(prediction, dtype=float)),
    }).groupby("molecule_id", sort=True).mean()
    loss = (work.predicted.to_numpy() - work.observed.to_numpy()) ** 2
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(work), size=(repeats, len(work)))
    values = np.sqrt(loss[sampled].mean(axis=1))
    return {"replicates": repeats, "seed": seed,
            "log10_rmse_95_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def validate_contract(frozen: Path) -> dict:
    meta = verify_stage(frozen, "clint_v6_frozen_candidate")
    registry = json.loads((frozen / "candidate_registry.json").read_text(encoding="utf-8"))
    if meta.get("test_labels_evaluated") or registry.get("test_labels_evaluated"):
        raise ValueError("Frozen CLint registry must precede test evaluation")
    if registry.get("task_id") != TASK or registry.get("algorithm") != "rf" or registry.get("aggregation") != "arithmetic_mean_log10_prediction":
        raise ValueError("Frozen CLint registry has an unexpected model contract")
    if len(registry.get("members", [])) != 3:
        raise ValueError("Frozen CLint registry must contain three RF members")
    for member in registry["members"]:
        run = Path(member["run_dir"])
        if sha256(run / "complete.json") != member["run_complete_sha256"] or sha256(run / "rf/model.joblib") != member["model_sha256"]:
            raise ValueError(f"Frozen CLint member changed: {run}")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/v6_rf3_v1")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v6/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v6/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final/CLint_v6_rf3_test_v1")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--confirm-frozen-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check([args.frozen / "complete.json", args.frozen / "candidate_registry.json",
                        args.datasets / "complete.json", args.splits / "complete.json"],
                       output=None if args.check_only else args.output)
    registry = validate_contract(args.frozen)
    if args.bootstrap < 1000:
        raise ValueError("--bootstrap must be at least 1000")
    if args.check_only:
        print("Frozen CLint registry and member hashes verified; test labels were not loaded.")
        return
    if not args.confirm_frozen_test:
        raise ValueError("One-time CLint test evaluation requires --confirm-frozen-test")
    frame, spec, _ = load_task(ROOT, args.datasets, args.splits, TASK)
    test = frame.loc[frame.split.eq("test")].reset_index(drop=True)
    if test.molecule_id.nunique() != 28:
        raise ValueError(f"Expected 28 frozen CLint test molecules, found {test.molecule_id.nunique()}")
    predictions, groups = [], None
    for member in registry["members"]:
        bundle = joblib.load(Path(member["run_dir"]) / "rf/model.joblib")
        if set(bundle.train_groups) & set(test.scaffold_group):
            raise ValueError("Frozen CLint model has training-scaffold overlap with test")
        if groups is None:
            groups = bundle.train_groups
        elif groups != bundle.train_groups:
            raise ValueError("Frozen CLint members disagree on training scaffolds")
        predictions.append(bundle.predict_smiles(test.smiles.tolist()))
    prediction = geometric_ensemble(predictions)
    result = metrics(test, prediction, spec)
    interval = bootstrap_log10_rmse(test, prediction, args.bootstrap, 20260915)
    table = prediction_table(test, prediction, spec, "frozen_rf3_test", groups)
    payload = {
        "task_id": TASK,
        "model": "RF seeds 2026-2028 log10 ensemble",
        "full_test": result,
        "bootstrap": interval,
        "interpretation": "One-time internal fixed-split test of a pre-test frozen CLint candidate; limited to 28 molecules and not an independent external validation.",
    }
    with stage_output(args.output) as out:
        table.to_csv(out / "test_predictions.csv", index=False)
        dump_json(out / "metrics.json", payload)
        pd.DataFrame([{
            "task_id": TASK, "records": result["records"], "molecules": result["molecules"],
            "log10_rmse": result["primary"], "log10_rmse_ci_lower": interval["log10_rmse_95_ci"][0],
            "log10_rmse_ci_upper": interval["log10_rmse_95_ci"][1],
            "spearman_molecule_mean": result["spearman_molecule_mean"],
            "gmfe": result["gmfe_positive_subset"], "within_2fold": result["within_2fold_positive_subset"],
            "within_3fold": result["within_3fold_positive_subset"],
        }]).to_csv(out / "test_summary.csv", index=False, float_format="%.6f")
        (out / "analysis_report.md").write_text(
            "# CLint v6 frozen test evaluation\n\n"
            f"The frozen three-seed RF ensemble was evaluated once on {result['molecules']} molecules without refitting "
            f"or reselection. Log10 RMSE was {result['primary']:.4f} with molecule-bootstrap 95% CI "
            f"[{interval['log10_rmse_95_ci'][0]:.4f}, {interval['log10_rmse_95_ci'][1]:.4f}]. "
            "This is a limited internal fixed-split result, not independent external validation.\n",
            encoding="utf-8")
        finish_stage(out, "clint_v6_frozen_test_evaluation", inputs={
            "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "splits_complete_sha256": sha256(args.splits / "complete.json"),
        }, task_id=TASK, test_evaluated=True, refit=False, calibration=False, selection_changed=False,
           bootstrap_replicates=args.bootstrap, partial=False)
    print(f"CLint frozen test evaluation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
