#!/usr/bin/env python3
"""Run the final documented Gate 5 CLint completion attempt with the frozen parent-ID contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress

from dmpk_toolkit import load_task, metrics, prediction_table
from gate5_clint_common import (ROOT, TASK, prediction_from_bundle, recompute_canonical_features,
    required_model_files, validate_frozen_bundle, verify_recomputed_feature_contract)
from pipeline_common import dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def molecule_bootstrap(frame: pd.DataFrame, prediction: np.ndarray, repeats: int, seed: int) -> dict:
    work = pd.DataFrame({
        "molecule_id": frame.molecule_id,
        "observed": np.log10(frame.raw_value.to_numpy(float)),
        "predicted": np.log10(np.asarray(prediction, dtype=float)),
    }).groupby("molecule_id", as_index=False).mean()
    squared = (work.predicted.to_numpy() - work.observed.to_numpy()) ** 2
    rng = np.random.default_rng(seed)
    values = np.sqrt(squared[rng.integers(0, len(work), size=(repeats, len(work)))].mean(axis=1))
    return {"unit": "molecule", "molecules": len(work), "repeats": repeats, "seed": seed,
            "log10_rmse_95_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def max_train_tanimoto(train_bits: np.ndarray, test_bits: np.ndarray) -> np.ndarray:
    train = np.asarray(train_bits, dtype=np.int32)
    test = np.asarray(test_bits, dtype=np.int32)
    inter = test @ train.T
    union = test.sum(axis=1)[:, None] + train.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0).max(axis=1)


def frozen_parent_ids(test: pd.DataFrame, split_manifest: Path) -> list[str]:
    """Map source identities to the separately frozen canonical-parent identities without reading labels."""
    columns = ["molecule_id", "parent_id", "smiles", "split", "eligible", "standardization_status"]
    manifest = pd.read_csv(split_manifest, usecols=columns, dtype=str, keep_default_na=False)
    if manifest.molecule_id.duplicated().any():
        raise ValueError("Frozen split manifest molecule_id is not unique")
    lookup = manifest.set_index("molecule_id")
    missing = set(test.molecule_id) - set(lookup.index)
    if missing:
        raise ValueError("A test source molecule is absent from the frozen split manifest")
    aligned = lookup.loc[test.molecule_id.tolist()].reset_index()
    eligible = aligned.eligible.str.lower().isin(["true", "1"])
    if (not aligned.split.eq("test").all() or not eligible.all()
            or not aligned.standardization_status.eq("ok").all()
            or aligned.parent_id.str.len().eq(0).any()):
        raise ValueError("Frozen test parent identity contract is ineligible or incomplete")
    if aligned.molecule_id.tolist() != test.molecule_id.tolist() or aligned.smiles.tolist() != test.smiles.tolist():
        raise ValueError("Frozen split source identity or SMILES ordering disagrees with the task table")
    return aligned.parent_id.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2")
    parser.add_argument("--recovery-protocol", type=Path,
                        default=ROOT / "data/public_development/gate5_clint_technical_recovery_v2")
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--stl-protocol", type=Path,
                        default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/final/CLint_gate5_stageA_et_c01_test_recovery_v2")
    parser.add_argument("--confirm-final-technical-recovery-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check(required_model_files(args.frozen) + [
        args.recovery_protocol / "complete.json", args.stl_protocol / "complete.json",
        args.splits / "complete.json",
    ], output=None if args.check_only else args.output)
    registry, _ = validate_frozen_bundle(args.frozen)
    recovery = verify_stage(args.recovery_protocol, "gate5_clint_technical_recovery_protocol_v2")
    manifest = json.loads((args.recovery_protocol / "recovery_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
        "model_sha256": sha256(args.frozen / "model.joblib"),
        "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
        "splits_complete_sha256": sha256(args.splits / "complete.json"),
        "recovery_evaluator_sha256": sha256(Path(__file__)),
        "recovery_common_sha256": sha256(ROOT / "scripts/gate5_clint_common.py"),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery v2 protocol no longer matches its frozen inputs or evaluator")
    if manifest.get("immutable_candidate_id") != registry["candidate_id"]:
        raise ValueError("Recovery v2 candidate differs from the frozen Gate 5 candidate")
    if manifest.get("prediction_or_metric_output_published") or recovery.get("test_evaluated"):
        raise ValueError("Recovery v2 is ineligible after a scored test")
    feature_audit = verify_recomputed_feature_contract(args.frozen, args.stl_protocol)
    if args.check_only:
        print(f"Gate 5 recovery v2 evaluator valid: train-only feature equivalence checked for {feature_audit['parents_checked']} parents; test labels not loaded.")
        return
    if not args.confirm_final_technical_recovery_test:
        raise ValueError("Final recovery requires --confirm-final-technical-recovery-test")
    startup_self_check([args.datasets / "complete.json", args.splits / "complete.json"], output=args.output)
    frame, spec, _ = load_task(ROOT, args.datasets, args.splits, TASK)
    test = frame.loc[frame.split.eq("test")].reset_index(drop=True)
    if test.empty or len(test) != 28 or test.molecule_id.nunique() != 28:
        raise ValueError(f"Expected 28 unique registered CLint test molecules, found {len(test)} rows / {test.molecule_id.nunique()} molecules")
    membership = pd.read_csv(args.frozen / "train_membership.csv")
    if set(test.scaffold_group) & set(membership.scaffold_group):
        raise ValueError("Frozen Gate 5 training scaffolds overlap the test set")
    parent_ids = frozen_parent_ids(test, args.splits / "split_manifest.csv")
    test_matrix = recompute_canonical_features(test.smiles.tolist(), parent_ids)
    bundle = joblib.load(args.frozen / "model.joblib")
    _, prediction = prediction_from_bundle(bundle, test_matrix)
    train_rows = membership.merge(
        pd.read_csv(args.stl_protocol / "feature_row_manifest.csv")[["molecule_id", "feature_index", "canonical_parent_smiles"]],
        on=["molecule_id", "feature_index"], validate="one_to_one")
    train_matrix = recompute_canonical_features(train_rows.canonical_parent_smiles.tolist(), train_rows.molecule_id.tolist())
    similarity = max_train_tanimoto(train_matrix[:, :2048], test_matrix[:, :2048])
    protocol = json.loads((args.frozen / "evaluation_protocol.json").read_text(encoding="utf-8"))
    result = metrics(test, prediction, spec)
    bootstrap = molecule_bootstrap(test, prediction, protocol["bootstrap"]["repeats"], protocol["bootstrap"]["seed"])
    observed_log, predicted_log = np.log10(test.raw_value.to_numpy(float)), np.log10(prediction)
    calibration = (None if np.unique(predicted_log).size < 2 or np.unique(observed_log).size < 2
                   else linregress(predicted_log, observed_log))
    calibration_payload = ({"slope": None, "intercept": None, "rvalue": None,
                            "status": "not_estimable_constant_prediction_or_observation"}
                           if calibration is None else {"slope": float(calibration.slope),
                           "intercept": float(calibration.intercept), "rvalue": float(calibration.rvalue),
                           "status": "estimated_descriptive_only"})
    table = prediction_table(test, prediction, spec, "gate5_final_technical_recovery_test", membership.scaffold_group.tolist())
    table["canonical_parent_id"] = parent_ids
    table["max_train_tanimoto"] = similarity
    table["inside_train_only_AD"] = similarity >= float(protocol["applicability_domain"]["train_only_floor"])
    documents = (test.assign(max_train_tanimoto=similarity).groupby("doc_id", as_index=False)
                 .agg(records=("row_id", "size"), molecules=("molecule_id", "nunique"),
                      mean_max_train_tanimoto=("max_train_tanimoto", "mean")))
    documents["descriptive_metric_eligible"] = documents.molecules.ge(3)
    failures = {
        "invalid_or_unmapped_test_feature": 0,
        "source_to_parent_identity_failures": 0,
        "train_scaffold_overlap": 0,
        "nonfinite_or_nonpositive_prediction": int((~np.isfinite(prediction) | (prediction <= 0)).sum()),
        "below_train_only_AD_floor": int((~table.inside_train_only_AD).sum()),
        "ad_floor": float(protocol["applicability_domain"]["train_only_floor"]),
        "final_technical_recovery": True,
    }
    payload = {
        "task_id": TASK,
        "model": registry["candidate_id"],
        "full_test": result,
        "bootstrap": bootstrap,
        "calibration_descriptive": calibration_payload,
        "technical_recovery_protocol_sha256": sha256(args.recovery_protocol / "complete.json"),
        "interpretation": (
            "Final documented completion after two pre-prediction technical aborts. The unchanged pre-test-frozen model "
            "used the frozen split parent_id solely for canonical feature identity validation. No refit, calibration, "
            "threshold update, or candidate reselection occurred."
        ),
    }
    with stage_output(args.output) as out:
        table.to_csv(out / "test_predictions.csv", index=False)
        documents.to_csv(out / "test_document_sensitivity.csv", index=False)
        dump_json(out / "metrics.json", payload)
        dump_json(out / "failure_report.json", failures)
        pd.DataFrame([{
            "task_id": TASK, "records": result["records"], "molecules": result["molecules"],
            "log10_rmse": result["primary"], "log10_rmse_ci_lower": bootstrap["log10_rmse_95_ci"][0],
            "log10_rmse_ci_upper": bootstrap["log10_rmse_95_ci"][1],
            "spearman_molecule_mean": result["spearman_molecule_mean"], "gmfe": result["gmfe_positive_subset"],
            "within_2fold": result["within_2fold_positive_subset"],
            "within_3fold": result["within_3fold_positive_subset"],
            "ad_covered_molecules": int(table.inside_train_only_AD.sum()), "technical_recovery": True,
        }]).to_csv(out / "test_summary.csv", index=False, float_format="%.6f")
        (out / "analysis_report.md").write_text(
            "# Gate 5 CLint final technical recovery test evaluation\n\n"
            f"The unchanged pre-test-frozen ExtraTrees c01 candidate was evaluated on {result['molecules']} molecules "
            "after two documented pre-prediction technical aborts. "
            f"Molecule-mean log10 RMSE was {result['primary']:.4f} with the pre-registered 95% bootstrap CI "
            f"[{bootstrap['log10_rmse_95_ci'][0]:.4f}, {bootstrap['log10_rmse_95_ci'][1]:.4f}]. "
            "No refit, calibration, threshold update, or candidate reselection occurred.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate5_clint_final_technical_recovery_test_evaluation", inputs={
            "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
            "recovery_protocol_complete_sha256": sha256(args.recovery_protocol / "complete.json"),
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "splits_complete_sha256": sha256(args.splits / "complete.json"),
        }, task_id=TASK, test_evaluated=True, technical_recovery=True, refit=False,
        calibration=False, selection_changed=False, bootstrap_replicates=protocol["bootstrap"]["repeats"],
        partial=False)
    print(f"Gate 5 CLint final technical recovery test evaluation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
