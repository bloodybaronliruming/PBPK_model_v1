#!/usr/bin/env python3
"""Evaluate the frozen OneADMET public candidate once on its blind source test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_split_manifest import structure_groups
from evaluate_pksmart_public_benchmark import aggregate_molecules, bootstrap_metrics, metric_values
from pipeline_common import ROOT, dump_json, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


TASK = "Half-Life_Human-Plasma-pHours_Public.csv"
REQUIRED_MEMBERSHIP = {"source_record_id", "SMILES", "Set", "parent_id", "scaffold_group", "cohort_usage"}


def read_source_test_labels(raw: Path, membership: pd.DataFrame) -> pd.DataFrame:
    """Open only the pre-registered source-test labels after all contracts pass."""
    wanted = set(membership.SMILES)
    parts = []
    for chunk in pd.read_csv(raw, usecols=["SMILES", "Set", TASK], chunksize=20_000, low_memory=False):
        chosen = chunk.loc[chunk.SMILES.isin(wanted), ["SMILES", "Set", TASK]]
        if not chosen.empty:
            parts.append(chosen)
    if not parts:
        raise ValueError("No frozen source-test structures are present in OneADMET raw CSV")
    labels = pd.concat(parts, ignore_index=True)
    if labels.SMILES.duplicated().any() or set(labels.SMILES) != wanted:
        raise ValueError("OneADMET raw source-test structure membership changed")
    attached = membership.merge(labels, on="SMILES", how="left", validate="one_to_one",
                                suffixes=("_membership", "_raw"))
    if (not attached["Set_membership"].eq("test").all()
            or not attached["Set_raw"].eq("test").all()
            or attached[TASK].isna().any()):
        raise ValueError("Frozen source-test membership no longer has complete author-test labels")
    expected_ids = [stable_id(f"oneadmet|{smiles}|{TASK}") for smiles in attached.SMILES]
    if not np.array_equal(attached.source_record_id.to_numpy(), np.asarray(expected_ids)):
        raise ValueError("OneADMET source-test record identifiers changed")
    parents, scaffolds = zip(*(structure_groups(smiles) for smiles in attached.SMILES))
    if not np.array_equal(attached.parent_id.to_numpy(), np.asarray(parents)) or not np.array_equal(attached.scaffold_group.to_numpy(), np.asarray(scaffolds)):
        raise ValueError("OneADMET source-test standardized structures changed")
    observed = pd.to_numeric(attached[TASK], errors="coerce")
    if observed.isna().any() or not np.isfinite(observed).all():
        raise ValueError("OneADMET source-test contains nonfinite pHours values")
    attached["observed_h"] = np.power(10.0, observed.to_numpy(float))
    if not np.isfinite(attached.observed_h).all() or (attached.observed_h <= 0).any():
        raise ValueError("OneADMET source-test contains invalid physical half-life values")
    return attached


def validate_contract(frozen: Path, cohort: Path, auxiliary: Path, raw: Path) -> tuple[dict, pd.DataFrame]:
    frozen_meta = verify_stage(frozen, "oneadmet_public_thalf_frozen_candidate")
    cohort_meta = verify_stage(cohort, "oneadmet_public_thalf_cohort")
    auxiliary_meta = verify_stage(auxiliary, "oneadmet_auxiliary")
    registry = json.loads((frozen / "candidate_registry.json").read_text(encoding="utf-8"))
    if frozen_meta.get("source_test_labels_evaluated") or registry.get("source_test_labels_evaluated"):
        raise ValueError("Frozen public candidate is not eligible for one-time source-test evaluation")
    if registry.get("cohort_complete_sha256") != sha256(cohort / "complete.json"):
        raise ValueError("Public cohort changed after candidate freeze")
    if registry.get("training_complete_sha256") != sha256(Path(registry["training_stage"]) / "complete.json"):
        raise ValueError("Public training stage changed after candidate freeze")
    model = Path(registry["training_stage"]) / registry["algorithm"] / "model.joblib"
    if registry.get("model_sha256") != sha256(model):
        raise ValueError("Frozen public model changed")
    if auxiliary_meta.get("inputs", {}).get("raw_sha256") != sha256(raw):
        raise ValueError("OneADMET raw CSV changed after auxiliary snapshot")
    membership = pd.read_csv(cohort / "source_test_membership_blinded.csv", keep_default_na=False)
    if REQUIRED_MEMBERSHIP - set(membership.columns) or not membership.cohort_usage.eq("public_source_test_scaffold_isolated").all():
        raise ValueError("OneADMET source-test membership contract is invalid")
    if {"target_value", "raw_value_h"} & set(membership.columns) or membership.source_record_id.duplicated().any():
        raise ValueError("OneADMET source-test membership is not label-blinded")
    if registry.get("source_test_membership_sha256") != sha256(cohort / "source_test_membership_blinded.csv"):
        raise ValueError("Source-test membership changed after candidate freeze")
    return registry, membership


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "models/frozen/Thalf__human__plasma_public/oneadmet_v1_et_v1")
    parser.add_argument("--cohort", type=Path,
                        default=ROOT / "data/public_benchmarks/oneadmet_human_plasma_thalf_v2")
    parser.add_argument("--auxiliary", type=Path,
                        default=ROOT / "data/external/oneadmet_auxiliary_v2")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/OneADMET/data/raw/oneADMET.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/benchmarks/oneadmet_human_plasma_source_test_v1")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--confirm-source-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check([args.frozen / "complete.json", args.frozen / "candidate_registry.json",
                        args.cohort / "complete.json", args.cohort / "source_test_membership_blinded.csv",
                        args.auxiliary / "complete.json", args.raw], output=None if args.check_only else args.output)
    registry, membership = validate_contract(args.frozen, args.cohort, args.auxiliary, args.raw)
    if args.check_only:
        print(f"Frozen OneADMET public source-test contract valid: members={len(membership)}; labels not loaded")
        return
    if not args.confirm_source_test:
        raise ValueError("One-time source-test scoring requires --confirm-source-test")
    attached = read_source_test_labels(args.raw, membership)
    bundle = joblib.load(Path(registry["training_stage"]) / registry["algorithm"] / "model.joblib")
    train_groups = set(bundle.train_groups)
    if train_groups & set(attached.scaffold_group):
        raise ValueError("Frozen model and source-test scaffolds overlap")
    attached["predicted_h"] = bundle.predict_smiles(attached.SMILES.tolist())
    molecules = aggregate_molecules(attached[["parent_id", "observed_h", "predicted_h"]].rename(columns={"parent_id": "molecule_id"}))
    metrics = metric_values(molecules)
    interval = bootstrap_metrics(molecules, args.bootstrap, 20260915)
    payload = {
        "endpoint": registry["endpoint"], "records": int(len(attached)), "molecules": int(len(molecules)),
        "metrics": metrics, "bootstrap": interval,
        "frozen_candidate_complete_sha256": sha256(args.frozen / "complete.json"),
        "raw_sha256": sha256(args.raw),
        "interpretation": (
            "One-time author-source-test evaluation of the frozen public human-plasma model. This is not an "
            "independent strict direct-IV terminal-half-life validation and cannot be merged with the formal "
            "primary-evidence external registry."),
    }
    with stage_output(args.output) as out:
        attached[["source_record_id", "SMILES", "parent_id", "scaffold_group", "observed_h", "predicted_h"]].to_csv(
            out / "source_test_predictions.csv", index=False, float_format="%.8g")
        pd.DataFrame([{"records": len(attached), "molecules": len(molecules), **metrics,
                       **{f"{name}_lower": values[0] for name, values in interval.items() if isinstance(values, list)},
                       **{f"{name}_upper": values[1] for name, values in interval.items() if isinstance(values, list)}}]).to_csv(
            out / "source_test_summary.csv", index=False, float_format="%.6f")
        dump_json(out / "metrics.json", payload)
        (out / "analysis_report.md").write_text(
            "# OneADMET public human-plasma source-test evaluation\n\n"
            f"The frozen public ExtraTrees model was evaluated once on {len(molecules)} scaffold-isolated author "
            f"source-test molecules without refitting, reselection, or calibration. Log10 RMSE was "
            f"{metrics['log10_rmse']:.4f}; molecule-level Spearman was {metrics['molecule_mean_spearman']:.4f}; "
            f"GMFE was {metrics['gmfe']:.3f}.\n\n"
            "This is a broad public human-plasma endpoint and not strict direct-IV terminal-half-life external validation.\n",
            encoding="utf-8")
        finish_stage(out, "oneadmet_public_thalf_source_test_evaluation", inputs={
            "frozen_candidate_complete_sha256": sha256(args.frozen / "complete.json"),
            "cohort_complete_sha256": sha256(args.cohort / "complete.json"), "raw_sha256": sha256(args.raw),
        }, source_test_labels_evaluated=True, refit=False, calibration=False, selection_changed=False,
           strict_iv_terminal=False, bootstrap_replicates=args.bootstrap, partial=False)
    print(f"OneADMET public source-test evaluation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
