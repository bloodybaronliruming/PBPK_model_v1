#!/usr/bin/env python3
"""Score the separately registered PKSmart public benchmark exactly once.

This evaluator validates the immutable, label-blinded registration before it
opens the public label column.  Its output is a new stage: it never alters the
registration, the frozen model, or the formal primary-evidence external set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr

from pipeline_common import ROOT, dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from register_pksmart_public_benchmark import PRIMARY_STATUS, model_contract


RAW_REQUIRED = {"smiles_r", "human_thalf"}
MEMBERSHIP_REQUIRED = {
    "candidate_id", "source_row", "parent_id", "scaffold_group", "molecule_id",
    "screening_status", "benchmark_cohort", "primary_metric_cohort", "endpoint_values_hidden",
}
PRIMARY_COHORT = "primary_structure_isolated_public_rows"
DIAGNOSTIC_COHORT = "diagnostic_public_rows_with_reference_overlap"


def canonical_smiles(value: str) -> str:
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f"Invalid PKSmart SMILES: {value}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def aggregate_molecules(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"molecule_id", "observed_h", "predicted_h"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metric frame misses columns: {sorted(missing)}")
    result = frame.groupby("molecule_id", sort=True, as_index=False)[["observed_h", "predicted_h"]].mean()
    if len(result) < 2:
        raise ValueError("At least two molecules are required for PKSmart benchmark metrics")
    values = result[["observed_h", "predicted_h"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("PKSmart benchmark labels and predictions must be positive finite values")
    return result


def metric_values(molecules: pd.DataFrame) -> dict[str, float]:
    observed = np.log10(molecules.observed_h.to_numpy(float))
    predicted = np.log10(molecules.predicted_h.to_numpy(float))
    error = predicted - observed
    absolute = np.abs(error)
    rho = spearmanr(observed, predicted).statistic
    return {
        "log10_rmse": float(np.sqrt(np.mean(error ** 2))),
        "log10_mae": float(np.mean(absolute)),
        "molecule_mean_spearman": float(rho) if np.isfinite(rho) else None,
        "gmfe": float(10 ** np.mean(absolute)),
        "within_2fold": float(np.mean(absolute <= np.log10(2))),
        "within_3fold": float(np.mean(absolute <= np.log10(3))),
    }


def bootstrap_metrics(molecules: pd.DataFrame, repeats: int, seed: int) -> dict[str, list[float] | int]:
    """Molecule-level percentile CIs for the metrics fixed in the registration."""
    if repeats < 1000:
        raise ValueError("Bootstrap repeats must be at least 1000")
    observed = np.log10(molecules.observed_h.to_numpy(float))
    predicted = np.log10(molecules.predicted_h.to_numpy(float))
    rng = np.random.default_rng(seed)
    n = len(molecules)
    take = rng.integers(0, n, size=(repeats, n))
    error = predicted[take] - observed[take]
    absolute = np.abs(error)
    values: dict[str, np.ndarray] = {
        "log10_rmse": np.sqrt(np.mean(error ** 2, axis=1)),
        "log10_mae": np.mean(absolute, axis=1),
        "gmfe": 10 ** np.mean(absolute, axis=1),
        "within_2fold": np.mean(absolute <= np.log10(2), axis=1),
        "within_3fold": np.mean(absolute <= np.log10(3), axis=1),
    }
    # With this small cohort, bootstrap samples can be constant. Such draws have
    # undefined Spearman and are excluded only from this interval calculation.
    rho = np.full(repeats, np.nan)
    for index, picked in enumerate(take):
        rho[index] = spearmanr(observed[picked], predicted[picked]).statistic
    values["molecule_mean_spearman"] = rho
    result: dict[str, list[float] | int] = {"replicates": repeats, "seed": seed}
    for name, sample in values.items():
        finite = sample[np.isfinite(sample)]
        if len(finite) < max(100, repeats // 2):
            raise ValueError(f"Too few finite bootstrap replicates for {name}")
        result[f"{name}_95_ci"] = [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
        if name == "molecule_mean_spearman":
            result["molecule_mean_spearman_finite_replicates"] = int(len(finite))
    return result


def validate_registration(registry_dir: Path, candidate_queue: Path, raw: Path,
                          frozen_model: Path) -> tuple[dict, pd.DataFrame]:
    meta = verify_stage(registry_dir, "pksmart_public_benchmark_registry")
    policy = json.loads((registry_dir / "benchmark_registry.json").read_text(encoding="utf-8"))
    if policy.get("evaluation_status") != "registered_not_scored" or policy.get("label_visibility_at_registration") != "hidden":
        raise ValueError("PKSmart benchmark must be a label-blinded, unscored registration")
    if policy.get("raw_sha256") != sha256(raw):
        raise ValueError("PKSmart raw file changed since benchmark registration")
    recorded_inputs = meta.get("inputs", {})
    if recorded_inputs.get("candidate_queue_complete_sha256") != sha256(candidate_queue / "complete.json"):
        raise ValueError("PKSmart candidate queue changed since benchmark registration")
    if recorded_inputs.get("frozen_model_complete_sha256") != sha256(frozen_model / "complete.json"):
        raise ValueError("Frozen model stage changed since PKSmart registration")
    if policy.get("raw_label_column") != "human_thalf" or policy.get("raw_smiles_column") != "smiles_r":
        raise ValueError("Unsupported registered PKSmart raw schema")
    if policy.get("primary_cohort") != PRIMARY_COHORT or policy.get("diagnostic_cohort") != DIAGNOSTIC_COHORT:
        raise ValueError("Unexpected registered PKSmart cohort names")
    expected = policy.get("frozen_model")
    actual = model_contract(frozen_model)
    if expected != actual:
        raise ValueError("Frozen model contract changed after PKSmart registration")
    members = pd.read_csv(registry_dir / "cohort_membership_blinded.csv", keep_default_na=False)
    missing = MEMBERSHIP_REQUIRED - set(members.columns)
    if missing or not members.endpoint_values_hidden.astype(bool).all():
        raise ValueError("Invalid or non-blinded PKSmart membership table")
    if members.candidate_id.duplicated().any() or members.source_row.duplicated().any():
        raise ValueError("PKSmart membership contains duplicate candidates or source rows")
    primary = members.primary_metric_cohort.astype(bool)
    if not primary.any() or not members.loc[primary, "benchmark_cohort"].eq(PRIMARY_COHORT).all():
        raise ValueError("PKSmart primary cohort membership is invalid")
    if not members.loc[~primary, "benchmark_cohort"].eq(DIAGNOSTIC_COHORT).all():
        raise ValueError("PKSmart diagnostic cohort membership is invalid")
    if int(meta.get("primary_structure_isolated_records", -1)) != int(primary.sum()):
        raise ValueError("PKSmart registry primary cohort count changed")
    return policy, members


def attach_public_labels(members: pd.DataFrame, candidate_queue: Path, raw: Path) -> pd.DataFrame:
    """Open labels only after immutable membership and raw hash checks succeed."""
    queue = pd.read_csv(candidate_queue / "candidate_records_blinded.csv", keep_default_na=False)
    required_queue = {"candidate_id", "source_row", "parent_canonical_smiles", "screening_status"}
    if required_queue - set(queue.columns) or queue.candidate_id.duplicated().any():
        raise ValueError("PKSmart candidate queue is not valid for identity verification")
    attached = members.merge(queue[list(required_queue)], on=["candidate_id", "source_row", "screening_status"],
                             how="left", validate="one_to_one")
    if attached.parent_canonical_smiles.isna().any():
        raise ValueError("A registered member is absent from the blinded candidate queue")
    raw_frame = pd.read_csv(raw)
    if RAW_REQUIRED - set(raw_frame.columns):
        raise ValueError("PKSmart raw file misses registered columns")
    rows = attached.source_row.astype(int).to_numpy()
    if (rows < 0).any() or (rows >= len(raw_frame)).any():
        raise ValueError("Registered PKSmart source_row is outside the raw file")
    selected = raw_frame.iloc[rows].reset_index(drop=True)
    raw_parent = selected.smiles_r.map(canonical_smiles)
    if not raw_parent.eq(attached.parent_canonical_smiles).all():
        raise ValueError("PKSmart raw structures no longer match registered candidate identities")
    observed = pd.to_numeric(selected.human_thalf, errors="coerce")
    if observed.isna().any() or (observed <= 0).any() or not np.isfinite(observed).all():
        raise ValueError("PKSmart benchmark cohort has missing, nonpositive, or nonfinite public labels")
    attached["smiles"] = raw_parent
    attached["observed_h"] = observed.to_numpy(float)
    return attached


def predict_registered_members(frame: pd.DataFrame, policy: dict) -> np.ndarray:
    member_predictions = []
    model_groups: set[str] | None = None
    for member in policy["frozen_model"]["members"]:
        run_dir = Path(member["run_dir"])
        if sha256(run_dir / "complete.json") != member["run_complete_sha256"]:
            raise ValueError(f"Frozen member completion hash changed: {run_dir}")
        model_path = run_dir / "extratrees/model.joblib"
        if sha256(model_path) != member["model_sha256"]:
            raise ValueError(f"Frozen member model hash changed: {run_dir}")
        bundle = joblib.load(model_path)
        groups = set(bundle.train_groups)
        model_groups = groups if model_groups is None else model_groups
        if groups != model_groups:
            raise ValueError("Frozen ensemble training scaffold groups disagree")
        member_predictions.append(bundle.predict_smiles(frame.smiles.tolist()))
    primary_groups = set(frame.loc[frame.primary_metric_cohort.astype(bool), "scaffold_group"])
    if model_groups and primary_groups & model_groups:
        raise ValueError("Registered primary PKSmart cohort overlaps frozen-model training scaffolds")
    values = np.asarray(member_predictions, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid frozen ensemble predictions")
    return np.power(10.0, np.log10(values).mean(axis=0))


def cohort_result(frame: pd.DataFrame, label: str, repeats: int, seed: int) -> tuple[dict, pd.DataFrame]:
    molecules = aggregate_molecules(frame)
    metrics = metric_values(molecules)
    return {
        "cohort": label,
        "records": int(len(frame)),
        "molecules": int(len(molecules)),
        "metrics": metrics,
        "bootstrap": bootstrap_metrics(molecules, repeats, seed),
    }, molecules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=ROOT / "results/benchmarks/pksmart_public_thalf_v1")
    parser.add_argument("--candidate-queue", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_candidate_queue_v1")
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/raw/External_test_315.csv")
    parser.add_argument("--frozen-model", type=Path,
                        default=ROOT / "models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/benchmarks/pksmart_public_thalf_v1_evaluation_v1")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--confirm-public-benchmark", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.candidate_queue, "pksmart_external_candidate_queue")
    startup_self_check([args.registration / "complete.json", args.registration / "benchmark_registry.json",
                        args.registration / "cohort_membership_blinded.csv", args.candidate_queue / "complete.json",
                        args.candidate_queue / "candidate_records_blinded.csv", args.raw,
                        args.frozen_model / "complete.json", args.frozen_model / "candidate_registry.json"],
                       output=None if args.check_only else args.output)
    policy, members = validate_registration(args.registration, args.candidate_queue, args.raw, args.frozen_model)
    if args.check_only:
        print(f"PKSmart benchmark contract valid: primary={int(members.primary_metric_cohort.sum())}; labels not loaded")
        return
    if not args.confirm_public_benchmark:
        raise ValueError("Scoring requires --confirm-public-benchmark after registry validation")
    attached = attach_public_labels(members, args.candidate_queue, args.raw)
    attached["predicted_h"] = predict_registered_members(attached, policy)
    primary = attached.loc[attached.primary_metric_cohort.astype(bool)].copy()
    primary_result, _ = cohort_result(primary, PRIMARY_COHORT, args.bootstrap, 20260915)
    diagnostic_result, _ = cohort_result(attached, "full_public_cohort_diagnostic", args.bootstrap, 20260916)
    keep = ["candidate_id", "source_row", "molecule_id", "scaffold_group", "benchmark_cohort",
            "primary_metric_cohort", "smiles", "observed_h", "predicted_h"]
    payload = {
        "benchmark_kind": policy["benchmark_kind"],
        "registration_complete_sha256": sha256(args.registration / "complete.json"),
        "raw_sha256": policy["raw_sha256"],
        "frozen_model": policy["frozen_model"],
        "primary": primary_result,
        "diagnostic_full_public_cohort": diagnostic_result,
        "interpretation": (
            "This is a separately registered PKSmart public-dataset-aligned benchmark. It is not a "
            "source-independent external validation, cannot be merged with the formal primary-evidence "
            "external registry, and did not select, calibrate, or retrain the frozen model."),
    }
    rows = []
    for result in [primary_result, diagnostic_result]:
        row = {"cohort": result["cohort"], "records": result["records"], "molecules": result["molecules"]}
        row.update(result["metrics"])
        for name, interval in result["bootstrap"].items():
            if isinstance(interval, list):
                row[f"{name}_lower"] = interval[0]
                row[f"{name}_upper"] = interval[1]
        rows.append(row)
    with stage_output(args.output) as out:
        attached[keep].to_csv(out / "public_benchmark_predictions.csv", index=False, float_format="%.8g")
        pd.DataFrame(rows).to_csv(out / "benchmark_summary.csv", index=False, float_format="%.6f")
        dump_json(out / "metrics.json", payload)
        (out / "analysis_report.md").write_text(
            "# PKSmart public-dataset benchmark evaluation\n\n"
            f"The pre-registered primary cohort contains {primary_result['molecules']} structure-isolated molecules "
            f"({primary_result['records']} records). Frozen v15 was evaluated without refitting, calibration, "
            "retraining, or label-based filtering. The full 38-row public cohort is reported only as a diagnostic.\n\n"
            "This result is aligned to a public dataset, not a source-independent external validation: the PKSmart "
            "CSV does not provide a row-level locator to a primary human PK study. It must remain separate from the "
            "formal primary-evidence external registry and cannot support superiority claims.\n",
            encoding="utf-8")
        finish_stage(out, "pksmart_public_benchmark_evaluation", inputs={
            "registration_complete_sha256": sha256(args.registration / "complete.json"),
            "raw_sha256": sha256(args.raw),
            "frozen_model_complete_sha256": sha256(args.frozen_model / "complete.json"),
        }, primary_records=primary_result["records"], primary_molecules=primary_result["molecules"],
           diagnostic_records=diagnostic_result["records"], bootstrap_replicates=args.bootstrap,
           refit=False, calibration=False, selection_changed=False, partial=False)
    print(f"PKSmart public benchmark evaluation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
