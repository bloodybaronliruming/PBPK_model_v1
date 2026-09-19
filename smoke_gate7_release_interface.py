"""Run the pre-registered Gate 7 label-free technical inference smoke.

This is an interface/integrity test, never a performance evaluation.  It only
accepts 2--8 label-free SMILES, loads the six contract-bound numeric assets,
and emits technical predictions plus provenance/AD fields.  F remains a
status-only row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gate5_clint_common import prediction_from_bundle, recompute_canonical_features
from infer_frozen_candidates import candidate_prediction, nearest_train_similarity
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


CONTRACT_STAGE = "gate7_release_contract_freeze"
AD_LEGACY_THRESHOLD = 0.30
LABEL_TOKENS = ("label", "target", "response", "measurement", "observed", "actual", "y_true")
ALLOWED_INPUT_COLUMNS = {"smiles", "input_id"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_hash(frame: pd.DataFrame) -> str:
    """Hash the label-free submitted input without treating a prediction as an input."""
    payload = frame[["input_id", "smiles"]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_label_free_input(path: Path, *, rows_min: int = 1, rows_max: int | None = None) -> pd.DataFrame:
    """Read a contract-conformant SMILES-only batch without accessing labels.

    This is shared by the fixed R2 smoke and the reusable R4 inference CLI so
    the latter cannot silently relax the frozen input boundary.
    """
    startup_self_check([path])
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    lowered = [str(column).strip().lower() for column in frame.columns]
    if any(token in column for column in lowered for token in LABEL_TOKENS):
        raise ValueError("Label-like columns are prohibited in Gate 7 no-label inference input")
    if not set(frame.columns) <= ALLOWED_INPUT_COLUMNS or "smiles" not in frame:
        raise ValueError("Gate 7 inference accepts only smiles and optional input_id columns")
    if len(frame) < rows_min or (rows_max is not None and len(frame) > rows_max):
        maximum = "unbounded" if rows_max is None else str(rows_max)
        raise ValueError(f"Gate 7 input requires {rows_min}--{maximum} rows")
    if frame.smiles.str.strip().eq("").any():
        raise ValueError("Gate 7 inference requires non-empty SMILES rows")
    if "input_id" not in frame:
        frame["input_id"] = [f"input_{index}" for index in range(len(frame))]
    if frame.input_id.duplicated().any():
        raise ValueError("input_id must be unique when supplied")
    # RDKit parsing occurs inside nearest_train_similarity; keep validation explicit and label-free here.
    from rdkit import Chem
    if any(Chem.MolFromSmiles(value) is None for value in frame.smiles):
        raise ValueError("All Gate 7 input SMILES must be RDKit-parseable")
    return frame[["input_id", "smiles"]].copy()


def read_smoke_input(path: Path | None) -> pd.DataFrame:
    if path is None:
        frame = pd.DataFrame({"input_id": ["probe_ethanol", "probe_benzene"], "smiles": ["CCO", "c1ccccc1"]})
        # Keep the default probes in a temporary label-free CSV-equivalent
        # frame; this legacy R2 path has already been executed and is not
        # altered by R4.
        if frame.input_id.duplicated().any() or frame.smiles.str.strip().eq("").any():
            raise ValueError("Invalid fixed Gate 7 R2 smoke probes")
        from rdkit import Chem
        if any(Chem.MolFromSmiles(value) is None for value in frame.smiles):
            raise ValueError("Invalid fixed Gate 7 R2 smoke probe SMILES")
        return frame[["input_id", "smiles"]].copy()
    return read_label_free_input(path, rows_min=2, rows_max=8)


def verified_dataset(root: Path, version: str, expected_hash: str) -> Path:
    folder = root / "data" / version / "datasets"
    verify_stage(folder, "datasets")
    if sha256(folder / "complete.json") != expected_hash:
        raise ValueError(f"Dataset version drift for technical AD source: {version}")
    return folder


def task_train_smiles_no_labels(dataset: Path, task_id: str, train_ids: set[str] | None = None) -> list[str]:
    path = dataset / "task_records.csv"
    startup_self_check([path])
    frame = pd.read_csv(path, usecols=["molecule_id", "smiles", "split", "task_id"], dtype=str, keep_default_na=False)
    frame = frame.loc[frame.task_id.eq(task_id)].copy()
    if train_ids is None:
        frame = frame.loc[frame.split.eq("train")]
    else:
        frame = frame.loc[frame.molecule_id.isin(train_ids)]
        if set(train_ids) - set(frame.molecule_id):
            raise ValueError(f"Missing frozen train molecules in label-free task table: {task_id}")
    result = frame.smiles.drop_duplicates().tolist()
    if not result:
        raise ValueError(f"No train structures for technical AD: {task_id}")
    return result


def legacy_candidate(root: Path, task_id: str) -> dict:
    folder = root / "models/frozen/final_candidates_v1"
    verify_stage(folder, "frozen_candidate_registry")
    registry = read_json(folder / "candidate_registry.json")
    items = [item for item in registry.get("candidates", []) if item.get("task_id") == task_id]
    if len(items) != 1:
        raise ValueError(f"No unique legacy candidate for {task_id}")
    return items[0]


def legacy_predict_and_ad(root: Path, task: dict, smiles: list[str]) -> tuple[np.ndarray, np.ndarray, float, str]:
    candidate = legacy_candidate(root, task["task_id"])
    if task["model_lane"] != "historical_test_locked":
        raise ValueError("Legacy registry may only serve the historical test-locked lane")
    if candidate["model_kind"] == "stl":
        run = Path(candidate["run_dir"])
        meta = verify_stage(run, "stl_training")
        dataset = verified_dataset(root, candidate["data_version"], meta["inputs"]["datasets_complete_sha256"])
        bundle = joblib.load(run / candidate["algorithm"] / "model.joblib")
        train = task_train_smiles_no_labels(dataset, task["task_id"], set(bundle.train_molecule_ids))
        version = f"legacy::{candidate['algorithm']}::{sha256(run / 'complete.json')[:12]}"
    elif candidate["model_kind"] == "multitask_seed_ensemble":
        runs = [Path(value) for value in candidate["run_dirs"]]
        metas = [verify_stage(run, "multitask_training") for run in runs]
        expected = metas[0]["inputs"]["datasets_complete_sha256"]
        if any(meta["inputs"].get("datasets_complete_sha256") != expected for meta in metas[1:]):
            raise ValueError("Legacy MTL members do not share one dataset version")
        dataset = verified_dataset(root, candidate["data_version"], expected)
        train = task_train_smiles_no_labels(dataset, task["task_id"])
        version = f"legacy::shared_mtl_ensemble::{sha256(runs[0] / 'complete.json')[:12]}"
    else:
        raise ValueError(f"Unsupported legacy candidate kind: {candidate['model_kind']}")
    prediction = np.asarray(candidate_prediction(candidate, smiles), dtype=float)
    similarity = nearest_train_similarity(smiles, train)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite technical prediction: {task['task_id']}")
    return prediction, similarity, AD_LEGACY_THRESHOLD, version


def clint_predict_and_ad(root: Path, task: dict, smiles: list[str]) -> tuple[np.ndarray, np.ndarray, float, str]:
    if task["model_lane"] != "current_frozen_final":
        raise ValueError("CLint asset lane differs from frozen contract")
    frozen = root / task["asset_source"]
    verify_stage(frozen, "gate5_clint_stagea_et_candidate")
    bundle = joblib.load(frozen / "model.joblib")
    matrix = recompute_canonical_features(smiles)
    _, prediction = prediction_from_bundle(bundle, matrix)
    membership = pd.read_csv(frozen / "train_membership.csv", usecols=["molecule_id"])
    protocol = root / "data/public_development/stl_benchmark_protocol_v1"
    verify_stage(protocol, "stl_benchmark_protocol")
    manifest = pd.read_csv(protocol / "feature_row_manifest.csv", usecols=["molecule_id", "canonical_parent_smiles"], dtype=str)
    train = membership.merge(manifest, on="molecule_id", validate="one_to_one").canonical_parent_smiles.tolist()
    similarity = nearest_train_similarity(smiles, train)
    evaluation = read_json(frozen / "evaluation_protocol.json")
    threshold = float(evaluation["applicability_domain"]["train_only_floor"])
    return prediction, similarity, threshold, f"gate5_stageA_et_c01_v2::{sha256(frozen / 'model.joblib')[:12]}"


def thalf_predict_and_ad(root: Path, task: dict, smiles: list[str]) -> tuple[np.ndarray, np.ndarray, float, str]:
    if task["model_lane"] != "historical_test_locked_with_source_overlap_limit":
        raise ValueError("Terminal t½ asset lane differs from frozen contract")
    frozen = root / task["asset_source"]
    verify_stage(frozen, "thalf_v15_frozen_candidate")
    registry = read_json(frozen / "candidate_registry.json")
    members = registry.get("members", [])
    if len(members) != 3:
        raise ValueError("Expected three terminal t½ frozen ensemble members")
    predictions, ids = [], None
    for member in members:
        run = Path(member["run_dir"])
        meta = verify_stage(run, "stl_training")
        if sha256(run / "complete.json") != member["run_complete_sha256"]:
            raise ValueError("Frozen terminal t½ run hash changed")
        if sha256(run / "extratrees/model.joblib") != member["model_sha256"]:
            raise ValueError("Frozen terminal t½ model hash changed")
        dataset = verified_dataset(root, "processed_v15", meta["inputs"]["datasets_complete_sha256"])
        bundle = joblib.load(run / "extratrees" / "model.joblib")
        ids = set(bundle.train_molecule_ids) if ids is None else ids
        if set(bundle.train_molecule_ids) != ids:
            raise ValueError("Frozen terminal t½ ensemble members have mismatched train identities")
        predictions.append(np.asarray(bundle.predict_smiles(smiles), dtype=float))
    train = task_train_smiles_no_labels(dataset, task["task_id"], ids)
    prediction = np.power(10.0, np.log10(np.stack(predictions)).mean(axis=0))
    similarity = nearest_train_similarity(smiles, train)
    return prediction, similarity, AD_LEGACY_THRESHOLD, f"v15_rdkit2d_et3_v1::{sha256(frozen / 'complete.json')[:12]}"


def technical_prediction(root: Path, task: dict, smiles: list[str]) -> tuple[np.ndarray, np.ndarray, float, str]:
    if task["task_id"] == "CLint__human__microsome":
        return clint_predict_and_ad(root, task, smiles)
    if task["task_id"] == "Thalf__human__terminal_iv":
        return thalf_predict_and_ad(root, task, smiles)
    return legacy_predict_and_ad(root, task, smiles)


def render_rows(root: Path, contract: dict, input_frame: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    smiles = input_frame.smiles.tolist()
    for task in contract["tasks"]:
        common = {
            "input_row": np.arange(len(input_frame)),
            "input_id": input_frame.input_id.tolist(),
            "smiles": smiles,
            "task_id": task["task_id"],
            "endpoint": task["endpoint"],
            "model_lane": task["model_lane"],
            "unit": task["unit"],
            "evidence_status": task["inferential_status"],
            "test_lifecycle": task["test_lifecycle"],
            "limitation": task["limitation"],
        }
        if not task["no_label_smoke_eligible"]:
            all_rows.append(pd.DataFrame({
                **common, "model_version": "none", "prediction_status": "not_available_exploratory",
                "predicted_physical": np.nan, "nearest_train_tanimoto": np.nan,
                "ad_threshold": np.nan, "applicability_domain": "not_applicable",
            }))
            continue
        predicted, similarity, threshold, version = technical_prediction(root, task, smiles)
        all_rows.append(pd.DataFrame({
            **common, "model_version": version, "prediction_status": "technical_prediction_available",
            "predicted_physical": predicted, "nearest_train_tanimoto": similarity,
            "ad_threshold": threshold,
            "applicability_domain": np.where(similarity >= threshold, "inside", "outside"),
        }))
    output = pd.concat(all_rows, ignore_index=True)
    if len(output) != len(input_frame) * 7 or set(output.columns) < set(contract["output"]["required_columns"]):
        raise ValueError("Technical smoke output does not meet the frozen row/schema contract")
    f_rows = output.endpoint.eq("F")
    if not (output.loc[f_rows, "prediction_status"].eq("not_available_exploratory").all()
            and output.loc[f_rows, "predicted_physical"].isna().all()):
        raise ValueError("F status-only contract was violated")
    if output.loc[~f_rows, "predicted_physical"].isna().any() or not np.isfinite(output.loc[~f_rows, "predicted_physical"]).all():
        raise ValueError("A numeric smoke-eligible endpoint did not return finite technical predictions")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--input", type=Path, help="Optional 2--8-row label-free CSV; default is the frozen two-probe input")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-no-label-smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    contract_dir = args.contract or root / "data/public_development/gate7_release_contract_v2"
    output = args.output or root / "results/analysis/gate7_release_no_label_smoke_v2"
    verify_stage(contract_dir, CONTRACT_STAGE)
    contract = read_json(contract_dir / "release_contract.json")
    smoke = read_json(contract_dir / "no_label_smoke_protocol.json")
    if not contract.get("numeric_release_authorized") is False or not smoke.get("confirmation_flag") == "--confirm-no-label-smoke":
        raise ValueError("Gate 7 release contract does not permit this technical smoke")
    input_frame = read_smoke_input(args.input)
    if args.check_only:
        print(f"Gate 7 R2 smoke preflight: rows={len(input_frame)} tasks={len(contract['tasks'])}; labels not read.")
        return
    if not args.confirm_no_label_smoke:
        raise ValueError("Gate 7 R2 requires --confirm-no-label-smoke")
    with stage_output(output) as folder:
        first = render_rows(root, contract, input_frame)
        second = render_rows(root, contract, input_frame)
        numeric = first.predicted_physical.notna()
        difference = np.abs(first.loc[numeric, "predicted_physical"].to_numpy(float)
                            - second.loc[numeric, "predicted_physical"].to_numpy(float))
        max_difference = float(difference.max(initial=0.0))
        if max_difference > 1e-12:
            raise ValueError(f"Technical repeat determinism failed: {max_difference:.3g}")
        first.to_csv(folder / "technical_smoke_predictions.csv", index=False)
        assertions = {
            "input_rows": int(len(input_frame)), "output_rows": int(len(first)), "expected_output_rows": int(len(input_frame) * 7),
            "numeric_prediction_rows": int(numeric.sum()), "f_status_only_rows": int((first.endpoint == "F").sum()),
            "all_required_output_columns_present": True, "all_numeric_predictions_finite": True,
            "f_status_only_rule_passed": True, "max_abs_repeat_prediction_difference": max_difference,
            "repeat_tolerance": 1e-12, "performance_metrics_computed": False,
        }
        (folder / "smoke_assertions.json").write_text(json.dumps(assertions, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "README.md").write_text(
            "# Gate 7 R2 no-label technical smoke\n\n"
            "Two contract-frozen label-free probes were sent through the six numeric assets; F emitted status-only rows. "
            "The stage verifies model loading, deterministic output, provenance fields, and applicability-domain schema only. "
            "It reads no labels and computes no performance metric, so it neither re-evaluates model performance nor authorizes a unified numeric release.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, "gate7_no_label_batch_inference_smoke",
            inputs={
                str((contract_dir / "complete.json").resolve()): sha256(contract_dir / "complete.json"),
                "label_free_smoke_input_sha256": frame_hash(input_frame),
            },
            partial=False, no_model_fitted=True, no_labels_accessed=True, no_performance_metrics_computed=True,
            no_calibration=True, no_candidate_selection_changed=True, technical_predictions_generated=True,
            unified_numeric_release_authorized=False, input_rows=int(len(input_frame)), output_rows=int(len(first)),
            max_abs_repeat_prediction_difference=max_difference,
        )
    print(f"Gate 7 R2 no-label technical smoke: {output}")


if __name__ == "__main__":
    run_cli(main)
