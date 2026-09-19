"""Shared test-sealed helpers for the Gate 5 CLint candidate and evaluator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, sha256, startup_self_check, verify_stage


TASK = "CLint__human__microsome"
CANDIDATE_ID = "ecfp4_rdkit2d::extra_trees__c01"
SEED = 20260916
PARAMETERS = {"n_estimators": 120, "max_features": 0.7, "min_samples_leaf": 3}


def load_gate4_candidate(gate4: Path) -> tuple[dict, dict]:
    """Return the Gate 4 metadata and exact, test-unread CLint registry row."""
    meta = verify_stage(gate4, "cross_gate_endpoint_candidate_freeze")
    startup_self_check([gate4 / "endpoint_candidate_freeze_registry.csv"])
    registry = pd.read_csv(gate4 / "endpoint_candidate_freeze_registry.csv", keep_default_na=False)
    row = registry.loc[registry.task_id.eq(TASK)]
    if len(row) != 1:
        raise ValueError("Gate 4 registry must contain exactly one CLint decision")
    candidate = row.iloc[0].to_dict()
    if candidate["unified_research_reference"] != CANDIDATE_ID:
        raise ValueError("Gate 4 CLint candidate is not the approved Stage-A ExtraTrees c01")
    if candidate["test_lifecycle"] != "test_unread" or meta.get("test_labels_read"):
        raise ValueError("Gate 5 may only start from an unread CLint test")
    params = json.loads(candidate["reference_parameters"])
    if params != PARAMETERS:
        raise ValueError(f"Gate 4 CLint parameters changed: {params}")
    return meta, candidate


def read_train_records(train_protocol: Path) -> pd.DataFrame:
    """Read only the published train-only table, never the all-split label table."""
    meta = verify_stage(train_protocol, "stl_train_only_protocol")
    if meta.get("test_labels_read") or meta.get("fixed_validation_targets_published"):
        raise ValueError("Gate 5 training input must remain target-closed outside train")
    startup_self_check([train_protocol / "benchmark_train_records.csv"])
    usecols = ["row_id", "molecule_id", "task_id", "endpoint", "smiles", "split", "scaffold_group",
               "target_value", "target_transform", "reporting_inverse", "feature_index"]
    records = pd.read_csv(train_protocol / "benchmark_train_records.csv", usecols=usecols)
    records = records.loc[records.task_id.eq(TASK)].copy()
    if records.empty or not records.split.eq("train").all() or not records.endpoint.eq("CLint").all():
        raise ValueError("CLint train-only input is malformed")
    records["feature_index"] = pd.to_numeric(records.feature_index, errors="raise").astype(int)
    records["target_value"] = pd.to_numeric(records.target_value, errors="raise")
    if records.target_transform.nunique() != 1 or records.target_transform.iloc[0] != "log10":
        raise ValueError("CLint requires log10 target transform")
    if records.reporting_inverse.nunique() != 1 or records.reporting_inverse.iloc[0] != "power10":
        raise ValueError("CLint requires power10 reporting inverse")
    if records.molecule_id.duplicated().any() or records.feature_index.duplicated().any():
        raise ValueError("Gate 5 CLint expects one train record and feature row per molecule")
    if not np.isfinite(records.target_value.to_numpy(float)).all():
        raise ValueError("Non-finite CLint training target")
    return records.sort_values("molecule_id").reset_index(drop=True)


def load_feature_matrix(protocol: Path, feature_set: str = "ecfp4_rdkit2d") -> np.ndarray:
    verify_stage(protocol, "stl_benchmark_protocol")
    startup_self_check([protocol / "canonical_parent_features_float32.npz", protocol / "feature_registry.json"])
    registry = json.loads((protocol / "feature_registry.json").read_text(encoding="utf-8"))
    if feature_set != "ecfp4_rdkit2d" or registry[feature_set]["dimensions"] != 2258:
        raise ValueError("Gate 5 CLint feature contract changed")
    cache = np.load(protocol / "canonical_parent_features_float32.npz")
    matrix = np.concatenate([cache["ecfp4"], cache["rdkit2d"]], axis=1)
    if matrix.shape[1] != 2258 or not np.isfinite(matrix).all(axis=None, where=~np.isnan(matrix)):
        raise ValueError("Feature cache has an unexpected shape or non-finite nonmissing value")
    return matrix.astype(np.float32, copy=False)


def recompute_canonical_features(smiles: list[str], expected_molecule_ids: list[str] | None = None) -> np.ndarray:
    """Recreate the registered deterministic feature view from canonical parents."""
    from build_stl_benchmark_protocol import canonical_parent_smiles
    from dmpk_toolkit import featurize

    canonical, identities = [], []
    for smi in smiles:
        parent, molecule_id = canonical_parent_smiles(smi)
        canonical.append(parent)
        identities.append(molecule_id)
    if expected_molecule_ids is not None and identities != list(expected_molecule_ids):
        raise ValueError("Canonical parent identity disagrees with the frozen feature manifest")
    matrix, _ = featurize(canonical, feature_set="ecfp4_rdkit2d", progress=False)
    if matrix.shape != (len(canonical), 2258) or not np.isfinite(matrix).all(axis=None, where=~np.isnan(matrix)):
        raise ValueError("Recomputed deterministic features are invalid")
    return matrix.astype(np.float32, copy=False)


def verify_recomputed_feature_contract(frozen: Path, protocol: Path) -> dict:
    """Prove train-only on-the-fly features match the frozen cache exactly."""
    membership = pd.read_csv(frozen / "train_membership.csv")
    manifest = pd.read_csv(protocol / "feature_row_manifest.csv")
    probe = membership.merge(manifest[["molecule_id", "feature_index", "canonical_parent_smiles"]],
                             on=["molecule_id", "feature_index"], how="inner", validate="one_to_one")
    if len(probe) != len(membership):
        raise ValueError("Frozen training membership is not fully represented in the feature manifest")
    recomputed = recompute_canonical_features(probe.canonical_parent_smiles.tolist(), probe.molecule_id.tolist())
    cached = load_feature_matrix(protocol)[probe.feature_index.to_numpy(int)]
    if not np.array_equal(recomputed, cached, equal_nan=True):
        raise ValueError("On-the-fly deterministic features do not exactly match the frozen cache")
    return {"parents_checked": int(len(probe)), "feature_dimensions": int(recomputed.shape[1]), "max_abs_difference": 0.0}


def train_nearest_similarity(ecfp4: np.ndarray) -> tuple[np.ndarray, float]:
    """Train-only max Tanimoto values and conservative 5th-percentile AD floor."""
    bits = np.asarray(ecfp4, dtype=np.int32)
    if bits.ndim != 2 or len(bits) < 2:
        raise ValueError("Need at least two binary fingerprint rows for applicability domain")
    intersections = bits @ bits.T
    counts = bits.sum(axis=1)
    unions = counts[:, None] + counts[None, :] - intersections
    similarity = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.max(axis=1)
    if not np.isfinite(nearest).all():
        raise ValueError("Unable to compute train-only nearest-neighbour similarity")
    return nearest, float(np.quantile(nearest, 0.05))


def prediction_from_bundle(bundle: dict, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed and physical predictions from a frozen Gate 5 bundle."""
    standard = np.asarray(bundle["estimator"].predict(matrix), dtype=float).reshape(-1)
    transformed = standard * float(bundle["target_standard_deviation"]) + float(bundle["target_mean"])
    transformed = np.clip(transformed, -300.0, 300.0)
    physical = np.power(10.0, transformed)
    if not np.isfinite(physical).all() or (physical <= 0).any():
        raise ValueError("Frozen CLint candidate produced nonpositive or non-finite predictions")
    return transformed, physical


def required_model_files(frozen: Path) -> list[Path]:
    return [frozen / name for name in [
        "complete.json", "candidate_registry.json", "model_contract.json", "evaluation_protocol.json",
        "train_membership.csv", "model.joblib",
    ]]


def validate_frozen_bundle(frozen: Path) -> tuple[dict, dict]:
    meta = verify_stage(frozen, "gate5_clint_stagea_et_candidate")
    startup_self_check(required_model_files(frozen))
    registry = json.loads((frozen / "candidate_registry.json").read_text(encoding="utf-8"))
    contract = json.loads((frozen / "model_contract.json").read_text(encoding="utf-8"))
    if registry.get("candidate_id") != CANDIDATE_ID or registry.get("test_labels_evaluated"):
        raise ValueError("Frozen Gate 5 registry does not preserve the unread candidate")
    if contract.get("task_id") != TASK or contract.get("algorithm") != "extra_trees":
        raise ValueError("Frozen Gate 5 model contract mismatch")
    if contract.get("parameters") != PARAMETERS or contract.get("seed") != SEED:
        raise ValueError("Frozen Gate 5 model parameters or seed changed")
    if sha256(frozen / "model.joblib") != registry.get("model_sha256"):
        raise ValueError("Frozen Gate 5 model hash mismatch")
    if meta.get("test_labels_read") or meta.get("test_evaluated"):
        raise ValueError("Frozen Gate 5 candidate must precede test evaluation")
    return registry, contract
