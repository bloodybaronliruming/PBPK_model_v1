"""Shared, train-fold-only data interface for the internal MMPK NCA track.

This module deliberately supports only the approved development cohort.  It
does not know the paths of the investigational or 2024 cohorts, so callers
cannot accidentally use them through this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from dmpk_toolkit import featurize
from pipeline_common import ROOT, sha256, stable_id, verify_stage


# The first item must use the exact N1b endpoint key; do not shorten it for
# display because that would make a frozen membership lookup ambiguous.
ENDPOINTS = (
    ("AUC [ng*h/mL]", "Log AUC [ng*h/mL]"),
    ("Cmax [ng/mL]", "Log Cmax [ng/mL]"),
    ("Tmax [h]", "Log Tmax [h]"),
    ("t1/2 [h]", "Log t1/2 [h]"),
    ("CL/F [L/h]", "Log CL/F [L/h]"),
    ("Vz/F [L]", "Log Vz/F [L]"),
    ("MRT [h]", "Log MRT [h]"),
    ("F [%]", "Log F [%]"),
)
TIER_TO_COLUMN = {
    "R1_direct": "direct_observation_eligible",
    "R1_plus_R2_augmented": "author_augmented_eligible",
}
# The source table stores a rounded display value for dose mg/kg but retains a
# log-dose computed before that display rounding (worst observed difference is
# 0.685%).  Keep the author-supplied log-dose as the conditional input and use
# this predeclared 1% reconciliation bound only to detect a material mismatch.
DOSE_DISPLAY_RECONCILIATION_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class StrictTrainFold:
    """In-memory training-only arrays; held-out labels are never returned."""

    outer_fold: int
    tier_view: str
    model_record_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    log_dose: np.ndarray
    input_state: dict


def approved_model_path(reference: Path = ROOT / "基准研究参考") -> Path:
    return reference / "mmpk_human_oral_pk_parameters_prediction_v3/model_dataset/approved_model.csv"


def _membership_hash(values: list[str] | tuple[str, ...]) -> str:
    return stable_id("\n".join(sorted(values)))


def _verify_protocol(n0: Path, n1b: Path, model_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify frozen inputs before touching the local approved source table."""
    verify_stage(n0, "mmpk_n0_read_only_data_code_license_audit")
    meta = verify_stage(n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    source_key = str(model_path.resolve())
    if meta["inputs"].get(source_key) != sha256(model_path):
        raise ValueError("Approved model table differs from the N1b-frozen source hash")
    records = pd.read_csv(n1b / "approved_record_registry_hashed.csv")
    labels = pd.read_csv(n1b / "approved_label_tier_registry_hashed.csv")
    required_records = {"model_record_id", "strict_outer_fold", "parent_id", "scaffold_id"}
    required_labels = {"model_record_id", "endpoint", "reliability_tier", *TIER_TO_COLUMN.values()}
    if not required_records <= set(records) or not required_labels <= set(labels):
        raise ValueError("N1b protocol schema is incomplete")
    if records.model_record_id.duplicated().any() or not records.strict_outer_fold.isin(range(5)).all():
        raise ValueError("N1b record membership is invalid")
    if labels.duplicated(["model_record_id", "endpoint"]).any() or not set(labels.endpoint) <= {x[0] for x in ENDPOINTS}:
        raise ValueError("N1b label membership is invalid")
    if not set(labels.model_record_id) <= set(records.model_record_id):
        raise ValueError("N1b labels refer to an unknown model record")
    return records, labels


def load_strict_train_fold(
    outer_fold: int,
    tier_view: str,
    *,
    reference: Path = ROOT / "基准研究参考",
    n0: Path = ROOT / "results/analysis/mmpk_n0_audit_v5",
    n1b: Path = ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3",
    progress: bool = False,
    feature_view: str = "ecfp4_logdose",
) -> StrictTrainFold:
    """Materialize one strict outer-*train* fold only.

    ``targets`` remain in the MMPK author-declared transformed space.  The
    default feature matrix is ECFP4 (2,048 bits) followed by log10 dose
    (mg/kg); N2a additionally permits the deterministic RDKit2D view.
    The returned dose scaling state is fit solely on these outer-training
    records; callers must not replace it with a global fit.
    """
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be one of 0..4")
    if tier_view not in TIER_TO_COLUMN:
        raise ValueError(f"Unknown tier view: {tier_view}")
    feature_specs = {
        "ecfp4_logdose": ("ecfp4", "ECFP4_radius2_2048_plus_log10_dose_mg_per_kg"),
        "ecfp4_rdkit2d_logdose": ("ecfp4_rdkit2d", "ECFP4_radius2_2048_plus_RDKit2D_plus_log10_dose_mg_per_kg"),
    }
    if feature_view not in feature_specs:
        raise ValueError(f"Unknown MMPK feature view: {feature_view}")
    model_path = approved_model_path(reference)
    records, labels = _verify_protocol(Path(n0), Path(n1b), model_path)
    target_columns = [column for _, column in ENDPOINTS]
    needed = ["SMILES", "Dose [mg/kg]", "Log Dose [mg/kg]", *target_columns]
    source = pd.read_csv(model_path, usecols=needed)
    expected_ids = [stable_id(f"mmpk_approved_model_row:{index}") for index in source.index]
    if len(source) != len(records) or set(expected_ids) != set(records.model_record_id):
        raise ValueError("Approved source row identity no longer matches the frozen N1b registry")
    source.insert(0, "model_record_id", expected_ids)
    # Slice rows before any labels are extracted.  The returned object cannot
    # carry labels for the held-out strict outer fold.
    train_ids = records.loc[~records.strict_outer_fold.eq(outer_fold), "model_record_id"].tolist()
    train = source[source.model_record_id.isin(train_ids)].copy()
    if len(train) != len(train_ids) or set(train.model_record_id) != set(train_ids):
        raise ValueError("Strict training membership does not match the approved source rows")
    if train["SMILES"].isna().any() or train["Log Dose [mg/kg]"].isna().any():
        raise ValueError("Strict training input contains missing SMILES or log-dose")
    dose = pd.to_numeric(train["Dose [mg/kg]"], errors="coerce").to_numpy(float)
    log_dose = pd.to_numeric(train["Log Dose [mg/kg]"], errors="coerce").to_numpy(float)
    implied_dose = np.power(10.0, log_dose)
    relative_dose_delta = np.abs(dose - implied_dose) / implied_dose
    if ((dose <= 0).any() or not np.isfinite(log_dose).all() or
            not np.isfinite(relative_dose_delta).all() or
            (relative_dose_delta > DOSE_DISPLAY_RECONCILIATION_RELATIVE_TOLERANCE).any()):
        raise ValueError("MMPK log-dose and displayed dose mg/kg have a material disagreement")
    label_lookup = labels.set_index(["model_record_id", "endpoint"])
    targets = np.full((len(train), len(ENDPOINTS)), np.nan, dtype=np.float64)
    mask = np.zeros_like(targets, dtype=bool)
    eligibility_column = TIER_TO_COLUMN[tier_view]
    for column_index, (endpoint, source_column) in enumerate(ENDPOINTS):
        values = pd.to_numeric(train[source_column], errors="coerce").to_numpy(float)
        membership = label_lookup.reindex(pd.MultiIndex.from_product([train.model_record_id, [endpoint]]))
        eligible = membership[eligibility_column].eq(True).fillna(False).to_numpy(bool)
        model_present = membership["author_augmented_eligible"].eq(True).fillna(False).to_numpy(bool)
        if np.any(eligible & ~model_present):
            raise ValueError("A tier mask includes a label not present in the approved model source")
        if np.any(eligible & ~np.isfinite(values)):
            raise ValueError(f"{endpoint}: frozen label tier disagrees with approved transformed target availability")
        targets[eligible, column_index] = values[eligible]
        mask[:, column_index] = eligible
    # R1 can only be a subset of author-augmented membership.  This check is
    # repeated at load time so a later caller cannot silently promote R2.
    if tier_view == "R1_direct":
        r1 = labels.direct_observation_eligible.astype(bool)
        r12 = labels.author_augmented_eligible.astype(bool)
        if np.any(r1 & ~r12):
            raise ValueError("R1 labels are not a subset of the augmented view")
    feature_set, feature_layout_name = feature_specs[feature_view]
    structure, _ = featurize(train.SMILES.tolist(), feature_set=feature_set, progress=progress)
    features = np.column_stack([structure, log_dose.astype(np.float32)]).astype(np.float32, copy=False)
    if features.shape[0] != len(train) or features.shape[1] <= 2048 or not np.isfinite(features[:, -1]).all():
        raise ValueError("Conditional structural feature matrix has an invalid dose layout")
    if np.isinf(features).any():
        raise ValueError("Conditional structural feature matrix contains infinite values")
    dose_mean = float(np.mean(log_dose))
    dose_std = max(float(np.std(log_dose)), 1e-8)
    input_state = {
        "structural_feature_set": feature_layout_name,
        "conditional_feature": "log10_dose_mg_per_kg",
        "feature_dimension": int(features.shape[1]),
        "dose_scaler_fit_scope": "strict_outer_train_only",
        "dose_display_reconciliation_relative_tolerance": DOSE_DISPLAY_RECONCILIATION_RELATIVE_TOLERANCE,
        "dose_scaler_state_hash": stable_id(f"{dose_mean:.17g}|{dose_std:.17g}|{len(train)}"),
        "training_membership_hash": _membership_hash(train.model_record_id.tolist()),
    }
    return StrictTrainFold(
        outer_fold=outer_fold, tier_view=tier_view,
        model_record_ids=tuple(train.model_record_id.tolist()), smiles=tuple(train.SMILES.tolist()),
        features=features, targets=targets, target_mask=mask, log_dose=log_dose, input_state=input_state,
    )


def load_strict_outer_inputs(
    outer_fold: int,
    feature_view: str,
    *,
    reference: Path = ROOT / "基准研究参考",
    n0: Path = ROOT / "results/analysis/mmpk_n0_audit_v5",
    n1b: Path = ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3",
    progress: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    """Return strict held-out *inputs only*; no target column is opened.

    This is intentionally separate from the train-fold loader so a technical
    smoke can exercise feature assembly without obtaining an outer label.
    """
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be one of 0..4")
    feature_sets = {
        "ecfp4_logdose": "ecfp4",
        "ecfp4_rdkit2d_logdose": "ecfp4_rdkit2d",
    }
    if feature_view not in feature_sets:
        raise ValueError(f"Unknown MMPK feature view: {feature_view}")
    model_path = approved_model_path(reference)
    records, _ = _verify_protocol(Path(n0), Path(n1b), model_path)
    source = pd.read_csv(model_path, usecols=["SMILES", "Log Dose [mg/kg]"])
    source.insert(0, "model_record_id", [stable_id(f"mmpk_approved_model_row:{idx}") for idx in source.index])
    held_ids = records.loc[records.strict_outer_fold.eq(outer_fold), "model_record_id"].tolist()
    held = source[source.model_record_id.isin(held_ids)].copy()
    if len(held) != len(held_ids) or set(held.model_record_id) != set(held_ids):
        raise ValueError("Strict outer input membership does not match N1b")
    log_dose = pd.to_numeric(held["Log Dose [mg/kg]"], errors="coerce").to_numpy(float)
    if not np.isfinite(log_dose).all():
        raise ValueError("Strict outer input has missing author log-dose")
    structure, _ = featurize(held.SMILES.tolist(), feature_set=feature_sets[feature_view], progress=progress)
    features = np.column_stack([structure, log_dose.astype(np.float32)]).astype(np.float32, copy=False)
    if np.isinf(features).any() or not np.isfinite(features[:, -1]).all():
        raise ValueError("Strict outer feature matrix is invalid")
    return tuple(held.model_record_id.tolist()), tuple(held.SMILES.tolist()), features, log_dose


def _canonical_parent_id(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Unparsable SMILES in approved MMPK data")
    return stable_id(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))


def _context_state(values: pd.Series) -> tuple[str, str, str]:
    cleaned = sorted({str(value).strip() for value in values.dropna() if str(value).strip()})
    if not cleaned:
        return "missing", "", "__MISSING__"
    if len(cleaned) == 1:
        return "single_value", stable_id(cleaned[0]), cleaned[0]
    return "multiple_values", stable_id("\n".join(cleaned)), ""


def load_approved_formulation_context(
    record_ids: tuple[str, ...] | list[str],
    *,
    reference: Path = ROOT / "基准研究参考",
    n0: Path = ROOT / "results/analysis/mmpk_n0_audit_v5",
    n1b: Path = ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3",
    n1d: Path = ROOT / "results/analysis/mmpk_n1d_context_derivation_audit_v1",
) -> pd.DataFrame:
    """Reproduce N1d's formulation mapping for requested records in memory.

    Raw formulation strings are needed transiently to form fold-local one-hot
    inputs, but are never written to a stage artifact.  The returned dataframe
    must therefore only be consumed by an internal trainer/smoke.
    """
    record_ids = tuple(record_ids)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Formulation context request contains duplicate record IDs")
    model_path = approved_model_path(reference)
    raw_path = Path(reference) / "mmpk_human_oral_pk_parameters_prediction_v3/raw_dataset/approved.csv"
    _verify_protocol(Path(n0), Path(n1b), model_path)
    n1d_meta = verify_stage(n1d, "mmpk_n1d_approved_context_and_derivation_audit")
    for source in (raw_path, model_path):
        if n1d_meta["inputs"].get(str(source.resolve())) != sha256(source):
            raise ValueError("MMPK source differs from the N1d-frozen context input")
    expected = pd.read_csv(n1d / "approved_model_context_registry_hashed.csv")
    raw = pd.read_csv(raw_path, usecols=["SMILES", "Dose [mg]", "Body Weight of Subjects", "Formulation"])
    model = pd.read_csv(model_path, usecols=["SMILES", "Dose [mg]", "Dose [mg/kg]"])
    model.insert(0, "model_record_id", [stable_id(f"mmpk_approved_model_row:{idx}") for idx in model.index])
    if not set(record_ids) <= set(model.model_record_id):
        raise ValueError("Formulation context request contains an unknown model record")
    raw["parent_id"] = raw.SMILES.map(_canonical_parent_id)
    raw["raw_mgkg"] = pd.to_numeric(raw["Dose [mg]"], errors="coerce") / pd.to_numeric(raw["Body Weight of Subjects"], errors="coerce")
    model["parent_id"] = model.SMILES.map(_canonical_parent_id)
    raw_by_parent = {key: value for key, value in raw.groupby("parent_id", sort=False)}
    rows = []
    for model_row in model[model.model_record_id.isin(record_ids)].itertuples(index=False):
        candidates = raw_by_parent.get(model_row.parent_id)
        if candidates is None:
            raise ValueError("A requested model record lacks raw parent candidates")
        dose_mg = getattr(model_row, "_2")  # tuple order: id, SMILES, Dose mg, Dose mg/kg, parent
        dose_mgkg = getattr(model_row, "_3")
        if pd.notna(dose_mg):
            selected = candidates[np.isclose(pd.to_numeric(candidates["Dose [mg]"], errors="coerce").to_numpy(float),
                                              float(dose_mg), rtol=0, atol=1e-9)]
            mapping_rule = "parent_plus_dose_mg"
        else:
            selected = candidates[np.isclose(candidates.raw_mgkg.to_numpy(float), float(dose_mgkg), rtol=0, atol=1e-6)]
            mapping_rule = "parent_plus_dose_mgkg_rounded_1e6"
        if selected.empty:
            raise ValueError("Cannot reproduce the N1d raw formulation mapping")
        state, value_hash, token = _context_state(selected["Formulation"])
        rows.append({"model_record_id": model_row.model_record_id, "formulation_state": state,
                     "formulation_value_hash": value_hash, "formulation_token": token,
                     "raw_candidate_records": int(len(selected)), "mapping_rule_reproduced": mapping_rule})
    result = pd.DataFrame(rows).set_index("model_record_id").reindex(record_ids).reset_index()
    check_columns = ["model_record_id", "formulation_state", "formulation_value_hash", "raw_candidate_records", "mapping_rule_reproduced"]
    frozen = expected[check_columns].set_index("model_record_id").reindex(record_ids).reset_index()
    # Empty hashes represent a predeclared missing formulation state.  CSV
    # parsing turns those empty cells into NaN in the hashed N1d registry, so
    # normalize only this sentinel before checking the immutable mapping.
    frozen["formulation_value_hash"] = frozen["formulation_value_hash"].fillna("")
    comparison = result[check_columns].eq(frozen[check_columns])
    if frozen.isna().any().any() or not comparison.all().all():
        differing = {column: int((~comparison[column]).sum()) for column in check_columns if not comparison[column].all()}
        raise ValueError(f"Recomputed formulation context disagrees with N1d's frozen hashed registry: {differing}")
    return result


def fit_formulation_one_hot(train_context: pd.DataFrame, held_context: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a non-exportable formulation encoder from outer-training inputs only."""
    required = {"model_record_id", "formulation_state", "formulation_token"}
    if not required <= set(train_context) or not required <= set(held_context):
        raise ValueError("Formulation context schema is incomplete")
    if train_context.formulation_state.eq("multiple_values").any() or held_context.formulation_state.eq("multiple_values").any():
        raise ValueError("Multiple formulation contexts must be filtered before one-hot encoding")
    train_tokens = train_context.formulation_token.astype(str).tolist()
    held_tokens = held_context.formulation_token.astype(str).tolist()
    vocabulary = sorted(set(train_tokens))
    # The unseen slot is fixed before looking at held tokens.  It prevents an
    # held-only raw category from changing the fitted vocabulary.
    columns = [*vocabulary, "__UNSEEN__"]
    position = {token: index for index, token in enumerate(columns)}
    def encode(tokens):
        result = np.zeros((len(tokens), len(columns)), dtype=np.float32)
        for row, token in enumerate(tokens):
            result[row, position.get(token, position["__UNSEEN__"])] = 1.0
        return result
    train_matrix, held_matrix = encode(train_tokens), encode(held_tokens)
    state = {
        "formulation_encoder_fit_scope": "strict_outer_train_only",
        "formulation_vocab_size": int(len(vocabulary)),
        "formulation_encoder_dimension": int(len(columns)),
        "formulation_vocab_hash": stable_id("\n".join(columns)),
        "multiple_context_rows_excluded": True,
        "missing_token": "__MISSING__",
        "unseen_token": "__UNSEEN__",
    }
    return train_matrix, held_matrix, state
