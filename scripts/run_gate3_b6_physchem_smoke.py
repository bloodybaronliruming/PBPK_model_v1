#!/usr/bin/env python3
"""Run the bounded, non-selective Gate-3 B6 nested-physchem engineering smoke.

The smoke uses one downstream endpoint and one outer scaffold fold.  It checks
the complete ancestry path (Stage-A OOF -> nested auxiliary OOF -> residual
late fusion), model serialization, finite predictions and zero parent/scaffold
overlap.  It deliberately does not calculate an outer-fold performance metric
and cannot be used to select or reject a route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dmpk_toolkit import featurize
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "CL__human__systemic_iv"
OUTER_FOLD = 0
AUX_TASKS = ["experimental_logP", "experimental_logD_pH7_4"]
PRODUCER_FOLDS = 5
META_ALPHAS = [0.1, 1.0, 10.0]
SEED = 20260917


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def digest_values(values) -> str:
    text = "\n".join(sorted(map(str, values)))
    return hashlib.sha256(text.encode()).hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1"])


def bounded_scaffolds(frame: pd.DataFrame, maximum: int, salt: str) -> pd.DataFrame:
    """Select deterministic whole scaffolds, never splitting a scaffold group."""
    if maximum < 8:
        raise ValueError("Smoke parent limit must be at least 8")
    counts = frame.groupby("scaffold_group").parent_id.nunique().reset_index(name="parents")
    counts["order"] = counts.scaffold_group.astype(str).map(
        lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()
    )
    counts = counts.sort_values(["order", "scaffold_group"], kind="stable")
    chosen, total = [], 0
    for row in counts.itertuples(index=False):
        if chosen and total >= maximum:
            break
        chosen.append(str(row.scaffold_group))
        total += int(row.parents)
    result = frame.loc[frame.scaffold_group.astype(str).isin(chosen)].copy()
    if result.parent_id.nunique() < 8 or result.scaffold_group.nunique() < 4:
        raise ValueError(f"Bounded selection is too small: {salt}")
    return result


def collapse_downstream(records: pd.DataFrame) -> pd.DataFrame:
    identity = records.groupby("parent_id").agg(
        features=("feature_index", "nunique"), scaffolds=("scaffold_group", "nunique"),
        structures=("canonical_parent_smiles", "nunique"), folds=("inner_fold_id", "nunique"),
    )
    if identity.max().max() != 1:
        raise ValueError("Downstream parent identity is inconsistent")
    return records.groupby("parent_id", as_index=False, sort=True).agg(
        scaffold_group=("scaffold_group", "first"),
        canonical_parent_smiles=("canonical_parent_smiles", "first"),
        feature_index=("feature_index", "first"),
        inner_fold_id=("inner_fold_id", "first"),
        parent_target=("target_value", "mean"),
        source_records=("row_id", "size"),
    )


def feature_view(cache: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "leakage_safe_rdkit2d":
        return cache["leakage_safe_rdkit2d"]
    if name == "ecfp4_plus_leakage_safe_rdkit2d":
        return np.concatenate([cache["ecfp4"], cache["leakage_safe_rdkit2d"]], axis=1)
    raise ValueError(f"Unknown auxiliary feature view: {name}")


def stagea_view(cache: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "rdkit2d":
        return cache["rdkit2d"]
    if name == "ecfp4":
        return cache["ecfp4"]
    if name == "ecfp4_rdkit2d":
        return np.concatenate([cache["ecfp4"], cache["rdkit2d"]], axis=1)
    raise ValueError(f"Unknown Stage-A feature view: {name}")


def fit_tree(x: np.ndarray, y: np.ndarray, seed: int, trees: int, max_features: float,
             min_samples_leaf: int, threads: int) -> tuple[dict, dict]:
    if len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Invalid tree training arrays")
    all_missing = np.isnan(x).all(axis=0)
    if all_missing.any():
        raise ValueError(f"Producer training view has {int(all_missing.sum())} all-missing columns")
    mean, std = float(np.mean(y)), float(np.std(y, ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid producer-training target scale")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=trees, max_features=max_features,
            min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=threads,
        )),
    ])
    pipeline.fit(x, (y - mean) / std)
    audit = {
        "training_rows": int(len(x)), "dimensions": int(x.shape[1]),
        "raw_missing_cells": int(np.isnan(x).sum()), "all_missing_columns": 0,
        "imputer_fit_scope": "current_model_training_parents_only",
    }
    return {"pipeline": pipeline, "target_mean": mean, "target_std": std}, audit


def predict_tree(bundle: dict, x: np.ndarray) -> np.ndarray:
    prediction = np.asarray(bundle["pipeline"].predict(x)).reshape(-1)
    prediction = prediction * float(bundle["target_std"]) + float(bundle["target_mean"])
    if not np.isfinite(prediction).all():
        raise ValueError("A model emitted non-finite predictions")
    return prediction


def dynamic_scaffold_folds(aux: pd.DataFrame, recipients: pd.DataFrame, salt: str) -> dict[str, int]:
    union = pd.concat([
        aux[["parent_id", "scaffold_group"]].assign(role="auxiliary"),
        recipients[["parent_id", "scaffold_group"]].assign(role="recipient"),
    ], ignore_index=True).drop_duplicates(["role", "parent_id"])
    cross = union.groupby("parent_id").scaffold_group.nunique()
    if cross.max() != 1:
        raise ValueError("Dynamic producer union contains conflicting scaffold identities")
    counts = union.groupby(["scaffold_group", "role"]).parent_id.nunique().unstack(fill_value=0)
    for role in ["auxiliary", "recipient"]:
        if role not in counts:
            counts[role] = 0
    counts = counts.reset_index()
    counts["total"] = counts.auxiliary + counts.recipient
    counts["tie"] = counts.scaffold_group.astype(str).map(
        lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()
    )
    counts = counts.sort_values(["total", "auxiliary", "recipient", "tie"],
                                ascending=[False, False, False, True], kind="stable")
    totals = counts[["total", "auxiliary", "recipient"]].sum().to_numpy(float)
    totals[totals == 0] = 1.0
    loads = np.zeros((PRODUCER_FOLDS, 3), dtype=float)
    assignment: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        addition = np.array([row.total, row.auxiliary, row.recipient], dtype=float)
        choices = []
        for fold in range(PRODUCER_FOLDS):
            projected = loads.copy()
            projected[fold] += addition
            normalized = projected / totals
            choices.append((float(np.square(normalized).sum()), float(normalized.max()), fold))
        chosen = min(choices)[2]
        assignment[str(row.scaffold_group)] = chosen
        loads[chosen] += addition
    if set(assignment.values()) != set(range(PRODUCER_FOLDS)):
        raise ValueError("Dynamic producer folds do not cover all five groups")
    return assignment


def overlap_row(component: str, task: str, fold: int | str, train: pd.DataFrame,
                recipient: pd.DataFrame, model_file: str) -> dict:
    parents = set(train.parent_id.astype(str)); rec_parents = set(recipient.parent_id.astype(str))
    scaffolds = set(train.scaffold_group.astype(str)); rec_scaffolds = set(recipient.scaffold_group.astype(str))
    return {
        "component": component, "task_id": task, "recipient_fold": fold,
        "training_parents": len(parents), "recipient_parents": len(rec_parents),
        "training_scaffolds": len(scaffolds), "recipient_scaffolds": len(rec_scaffolds),
        "parent_overlap": len(parents & rec_parents), "scaffold_overlap": len(scaffolds & rec_scaffolds),
        "training_parent_sha256": digest_values(parents),
        "training_scaffold_sha256": digest_values(scaffolds),
        "recipient_parent_sha256": digest_values(rec_parents),
        "recipient_scaffold_sha256": digest_values(rec_scaffolds),
        "model_file": model_file,
    }


def computed_control(smiles: list[str]) -> np.ndarray:
    rows = []
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError("Computed-control structure parsing failed")
        rows.append([Crippen.MolLogP(mol), Descriptors.TPSA(mol), 1.0, 1.0])
    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Computed-control branch contains non-finite values")
    return values


def select_meta_alpha(features: np.ndarray, residual: np.ndarray, folds: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    unique = sorted(set(folds.tolist()))
    for alpha in META_ALPHAS:
        errors = []
        for fold in unique:
            train = folds != fold; valid = folds == fold
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(features[train], residual[train])
            predicted = model.predict(features[valid])
            errors.append(float(np.sqrt(np.mean(np.square(residual[valid] - predicted)))))
        rows.append({"alpha": alpha, "mean_inner_residual_rmse": float(np.mean(errors)), "inner_folds": len(unique)})
    table = pd.DataFrame(rows).sort_values(["mean_inner_residual_rmse", "alpha"], kind="stable")
    return float(table.iloc[0].alpha), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_protocol_v2")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_cohort_v1")
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--p1-freeze", type=Path, default=ROOT / "data/public_development/stl_p1_decision_freeze_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/gate3_b6_physchem_smoke_v2")
    parser.add_argument("--downstream-parents-per-fold", type=int, default=24)
    parser.add_argument("--auxiliary-parents-per-task", type=int, default=160)
    parser.add_argument("--trees", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.downstream_parents_per_fold, args.auxiliary_parents_per_task, args.trees, args.threads) < 1:
        raise ValueError("Smoke budgets must be positive")
    RDLogger.DisableLog("rdApp.error"); RDLogger.DisableLog("rdApp.warning")

    required = [
        args.protocol / "complete.json", args.protocol / "protocol.json",
        args.protocol / "auxiliary_producer_registry.csv",
        args.cohort / "complete.json", args.cohort / "parent_targets.csv",
        args.cohort / "feature_parent_manifest.csv", args.cohort / "parent_features_float32.npz",
        args.cohort / "outer_fold_auxiliary_parent_membership.csv", args.cohort / "feature_registry.json",
        args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
        args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
        args.p1_freeze / "complete.json", args.p1_freeze / "p1_decision_registry.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_meta = verify_stage(args.protocol, "gate3_b6_physchem_protocol")
    cohort_meta = verify_stage(args.cohort, "gate3_b6_physchem_cohort")
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    metas = [protocol_meta, cohort_meta, train_meta, stl_meta, p1_meta]
    if any(meta.get("test_labels_read", False) or meta.get("validation_target_file_opened", False) for meta in metas):
        raise ValueError("Smoke accepts only fixed-validation/test-closed inputs")
    protocol = json.loads((args.protocol / "protocol.json").read_text())
    if not protocol.get("bounded_smoke_authorized_after_cohort") or protocol.get("formal_model_training_authorized"):
        raise ValueError("Protocol authorization does not match bounded-smoke scope")

    p1 = pd.read_csv(args.p1_freeze / "p1_decision_registry.csv")
    stage = p1.loc[p1.task_id.eq(TASK)]
    if len(stage) != 1 or stage.iloc[0].reference_for_next_stage != "corrected_stageA_matched_control":
        raise ValueError("Frozen Stage-A reference is absent")
    stage_spec = stage.iloc[0]
    stage_params = json.loads(stage_spec.stageA_parameters)
    if stage_spec.stageA_algorithm != "extra_trees":
        raise ValueError("Smoke runner currently requires the frozen ExtraTrees leader")

    if args.check_only:
        expected = 4 + 1 + len(AUX_TASKS) * (2 * PRODUCER_FOLDS + PRODUCER_FOLDS + 1) + 2 * 4 + 2
        print(f"Gate 3 B6 smoke ready: task={TASK} outer_fold={OUTER_FOLD} approximate_fits={expected}")
        return

    # This train-only CSV is a shared target-bearing container.  Only outer-train
    # values are materialized into fitting arrays; outer-fold targets are never
    # copied, scored, or written.  The shared-container fact is published below.
    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv")
    records = records.loc[records.task_id.eq(TASK)].copy()
    for column in ["target_value", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records.feature_index = records.feature_index.astype(int); records.inner_fold_id = records.inner_fold_id.astype(int)
    downstream = collapse_downstream(records)
    selected = []
    for fold, frame in downstream.groupby("inner_fold_id", sort=True):
        selected.append(bounded_scaffolds(frame, args.downstream_parents_per_fold, f"downstream|{fold}"))
    downstream = pd.concat(selected, ignore_index=True)
    outer_train = downstream.loc[downstream.inner_fold_id.ne(OUTER_FOLD)].copy().sort_values("parent_id")
    outer_eval = downstream.loc[downstream.inner_fold_id.eq(OUTER_FOLD)].copy().sort_values("parent_id")
    if set(outer_train.parent_id) & set(outer_eval.parent_id) or set(outer_train.scaffold_group) & set(outer_eval.scaffold_group):
        raise ValueError("Downstream outer partition leakage")

    stage_npz = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    stage_cache = {key: stage_npz[key] for key in stage_npz.files}
    stage_matrix = stagea_view(stage_cache, str(stage_spec.stageA_feature_set))
    cohort_npz = np.load(args.cohort / "parent_features_float32.npz")
    cohort_cache = {key: cohort_npz[key] for key in cohort_npz.files}
    feature_registry = json.loads((args.cohort / "feature_registry.json").read_text())
    safe_names = feature_registry["leakage_safe_rdkit2d"]["descriptor_names"]
    downstream_aux_x, returned_names = featurize(
        downstream.canonical_parent_smiles.tolist(), descriptor_names=safe_names,
        progress=False, feature_set="ecfp4_rdkit2d",
    )
    if returned_names != safe_names:
        raise ValueError("Downstream auxiliary feature layout changed")
    downstream_ecfp = downstream_aux_x[:, :2048].astype(np.float32)
    downstream_safe = downstream_aux_x[:, 2048:].astype(np.float32)
    downstream_aux_cache = {
        "ecfp4": downstream_ecfp,
        "leakage_safe_rdkit2d": downstream_safe,
    }
    downstream["smoke_feature_index"] = np.arange(len(downstream), dtype=int)
    smoke_index = downstream.set_index("parent_id").smoke_feature_index.to_dict()
    outer_train["smoke_feature_index"] = outer_train.parent_id.map(smoke_index).astype(int)
    outer_eval["smoke_feature_index"] = outer_eval.parent_id.map(smoke_index).astype(int)

    parent_targets = pd.read_csv(args.cohort / "parent_targets.csv")
    membership = pd.read_csv(args.cohort / "outer_fold_auxiliary_parent_membership.csv")
    membership.retained_for_outer_fold = bool_series(membership.retained_for_outer_fold)
    membership = membership.loc[
        membership.downstream_task.eq(TASK) & membership.outer_fold.eq(OUTER_FOLD)
        & membership.retained_for_outer_fold
    ].copy()
    producer_registry = pd.read_csv(args.protocol / "auxiliary_producer_registry.csv")

    input_hashes = {str(path.resolve()): sha256(path) for path in required}
    with stage_output(args.output) as out:
        (out / "models").mkdir()
        ancestry, imputation, reloads, aux_cv, aux_selected, meta_cv = [], [], [], [], [], []

        # Corrected Stage-A crossfit on frozen downstream folds.
        stage_oof = np.full(len(outer_train), np.nan)
        for recipient_fold in sorted(outer_train.inner_fold_id.unique()):
            train = outer_train.loc[outer_train.inner_fold_id.ne(recipient_fold)]
            recipient = outer_train.loc[outer_train.inner_fold_id.eq(recipient_fold)]
            model, audit = fit_tree(
                stage_matrix[train.feature_index], train.parent_target.to_numpy(float),
                stable_seed("stageA", recipient_fold), args.trees,
                float(stage_params["max_features"]), int(stage_params["min_samples_leaf"]), args.threads,
            )
            name = f"models/stageA_crossfit_fold{recipient_fold}.joblib"
            joblib.dump(model, out / name, compress=3)
            prediction = predict_tree(model, stage_matrix[recipient.feature_index])
            stage_oof[outer_train.inner_fold_id.to_numpy() == recipient_fold] = prediction
            ancestry_row = overlap_row("stageA_crossfit", TASK, int(recipient_fold), train, recipient, name)
            ancestry_row["model_sha256"] = sha256(out / name)
            ancestry.append(ancestry_row)
            imputation.append({"component": "stageA_crossfit", "task_id": TASK, "fold": int(recipient_fold), **audit})
        if not np.isfinite(stage_oof).all():
            raise ValueError("Stage-A crossfit did not cover every outer-training parent")
        stage_final, audit = fit_tree(
            stage_matrix[outer_train.feature_index], outer_train.parent_target.to_numpy(float),
            stable_seed("stageA", "outer_eval"), args.trees,
            float(stage_params["max_features"]), int(stage_params["min_samples_leaf"]), args.threads,
        )
        stage_final_name = "models/stageA_outer_train.joblib"
        joblib.dump(stage_final, out / stage_final_name, compress=3)
        stage_eval = predict_tree(stage_final, stage_matrix[outer_eval.feature_index])
        ancestry_row = overlap_row("stageA_outer_eval", TASK, OUTER_FOLD, outer_train, outer_eval, stage_final_name)
        ancestry_row["model_sha256"] = sha256(out / stage_final_name)
        ancestry.append(ancestry_row)
        imputation.append({"component": "stageA_outer_eval", "task_id": TASK, "fold": OUTER_FOLD, **audit})
        stage_reload = predict_tree(joblib.load(out / stage_final_name), stage_matrix[outer_eval.feature_index])
        reloads.append({"component": "stageA_outer_eval", "task_id": TASK,
                        "maximum_absolute_difference": float(np.max(np.abs(stage_eval - stage_reload)))})

        aux_train_predictions: dict[str, np.ndarray] = {}
        aux_eval_predictions: dict[str, np.ndarray] = {}
        for aux_task in AUX_TASKS:
            eligible = membership.loc[membership.auxiliary_task.eq(aux_task), ["parent_id", "scaffold_group", "feature_index"]]
            targets = parent_targets.loc[parent_targets.task_id.eq(aux_task), ["parent_id", "target_value"]]
            aux = eligible.merge(targets, on="parent_id", validate="one_to_one")
            aux = bounded_scaffolds(aux, args.auxiliary_parents_per_task, f"auxiliary|{aux_task}").sort_values("parent_id")
            assignment = dynamic_scaffold_folds(aux, outer_train, f"producer|{aux_task}|{TASK}|{OUTER_FOLD}")
            aux["producer_fold"] = aux.scaffold_group.astype(str).map(assignment).astype(int)
            recipient_folds = outer_train.scaffold_group.astype(str).map(assignment)
            if recipient_folds.isna().any():
                raise ValueError("A downstream outer-training scaffold lacks a dynamic producer fold")
            recipient_folds = recipient_folds.astype(int).to_numpy()
            candidates = producer_registry.loc[producer_registry.auxiliary_task.eq(aux_task)]
            candidate_scores = []
            for spec in candidates.itertuples(index=False):
                matrix = feature_view(cohort_cache, str(spec.feature_set))
                fold_errors = []
                for fold in range(PRODUCER_FOLDS):
                    train = aux.loc[aux.producer_fold.ne(fold)]; valid = aux.loc[aux.producer_fold.eq(fold)]
                    model, audit = fit_tree(
                        matrix[train.feature_index], train.target_value.to_numpy(float),
                        stable_seed("aux-select", aux_task, spec.producer_id, fold), args.trees,
                        float(spec.max_features), int(spec.min_samples_leaf), args.threads,
                    )
                    pred = predict_tree(model, matrix[valid.feature_index])
                    rmse = float(np.sqrt(np.mean(np.square(valid.target_value.to_numpy(float) - pred))))
                    fold_errors.append(rmse)
                    aux_cv.append({"auxiliary_task": aux_task, "producer_id": spec.producer_id,
                                   "feature_set": spec.feature_set, "fold": fold, "validation_parents": len(valid),
                                   "validation_rmse": rmse, "scaffold_overlap": 0})
                candidate_scores.append((float(np.mean(fold_errors)), spec))
            candidate_scores.sort(key=lambda item: (item[0], str(item[1].producer_id)))
            best_score = candidate_scores[0][0]
            within = [item for item in candidate_scores if item[0] <= best_score * 1.01]
            within.sort(key=lambda item: (item[1].feature_set != "leakage_safe_rdkit2d", item[0], item[1].producer_id))
            selected_score, selected_spec = within[0]
            aux_selected.append({
                "auxiliary_task": aux_task, "selected_producer_id": selected_spec.producer_id,
                "selected_feature_set": selected_spec.feature_set, "mean_internal_rmse": selected_score,
                "best_observed_internal_rmse": best_score,
                "selection_rule": "within_1pct_choose_leakage_safe_rdkit2d_else_min_RMSE",
                "downstream_labels_used_for_selection": 0, "source_test_labels_used_for_selection": 0,
            })
            aux_matrix = feature_view(cohort_cache, str(selected_spec.feature_set))
            downstream_matrix = feature_view(downstream_aux_cache, str(selected_spec.feature_set))
            train_pred = np.full(len(outer_train), np.nan)
            for fold in range(PRODUCER_FOLDS):
                producer_train = aux.loc[aux.producer_fold.ne(fold)]
                recipient = outer_train.loc[recipient_folds == fold]
                if recipient.empty:
                    continue
                model, audit = fit_tree(
                    aux_matrix[producer_train.feature_index], producer_train.target_value.to_numpy(float),
                    stable_seed("aux-crossfit", aux_task, fold), args.trees,
                    float(selected_spec.max_features), int(selected_spec.min_samples_leaf), args.threads,
                )
                name = f"models/{aux_task}_crossfit_fold{fold}.joblib"
                joblib.dump(model, out / name, compress=3)
                mask = recipient_folds == fold
                prediction = predict_tree(model, downstream_matrix[outer_train.loc[mask, "smoke_feature_index"]])
                train_pred[mask] = prediction
                ancestry_row = overlap_row("auxiliary_crossfit", aux_task, fold, producer_train, recipient, name)
                ancestry_row["model_sha256"] = sha256(out / name)
                ancestry.append(ancestry_row)
                imputation.append({"component": "auxiliary_crossfit", "task_id": aux_task, "fold": fold, **audit})
            if not np.isfinite(train_pred).all():
                raise ValueError(f"Nested auxiliary crossfit is incomplete: {aux_task}")
            aux_train_predictions[aux_task] = train_pred
            final_model, audit = fit_tree(
                aux_matrix[aux.feature_index], aux.target_value.to_numpy(float),
                stable_seed("aux-final", aux_task), args.trees,
                float(selected_spec.max_features), int(selected_spec.min_samples_leaf), args.threads,
            )
            name = f"models/{aux_task}_outer_train.joblib"
            joblib.dump(final_model, out / name, compress=3)
            evaluation_prediction = predict_tree(final_model, downstream_matrix[outer_eval.smoke_feature_index])
            aux_eval_predictions[aux_task] = evaluation_prediction
            ancestry_row = overlap_row("auxiliary_outer_eval", aux_task, OUTER_FOLD, aux, outer_eval, name)
            ancestry_row["model_sha256"] = sha256(out / name)
            ancestry.append(ancestry_row)
            imputation.append({"component": "auxiliary_outer_eval", "task_id": aux_task, "fold": OUTER_FOLD, **audit})
            reload_prediction = predict_tree(joblib.load(out / name), downstream_matrix[outer_eval.smoke_feature_index])
            reloads.append({"component": "auxiliary_outer_eval", "task_id": aux_task,
                            "maximum_absolute_difference": float(np.max(np.abs(evaluation_prediction - reload_prediction)))})

        c1_train = computed_control(outer_train.canonical_parent_smiles.tolist())
        c1_eval = computed_control(outer_eval.canonical_parent_smiles.tolist())
        b6_train = np.column_stack([
            aux_train_predictions["experimental_logP"], aux_train_predictions["experimental_logD_pH7_4"],
            np.ones(len(outer_train)), np.ones(len(outer_train)),
        ])
        b6_eval = np.column_stack([
            aux_eval_predictions["experimental_logP"], aux_eval_predictions["experimental_logD_pH7_4"],
            np.ones(len(outer_eval)), np.ones(len(outer_eval)),
        ])
        residual = outer_train.parent_target.to_numpy(float) - stage_oof
        route_eval = {"C0_corrected_StageA": stage_eval.copy()}
        route_train = {"C0_corrected_StageA": stage_oof.copy()}
        for route, x_train, x_eval in [
            ("C1_structure_only_control", c1_train, c1_eval),
            ("B6_physchem_candidate", b6_train, b6_eval),
        ]:
            alpha, cv = select_meta_alpha(x_train, residual, outer_train.inner_fold_id.to_numpy(int))
            cv.insert(0, "route", route); meta_cv.append(cv)
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(x_train, residual)
            name = f"models/{route}_meta.joblib"
            joblib.dump(model, out / name, compress=3)
            route_train[route] = stage_oof + model.predict(x_train)
            route_eval[route] = stage_eval + model.predict(x_eval)
            reload_pred = stage_eval + joblib.load(out / name).predict(x_eval)
            reloads.append({"component": "meta_outer_eval", "task_id": route,
                            "maximum_absolute_difference": float(np.max(np.abs(route_eval[route] - reload_pred)))})
            ancestry_row = overlap_row("meta_model_fit_and_evaluation", route, OUTER_FOLD,
                                       outer_train, outer_eval, name)
            ancestry_row["model_sha256"] = sha256(out / name)
            ancestry.append(ancestry_row)

        ancestry_table = pd.DataFrame(ancestry)
        if ancestry_table[["parent_overlap", "scaffold_overlap"]].to_numpy().max() != 0:
            raise ValueError("A saved ancestry row has parent/scaffold leakage")
        reload_table = pd.DataFrame(reloads)
        if reload_table.maximum_absolute_difference.max() > 1e-12:
            raise ValueError("Serialized model reload tolerance failed")
        if not all(np.isfinite(values).all() for values in [*route_train.values(), *route_eval.values()]):
            raise ValueError("A route emitted non-finite predictions")

        train_predictions = outer_train[["parent_id", "scaffold_group", "inner_fold_id"]].copy()
        train_predictions["observed_outer_train_target"] = outer_train.parent_target.to_numpy(float)
        for route, values in route_train.items():
            train_predictions[route] = values
        for task, values in aux_train_predictions.items():
            train_predictions[f"nested_{task}"] = values
        eval_predictions = outer_eval[["parent_id", "scaffold_group", "inner_fold_id"]].copy()
        for route, values in route_eval.items():
            eval_predictions[route] = values
        for task, values in aux_eval_predictions.items():
            eval_predictions[f"nested_{task}"] = values
        if any("target" in column.lower() for column in eval_predictions.columns):
            raise ValueError("Blinded outer-evaluation output unexpectedly contains a target column")

        configuration = {
            "scope": "bounded_engineering_smoke_only",
            "downstream_task": TASK, "outer_fold": OUTER_FOLD,
            "downstream_parent_cap_per_fold": args.downstream_parents_per_fold,
            "auxiliary_parent_cap_per_task": args.auxiliary_parents_per_task,
            "smoke_trees": args.trees, "formal_protocol_trees": 300,
            "routes": ["C0_corrected_StageA", "C1_structure_only_control", "B6_physchem_candidate"],
            "outer_evaluation_metric_computed": False,
            "outer_evaluation_target_values_used": 0,
            "outer_evaluation_target_container_shared_with_outer_train": True,
            "architecture_selection_authorized": False,
            "formal_training_authorized": False,
            "fixed_validation_status": "closed", "test_status": "closed",
        }
        (out / "configuration.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n")
        pd.DataFrame(aux_cv).to_csv(out / "auxiliary_internal_cv.csv", index=False)
        pd.DataFrame(aux_selected).to_csv(out / "selected_auxiliary_producers.csv", index=False)
        pd.concat(meta_cv, ignore_index=True).to_csv(out / "meta_alpha_internal_cv.csv", index=False)
        ancestry_table.to_csv(out / "nested_ancestry_audit.csv", index=False)
        pd.DataFrame(imputation).to_csv(out / "fold_local_imputation_audit.csv", index=False)
        reload_table.to_csv(out / "model_reload_audit.csv", index=False)
        train_predictions.to_csv(out / "outer_train_crossfit_predictions.csv", index=False)
        eval_predictions.to_csv(out / "outer_eval_predictions_blinded.csv", index=False)
        pd.DataFrame([
            {"route": "C0_corrected_StageA", "late_branch_dimensions": 0, "role": "reference"},
            {"route": "C1_structure_only_control", "late_branch_dimensions": 4, "role": "capacity_matched_control"},
            {"route": "B6_physchem_candidate", "late_branch_dimensions": 4, "role": "single_preregistered_candidate"},
        ]).to_csv(out / "route_registry.csv", index=False)
        finish_stage(
            out, "gate3_b6_physchem_smoke", inputs=input_hashes, partial=False,
            limited_smoke=True, downstream_task=TASK, outer_fold=OUTER_FOLD,
            downstream_outer_train_parents=len(outer_train), downstream_outer_eval_parents=len(outer_eval),
            auxiliary_tasks=len(AUX_TASKS), routes=3,
            ancestry_rows=len(ancestry_table), maximum_parent_overlap=0, maximum_scaffold_overlap=0,
            maximum_reload_difference=float(reload_table.maximum_absolute_difference.max()),
            outer_evaluation_metric_computed=False, outer_evaluation_target_values_used=0,
            outer_evaluation_target_container_shared_with_outer_train=True,
            source_test_labels_read=False, validation_target_file_opened=False, test_labels_read=False,
            architecture_selection_authorized=False, formal_model_training_authorized=False,
            engineering_smoke_passed=True,
        )
    print(f"Gate 3 B6 physchem smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
