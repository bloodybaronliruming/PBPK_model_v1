#!/usr/bin/env python3
"""N2b train-only nested-baseline runner smoke; it never scores an outer fold."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from mmpk_nca_common import ENDPOINTS, load_strict_train_fold
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")


def fit_predict_smoke(algorithm: str, feature_view: str, x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, str]:
    """Fit a deliberately tiny smoke estimator, returning no score or model file."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    a = imputer.fit_transform(x_fit)
    b = imputer.transform(x_eval)
    mean, std = float(np.mean(y_fit)), max(float(np.std(y_fit)), 1e-8)
    z = (y_fit - mean) / std
    if algorithm == "dummy_mean":
        predicted = np.full(len(b), mean, dtype=float)
    elif algorithm == "ridge":
        scaler = StandardScaler()
        estimator = Ridge(alpha=1.0, solver="lsqr", tol=1e-6).fit(scaler.fit_transform(a), z)
        predicted = estimator.predict(scaler.transform(b)) * std + mean
    elif algorithm == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=8, max_features=0.3, min_samples_leaf=1,
                                        random_state=20260918, n_jobs=1).fit(a, z)
        predicted = estimator.predict(b) * std + mean
    else:
        raise ValueError(f"Unsupported smoke algorithm: {algorithm}")
    if not np.isfinite(predicted).all():
        raise ValueError("Smoke prediction is non-finite")
    state_hash = sha256_bytes(f"{algorithm}|{feature_view}|{mean:.17g}|{std:.17g}|{len(y_fit)}".encode())
    return predicted, state_hash


def sha256_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2a_baseline_protocol_v2")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--n1c", type=Path, default=ROOT / "results/analysis/mmpk_n1c_train_fold_loader_smoke_v1")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/mmpk_n2b_strict_baseline_runner_smoke_v2")
    args = parser.parse_args()
    if args.outer_fold not in range(5) or args.inner_fold not in range(4):
        raise ValueError("outer-fold must be 0..4 and inner-fold must be 0..3")
    startup_self_check([args.protocol / "complete.json", args.n1b / "complete.json", args.n1c / "complete.json"], output=args.output)
    protocol = verify_stage(args.protocol, "mmpk_n2a_strict_conditional_baseline_protocol")
    verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    verify_stage(args.n1c, "mmpk_n1c_strict_train_fold_conditional_loader_smoke")
    membership = pd.read_csv(args.protocol / "strict_nested_inner_membership_hashed.csv")
    membership = membership[membership.outer_test_fold.eq(args.outer_fold)].copy()
    if membership.model_record_id.duplicated().any() or set(membership.inner_fold) != set(range(4)):
        raise ValueError("N2a nested membership is not a complete partition")
    rows = []
    for feature_view in ("ecfp4_logdose", "ecfp4_rdkit2d_logdose"):
        loaded = load_strict_train_fold(args.outer_fold, "R1_direct", n1b=args.n1b, feature_view=feature_view)
        local = pd.DataFrame({"model_record_id": loaded.model_record_ids, "row": np.arange(len(loaded.model_record_ids))})
        local = local.merge(membership[["model_record_id", "inner_fold", "parent_id", "scaffold_id", "strict_component_id"]], on="model_record_id", validate="one_to_one")
        if len(local) != len(loaded.model_record_ids) or set(local.model_record_id) != set(membership.model_record_id):
            raise ValueError("Loader and N2a nested membership disagree")
        fitting = local[local.inner_fold.ne(args.inner_fold)]
        evaluation = local[local.inner_fold.eq(args.inner_fold)]
        if (set(fitting.parent_id) & set(evaluation.parent_id) or set(fitting.scaffold_id) & set(evaluation.scaffold_id) or
                set(fitting.strict_component_id) & set(evaluation.strict_component_id)):
            raise ValueError("N2b smoke nested fitting/evaluation leak")
        for endpoint in PRIMARY:
            target_index = [x[0] for x in ENDPOINTS].index(endpoint)
            fit_rows = fitting.row.to_numpy()[loaded.target_mask[fitting.row.to_numpy(), target_index]]
            eval_rows = evaluation.row.to_numpy()[loaded.target_mask[evaluation.row.to_numpy(), target_index]]
            if len(fit_rows) == 0 or len(eval_rows) == 0:
                raise ValueError(f"N2b smoke has empty R1 labels for {endpoint}")
            for algorithm in ("dummy_mean", "ridge", "extra_trees"):
                prediction, state_hash = fit_predict_smoke(algorithm, feature_view, loaded.features[fit_rows],
                                                           loaded.targets[fit_rows, target_index], loaded.features[eval_rows])
                # Predictions are intentionally discarded. Their finite shape is
                # the interface check; no score is calculated or persisted.
                rows.append({"outer_test_fold": args.outer_fold, "inner_fold": args.inner_fold,
                             "endpoint": endpoint, "feature_view": feature_view, "algorithm": algorithm,
                             "fit_R1_labels": int(len(fit_rows)), "evaluation_R1_labels": int(len(eval_rows)),
                             "feature_dimension": int(loaded.features.shape[1]), "target_state_hash": state_hash,
                             "predictions_discarded": True})
    audit = pd.DataFrame(rows)
    if len(audit) != len(PRIMARY) * 2 * 3 or not audit.predictions_discarded.all():
        raise ValueError("Incomplete N2b smoke fit matrix")
    with stage_output(args.output) as out:
        audit.to_csv(out / "train_only_nested_smoke_fit_audit.csv", index=False)
        (out / "README.md").write_text(
            "# MMPK N2b strict nested baseline runner smoke\n\n"
            "A single N2a outer/inner context exercises R1 training-only data loading, two frozen feature views, fold-local imputation/scaling, fold-local target standardization and three tiny estimators across four endpoints. Inner-evaluation predictions are checked for finiteness then discarded. No metric, selected model, outer-fold prediction or external label is created. Eight-tree ExtraTrees is an engineering-only smoke surrogate, not an N2a candidate result.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n2b_strict_nested_baseline_runner_smoke", inputs={
            str((args.protocol / "complete.json").resolve()): sha256(args.protocol / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
            str((args.n1c / "complete.json").resolve()): sha256(args.n1c / "complete.json")},
            smoke_only=True, no_outer_fold_scoring=True, no_performance_metrics=True, no_model_selection=True,
            external_labels_accessed=False, predictions_persisted=False, models_persisted=False,
            outer_test_fold=args.outer_fold, inner_fold=args.inner_fold, partial=False)
    print(f"MMPK N2b strict nested baseline runner smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
