#!/usr/bin/env python3
"""Freeze the MMPK N2a strict conditional-baseline protocol without fitting models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")
INNER_FOLDS = 4


def balanced_inner_components(records: pd.DataFrame, labels: pd.DataFrame) -> dict[str, int]:
    """Allocate outer-train components using counts/presence, never target values."""
    direct = labels[(labels.endpoint.isin(PRIMARY)) & labels.direct_observation_eligible.astype(bool)]
    summary = records[["strict_component_id"]].drop_duplicates().set_index("strict_component_id")
    summary["records"] = records.groupby("strict_component_id").size()
    for endpoint in PRIMARY:
        summary[endpoint] = direct[direct.endpoint.eq(endpoint)].groupby("strict_component_id").size()
    summary = summary.fillna(0).astype(int)
    columns = ["records", *PRIMARY]
    weights = np.asarray([1.0, 1.5, 1.5, 1.5, 1.5])
    target = summary[columns].sum().to_numpy(float) / INNER_FOLDS
    counts = np.zeros((INNER_FOLDS, len(columns)), dtype=float)
    assignment: dict[str, int] = {}
    order = summary.assign(_weight=summary[columns].mul(weights, axis=1).sum(axis=1)).sort_values(
        ["_weight", "records"], ascending=[False, False], kind="stable")
    for rank, (component, values) in enumerate(order.iterrows()):
        vector = values[columns].to_numpy(float)
        if rank < INNER_FOLDS:
            chosen = rank
        else:
            scale = np.where(target > 0, target, 1.0)
            score = []
            for fold in range(INNER_FOLDS):
                proposed = counts.copy()
                proposed[fold] += vector
                score.append(float(np.sum(weights * ((proposed - target) / scale) ** 2)))
            chosen = min(range(INNER_FOLDS), key=lambda fold: (score[fold], counts[fold, 0], fold))
        counts[chosen] += vector
        assignment[component] = chosen
    return assignment


def candidate_registry(seed: int) -> pd.DataFrame:
    """Fully enumerate the bounded screen so the runner has no implicit grid."""
    rows = [{"candidate_id": "dummy_mean__dose_only", "algorithm": "dummy_mean", "feature_view": "dose_only",
             "parameters_json": "{}", "random_seed": seed, "preprocessing": "none", "role": "reference only; never wins by default"}]
    for view in ("ecfp4_logdose", "ecfp4_rdkit2d_logdose"):
        for alpha in (0.1, 1.0, 10.0, 100.0):
            rows.append({"candidate_id": f"ridge__{view}__alpha{alpha:g}", "algorithm": "ridge", "feature_view": view,
                         "parameters_json": json.dumps({"alpha": alpha, "solver": "lsqr", "tol": 1e-6}, sort_keys=True),
                         "random_seed": seed, "preprocessing": "train-fold median imputation + scaling", "role": "linear regularized baseline"})
        for family, settings, role in (
            ("random_forest", (("a", {"n_estimators": 400, "max_features": 0.3, "min_samples_leaf": 1, "max_depth": None}),
                                ("b", {"n_estimators": 400, "max_features": 0.7, "min_samples_leaf": 3, "max_depth": 24})), "bounded tree baseline"),
            ("extra_trees", (("a", {"n_estimators": 400, "max_features": 0.3, "min_samples_leaf": 1, "max_depth": None}),
                              ("b", {"n_estimators": 400, "max_features": 0.7, "min_samples_leaf": 3, "max_depth": 24})), "bounded tree baseline"),
            ("xgboost", (("a", {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}),
                          ("b", {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 0.8})), "GPU only after explicit device smoke; otherwise deferred"),
        ):
            for suffix, parameters in settings:
                rows.append({"candidate_id": f"{family}__{view}__{suffix}", "algorithm": family, "feature_view": view,
                             "parameters_json": json.dumps(parameters, sort_keys=True), "random_seed": seed,
                             "preprocessing": "train-fold median imputation; scaling only for linear models", "role": role})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--n1c", type=Path, default=ROOT / "results/analysis/mmpk_n1c_train_fold_loader_smoke_v1")
    parser.add_argument("--n1d", type=Path, default=ROOT / "results/analysis/mmpk_n1d_context_derivation_audit_v1")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2a_baseline_protocol_v2")
    args = parser.parse_args()
    required = [args.n0 / "complete.json", args.n1b / "complete.json", args.n1c / "complete.json", args.n1d / "complete.json"]
    startup_self_check(required, output=args.output)
    verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    n1b = verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    verify_stage(args.n1c, "mmpk_n1c_strict_train_fold_conditional_loader_smoke")
    verify_stage(args.n1d, "mmpk_n1d_approved_context_and_derivation_audit")
    records = pd.read_csv(args.n1b / "approved_record_registry_hashed.csv")
    labels = pd.read_csv(args.n1b / "approved_label_tier_registry_hashed.csv")
    author = pd.read_csv(args.n1b / "author_exact_smiles_split_membership_hashed.csv")
    if not records.strict_outer_fold.isin(range(5)).all() or records.model_record_id.duplicated().any():
        raise ValueError("N1b strict outer membership is invalid")
    if set(PRIMARY) - set(labels.endpoint):
        raise ValueError("N1b lacks a primary endpoint")
    if not labels.author_augmented_eligible.astype(bool).all():
        raise ValueError("N1b v3 label registry must contain only finite transformed targets")
    inner_rows, audits = [], []
    for outer in range(5):
        outer_train = records[records.strict_outer_fold.ne(outer)].copy()
        outer_train["outer_test_fold"] = outer
        outer_labels = labels[labels.model_record_id.isin(outer_train.model_record_id)].copy()
        assignment = balanced_inner_components(outer_train, outer_labels)
        outer_train["inner_fold"] = outer_train.strict_component_id.map(assignment)
        if outer_train.inner_fold.isna().any() or not outer_train.inner_fold.isin(range(INNER_FOLDS)).all():
            raise ValueError("An outer-train component lacks an inner-fold assignment")
        for fold in range(INNER_FOLDS):
            evaluation = outer_train[outer_train.inner_fold.eq(fold)]
            fitting = outer_train[outer_train.inner_fold.ne(fold)]
            overlap = {
                "component_overlap": len(set(evaluation.strict_component_id) & set(fitting.strict_component_id)),
                "parent_overlap": len(set(evaluation.parent_id) & set(fitting.parent_id)),
                "scaffold_overlap": len(set(evaluation.scaffold_id) & set(fitting.scaffold_id)),
            }
            if any(overlap.values()) or evaluation.empty or fitting.empty:
                raise ValueError("Nested strict inner fold leaks or is empty")
            for endpoint in PRIMARY:
                eligible = outer_labels[(outer_labels.endpoint.eq(endpoint)) & outer_labels.direct_observation_eligible.astype(bool)]
                evaluation_labels = int(eligible.model_record_id.isin(evaluation.model_record_id).sum())
                fitting_labels = int(eligible.model_record_id.isin(fitting.model_record_id).sum())
                if evaluation_labels <= 0 or fitting_labels <= 0:
                    raise ValueError(f"{outer}/{fold}/{endpoint} has empty R1 nested membership")
                audits.append({"outer_fold": outer, "inner_fold": fold, "endpoint": endpoint,
                               "inner_evaluation_records": len(evaluation), "inner_fitting_records": len(fitting),
                               "inner_evaluation_R1_labels": evaluation_labels, "inner_fitting_R1_labels": fitting_labels,
                               **overlap})
        inner_rows.append(outer_train[["model_record_id", "outer_test_fold", "strict_outer_fold", "inner_fold", "strict_component_id", "parent_id", "scaffold_id"]])
    inner = pd.concat(inner_rows, ignore_index=True).sort_values(["outer_test_fold", "inner_fold", "model_record_id"])
    audit = pd.DataFrame(audits)
    if (audit[["component_overlap", "parent_overlap", "scaffold_overlap"]] != 0).any().any():
        raise ValueError("Nested leak audit failed")
    author_summary = (author.merge(records[["model_record_id", "author_exact_smiles_id"]], on="model_record_id", validate="many_to_one")
                      .groupby(["fold", "role"]).agg(records=("model_record_id", "size"),
                      unique_exact_smiles=("author_exact_smiles_id", "nunique")).reset_index())
    # Exact-SMILES author membership is a B-grade reference, not a strict test.
    if set(author.role) != {"train", "validation", "test"} or author.groupby("fold").size().ne(len(records)).any():
        raise ValueError("N1b author membership is incomplete")
    features = pd.DataFrame([
        ("dose_only", "author Log Dose [mg/kg]", 1, "all primary records", "train-fold standardize when algorithm requires"),
        ("ecfp4_logdose", "ECFP4 radius=2, 2048 bits + author Log Dose [mg/kg]", 2049, "all primary records", "only log-dose scaling for linear models"),
        ("ecfp4_rdkit2d_logdose", "ECFP4 + deterministic RDKit2D + author Log Dose [mg/kg]", "2049+RDKit2D", "all primary records", "imputer/scaler fit on each fitting fold only"),
        ("formulation_sensitivity", "ecfp4_logdose + train-only one-hot formulation", "2049+fold-local categories", "single formulation or explicit missing; multiple context values excluded", "sensitivity only; no primary selection"),
    ], columns=["feature_view", "definition", "dimensions", "eligible_context", "preprocessing_rule"])
    metrics = pd.DataFrame([
        ("log_RMSE", "primary", "record-level transformed target", "all primary endpoints"),
        ("GMFE", "secondary", "10**mean(abs(log prediction - log observation))", "all primary endpoints"),
        ("AFE", "secondary bias", "10**mean(log prediction - log observation)", "all primary endpoints"),
        ("two_fold_accuracy", "secondary", "fraction abs(log error)<=log10(2)", "all primary endpoints"),
        ("log_R2", "secondary", "transformed-space coefficient of determination", "all primary endpoints"),
        ("parent_equal_dose_arm_sensitivity", "sensitivity", "mean within parent then compute above metrics", "report separately; never replaces primary"),
    ], columns=["metric", "role", "definition", "scope"])
    sensitivity = pd.DataFrame([
        ("R1_primary", "R1 direct labels in train; R1 direct labels in outer evaluation", "primary development and selection"),
        ("R1_plus_R2_train_sensitivity", "R1+R2 train labels; R1 direct labels in same outer evaluation", "use R1-selected configuration; no re-selection"),
        ("formulation_matched_sensitivity", "R1 direct, one fixed train-only formulation encoding", "same R1-selected configuration; no re-selection"),
        ("author_like_B_reference", "author exact-SMILES 10-fold; transformed labels", "separate B-grade column; cannot rank against strict"),
    ], columns=["analysis", "membership_and_labels", "interpretation"])
    with stage_output(args.output) as out:
        inner.to_csv(out / "strict_nested_inner_membership_hashed.csv", index=False)
        audit.to_csv(out / "strict_nested_inner_leakage_and_coverage.csv", index=False)
        candidates = candidate_registry(args.seed)
        if len(candidates) != 21 or candidates.candidate_id.duplicated().any():
            raise ValueError("N2a candidate enumeration is not the frozen 21-cell screen")
        candidates.to_csv(out / "baseline_candidate_registry.csv", index=False)
        features.to_csv(out / "feature_view_registry.csv", index=False)
        metrics.to_csv(out / "metric_registry.csv", index=False)
        sensitivity.to_csv(out / "tier_and_context_sensitivity_registry.csv", index=False)
        author_summary.to_csv(out / "author_like_B_membership_summary.csv", index=False)
        protocol = {
            "scope": "internal_only_approved_human_oral_cohort_SMILES_plus_author_log_dose",
            "strict_primary": "5 strict outer folds; 4 frozen parent+source+scaffold inner folds within each outer train; R1 direct labels only",
            "primary_endpoints": list(PRIMARY),
            "selection": "select candidate only by mean inner-fold log_RMSE within each outer-train; outer evaluation is not used for selection",
            "random_seed": args.seed,
            "candidate_cells": int(len(candidates)),
            "outer_reporting": "R1 direct outer evaluation only; report record-primary and parent-equal dose-arm sensitivity separately",
            "augmented_rule": "R1+R2 is sensitivity only, using R1-selected configuration and evaluated on the same R1 outer membership",
            "formulation_rule": "matched sensitivity only; explicit missing category, exclude multiple-context rows, one-hot vocabulary fit only on fitting data",
            "author_like_rule": "exact-SMILES 10-fold B-grade reference only; never rank against strict results",
            "forbidden": ["external cohort loading or scoring", "outer-test-driven selection", "use of AUC Type/free-text/salt/prodrug/body-weight/subject-count as default predictors", "treating R2 as independent direct evidence", "data/model release before data rights clarification"],
        }
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# MMPK N2a conditional-baseline protocol\n\n"
            "This package freezes the first internal-only baseline comparison. It contains hashed memberships, feature/metric/candidate registries and no targets, models, predictions or performance. The strict R1 route is primary; R1+R2 and formulation are fixed sensitivity analyses, while author exact-SMILES CV is a separate B-grade reference.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n2a_strict_conditional_baseline_protocol", inputs={
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
            str((args.n1c / "complete.json").resolve()): sha256(args.n1c / "complete.json"),
            str((args.n1d / "complete.json").resolve()): sha256(args.n1d / "complete.json")},
            no_training=True, no_predictions=True, no_performance_metrics=True, external_labels_accessed=False,
            raw_labels_exported=False, strict_outer_folds=5, strict_inner_folds=INNER_FOLDS,
            primary_endpoints=list(PRIMARY), candidate_cells=int(len(candidates)), random_seed=args.seed,
            author_track_grade="B_exact_smiles_reference_only", partial=False)
    print(f"MMPK N2a strict conditional-baseline protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
