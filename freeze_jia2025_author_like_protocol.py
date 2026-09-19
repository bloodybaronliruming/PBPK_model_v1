#!/usr/bin/env python3
"""Freeze the B-grade, author-like Jia 2025 Fu/CL/VDss baseline protocol.

No estimator is fitted here.  The script intentionally makes all choices that
would otherwise be made after inspecting the author holdout: cohort role,
duplicate policy, feature recipe, model budget, scoring and lifecycle rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import (ROOT, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check,
                             verify_stage)


PAPER_ENDPOINTS = {"lgFu": "Fu", "lgCL": "CL", "lgVDss": "VDss"}
PAPER_PRIMARY = {
    "Fu": {"primary_id": "paper_svr_merged", "reason": "Published best configuration: SVM with merged descriptors."},
    "CL": {"primary_id": "paper_consensus_15", "reason": "Published best configuration: mean consensus of all 15 paper cells."},
    "VDss": {"primary_id": "paper_consensus_15", "reason": "Published best configuration: mean consensus of all 15 paper cells."},
}
TASK_FIELDS = {
    "Fu": {"raw": "Fu_final", "log": "lgFu", "split": "Fu_train_test", "unit": "fraction (unitless)"},
    "CL": {"raw": "CL_final(L/hour/kg)", "log": "lgCL", "split": "CL_train_test", "unit": "L/hour/kg"},
    "VDss": {"raw": "VD_final(L/kg)", "log": "lgVD", "split": "VD_train_test", "unit": "L/kg"},
}


def read_paper_hyperparameters(workbook: Path) -> pd.DataFrame:
    """Extract the 3 x 5 fixed configurations from the author SI sheet."""
    raw = pd.read_excel(workbook, sheet_name="hyperparameter for models", header=None)
    rows = []
    current = None
    for _, row in raw.iloc[2:].iterrows():
        endpoint = row.iloc[0]
        if pd.notna(endpoint):
            current = str(endpoint).strip()
        if current not in PAPER_ENDPOINTS:
            continue
        descriptor = row.iloc[1]
        if pd.isna(descriptor):
            continue
        descriptor = str(descriptor).strip()
        if descriptor not in {"rdkit", "MACCS", "FCFP6", "mordred", "merged"}:
            continue
        rows.append({
            "endpoint": PAPER_ENDPOINTS[current], "paper_target": current,
            "descriptor_set": descriptor,
            "rf_n_estimators": int(row.iloc[2]), "rf_min_samples_leaf": int(row.iloc[3]), "rf_max_depth": int(row.iloc[4]),
            "svr_kernel": str(row.iloc[5]), "svr_gamma": str(row.iloc[6]), "svr_C": float(row.iloc[7]),
            "xgb_learning_rate": float(row.iloc[8]), "xgb_max_depth": int(row.iloc[9]), "xgb_n_estimators": int(row.iloc[10]),
        })
    table = pd.DataFrame(rows).sort_values(["endpoint", "descriptor_set"]).reset_index(drop=True)
    if len(table) != 15 or table.groupby("endpoint").size().to_dict() != {"CL": 5, "Fu": 5, "VDss": 5}:
        raise ValueError("Could not recover the expected 15 Jia hyperparameter cells from SI workbook")
    return table


def main():
    parser = argparse.ArgumentParser(description="Freeze, but do not train, the Jia 2025 author-like baseline protocol.")
    parser.add_argument("--workbook", type=Path, default=ROOT / "基准研究参考/jm5c00340_si_001.xlsx")
    parser.add_argument("--n0-audit", type=Path, default=ROOT / "results/analysis/jia2025_n0_audit_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/jia2025_author_like_protocol_v1")
    args = parser.parse_args()
    startup_self_check([args.workbook, args.n0_audit / "complete.json"], output=args.output)
    configure_logging(ROOT, "freeze_jia2025_author_like_protocol")
    n0 = verify_stage(args.n0_audit, "jia2025_n0_reproducibility_audit")
    if n0.get("provisional_reproduction_grade") != "B":
        raise ValueError("This protocol is only valid for the audited B-grade author-like route")

    paper_cells = read_paper_hyperparameters(args.workbook)
    inputs = {
        str(args.workbook.resolve()): sha256(args.workbook),
        str((args.n0_audit / "complete.json").resolve()): sha256(args.n0_audit / "complete.json"),
        str((args.n0_audit / "author_split_membership_hashed.csv").resolve()): sha256(args.n0_audit / "author_split_membership_hashed.csv"),
        str((args.n0_audit / "author_split_overlap_audit.csv").resolve()): sha256(args.n0_audit / "author_split_overlap_audit.csv"),
    }
    with stage_output(args.output) as out:
        paper_cells.to_csv(out / "paper_hyperparameter_cells.csv", index=False)
        budget = []
        for endpoint in ("Fu", "CL", "VDss"):
            for descriptor in ("rdkit", "MACCS", "FCFP6", "mordred", "merged"):
                budget.extend([
                    {"endpoint": endpoint, "family": "paper_aligned", "model_id": f"paper_rf_{descriptor}", "representation": descriptor, "selection_role": "secondary_all_cells", "budget_origin": "local_Jia_SI_fixed_hyperparameters"},
                    {"endpoint": endpoint, "family": "paper_aligned", "model_id": f"paper_svr_{descriptor}", "representation": descriptor, "selection_role": "paper_primary_only_if_published_choice", "budget_origin": "local_Jia_SI_fixed_hyperparameters"},
                    {"endpoint": endpoint, "family": "paper_aligned", "model_id": f"paper_xgb_{descriptor}", "representation": descriptor, "selection_role": "secondary_all_cells", "budget_origin": "local_Jia_SI_fixed_hyperparameters"},
                ])
            budget.extend([
                {"endpoint": endpoint, "family": "fixed_project_reference", "model_id": "dummy_log_mean", "representation": "none", "selection_role": "mandatory_reference", "budget_origin": "predeclared"},
                {"endpoint": endpoint, "family": "fixed_project_reference", "model_id": "ridge_rdkit2d", "representation": "rdkit2d", "selection_role": "mandatory_reference", "budget_origin": "predeclared_alpha_10"},
                {"endpoint": endpoint, "family": "fixed_project_reference", "model_id": "extra_trees_ecfp4_rdkit2d", "representation": "ecfp4_rdkit2d", "selection_role": "mandatory_reference", "budget_origin": "predeclared_300trees_maxfeatures0.7_leaf3_seed20260916"},
            ])
        budget = pd.DataFrame(budget)
        budget.to_csv(out / "predeclared_model_budget.csv", index=False)

        cohort_rules = pd.DataFrame([
            {"cohort": "author_native_record", "role": "primary_author_like", "membership": "exact author train/test fields after finite raw+log consistency check", "duplicate_parent_policy": "retain records exactly as authored; report parent counts and duplication", "fu_cross_split_parent_policy": "retain in primary author-like result; flag non-strict leakage", "test_role": "one descriptive post-freeze score only"},
            {"cohort": "author_parent_purged_sensitivity", "role": "mandatory_sensitivity_not_selection", "membership": "author test after removing any parent present in author train", "duplicate_parent_policy": "median log target per parent within split", "fu_cross_split_parent_policy": "purge all 10 overlapping Fu test parents", "test_role": "report alongside primary; never choose a winner"},
            {"cohort": "project_strict_protocol", "role": "separate_noncomparable_column", "membership": "existing project scaffold/source-cluster protocol only", "duplicate_parent_policy": "existing project rules", "fu_cross_split_parent_policy": "not applicable", "test_role": "never mix with Jia author-test results"},
        ])
        cohort_rules.to_csv(out / "cohort_and_duplicate_rules.csv", index=False)

        feature_rules = pd.DataFrame([
            {"representation": "rdkit", "implementation": "RDKit 2D descriptor family; exact output dimension is environment-locked at feature-build time", "train_only_operations": "drop nonfinite columns, median impute; no test-informed filtering", "scale_for": "SVR only"},
            {"representation": "MACCS", "implementation": "RDKit MACCS keys", "train_only_operations": "none beyond finite check", "scale_for": "SVR only"},
            {"representation": "FCFP6", "implementation": "RDKit feature-invariant Morgan radius=3, nBits=1024", "train_only_operations": "none beyond finite check", "scale_for": "SVR only"},
            {"representation": "mordred", "implementation": "Mordred 2D descriptors; package/version recorded at execution", "train_only_operations": "drop columns with training missingness; variance >=0.2; greedily remove one of pairs with |Pearson r|>0.85; median impute surviving features", "scale_for": "SVR only"},
            {"representation": "merged", "implementation": "concatenate MACCS, FCFP6, RDKit2D and Mordred after train-only filtering", "train_only_operations": "same train-only Mordred/merged filtering; no test-informed descriptor identity purge", "scale_for": "SVR only"},
            {"representation": "ecfp4_rdkit2d", "implementation": "project reference: ECFP4 2048 bits plus project deterministic RDKit2D", "train_only_operations": "median impute", "scale_for": "ridge only"},
        ])
        feature_rules.to_csv(out / "feature_and_preprocessing_rules.csv", index=False)

        metrics = pd.DataFrame([
            {"metric": "R2_log", "space": "log10", "primary": True, "definition": "coefficient of determination on author-test records"},
            {"metric": "MAE_log", "space": "log10", "primary": True, "definition": "mean absolute error on author-test records"},
            {"metric": "RMSE_log", "space": "log10", "primary": True, "definition": "root mean squared error on author-test records"},
            {"metric": "GMFE", "space": "inverse_transformed", "primary": True, "definition": "10^(mean absolute log10 error); no prediction clipping"},
            {"metric": "within_2fold", "space": "inverse_transformed", "primary": True, "definition": "fraction with absolute log10 error <= log10(2)"},
            {"metric": "Spearman", "space": "log10", "primary": False, "definition": "descriptive ranking diagnostic"},
        ])
        metrics.to_csv(out / "metric_and_reporting_rules.csv", index=False)

        lifecycle = {
            "author_test_status_at_freeze": "technical_label_presence_accessed_in_N0_not_scored",
            "N0_raw_label_values_exported": False,
            "N0_predictive_metrics_computed": False,
            "model_selection_after_N0": "prohibited",
            "permitted_next_action": "fit only frozen configurations using author-train; then score author-test once descriptively",
            "author_test_re_evaluation": "prohibited after the single scored release",
            "selection_criterion": "none on author-test; paper-primary choices are predeclared from the published article, all other cells reported without winner selection",
            "project_consumed_test_access": False,
            "project_test_usage": "prohibited",
        }
        dump_json(out / "author_test_lifecycle.json", lifecycle)
        protocol = {
            "protocol_id": "jia2025_author_like_protocol_v1",
            "grade": "B_author_like_not_exact_reproduction",
            "scope": "human IV Fu/CL/VDss external literature comparison",
            "tasks": TASK_FIELDS,
            "paper_primary_models": PAPER_PRIMARY,
            "train_cv": {"folds": 10, "purpose": "diagnostic only; does not choose a model after protocol freeze", "grouping": "record-level native cohort; parent-purged sensitivity reported separately"},
            "target_contract": "fit and score log10 targets; verify log column against raw units before fitting; no clipping before inverse-transform metrics",
            "author_paper_difference": "Do not remove author-train records based on identity with author-test descriptors; such test-informed preprocessing is forbidden here.",
            "reporting_contract": "author-native primary, parent-purged sensitivity, and project strict protocol must be separate columns; no cross-protocol winner declaration.",
            "no_training_in_this_stage": True,
        }
        dump_json(out / "protocol.json", protocol)
        report = """# Jia 2025 author-like baseline protocol v1

## Status

Frozen B-grade author-like protocol; no estimator was fitted and no author-test score was calculated.

## Why B-grade

The local workbook provides endpoint values and author train/test fields, but not the exact code or split-generation environment. N0 further showed parent overlap in Fu and scaffold overlap in all three endpoints. This protocol therefore reproduces a transparent approximation of the published comparison, not an exact or scaffold-strict reproduction.

## Cohorts and leakage disclosure

`author_native_record` is the fixed primary literature-comparison cohort because it retains the author membership. Its result must disclose Fu parent overlap and all scaffold overlap. `author_parent_purged_sensitivity` is mandatory and reports a leakage-reduced sensitivity; it is not a replacement selected by test performance. Project strict scaffold/source-cluster results remain entirely separate.

## Frozen model budget

The SI's 15 RF/SVR/XGBoost descriptor cells are fixed in `paper_hyperparameter_cells.csv`. Published primary choices are predeclared: Fu is SVR/merged; CL and VDss are the arithmetic mean of all 15 paper-aligned predictions. Dummy, Ridge and fixed ExtraTrees are comparison references, not a search space. No author-test result may determine a winner.

## Preprocessing and scoring

All imputation, filtering, correlation pruning and scaling are fitted inside the relevant author training partition only. In particular, unlike the paper's description, no author-train row is removed because of its relationship to author-test features. Report log-space R²/MAE/RMSE and inverse-scale GMFE/within-2-fold without clipping. The author test may be scored once only after all configurations are frozen and fitted.
"""
        (out / "protocol_report.md").write_text(report, encoding="utf-8")
        finish_stage(out, "jia2025_author_like_protocol_freeze", inputs=inputs,
                     no_training=True, author_test_scored=False, project_test_accessed=False,
                     protocol_grade="B_author_like", paper_hyperparameter_cells=int(len(paper_cells)),
                     predeclared_models=int(len(budget)), partial=False)
    print(f"Jia 2025 author-like protocol freeze: {args.output}")


if __name__ == "__main__":
    run_cli(main)
