#!/usr/bin/env python3
"""Freeze N2c paired sensitivity analyses from the completed MMPK N2b screen.

This is a registration-only step.  It neither fits a model nor reads any
target from an outer strict fold.  The selected N2b configuration is bound
per endpoint and outer fold before either sensitivity arm can be run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")


def _require_unique_selection(selection: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"outer_test_fold", "endpoint", "candidate_id", "algorithm", "feature_view",
                "mean_inner_log_RMSE", "inner_log_RMSE_sd", "inner_folds", "selected"}
    if not required <= set(selection):
        raise ValueError("N2b inner candidate selection schema is incomplete")
    if not set(selection.endpoint) <= set(PRIMARY) or not selection.outer_test_fold.isin(range(5)).all():
        raise ValueError("N2b selection contains an unexpected endpoint or strict fold")
    if selection.groupby(["outer_test_fold", "endpoint"]).size().ne(len(candidates)).any():
        raise ValueError("N2b does not contain every frozen candidate in every outer endpoint cell")
    winners = selection[selection.selected.astype(bool)].copy()
    if len(winners) != 5 * len(PRIMARY) or winners.duplicated(["outer_test_fold", "endpoint"]).any():
        raise ValueError("N2b must have exactly one selected candidate per outer endpoint cell")
    checked = winners.merge(candidates[["candidate_id", "algorithm", "feature_view", "parameters_json"]],
                            on=["candidate_id", "algorithm", "feature_view"], how="left", validate="many_to_one")
    if checked.parameters_json.isna().any():
        raise ValueError("A selected N2b candidate is absent from the frozen N2a registry")
    if not checked.feature_view.eq("ecfp4_rdkit2d_logdose").all():
        raise ValueError("Unexpected N2b view; N2c must bind the completed N2b configuration exactly")
    return checked.sort_values(["endpoint", "outer_test_fold"]).reset_index(drop=True)


def _pooled_inner_summary(selection: pd.DataFrame) -> pd.DataFrame:
    """Summarize inner-only evidence; it is not an outer-performance model choice."""
    columns = ["endpoint", "candidate_id", "algorithm", "feature_view"]
    summary = (selection.groupby(columns, as_index=False)
               .agg(mean_outer_train_inner_log_RMSE=("mean_inner_log_RMSE", "mean"),
                    sd_outer_train_inner_log_RMSE=("mean_inner_log_RMSE", "std"),
                    outer_train_repetitions=("outer_test_fold", "nunique")))
    summary["inner_only_rank"] = summary.groupby("endpoint")["mean_outer_train_inner_log_RMSE"].rank(
        method="first", ascending=True).astype(int)
    summary["role"] = "inner-only descriptive deployment candidate; never an outer-score-selected champion"
    return summary.sort_values(["endpoint", "inner_only_rank", "candidate_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--n1d", type=Path, default=ROOT / "results/analysis/mmpk_n1d_context_derivation_audit_v1")
    parser.add_argument("--n2a", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2a_baseline_protocol_v2")
    parser.add_argument("--n2b", type=Path, default=ROOT / "results/benchmarks/mmpk_n2b_strict_r1_baseline_screen_v1")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2c_paired_sensitivity_protocol_v1")
    args = parser.parse_args()
    required = [args.n0 / "complete.json", args.n1b / "complete.json", args.n1d / "complete.json",
                args.n2a / "complete.json", args.n2b / "complete.json"]
    startup_self_check(required, output=args.output)
    verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    n1d = verify_stage(args.n1d, "mmpk_n1d_approved_context_and_derivation_audit")
    n2a = verify_stage(args.n2a, "mmpk_n2a_strict_conditional_baseline_protocol")
    n2b = verify_stage(args.n2b, "mmpk_n2b_strict_r1_conditional_baseline_screen")
    if not n2b.get("formal_r1_screen") or n2b.get("R2_used_for_selection") or n2b.get("formulation_used_for_selection"):
        raise ValueError("N2b is not the required R1-only, pre-sensitivity formal screen")
    if n2b.get("external_labels_accessed") or n2b.get("strict_outer_folds") != 5 or n2b.get("strict_inner_folds") != 4:
        raise ValueError("N2b scope or split contract differs from N2c requirements")
    if n1d.get("time_window_row_context_available") or n1d.get("analyte_matrix_row_context_available"):
        raise ValueError("N1d context contract unexpectedly changed")
    candidates = pd.read_csv(args.n2a / "baseline_candidate_registry.csv")
    selection = pd.read_csv(args.n2b / "inner_candidate_selection.csv")
    winners = _require_unique_selection(selection, candidates)
    pooled = _pooled_inner_summary(selection)
    selected = winners[["outer_test_fold", "endpoint", "candidate_id", "algorithm", "feature_view", "parameters_json",
                        "mean_inner_log_RMSE", "inner_log_RMSE_sd", "inner_folds"]].copy()
    selected["selection_scope"] = "N2b frozen outer-train inner-fold R1 log_RMSE only"
    selected["outer_score_used_for_selection"] = False
    arms = pd.DataFrame([
        ("baseline_reuse", "N2b saved R1 prediction", "R1 direct outer labels", "all R1 outer records", "No refit; paired reference only"),
        ("R1_plus_R2_train", "R1+R2 train labels; N2b-selected feature/configuration", "same R1 direct outer labels", "all R1 outer records", "R2 never independently interpreted; no reselection"),
        ("formulation_matched", "R1 train labels; N2b-selected feature/configuration plus formulation one-hot", "same R1 direct outer labels", "outer rows with single formulation or explicit missing state; multiple-context rows excluded", "one-hot vocabulary fit on outer train only; baseline is restricted to identical paired rows"),
    ], columns=["arm", "training_rule", "evaluation_rule", "paired_population", "nonnegotiable_constraint"])
    reporting = pd.DataFrame([
        ("primary_delta", "sensitivity log_RMSE minus baseline log_RMSE", "same endpoint, outer-fold union, and arm-specific paired record set"),
        ("outer_direction", "count of outer folds with lower sensitivity log_RMSE", "reported as descriptive 0..5; no fold is dropped"),
        ("clustered_bootstrap", "canonical-parent clustered paired bootstrap", f"{args.bootstrap_replicates} resamples, seed {args.bootstrap_seed}, percentile 95% CI"),
        ("sensitivity_signal", "eligible only when relative pooled log_RMSE improvement >=2%, at least 3/5 outer folds improve, and bootstrap 95% CI upper bound <0", "does not replace the N2b baseline or authorize an architecture claim by itself"),
        ("prohibited", "outer-test-driven reselection, combined R1+R2+formulation arm, unpaired formulation comparison, external scoring", "hard failure"),
    ], columns=["item", "definition", "scope_or_decision"])
    protocol = {
        "scope": "internal_only_approved_human_oral_MMPK; paired sensitivity after completed strict R1 screen",
        "parent_protocol": "mmpk_n2a_strict_conditional_baseline_protocol",
        "bound_baseline": "mmpk_n2b_strict_r1_conditional_baseline_screen",
        "primary_endpoints": list(PRIMARY),
        "selected_cells": int(len(selected)),
        "selection": "exact N2b outer-specific candidate configuration; no configuration, feature-view, or hyperparameter reselection",
        "arms": ["R1_plus_R2_train", "formulation_matched"],
        "formulation_contract": "derive from N1d-reproduced raw-model mapping; only single value or explicit missing allowed; multiple raw-context values excluded; categories and one-hot vocabulary fit within outer training records only",
        "evaluation": "same frozen strict outer R1 membership after selection; formulation comparison is paired on its predeclared eligible subset",
        "bootstrap": {"unit": "canonical_parent", "replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                      "interval": "two-sided percentile 95%", "delta": "sensitivity minus baseline log_RMSE"},
        "forbidden": ["outer-score-driven selection", "R2/formulation use for selection", "combined augmented-plus-formulation arm", "external cohort loading/scoring", "public data/model release before rights clarification"],
    }
    with stage_output(args.output) as out:
        selected.to_csv(out / "outer_specific_R1_selected_configuration.csv", index=False)
        pooled.to_csv(out / "pooled_inner_candidate_summary.csv", index=False)
        arms.to_csv(out / "sensitivity_arm_registry.csv", index=False)
        reporting.to_csv(out / "paired_reporting_and_bootstrap_registry.csv", index=False)
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# MMPK N2c paired sensitivity protocol\n\n"
            "This immutable, internal-only registration binds every N2c fit to the R1-only N2b winner for its endpoint and strict outer fold. It contains no targets, predictions, performance metrics or fitted model. R1+R2 and formulation are separate paired sensitivity arms; neither can reselect or replace N2b.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n2c_paired_sensitivity_protocol", inputs={
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
            str((args.n1d / "complete.json").resolve()): sha256(args.n1d / "complete.json"),
            str((args.n2a / "complete.json").resolve()): sha256(args.n2a / "complete.json"),
            str((args.n2b / "complete.json").resolve()): sha256(args.n2b / "complete.json")},
            no_training=True, no_predictions=True, no_performance_metrics=True, external_labels_accessed=False,
            outer_selection_forbidden=True, R2_used_for_selection=False, formulation_used_for_selection=False,
            paired_bootstrap_unit="canonical_parent", bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed, partial=False)
    print(f"MMPK N2c paired sensitivity protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
