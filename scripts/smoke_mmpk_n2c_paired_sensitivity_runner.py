#!/usr/bin/env python3
"""Technical, no-label smoke for the frozen MMPK N2c paired sensitivities."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mmpk_nca_common import (ENDPOINTS, fit_formulation_one_hot, load_approved_formulation_context,
                             load_strict_outer_inputs, load_strict_train_fold)
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--n1d", type=Path, default=ROOT / "results/analysis/mmpk_n1d_context_derivation_audit_v1")
    parser.add_argument("--n2c", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2c_paired_sensitivity_protocol_v1")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/mmpk_n2c_paired_sensitivity_runner_smoke_v1")
    args = parser.parse_args()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    required = [args.n0 / "complete.json", args.n1b / "complete.json", args.n1d / "complete.json", args.n2c / "complete.json"]
    startup_self_check(required, output=args.output)
    verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    verify_stage(args.n1d, "mmpk_n1d_approved_context_and_derivation_audit")
    protocol = verify_stage(args.n2c, "mmpk_n2c_paired_sensitivity_protocol")
    selected = pd.read_csv(args.n2c / "outer_specific_R1_selected_configuration.csv")
    selected = selected[selected.outer_test_fold.eq(args.outer_fold)].copy()
    if set(selected.endpoint) != set(PRIMARY) or selected.feature_view.nunique() != 1:
        raise ValueError("N2c selected configuration is incomplete or has an unexpected feature view")
    feature_view = selected.feature_view.iloc[0]
    r1 = load_strict_train_fold(args.outer_fold, "R1_direct", n1b=args.n1b, feature_view=feature_view)
    r12 = load_strict_train_fold(args.outer_fold, "R1_plus_R2_augmented", n1b=args.n1b, feature_view=feature_view)
    outer_ids, _, outer_x, _ = load_strict_outer_inputs(args.outer_fold, feature_view, n1b=args.n1b)
    train_context = load_approved_formulation_context(r1.model_record_ids, n1b=args.n1b, n1d=args.n1d)
    outer_context = load_approved_formulation_context(outer_ids, n1b=args.n1b, n1d=args.n1d)
    train_keep = ~train_context.formulation_state.eq("multiple_values").to_numpy()
    outer_keep = ~outer_context.formulation_state.eq("multiple_values").to_numpy()
    train_onehot, outer_onehot, encoder = fit_formulation_one_hot(train_context.loc[train_keep], outer_context.loc[outer_keep])
    if train_onehot.shape[0] != int(train_keep.sum()) or outer_onehot.shape[0] != int(outer_keep.sum()):
        raise ValueError("Formulation encoder row accounting failed")
    if not np.isfinite(train_onehot).all() or not np.isfinite(outer_onehot).all():
        raise ValueError("Formulation one-hot matrix is non-finite")
    endpoint_index = {endpoint: index for index, (endpoint, _) in enumerate(ENDPOINTS)}
    endpoint_rows = []
    for endpoint in PRIMARY:
        index = endpoint_index[endpoint]
        endpoint_rows.append({
            "outer_fold": args.outer_fold,
            "endpoint": endpoint,
            "selected_candidate_id": selected.loc[selected.endpoint.eq(endpoint), "candidate_id"].iloc[0],
            "R1_train_labels_before_formulation_filter": int(r1.target_mask[:, index].sum()),
            "R1_train_labels_after_formulation_filter": int((r1.target_mask[:, index] & train_keep).sum()),
            "R1_plus_R2_train_labels": int(r12.target_mask[:, index].sum()),
            "base_feature_dimension": int(r1.features.shape[1]),
            "formulation_feature_dimension": int(r1.features.shape[1] + train_onehot.shape[1]),
            "outer_input_records_before_formulation_filter": int(len(outer_ids)),
            "outer_input_records_after_formulation_filter": int(outer_keep.sum()),
        })
    summary = pd.DataFrame([{
        "outer_fold": args.outer_fold,
        "feature_view": feature_view,
        "train_records_before_formulation_filter": int(len(r1.model_record_ids)),
        "train_records_after_formulation_filter": int(train_keep.sum()),
        "outer_input_records_before_formulation_filter": int(len(outer_ids)),
        "outer_input_records_after_formulation_filter": int(outer_keep.sum()),
        **encoder,
        "train_context_membership_hash": stable_id("\n".join(r1.model_record_ids)),
        "outer_input_membership_hash": stable_id("\n".join(outer_ids)),
    }])
    with stage_output(args.output) as out:
        pd.DataFrame(endpoint_rows).to_csv(out / "endpoint_no_label_shape_audit.csv", index=False)
        summary.to_csv(out / "formulation_no_label_encoder_audit.csv", index=False)
        (out / "README.md").write_text(
            "# MMPK N2c paired sensitivity runner smoke\n\n"
            "This technical smoke reconstructs only strict train/held inputs and formulation context for one predeclared outer fold. It does not open target columns for the held fold, fit a model, create predictions, calculate performance, or access any external cohort. Raw formulation strings remain in memory and are not exported.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n2c_paired_sensitivity_runner_smoke", inputs={
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
            str((args.n1d / "complete.json").resolve()): sha256(args.n1d / "complete.json"),
            str((args.n2c / "complete.json").resolve()): sha256(args.n2c / "complete.json")},
            outer_fold=args.outer_fold, no_model_fit=True, no_predictions=True, no_performance_metrics=True,
            outer_target_columns_opened=False, external_labels_accessed=False, raw_formulation_exported=False,
            selected_configuration_reselected=False, partial=False)
    print(f"MMPK N2c paired sensitivity runner smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
