#!/usr/bin/env python3
"""Run only the pre-registered MMPK N2c paired sensitivity refits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mmpk_nca_common import (ENDPOINTS, fit_formulation_one_hot, load_approved_formulation_context,
                             load_strict_outer_inputs, load_strict_train_fold)
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage
from run_mmpk_n2b_strict_baseline_screen import load_outer_r1_targets_after_selection, make_prediction, metric_row


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")
ARMS = ("R1_plus_R2_train", "formulation_matched")


def _checked_baseline(predictions: pd.DataFrame, outer: int, endpoint: str, candidate_id: str) -> pd.DataFrame:
    rows = predictions[(predictions.outer_test_fold.eq(outer)) & predictions.endpoint.eq(endpoint)].copy()
    required = {"model_record_id", "selected_candidate_id", "predicted_transformed_value"}
    if not required <= set(rows) or rows.model_record_id.duplicated().any() or rows.empty:
        raise ValueError("N2b saved baseline prediction membership is invalid")
    if not rows.selected_candidate_id.eq(candidate_id).all():
        raise ValueError("N2b saved baseline prediction uses a different selected configuration")
    return rows[["model_record_id", "predicted_transformed_value"]].rename(columns={"predicted_transformed_value": "baseline_prediction"})


def _make_formulation_features(base_train, base_outer, train_context, outer_context):
    train_keep = ~train_context.formulation_state.eq("multiple_values").to_numpy()
    outer_keep = ~outer_context.formulation_state.eq("multiple_values").to_numpy()
    train_onehot, outer_onehot, state = fit_formulation_one_hot(train_context.loc[train_keep], outer_context.loc[outer_keep])
    return (np.column_stack([base_train[train_keep], train_onehot]).astype(np.float32, copy=False),
            np.column_stack([base_outer[outer_keep], outer_onehot]).astype(np.float32, copy=False),
            train_keep, outer_keep, state)


def _score_pair(y_by_id: dict[str, float], pair: pd.DataFrame, arm: str, outer: int, endpoint: str) -> tuple[dict, list[dict]]:
    if pair.model_record_id.duplicated().any() or not set(pair.model_record_id) <= set(y_by_id):
        raise ValueError("Paired prediction membership differs from delayed R1 target membership")
    y = np.asarray([y_by_id[record] for record in pair.model_record_id], dtype=float)
    baseline = pair.baseline_prediction.to_numpy(float)
    sensitivity = pair.sensitivity_prediction.to_numpy(float)
    baseline_metrics = metric_row(y, baseline)
    sensitivity_metrics = metric_row(y, sensitivity)
    row = {"outer_test_fold": outer, "endpoint": endpoint, "arm": arm, "paired_R1_records": int(len(pair)),
           "baseline_log_RMSE": baseline_metrics["log_RMSE"], "sensitivity_log_RMSE": sensitivity_metrics["log_RMSE"],
           "delta_log_RMSE": sensitivity_metrics["log_RMSE"] - baseline_metrics["log_RMSE"],
           "relative_log_RMSE_improvement": 1.0 - sensitivity_metrics["log_RMSE"] / baseline_metrics["log_RMSE"],
           **{f"baseline_{key}": value for key, value in baseline_metrics.items() if key != "R1_outer_records"},
           **{f"sensitivity_{key}": value for key, value in sensitivity_metrics.items() if key != "R1_outer_records"}}
    prediction_rows = []
    for record, base, sens in zip(pair.model_record_id, baseline, sensitivity):
        prediction_rows.extend([
            {"outer_test_fold": outer, "endpoint": endpoint, "comparison_arm": arm, "prediction_role": "baseline_reuse",
             "model_record_id": record, "predicted_transformed_value": float(base)},
            {"outer_test_fold": outer, "endpoint": endpoint, "comparison_arm": arm, "prediction_role": "sensitivity",
             "model_record_id": record, "predicted_transformed_value": float(sens)},
        ])
    return row, prediction_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--n1d", type=Path, default=ROOT / "results/analysis/mmpk_n1d_context_derivation_audit_v1")
    parser.add_argument("--n2c", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2c_paired_sensitivity_protocol_v1")
    parser.add_argument("--n2b", type=Path, default=ROOT / "results/benchmarks/mmpk_n2b_strict_r1_baseline_screen_v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke-no-label", action="store_true")
    parser.add_argument("--confirm-formal-paired-sensitivity", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.smoke_no_label and args.confirm_formal_paired_sensitivity:
        raise ValueError("Choose either --smoke-no-label or --confirm-formal-paired-sensitivity")
    if not args.smoke_no_label and not args.confirm_formal_paired_sensitivity:
        raise ValueError("Formal sensitivity reads strict outer labels; pass --confirm-formal-paired-sensitivity after smoke")
    if args.output is None:
        args.output = (ROOT / "results/analysis/mmpk_n2c_paired_sensitivity_fit_smoke_v1" if args.smoke_no_label
                       else ROOT / "results/benchmarks/mmpk_n2c_paired_sensitivity_v1")
    required = [args.n0 / "complete.json", args.n1b / "complete.json", args.n1d / "complete.json",
                args.n2c / "complete.json", args.n2b / "complete.json"]
    startup_self_check(required, output=args.output)
    verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    verify_stage(args.n1d, "mmpk_n1d_approved_context_and_derivation_audit")
    n2c = verify_stage(args.n2c, "mmpk_n2c_paired_sensitivity_protocol")
    n2b = verify_stage(args.n2b, "mmpk_n2b_strict_r1_conditional_baseline_screen")
    if n2b.get("external_labels_accessed") or n2b.get("R2_used_for_selection") or n2b.get("formulation_used_for_selection"):
        raise ValueError("N2b baseline is incompatible with the registered N2c sensitivity")
    selected = pd.read_csv(args.n2c / "outer_specific_R1_selected_configuration.csv")
    baseline_predictions = pd.read_csv(args.n2b / "outer_R1_predictions_internal.csv")
    if len(selected) != 20 or selected.duplicated(["outer_test_fold", "endpoint"]).any():
        raise ValueError("N2c selected configuration registry is incomplete")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing silent fallback for frozen XGBoost configurations")
        cuda_name = torch.cuda.get_device_name(0)
    else:
        cuda_name = "not_used"
    outer_values = [0] if args.smoke_no_label else list(range(5))
    endpoint_values = [PRIMARY[0]] if args.smoke_no_label else list(PRIMARY)
    progress = tqdm(total=len(outer_values) * len(endpoint_values) * 2,
                    desc="MMPK_N2C_SMOKE" if args.smoke_no_label else "MMPK_N2C_PROGRESS", unit="fit")
    metric_rows, prediction_rows, smoke_rows = [], [], []
    with stage_output(args.output) as out:
        for outer in outer_values:
            local = selected[selected.outer_test_fold.eq(outer)].set_index("endpoint")
            if set(local.index) != set(PRIMARY) or local.feature_view.nunique() != 1:
                raise ValueError("N2c selected configuration is incomplete or view-inconsistent within an outer fold")
            feature_view = local.feature_view.iloc[0]
            r1 = load_strict_train_fold(outer, "R1_direct", n1b=args.n1b, feature_view=feature_view)
            r12 = load_strict_train_fold(outer, "R1_plus_R2_augmented", n1b=args.n1b, feature_view=feature_view)
            if r1.model_record_ids != r12.model_record_ids or not np.array_equal(r1.features, r12.features):
                raise ValueError("R1 and R1+R2 loaders disagree on strict training inputs")
            outer_ids, _, outer_x, _ = load_strict_outer_inputs(outer, feature_view, n1b=args.n1b)
            train_context = load_approved_formulation_context(r1.model_record_ids, n1b=args.n1b, n1d=args.n1d)
            outer_context = load_approved_formulation_context(outer_ids, n1b=args.n1b, n1d=args.n1d)
            form_train_x, form_outer_x, train_keep, outer_keep, encoder = _make_formulation_features(
                r1.features, outer_x, train_context, outer_context)
            outer_index = {record: index for index, record in enumerate(outer_ids)}
            endpoint_index = {endpoint: index for index, (endpoint, _) in enumerate(ENDPOINTS)}
            delayed_pairs: list[tuple[str, str, pd.DataFrame]] = []
            for endpoint in endpoint_values:
                config = local.loc[endpoint]
                parameters = json.loads(config.parameters_json)
                baseline = _checked_baseline(baseline_predictions, outer, endpoint, config.candidate_id)
                base_positions = np.asarray([outer_index[record] for record in baseline.model_record_id], dtype=int)
                target_i = endpoint_index[endpoint]
                # R1+R2 arm: exact N2b configuration, only the training label mask changes.
                augmented_fit = np.flatnonzero(r12.target_mask[:, target_i])
                augmented_prediction = make_prediction(config.algorithm, parameters, 20260918, args.threads, args.device,
                                                        r12.features[augmented_fit], r12.targets[augmented_fit, target_i], outer_x[base_positions])
                r12_pair = baseline.copy()
                r12_pair["sensitivity_prediction"] = augmented_prediction
                progress.update(1)
                # Formulation arm: multiple-context records are excluded before fitting and before pairing.
                form_fit_mask = r1.target_mask[:, target_i] & train_keep
                if not form_fit_mask.any():
                    raise ValueError(f"{outer}/{endpoint}: formulation arm has no R1 train labels")
                form_base_mask = np.asarray([outer_keep[position] for position in base_positions], dtype=bool)
                form_baseline = baseline.loc[form_base_mask].copy()
                form_positions = base_positions[form_base_mask]
                compact_positions = np.flatnonzero(outer_keep)
                compact_lookup = {position: index for index, position in enumerate(compact_positions)}
                compact_selected = np.asarray([compact_lookup[position] for position in form_positions], dtype=int)
                form_prediction = make_prediction(config.algorithm, parameters, 20260918, args.threads, args.device,
                                                    form_train_x[form_fit_mask[train_keep]], r1.targets[train_keep, target_i][form_fit_mask[train_keep]],
                                                    form_outer_x[compact_selected])
                form_pair = form_baseline.copy()
                form_pair["sensitivity_prediction"] = form_prediction
                progress.update(1)
                smoke_rows.append({"outer_test_fold": outer, "endpoint": endpoint, "candidate_id": config.candidate_id,
                                   "R1_plus_R2_train_labels": int(len(augmented_fit)),
                                   "formulation_R1_train_labels": int(form_fit_mask.sum()),
                                   "R1_outer_input_predictions": int(len(r12_pair)),
                                   "formulation_paired_outer_input_predictions": int(len(form_pair)),
                                   "formulation_feature_dimension": int(form_train_x.shape[1]),
                                   "formulation_vocab_hash": encoder["formulation_vocab_hash"]})
                if not args.smoke_no_label:
                    delayed_pairs.extend([(endpoint, "R1_plus_R2_train", r12_pair), (endpoint, "formulation_matched", form_pair)])
            if not args.smoke_no_label:
                # This is deliberately after both arms have made all predictions for this outer fold.
                targets = load_outer_r1_targets_after_selection(outer, list(outer_ids), args.n1b)
                for endpoint, arm, pair in delayed_pairs:
                    target_mask, target_values = targets[endpoint]
                    y_by_id = {record: float(value) for record, flag, value in zip(outer_ids, target_mask, target_values) if flag}
                    row, rows = _score_pair(y_by_id, pair, arm, outer, endpoint)
                    metric_rows.append(row)
                    prediction_rows.extend(rows)
        progress.close()
        pd.DataFrame(smoke_rows).to_csv(out / "fit_and_input_audit.csv", index=False)
        if args.smoke_no_label:
            (out / "README.md").write_text(
                "# MMPK N2c paired sensitivity fit smoke\n\n"
                "Two pre-registered arms are fitted/predicted for one fixed outer-fold/endpoint only to test the engineering path. Held target columns are never opened; no predictions are persisted and no metric is calculated.\n", encoding="utf-8")
            finish_stage(out, "mmpk_n2c_paired_sensitivity_fit_smoke", inputs={
                str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
                str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
                str((args.n1d / "complete.json").resolve()): sha256(args.n1d / "complete.json"),
                str((args.n2c / "complete.json").resolve()): sha256(args.n2c / "complete.json"),
                str((args.n2b / "complete.json").resolve()): sha256(args.n2b / "complete.json")},
                no_persisted_predictions=True, no_performance_metrics=True, outer_target_columns_opened=False,
                external_labels_accessed=False, selected_configuration_reselected=False, partial=False)
        else:
            metrics = pd.DataFrame(metric_rows)
            if len(metrics) != 40 or metrics.duplicated(["outer_test_fold", "endpoint", "arm"]).any():
                raise ValueError("Formal N2c metric registry is incomplete")
            metrics.to_csv(out / "paired_outer_R1_metric_summary.csv", index=False)
            pd.DataFrame(prediction_rows).to_csv(out / "paired_predictions_internal.csv", index=False)
            (out / "README.md").write_text(
                "# MMPK N2c paired sensitivities\n\n"
                "Internal-only formal paired sensitivity results. Every fit uses an N2b outer-specific R1-selected configuration without reselection. R1+R2 and formulation are separate arms, each evaluated only against its same-record saved N2b baseline. Bootstrap inference is intentionally deferred to a read-only analysis stage.\n", encoding="utf-8")
            finish_stage(out, "mmpk_n2c_paired_sensitivity", inputs={
                str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
                str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
                str((args.n1d / "complete.json").resolve()): sha256(args.n1d / "complete.json"),
                str((args.n2c / "complete.json").resolve()): sha256(args.n2c / "complete.json"),
                str((args.n2b / "complete.json").resolve()): sha256(args.n2b / "complete.json")},
                formal_paired_sensitivity=True, paired_arms=list(ARMS), total_fits=40, device=args.device,
                cuda_device_name=cuda_name, outer_selection_forbidden=True, R2_used_for_selection=False,
                formulation_used_for_selection=False, external_labels_accessed=False, bootstrap_deferred_read_only=True,
                partial=False)
    print(f"MMPK N2c {'fit smoke' if args.smoke_no_label else 'formal paired sensitivity'}: {args.output}")


if __name__ == "__main__":
    run_cli(main)
