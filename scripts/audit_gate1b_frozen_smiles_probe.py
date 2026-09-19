#!/usr/bin/env python3
"""Audit frozen-MoLFormer OOF probes against frozen Stage-A leaders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import metric_row


VIEW_PATHS = {
    "rdkit2d": "stl_stageA_first_batch_rdkit2d_v1",
    "ecfp4": "stl_stageA_ecfp4_v1",
    "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--probe", type=Path, default=ROOT / "results/benchmarks/gate1b_frozen_smiles_probe_v1")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/gate1b_frozen_smiles_probe_audit_v1")
    args = parser.parse_args()
    baseline_paths = {view: args.benchmarks / name for view, name in VIEW_PATHS.items()}
    required = [args.stageA_audit / "complete.json", args.stageA_audit / "endpoint_decisions.csv",
                args.probe / "complete.json", args.probe / "predictions.csv", args.probe / "candidate_ranking.csv"]
    required += [path / "complete.json" for path in baseline_paths.values()]
    startup_self_check(required, output=args.output)
    stage_a = verify_stage(args.stageA_audit, "stl_stageA_three_view_audit")
    probe = verify_stage(args.probe, "gate1b_frozen_smiles_screen")
    baseline_meta = {view: verify_stage(path, "stl_benchmark_screen") for view, path in baseline_paths.items()}
    if probe.get("partial") or probe.get("evaluation_scope") != "train_cv" or probe.get("test_labels_read"):
        raise ValueError("Frozen-SMILES input must be a complete train-CV-only screen")
    if stage_a.get("validation_labels_read") or any(meta.get("test_labels_read") for meta in baseline_meta.values()):
        raise ValueError("Stage-A references are not protected-label safe")

    leaders = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv")
    predictions = pd.read_csv(args.probe / "predictions.csv")
    ranking = pd.read_csv(args.probe / "candidate_ranking.csv")
    baselines = {view: pd.read_csv(path / "predictions.csv") for view, path in baseline_paths.items()}
    if not np.isfinite(predictions.predicted_interface_target).all():
        raise ValueError("Frozen-SMILES predictions contain non-finite values")

    candidate_rows, fold_rows, endpoint_rows, residual_rows = [], [], [], []
    for leader in leaders.itertuples(index=False):
        baseline = baselines[leader.stageA_leading_feature_view]
        baseline = baseline.loc[
            baseline.task_id.eq(leader.task_id) & baseline.candidate_id.eq(leader.stageA_leading_candidate)
        ].copy()
        if baseline.row_id.duplicated().any() or baseline.row_id.nunique() == 0:
            raise ValueError(f"Invalid Stage-A leader rows: {leader.task_id}")
        task_candidates = ranking.loc[ranking.task_id.eq(leader.task_id)]
        task_comparisons = []
        for candidate in task_candidates.itertuples(index=False):
            current = predictions.loc[
                predictions.task_id.eq(leader.task_id) & predictions.candidate_id.eq(candidate.candidate_id)
            ].copy()
            if current.row_id.duplicated().any() or set(current.row_id) != set(baseline.row_id):
                raise ValueError(f"OOF membership mismatch: {leader.task_id} {candidate.candidate_id}")
            nonworse, changes = 0, []
            for fold in sorted(current.inner_fold_id.unique()):
                cpart = current.loc[current.inner_fold_id.eq(fold)]
                bpart = baseline.loc[baseline.inner_fold_id.eq(fold)]
                cm = metric_row(cpart, cpart.predicted_interface_target.to_numpy(), leader.task_id,
                                candidate.algorithm, "frozen_molformer", f"fold_{fold}")
                bm = metric_row(bpart, bpart.predicted_interface_target.to_numpy(), leader.task_id,
                                "stageA_leader", leader.stageA_leading_feature_view, f"fold_{fold}")
                change = 100.0 * (cm["primary_metric"] / bm["primary_metric"] - 1.0)
                nonworse += int(cm["primary_metric"] <= bm["primary_metric"])
                changes.append(change)
                fold_rows.append({
                    "task_id": leader.task_id, "endpoint": leader.endpoint,
                    "candidate_id": candidate.candidate_id, "algorithm": candidate.algorithm,
                    "inner_fold_id": int(fold), "sequence_primary_metric": cm["primary_metric"],
                    "stageA_primary_metric": bm["primary_metric"], "relative_change_percent": change,
                    "sequence_nonworse": cm["primary_metric"] <= bm["primary_metric"],
                })
            relative = 100.0 * (candidate.primary_metric / leader.stageA_leading_primary_metric - 1.0)
            maximum = float(current.predicted_interface_target.abs().max())
            stable = maximum < 25.0
            row = {
                "task_id": leader.task_id, "endpoint": leader.endpoint,
                "candidate_id": candidate.candidate_id, "algorithm": candidate.algorithm,
                "sequence_primary_metric": candidate.primary_metric,
                "stageA_feature_view": leader.stageA_leading_feature_view,
                "stageA_candidate_id": leader.stageA_leading_candidate,
                "stageA_primary_metric": leader.stageA_leading_primary_metric,
                "relative_change_percent": relative, "folds_nonworse_out_of_5": nonworse,
                "best_fold_relative_change_percent": min(changes),
                "worst_fold_relative_change_percent": max(changes),
                "max_abs_interface_prediction": maximum,
                "numerical_stability": "pass" if stable else "fail_extreme_prediction",
                "strong_advance_gate": stable and relative <= -2.0 and nonworse >= 3,
                "diversity_advance_gate": stable and relative <= 1.0,
                "fixed_validation_authorized": False,
            }
            candidate_rows.append(row)
            task_comparisons.append(row)
        best = min(task_comparisons, key=lambda item: (item["sequence_primary_metric"], item["candidate_id"]))
        endpoint_rows.append({
            "task_id": leader.task_id, "endpoint": leader.endpoint,
            "best_sequence_candidate_id": best["candidate_id"],
            "best_sequence_primary_metric": best["sequence_primary_metric"],
            "stageA_primary_metric": best["stageA_primary_metric"],
            "relative_change_percent": best["relative_change_percent"],
            "folds_nonworse_out_of_5": best["folds_nonworse_out_of_5"],
            "decision": "advance_to_fixed_validation" if best["strong_advance_gate"]
                        else "retain_as_negative_ablation_no_fixed_validation",
        })

        current = predictions.loc[
            predictions.task_id.eq(leader.task_id) & predictions.candidate_id.eq(best["candidate_id"])
        ].copy()
        joined = current.merge(
            baseline[["row_id", "predicted_interface_target"]].rename(
                columns={"predicted_interface_target": "stageA_prediction"}
            ), on="row_id", validate="one_to_one",
        )
        parent_errors = pd.DataFrame({
            "molecule_id": joined.molecule_id,
            "sequence_error": joined.predicted_interface_target - joined.interface_target,
            "stageA_error": joined.stageA_prediction - joined.interface_target,
        }).groupby("molecule_id", as_index=False).mean(numeric_only=True)
        correlation = float(parent_errors.sequence_error.corr(parent_errors.stageA_error))
        equal_blend = 0.5 * (joined.predicted_interface_target.to_numpy() + joined.stageA_prediction.to_numpy())
        blend_metric = metric_row(joined, equal_blend, leader.task_id, "equal_blend",
                                  "posthoc_diagnostic", "train_cv_diagnostic")["primary_metric"]
        residual_rows.append({
            "task_id": leader.task_id, "endpoint": leader.endpoint,
            "best_sequence_candidate_id": best["candidate_id"],
            "parent_error_correlation": correlation,
            "stageA_primary_metric": best["stageA_primary_metric"],
            "posthoc_equal_blend_primary_metric": blend_metric,
            "posthoc_equal_blend_relative_change_percent": 100.0 * (blend_metric / best["stageA_primary_metric"] - 1.0),
            "selection_authorized": False,
        })

    candidates = pd.DataFrame(candidate_rows).sort_values(["task_id", "sequence_primary_metric"])
    endpoints = pd.DataFrame(endpoint_rows).sort_values("task_id")
    summary = {
        "tasks": len(endpoints), "candidate_task_evaluations": len(candidates),
        "strong_advancing_candidates": int(candidates.strong_advance_gate.sum()),
        "diversity_advancing_candidates": int(candidates.diversity_advance_gate.sum()),
        "tasks_advancing_to_fixed_validation": int(endpoints.decision.eq("advance_to_fixed_validation").sum()),
        "validation_rows_evaluated": False,
        "validation_targets_used_for_fit_scoring_or_selection": False,
        "validation_target_values_materialized_by_original_runner_before_train_filter": True,
        "original_validation_labels_read_false_claim_corrected": True,
        "test_labels_read": False,
        "next_action": "use_train_only_protocol_and_run_fold_local_PCA_Ridge_MLP_adaptation",
    }
    with stage_output(args.output) as out:
        candidates.to_csv(out / "candidate_comparison.csv", index=False)
        pd.DataFrame(fold_rows).sort_values(["task_id", "candidate_id", "inner_fold_id"]).to_csv(
            out / "fold_paired_comparison.csv", index=False
        )
        endpoints.to_csv(out / "endpoint_decisions.csv", index=False)
        pd.DataFrame(residual_rows).sort_values("task_id").to_csv(out / "residual_complementarity_diagnostic.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Gate 1B frozen-SMILES probe audit v1\n\nPaired train-OOF comparison against frozen Stage-A leaders. "
            "The original runner evaluated no validation rows and used no validation target for fitting or selection, but loaded the target-bearing combined CSV before filtering train; this audit corrects the earlier overly broad no-read claim. "
            "Equal blending is a post-hoc complementarity diagnostic and cannot select a candidate.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate1b_frozen_smiles_probe_audit", inputs={
            "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json"),
            "probe_complete_sha256": sha256(args.probe / "complete.json"),
            **{f"{view}_complete_sha256": sha256(path / "complete.json") for view, path in baseline_paths.items()},
        }, partial=False, **summary)
    print(f"Gate 1B frozen-SMILES probe audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
