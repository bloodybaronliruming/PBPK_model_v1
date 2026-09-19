#!/usr/bin/env python3
"""Audit Gate 1B D-MPNN against frozen Stage-A train-CV leaders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import metric_row


SCREEN_BY_VIEW = {
    "rdkit2d": "rdkit",
    "ecfp4": "ecfp",
    "ecfp4_rdkit2d": "combined",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--multimodal-protocol", type=Path, default=ROOT / "data/public_development/gate1b_multimodal_protocol_v2")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--dmpnn", type=Path, default=ROOT / "results/benchmarks/gate1b_dmpnn_stage1_v1")
    parser.add_argument("--rdkit", type=Path, default=ROOT / "results/benchmarks/stl_stageA_first_batch_rdkit2d_v1")
    parser.add_argument("--ecfp", type=Path, default=ROOT / "results/benchmarks/stl_stageA_ecfp4_v1")
    parser.add_argument("--combined", type=Path, default=ROOT / "results/benchmarks/stl_stageA_ecfp4_rdkit2d_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/gate1b_dmpnn_stage1_audit_v1")
    args = parser.parse_args()
    screens = {"rdkit": args.rdkit, "ecfp": args.ecfp, "combined": args.combined}
    required = [args.protocol / "complete.json", args.multimodal_protocol / "complete.json",
                args.stageA_audit / "complete.json", args.dmpnn / "complete.json"]
    required += [p / "complete.json" for p in screens.values()]
    startup_self_check(required, output=args.output)
    verify_stage(args.protocol, "stl_benchmark_protocol")
    multi_meta = verify_stage(args.multimodal_protocol, "gate1b_multimodal_input_protocol")
    audit_meta = verify_stage(args.stageA_audit, "stl_stageA_three_view_audit")
    dmpnn_meta = verify_stage(args.dmpnn, "gate1b_dmpnn_screen")
    screen_meta = {name: verify_stage(path, "stl_benchmark_screen") for name, path in screens.items()}
    if multi_meta.get("fixed_validation_authorized_now") or audit_meta.get("validation_labels_read"):
        raise ValueError("Gate 1B D-MPNN audit requires validation to remain closed")
    if dmpnn_meta.get("partial") or dmpnn_meta.get("validation_labels_read") or dmpnn_meta.get("test_labels_read"):
        raise ValueError("D-MPNN input must be a complete train-CV-only screen")
    if any(m.get("evaluation_scope") != "train_cv" or m.get("test_labels_read") for m in screen_meta.values()):
        raise ValueError("Stage-A references must be complete train-CV-only screens")

    leaders = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv")
    dmetrics = pd.read_csv(args.dmpnn / "metrics.csv")
    dpred = pd.read_csv(args.dmpnn / "predictions.csv")
    baseline_predictions = {name: pd.read_csv(path / "predictions.csv") for name, path in screens.items()}
    if not np.isfinite(dpred.predicted_interface_target).all():
        raise ValueError("Non-finite D-MPNN predictions")

    candidate_rows, fold_rows, endpoint_rows = [], [], []
    for leader in leaders.itertuples(index=False):
        screen_key = SCREEN_BY_VIEW[leader.stageA_leading_feature_view]
        base = baseline_predictions[screen_key]
        base = base.loc[(base.task_id == leader.task_id) & (base.candidate_id == leader.stageA_leading_candidate)].copy()
        if base.row_id.nunique() == 0 or base.row_id.duplicated().any():
            raise ValueError(f"Invalid Stage-A leader predictions for {leader.task_id}")
        task_metrics = dmetrics.loc[dmetrics.task_id.eq(leader.task_id)].sort_values(["primary_metric", "candidate_id"])
        for candidate in task_metrics.itertuples(index=False):
            current = dpred.loc[(dpred.task_id == leader.task_id) & (dpred.candidate_id == candidate.candidate_id)].copy()
            if current.row_id.duplicated().any() or set(current.row_id) != set(base.row_id):
                raise ValueError(f"OOF membership mismatch: {leader.task_id} {candidate.candidate_id}")
            fold_nonworse = 0
            for fold in sorted(current.inner_fold_id.unique()):
                dpart = current.loc[current.inner_fold_id.eq(fold)]
                bpart = base.loc[base.inner_fold_id.eq(fold)]
                dm = metric_row(dpart, dpart.predicted_interface_target.to_numpy(), leader.task_id,
                                "dmpnn", "graph_rdkit2d", f"fold_{fold}")
                bm = metric_row(bpart, bpart.predicted_interface_target.to_numpy(), leader.task_id,
                                "stageA_leader", leader.stageA_leading_feature_view, f"fold_{fold}")
                nonworse = dm["primary_metric"] <= bm["primary_metric"]
                fold_nonworse += int(nonworse)
                fold_rows.append({
                    "task_id": leader.task_id, "endpoint": leader.endpoint,
                    "dmpnn_candidate_id": candidate.candidate_id, "inner_fold_id": int(fold),
                    "dmpnn_primary_metric": dm["primary_metric"],
                    "stageA_primary_metric": bm["primary_metric"],
                    "relative_change_percent": 100.0 * (dm["primary_metric"] / bm["primary_metric"] - 1.0),
                    "dmpnn_nonworse": nonworse,
                })
            relative = 100.0 * (candidate.primary_metric / leader.stageA_leading_primary_metric - 1.0)
            stable = float(current.predicted_interface_target.abs().max()) < 25.0
            strong = stable and relative <= -2.0 and fold_nonworse >= 3
            diversity = stable and relative <= 1.0
            candidate_rows.append({
                "task_id": leader.task_id, "endpoint": leader.endpoint,
                "dmpnn_candidate_id": candidate.candidate_id,
                "dmpnn_primary_metric": candidate.primary_metric,
                "stageA_feature_view": leader.stageA_leading_feature_view,
                "stageA_candidate_id": leader.stageA_leading_candidate,
                "stageA_primary_metric": leader.stageA_leading_primary_metric,
                "relative_change_percent": relative,
                "folds_nonworse_out_of_5": fold_nonworse,
                "max_abs_interface_prediction": float(current.predicted_interface_target.abs().max()),
                "numerical_stability": "pass" if stable else "fail_extreme_prediction",
                "strong_advance_gate": strong,
                "diversity_advance_gate": diversity,
                "validation_authorized": False,
            })
        best = min((x for x in candidate_rows if x["task_id"] == leader.task_id), key=lambda x: x["dmpnn_primary_metric"])
        endpoint_rows.append({
            "task_id": leader.task_id, "endpoint": leader.endpoint,
            "best_dmpnn_candidate_id": best["dmpnn_candidate_id"],
            "best_dmpnn_primary_metric": best["dmpnn_primary_metric"],
            "stageA_primary_metric": best["stageA_primary_metric"],
            "relative_change_percent": best["relative_change_percent"],
            "folds_nonworse_out_of_5": best["folds_nonworse_out_of_5"],
            "decision": "advance_to_bounded_fixed_validation" if best["strong_advance_gate"]
                else "stop_current_DMPNN_configs; retain_StageA_structure_leader",
            "validation_labels_read": False,
        })
    candidates = pd.DataFrame(candidate_rows).sort_values(["task_id", "dmpnn_primary_metric"])
    folds = pd.DataFrame(fold_rows).sort_values(["task_id", "dmpnn_candidate_id", "inner_fold_id"])
    endpoints = pd.DataFrame(endpoint_rows).sort_values("task_id")
    summary = {
        "tasks": len(endpoints),
        "candidate_task_evaluations": len(candidates),
        "numerically_stable_candidate_tasks": int(candidates.numerical_stability.eq("pass").sum()),
        "tasks_passing_strong_advance_gate": int(endpoints.decision.eq("advance_to_bounded_fixed_validation").sum()),
        "tasks_with_best_dmpnn_better_than_stageA": int(endpoints.relative_change_percent.lt(0).sum()),
        "validation_authorized": False,
        "validation_labels_read": False,
        "test_labels_read": False,
        "next_action": "retain_stageA_leaders_and_move_to_frozen_SMILES_provider_audit",
    }
    with stage_output(args.output) as out:
        candidates.to_csv(out / "candidate_comparison.csv", index=False)
        folds.to_csv(out / "fold_paired_comparison.csv", index=False)
        endpoints.to_csv(out / "endpoint_decisions.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Gate 1B D-MPNN Stage-1 audit\n\nTrain-CV-only paired audit against each endpoint's frozen Stage-A leader. "
            "The preregistered strong gate requires at least 2% lower primary error and non-worse performance in at least 3/5 folds. "
            "Fixed validation and test labels remain unread.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate1b_dmpnn_stage1_audit", inputs={
            "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
            "multimodal_protocol_complete_sha256": sha256(args.multimodal_protocol / "complete.json"),
            "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json"),
            "dmpnn_complete_sha256": sha256(args.dmpnn / "complete.json"),
            **{f"{name}_complete_sha256": sha256(path / "complete.json") for name, path in screens.items()},
        }, **summary, partial=False)
    print(f"Gate 1B D-MPNN Stage-1 audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
