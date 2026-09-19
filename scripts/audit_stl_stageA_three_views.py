#!/usr/bin/env python3
"""Unify frozen-train Stage-A audits across RDKit2D, ECFP4, and both views.

This stage never reads fixed validation or test labels. It excludes numerical
instability and unattributed convergence warnings before freezing a provisional
classical/multiview shortlist for each endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_stl_stageA_results import audit
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


RELATIVE_PERFORMANCE_BAND = 0.05
VIEWS = ("rdkit2d", "ecfp4", "ecfp4_rdkit2d")


def freeze_shortlist(candidates: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    rows = []
    for task, group in candidates.groupby("task_id", sort=True):
        eligible = group.loc[group.stageA_eligibility.eq("eligible")].sort_values(
            ["primary_metric", "feature_set", "candidate_id"]
        ).copy()
        if eligible.empty:
            raise ValueError(f"No eligible candidate for {task}")
        best_metric = float(eligible.primary_metric.iloc[0])
        eligible["relative_gap_from_task_best"] = eligible.primary_metric / best_metric - 1.0
        band = eligible.loc[eligible.relative_gap_from_task_best.le(RELATIVE_PERFORMANCE_BAND)]
        selected: list[pd.Series] = []

        def add(row: pd.Series, reason: str) -> None:
            item = row.copy()
            item["shortlist_reason"] = reason
            selected.append(item)

        add(eligible.iloc[0], "best_eligible_train_CV")
        different_algorithm = band.loc[~band.algorithm.isin({selected[0].algorithm})]
        if not different_algorithm.empty:
            add(different_algorithm.iloc[0], "best_distinct_algorithm_within_5pct")
        used_keys = {(x.feature_set, x.candidate_id) for x in selected}
        used_views = {x.feature_set for x in selected}
        different_view = band.loc[~band.feature_set.isin(used_views)].copy()
        different_view = different_view.loc[
            ~different_view.apply(lambda x: (x.feature_set, x.candidate_id) in used_keys, axis=1)
        ]
        if len(selected) < n and not different_view.empty:
            add(different_view.iloc[0], "best_distinct_feature_view_within_5pct")
        while len(selected) < n:
            used_keys = {(x.feature_set, x.candidate_id) for x in selected}
            used_algorithms = {x.algorithm for x in selected}
            remaining = eligible.loc[
                ~eligible.apply(lambda x: (x.feature_set, x.candidate_id) in used_keys, axis=1)
            ]
            if remaining.empty:
                break
            distinct_algorithm_in_band = remaining.loc[
                remaining.relative_gap_from_task_best.le(RELATIVE_PERFORMANCE_BAND)
                & ~remaining.algorithm.isin(used_algorithms)
            ]
            if not distinct_algorithm_in_band.empty:
                add(distinct_algorithm_in_band.iloc[0], "next_distinct_algorithm_within_5pct")
            else:
                reason = "next_eligible_within_5pct" if remaining.iloc[0].relative_gap_from_task_best <= RELATIVE_PERFORMANCE_BAND else "fallback_eligible_above_5pct"
                add(remaining.iloc[0], reason)
        for position, row in enumerate(selected, start=1):
            row["shortlist_position"] = position
            row["shortlist_status"] = "frozen_stageA_train_CV_only; validation_and_test_unread"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["task_id", "shortlist_position"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--rdkit", type=Path, default=ROOT / "results/benchmarks/stl_stageA_first_batch_rdkit2d_v1")
    parser.add_argument("--ecfp", type=Path, default=ROOT / "results/benchmarks/stl_stageA_ecfp4_v1")
    parser.add_argument("--combined", type=Path, default=ROOT / "results/benchmarks/stl_stageA_ecfp4_rdkit2d_v1")
    parser.add_argument("--combined-log", type=Path, default=ROOT / "results/pipeline_logs/stl_long_tasks/run_stl_combined_stageA_v1_20260916_134803.log")
    parser.add_argument("--linear-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_linear_stabilization_audit_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    args = parser.parse_args()
    required = [args.protocol / "complete.json", args.rdkit / "complete.json", args.ecfp / "complete.json",
                args.combined / "complete.json", args.combined_log, args.linear_audit / "complete.json"]
    startup_self_check(required, output=args.output)
    verify_stage(args.protocol, "stl_benchmark_protocol")
    for screen in [args.rdkit, args.ecfp, args.combined]:
        meta = verify_stage(screen, "stl_benchmark_screen")
        if meta.get("evaluation_scope") != "train_cv" or meta.get("validation_confirmation") or meta.get("test_labels_read"):
            raise ValueError(f"Only blinded train-CV screens are allowed: {screen}")
    linear_meta = verify_stage(args.linear_audit, "stl_linear_stabilization_audit")
    if linear_meta.get("validation_labels_read") or linear_meta.get("test_labels_read"):
        raise ValueError("Linear reference audit is not blinded")

    frames = []
    for expected_view, screen in zip(VIEWS, [args.rdkit, args.ecfp, args.combined]):
        tables, _ = audit(args.protocol, screen)
        frame = tables["candidate_audit.csv"].copy()
        if set(frame.feature_set) != {expected_view}:
            raise ValueError(f"Unexpected feature view in {screen}")
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)
    candidates["convergence_status"] = "pass_no_logged_warning"
    warning_count = args.combined_log.read_text(encoding="utf-8", errors="replace").count("ConvergenceWarning")
    warning_mask = candidates.feature_set.eq("ecfp4_rdkit2d") & candidates.algorithm.eq("elasticnet")
    if warning_count:
        candidates.loc[warning_mask, "convergence_status"] = "exclude_unattributed_convergence_warning"
    candidates["stageA_eligibility"] = "eligible"
    candidates.loc[candidates.numerical_stability_status.ne("pass"), "stageA_eligibility"] = "exclude_numerical_instability"
    candidates.loc[candidates.convergence_status.ne("pass_no_logged_warning"), "stageA_eligibility"] = "exclude_convergence_not_auditable"
    candidates["candidate_uid"] = candidates.feature_set + "::" + candidates.candidate_id

    eligible = candidates.loc[candidates.stageA_eligibility.eq("eligible")]
    view_best = (eligible.sort_values(["task_id", "feature_set", "primary_metric", "candidate_id"])
                 .groupby(["task_id", "feature_set"], as_index=False).head(1).copy())
    rdkit_metric = view_best.loc[view_best.feature_set.eq("rdkit2d"), ["task_id", "primary_metric"]].rename(
        columns={"primary_metric": "rdkit2d_reference_metric"}
    )
    view_best = view_best.merge(rdkit_metric, on="task_id", validate="many_to_one")
    view_best["relative_change_vs_rdkit2d_percent"] = 100.0 * (
        view_best.primary_metric / view_best.rdkit2d_reference_metric - 1.0
    )
    shortlist = freeze_shortlist(candidates)
    linear_reference = pd.read_csv(args.linear_audit / "endpoint_comparison.csv")

    endpoint_rows = []
    for task, group in view_best.groupby("task_id", sort=True):
        leader = group.sort_values(["primary_metric", "feature_set"]).iloc[0]
        combined = group.loc[group.feature_set.eq("ecfp4_rdkit2d")].iloc[0]
        endpoint_rows.append({
            "task_id": task, "endpoint": leader.endpoint,
            "stageA_leading_feature_view": leader.feature_set,
            "stageA_leading_candidate": leader.candidate_id,
            "stageA_leading_primary_metric": leader.primary_metric,
            "combined_relative_change_vs_rdkit2d_percent": combined.relative_change_vs_rdkit2d_percent,
            "combined_interpretation": "material_train_CV_gain" if combined.relative_change_vs_rdkit2d_percent <= -2.0
                else "near_tie" if combined.relative_change_vs_rdkit2d_percent <= 2.0 else "worse_than_rdkit2d",
            "next_use": "classical_multiview_reference_for_Gate1B",
            "fixed_validation_status": "locked_until_multimodal_Gate1B_shortlist",
        })
    endpoints = pd.DataFrame(endpoint_rows)
    summary = {
        "feature_views": 3,
        "candidate_task_evaluations": len(candidates),
        "eligible_candidate_tasks": int(candidates.stageA_eligibility.eq("eligible").sum()),
        "numerically_unstable_candidate_tasks": int(candidates.numerical_stability_status.ne("pass").sum()),
        "combined_elasticnet_candidate_tasks_excluded_for_unattributed_warnings": int(warning_mask.sum()) if warning_count else 0,
        "combined_log_convergence_warnings": warning_count,
        "shortlist_entries": len(shortlist),
        "tasks_with_material_combined_gain": int(endpoints.combined_interpretation.eq("material_train_CV_gain").sum()),
        "validation_labels_read": False,
        "test_labels_read": False,
        "fixed_validation_authorized_next": False,
        "next_gate": "Gate1B_multimodal_incremental_ablation",
    }
    with stage_output(args.output) as out:
        candidates.sort_values(["task_id", "feature_set", "primary_metric", "candidate_id"]).to_csv(out / "all_candidate_audit.csv", index=False)
        view_best.sort_values(["task_id", "feature_set"]).to_csv(out / "feature_view_best.csv", index=False)
        shortlist.to_csv(out / "frozen_stageA_shortlist.csv", index=False)
        endpoints.to_csv(out / "endpoint_decisions.csv", index=False)
        linear_reference.to_csv(out / "stable_linear_references.csv", index=False)
        candidates.loc[candidates.stageA_eligibility.ne("eligible")].to_csv(out / "excluded_candidates.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Unified three-view STL Stage-A audit\n\nFrozen-train OOF only. Numerical instability and "
            "unattributed ElasticNet convergence warnings are excluded before a three-candidate-per-endpoint shortlist is frozen. "
            "Fixed validation and test labels remain unread.\n", encoding="utf-8")
        finish_stage(out, "stl_stageA_three_view_audit", inputs={
            "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
            "rdkit_complete_sha256": sha256(args.rdkit / "complete.json"),
            "ecfp_complete_sha256": sha256(args.ecfp / "complete.json"),
            "combined_complete_sha256": sha256(args.combined / "complete.json"),
            "combined_log_sha256": sha256(args.combined_log),
            "linear_audit_complete_sha256": sha256(args.linear_audit / "complete.json"),
        }, **summary, partial=False)
    print(f"Unified STL Stage-A audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
