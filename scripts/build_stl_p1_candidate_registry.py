#!/usr/bin/env python3
"""Freeze the finite P1 strong-STL candidate and parameter registry.

This stage translates the registered Stage-A shortlist into a small, fixed
nested-CV search space. It reads no validation or test labels and fits no model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from stl_benchmark_common import parameter_candidates


P1_FAMILIES = {
    "CL__human__systemic_iv": [
        ("extra_trees", "rdkit2d", "Stage-A leader"),
        ("random_forest", "ecfp4_rdkit2d", "near-tied distinct tree family"),
        ("xgboost", "ecfp4", "distinct boosting and feature-view reference"),
    ],
    "CLint__human__microsome": [
        ("extra_trees", "ecfp4_rdkit2d", "Stage-A leader"),
        ("lightgbm", "rdkit2d", "distinct boosting family"),
        ("extra_trees", "rdkit2d", "feature-view control"),
    ],
    "Papp__human__caco2_ab": [
        ("extra_trees", "ecfp4_rdkit2d", "Stage-A leader"),
        ("random_forest", "ecfp4_rdkit2d", "near-tied distinct tree family"),
    ],
    "Thalf__human__terminal_iv": [
        ("extra_trees", "ecfp4_rdkit2d", "Stage-A leader"),
        ("random_forest", "ecfp4_rdkit2d", "near-tied distinct tree family"),
    ],
    "VDss__human__steady_state_iv": [
        ("extra_trees", "rdkit2d", "Stage-A leader"),
        ("random_forest", "rdkit2d", "near-tied distinct tree family"),
        ("xgboost", "ecfp4_rdkit2d", "distinct boosting family"),
    ],
    "fu__human__plasma": [
        ("extra_trees", "ecfp4_rdkit2d", "Stage-A leader and physical-MAE reference"),
        ("lightgbm", "ecfp4_rdkit2d", "distinct boosting family"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--stagea-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/stl_p1_candidate_registry_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.protocol / "complete.json", args.protocol / "algorithm_registry.csv",
                args.protocol / "task_manifest.csv", args.stagea_audit / "complete.json",
                args.stagea_audit / "frozen_stageA_shortlist.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_meta = verify_stage(args.protocol, "stl_benchmark_protocol")
    audit_meta = verify_stage(args.stagea_audit, "stl_stageA_three_view_audit")
    if protocol_meta.get("test_labels_read") or audit_meta.get("test_labels_read"):
        raise ValueError("P1 registry requires test-closed frozen inputs")
    algorithms = pd.read_csv(args.protocol / "algorithm_registry.csv")
    tasks = pd.read_csv(args.protocol / "task_manifest.csv")
    if set(P1_FAMILIES) != set(tasks.task_id):
        raise ValueError("P1 family map must cover exactly the six authorized tasks")
    rows = []
    for task_id, endpoint in tasks.set_index("task_id").endpoint.items():
        for family_index, (algorithm, feature_set, basis) in enumerate(P1_FAMILIES[task_id], start=1):
            registry = algorithms.loc[algorithms.algorithm.eq(algorithm)]
            if len(registry) != 1 or not bool(registry.runtime_available.iloc[0]):
                raise ValueError(f"P1 candidate is unavailable: {task_id} {algorithm}")
            allowed = set(str(registry.allowed_feature_views.iloc[0]).split(","))
            if "all" not in allowed and feature_set not in allowed:
                raise ValueError(f"P1 feature view not registered: {task_id} {algorithm} {feature_set}")
            parameters = parameter_candidates(algorithm, maximum=4, seed=20260916, profile="stage_a")
            for parameter_index, parameter in enumerate(parameters):
                rows.append({
                    "task_id": task_id, "endpoint": endpoint, "algorithm": algorithm, "feature_set": feature_set,
                    "family_index": family_index, "parameter_index": parameter_index,
                    "candidate_uid": f"{task_id}::{algorithm}::{feature_set}::c{parameter_index:02d}",
                    "shortlist_basis": basis,
                    "parameters": json.dumps(parameter, sort_keys=True, default=list),
                    "primary_metric_name": "physical_MAE" if endpoint == "fu" else "transformed_RMSE",
                    "inner_selection": "mean_primary_metric_across_four_grouped_scaffold_folds",
                    "outer_confirmation": "three_fixed_estimator_seeds_after_nested_selection",
                    "advance_gate": "mean_improvement_at_least_2pct; majority_outer_folds_noninferior; no_material_registered_sensitivity_harm",
                })
    table = pd.DataFrame(rows).sort_values(["task_id", "family_index", "parameter_index"]).reset_index(drop=True)
    expected = sum(len(families) * 4 for families in P1_FAMILIES.values())
    if len(table) != expected or table.candidate_uid.duplicated().any():
        raise ValueError("Unexpected P1 candidate registry cardinality")
    if args.check_only:
        print(f"P1 candidate registry ready: tasks={table.task_id.nunique()} candidates={len(table)}")
        return
    with stage_output(args.output) as out:
        table.to_csv(out / "candidate_registry.csv", index=False)
        family_table = table.drop_duplicates(["task_id", "algorithm", "feature_set"]).copy()
        family_table.to_csv(out / "family_registry.csv", index=False)
        (out / "README.md").write_text(
            "# P1 finite strong-STL candidate registry\n\n"
            "This registry freezes 15 algorithm/feature families and four bounded Stage-A parameter settings per family. "
            "Nested scaffold CV selects parameters; fixed validation and test labels are not read by this stage.\n",
            encoding="utf-8",
        )
        finish_stage(out, "stl_p1_candidate_registry", inputs={
            "stl_protocol_complete_sha256": sha256(args.protocol / "complete.json"),
            "stagea_audit_complete_sha256": sha256(args.stagea_audit / "complete.json"),
        }, tasks=int(table.task_id.nunique()), families=int(len(family_table)), candidates=int(len(table)),
           test_labels_read=False, validation_target_file_opened=False, model_fitted=False,
           model_selection_authorized=False, partial=False)
    print(f"P1 candidate registry: {args.output}")


if __name__ == "__main__":
    run_cli(main)
