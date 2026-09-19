#!/usr/bin/env python3
"""Freeze the resumable 30-cell Gate-3 B6 formal train-CV run manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASKS = [
    "CL__human__systemic_iv",
    "CLint__human__microsome",
    "Papp__human__caco2_ab",
    "Thalf__human__terminal_iv",
    "VDss__human__steady_state_iv",
    "fu__human__plasma",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_protocol_v2")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_cohort_v1")
    parser.add_argument("--smoke", type=Path, default=ROOT / "results/benchmarks/gate3_b6_physchem_smoke_v2")
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--p1-freeze", type=Path, default=ROOT / "data/public_development/stl_p1_decision_freeze_v1")
    parser.add_argument("--p1-confirmation", type=Path, default=ROOT / "results/benchmarks/stl_p1_nested_confirmation_v2")
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts/run_gate3_b6_physchem_outer_fold.py")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/gate3_b6_formal_run_v4")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.protocol / "complete.json", args.protocol / "endpoint_fusion_registry.csv",
        args.protocol / "auxiliary_producer_registry.csv", args.protocol / "endpoint_advance_gates.csv",
        args.cohort / "complete.json", args.cohort / "outer_fold_purge_summary.csv",
        args.smoke / "complete.json", args.smoke / "nested_ancestry_audit.csv",
        args.smoke / "model_reload_audit.csv", args.smoke / "outer_eval_predictions_blinded.csv",
        args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
        args.p1_freeze / "complete.json", args.p1_freeze / "p1_decision_registry.csv",
        args.p1_confirmation / "complete.json", args.runner,
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_meta = verify_stage(args.protocol, "gate3_b6_physchem_protocol")
    cohort_meta = verify_stage(args.cohort, "gate3_b6_physchem_cohort")
    smoke_meta = verify_stage(args.smoke, "gate3_b6_physchem_smoke")
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    p1_confirmation_meta = verify_stage(args.p1_confirmation, "stl_p1_nested_confirmation")
    metas = [protocol_meta, cohort_meta, smoke_meta, train_meta, p1_meta, p1_confirmation_meta]
    if any(meta.get("test_labels_read", False) or meta.get("validation_target_file_opened", False) for meta in metas):
        raise ValueError("Formal run freeze accepts only fixed-validation/test-closed evidence")
    required_smoke = {
        "engineering_smoke_passed": True,
        "limited_smoke": True,
        "outer_evaluation_metric_computed": False,
        "outer_evaluation_target_values_used": 0,
        "architecture_selection_authorized": False,
        "formal_model_training_authorized": False,
    }
    for key, expected in required_smoke.items():
        if smoke_meta.get(key) != expected:
            raise ValueError(f"Smoke contract mismatch: {key}={smoke_meta.get(key)!r}")
    if smoke_meta.get("maximum_parent_overlap") != 0 or smoke_meta.get("maximum_scaffold_overlap") != 0:
        raise ValueError("Smoke ancestry has nonzero overlap")
    if float(smoke_meta.get("maximum_reload_difference", float("inf"))) > 1e-12:
        raise ValueError("Smoke reload tolerance failed")
    stagea_ensemble_seeds = p1_confirmation_meta.get("confirmation_seeds")
    if stagea_ensemble_seeds != [20260920, 20260921, 20260922]:
        raise ValueError(f"Unexpected corrected Stage-A ensemble seeds: {stagea_ensemble_seeds}")
    ancestry = pd.read_csv(args.smoke / "nested_ancestry_audit.csv")
    expected_components = {
        "stageA_crossfit": 4,
        "stageA_outer_eval": 1,
        "auxiliary_crossfit": 10,
        "auxiliary_outer_eval": 2,
        "meta_model_fit_and_evaluation": 2,
    }
    if ancestry.groupby("component").size().to_dict() != expected_components:
        raise ValueError("Smoke v2 does not contain the complete five-level ancestry contract")
    if ancestry.model_sha256.isna().any():
        raise ValueError("Smoke ancestry lacks model hashes")
    for row in ancestry.itertuples(index=False):
        if sha256(args.smoke / row.model_file) != row.model_sha256:
            raise ValueError(f"Smoke ancestry model hash mismatch: {row.model_file}")
    blinded = pd.read_csv(args.smoke / "outer_eval_predictions_blinded.csv", nrows=1)
    if any("target" in column.lower() or "observed" in column.lower() for column in blinded.columns):
        raise ValueError("Smoke outer-evaluation output is not label-blinded")

    records = pd.read_csv(
        args.train_protocol / "benchmark_train_records.csv",
        usecols=["task_id", "parent_id", "inner_fold_id"],
    ).drop_duplicates()
    if set(records.task_id) != set(TASKS) or set(records.inner_fold_id) != set(range(5)):
        raise ValueError("Formal downstream task/fold membership differs from the frozen scope")
    downstream_counts = records.groupby(["task_id", "inner_fold_id"]).parent_id.nunique().to_dict()
    purge = pd.read_csv(args.cohort / "outer_fold_purge_summary.csv")
    purge_lookup = purge.set_index(["downstream_scope", "outer_fold", "auxiliary_task"])
    endpoint = pd.read_csv(args.protocol / "endpoint_fusion_registry.csv").set_index("task_id")
    if set(endpoint.index) != set(TASKS):
        raise ValueError("Endpoint fusion registry does not cover the formal task set")

    cells = []
    for task in TASKS:
        slug = task.replace("__", "_").lower()
        for fold in range(5):
            aux_logp = purge_lookup.loc[(task, fold, "experimental_logP")]
            aux_logd = purge_lookup.loc[(task, fold, "experimental_logD_pH7_4")]
            output = f"results/benchmarks/gate3_b6_physchem_formal_cells_v3/{slug}_fold{fold}"
            command = (
                "conda run --no-capture-output -n oneadmet python "
                f"scripts/run_gate3_b6_physchem_outer_fold.py --task {task} --outer-fold {fold} --output {output}"
            )
            cells.append({
                "cell_id": f"{slug}_fold{fold}", "task_id": task,
                "endpoint": endpoint.loc[task, "endpoint"], "outer_fold": fold,
                "outer_evaluation_parents": int(downstream_counts[(task, fold)]),
                "auxiliary_logP_parents": int(aux_logp.parents_after_fold_purge),
                "auxiliary_logD_parents": int(aux_logd.parents_after_fold_purge),
                "stageA_fits": 15, "auxiliary_candidate_cv_fits": 20,
                "auxiliary_selected_crossfit_fits": 10, "auxiliary_final_fits": 2,
                "meta_inner_selection_fits": 24, "meta_final_fits": 2,
                "total_model_fits": 73, "stageA_trees": 1800, "auxiliary_trees": 9600,
                "output_directory": output, "command": command,
            })
    registry = pd.DataFrame(cells)
    if len(registry) != 30 or registry.cell_id.duplicated().any():
        raise ValueError("Formal cell registry must contain exactly 30 unique cells")

    resources = pd.DataFrame([
        {
            "scope": "one_endpoint_outer_fold_cell", "cells": 1,
            "model_fits": 73, "tree_estimators": 11400,
            "expected_wall_time": "unknown_until_first_full_cell_pilot; likely tens_of_minutes",
            "preferred_device": "CPU_multithreaded_sklearn",
            "recoverability": "independent_atomic_cell_output",
        },
        {
            "scope": "complete_formal_train_cv", "cells": 30,
            "model_fits": int(registry.total_model_fits.sum()),
            "tree_estimators": int((registry.stageA_trees + registry.auxiliary_trees).sum()),
            "expected_wall_time": "hours; revise_after_first_full_cell_pilot",
            "preferred_device": "CPU_multithreaded_sklearn; GPU_not_used_by_ExtraTrees_or_Ridge",
            "recoverability": "resume_by_skipping_verified_complete_cells",
        },
    ])
    decision = {
        "decision_id": "gate3_b6_formal_traincv_authorization_v4",
        "evidence": "complete bounded smoke v2 plus versioned cohort/protocol",
        "formal_train_cv_authorized": True,
        "first_full_cell_pilot_required": True,
        "full_30_cell_background_execution_authorized_after_pilot_resource_check": True,
        "architecture_selection_before_all_30_cells": False,
        "fixed_validation_authorized": False,
        "test_authorized": False,
        "human_F_included": False,
        "source_test_labels_allowed": False,
        "cell_outputs_are_independent_and_non_overwriting": True,
        "stageA_ensemble_seeds": stagea_ensemble_seeds,
        "outer_fold_runner_sha256": sha256(args.runner),
        "required_imputation_audit_rows_per_cell": 47,
        "required_candidate_cv_imputation_audit_rows_per_cell": 20,
        "progress_reporting": {
            "log_friendly": True,
            "dynamic_eta": True,
            "total_fits_per_cell": 73,
            "stage_fit_budgets": [15, 16, 16, 13, 13],
        },
    }
    json.dumps(decision, allow_nan=False)
    if args.check_only:
        print(
            "Gate 3 B6 formal run freeze ready: "
            f"cells={len(registry)} fits={registry.total_model_fits.sum()} "
            f"tree_estimators={(registry.stageA_trees + registry.auxiliary_trees).sum()}"
        )
        return
    inputs = {str(path.resolve()): sha256(path) for path in required}
    with stage_output(args.output) as out:
        registry.to_csv(out / "formal_cell_registry.csv", index=False)
        resources.to_csv(out / "resource_budget.csv", index=False)
        (out / "formal_authorization.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
        (out / "README.md").write_text(
            "# Gate 3 B6 formal train-CV run freeze v4\n\n"
            "Thirty independent endpoint-by-outer-fold cells are authorized after the complete smoke-v2 contract. "
            "The runner emits durable per-stage, per-cell and whole-batch progress with dynamic ETA. The model, "
            "split, seed and leakage-control contracts are unchanged from v3. No route may be selected until all 30 cells and the "
            "registered aggregate analysis finish. Fixed validation, test, source-test labels and human F remain closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "gate3_b6_formal_run_freeze", inputs=inputs, partial=False,
            cells=len(registry), tasks=len(TASKS), outer_folds=5,
            total_model_fits=int(registry.total_model_fits.sum()),
            total_tree_estimators=int((registry.stageA_trees + registry.auxiliary_trees).sum()),
            smoke_ancestry_rows=len(ancestry), smoke_model_hashes_verified=True,
            stageA_ensemble_seeds=stagea_ensemble_seeds,
            outer_fold_runner_sha256=sha256(args.runner),
            progress_reporting=True, total_fits_per_cell=73,
            formal_train_cv_authorized=True, first_full_cell_pilot_required=True,
            architecture_selection_authorized=False, fixed_validation_authorized=False,
            test_authorized=False, validation_target_file_opened=False, test_labels_read=False,
        )
    print(f"Gate 3 B6 formal run freeze: {args.output}")


if __name__ == "__main__":
    run_cli(main)
