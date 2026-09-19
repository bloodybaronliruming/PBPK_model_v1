#!/usr/bin/env python3
"""Register a single technical recovery after a pre-metric Gate 5 test failure."""
from __future__ import annotations

import argparse
from pathlib import Path

from gate5_clint_common import ROOT, required_model_files, validate_frozen_bundle, verify_recomputed_feature_contract
from pipeline_common import dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--failed-output", type=Path, default=ROOT / "results/final/CLint_gate5_stageA_et_c01_test_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/gate5_clint_technical_recovery_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check(required_model_files(args.frozen) + [args.stl_protocol / "complete.json"],
                       output=None if args.check_only else args.output)
    if args.failed_output.exists():
        raise ValueError("Technical recovery is forbidden because the failed evaluation output exists")
    registry, _ = validate_frozen_bundle(args.frozen)
    feature_audit = verify_recomputed_feature_contract(args.frozen, args.stl_protocol)
    if args.check_only:
        print("Gate 5 technical recovery preflight valid: exact frozen candidate, no published failed output, train-only feature equivalence.")
        return
    with stage_output(args.output) as out:
        dump_json(out / "recovery_manifest.json", {
            "protocol_version": 1,
            "status": "single_recovery_authorized_before_any_metric_or_prediction_output",
            "trigger": "feature_cache_missing_test_parent",
            "failed_command": "evaluate_gate5_clint_frozen.py --confirm-one-time-test",
            "failure_phase": "after_programmatic_test_table_load_before_feature_construction_prediction_metric_bootstrap_or_output",
            "programmatic_test_table_loaded": True,
            "test_values_logged_or_persisted": False,
            "prediction_or_metric_output_published": False,
            "failed_output_path": str(args.failed_output.resolve()),
            "immutable_candidate_id": registry["candidate_id"],
            "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
            "model_sha256": sha256(args.frozen / "model.joblib"),
            "candidate_registry_sha256": sha256(args.frozen / "candidate_registry.json"),
            "recovery_evaluator_sha256": sha256(ROOT / "scripts/evaluate_gate5_clint_technical_recovery.py"),
            "recovery_common_sha256": sha256(ROOT / "scripts/gate5_clint_common.py"),
            "allowed_change": "replace unavailable cached test feature lookup with registered on-the-fly canonical-parent featurization",
            "prohibitions": ["model_refit", "candidate_change", "feature_family_change", "hyperparameter_change", "calibration", "threshold_change", "second_recovery"],
            "required_confirmation_flag": "--confirm-technical-recovery-test",
            "required_new_output": str((ROOT / "results/final/CLint_gate5_stageA_et_c01_test_recovery_v1").resolve()),
        })
        dump_json(out / "train_feature_equivalence_audit.json", feature_audit)
        (out / "README.md").write_text(
            "# Gate 5 CLint technical recovery protocol\n\n"
            "The first confirmed evaluator loaded the test table but failed before feature construction, prediction, metrics, bootstrap, or output because test parents were absent from the train/validation feature cache. "
            "This protocol records the access conservatively and permits exactly one completion retry using the unchanged frozen model and on-the-fly canonical-parent features proven identical to the train cache.\n",
            encoding="utf-8")
        finish_stage(out, "gate5_clint_technical_recovery_protocol", inputs={
            "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
        }, candidate_id=registry["candidate_id"], programmatic_test_table_loaded=True,
        prediction_or_metric_output_published=False, model_refit=False, test_evaluated=False, partial=False)
    print(f"Gate 5 CLint technical recovery protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
