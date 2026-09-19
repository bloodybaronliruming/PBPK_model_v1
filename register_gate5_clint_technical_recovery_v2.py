#!/usr/bin/env python3
"""Register the final Gate 5 completion attempt after two pre-metric technical aborts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gate5_clint_common import ROOT, required_model_files, validate_frozen_bundle, verify_recomputed_feature_contract
from pipeline_common import dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2")
    parser.add_argument("--stl-protocol", type=Path,
                        default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--recovery-v1", type=Path,
                        default=ROOT / "data/public_development/gate5_clint_technical_recovery_v1")
    parser.add_argument("--original-output", type=Path,
                        default=ROOT / "results/final/CLint_gate5_stageA_et_c01_test_v1")
    parser.add_argument("--recovery-v1-output", type=Path,
                        default=ROOT / "results/final/CLint_gate5_stageA_et_c01_test_recovery_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/public_development/gate5_clint_technical_recovery_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    evaluator = ROOT / "scripts/evaluate_gate5_clint_technical_recovery_v2.py"
    common = ROOT / "scripts/gate5_clint_common.py"
    startup_self_check(required_model_files(args.frozen) + [
        args.stl_protocol / "complete.json", args.splits / "complete.json",
        args.recovery_v1 / "complete.json", evaluator, common,
    ], output=None if args.check_only else args.output)
    if args.original_output.exists() or args.recovery_v1_output.exists():
        raise ValueError("A Gate 5 test output exists; the final technical amendment is forbidden")
    registry, _ = validate_frozen_bundle(args.frozen)
    v1_meta = verify_stage(args.recovery_v1, "gate5_clint_technical_recovery_protocol")
    v1_manifest = json.loads((args.recovery_v1 / "recovery_manifest.json").read_text(encoding="utf-8"))
    if v1_meta.get("test_evaluated") or v1_manifest.get("prediction_or_metric_output_published"):
        raise ValueError("Recovery v1 produced a scored result; v2 is forbidden")
    feature_audit = verify_recomputed_feature_contract(args.frozen, args.stl_protocol)
    split_columns = set(pd.read_csv(args.splits / "split_manifest.csv", nrows=0).columns)
    required_columns = {"molecule_id", "parent_id", "smiles", "split", "eligible", "standardization_status"}
    if not required_columns.issubset(split_columns):
        raise ValueError("Frozen split manifest lacks the source-ID to parent-ID contract required by v2")
    if args.check_only:
        print("Gate 5 recovery v2 preflight valid: frozen model unchanged, both failed outputs absent, parent-ID contract available, test labels not loaded.")
        return
    manifest = {
        "protocol_version": 2,
        "status": "final_completion_attempt_authorized_before_any_prediction_or_metric_output",
        "supersedes": str(args.recovery_v1.resolve()),
        "why_v1_prohibition_is_amended": (
            "Recovery v1 produced no feature-complete test matrix, prediction, metric, bootstrap, or output; "
            "its source-molecule-ID identity assertion was demonstrated invalid against the already-frozen split parent_id contract."
        ),
        "attempt_history": [
            {
                "command": "evaluate_gate5_clint_frozen.py --confirm-one-time-test",
                "failure": "test parents absent from train/validation feature cache",
                "phase": "after test table load; before feature construction, prediction, metrics, bootstrap, or output",
            },
            {
                "command": "evaluate_gate5_clint_technical_recovery.py --confirm-technical-recovery-test",
                "failure": "source molecule_id was incorrectly required to equal canonical parent identity",
                "phase": "after test table load and feature recomputation; before prediction, metrics, bootstrap, or output",
            },
        ],
        "programmatic_test_table_loaded": True,
        "test_values_logged_or_persisted": False,
        "prediction_or_metric_output_published": False,
        "failed_output_paths": [str(args.original_output.resolve()), str(args.recovery_v1_output.resolve())],
        "diagnostic_metadata_findings": {
            "test_molecules": 28,
            "source_molecule_id_not_equal_to_canonical_parent_id": 15,
            "canonical_identity_equal_to_frozen_split_parent_id": 28,
            "standardization_failures": 0,
        },
        "identity_contract": {
            "molecule_id": "source-level identity retained for observation grouping and traceability",
            "parent_id": "canonical-parent identity used only to validate reconstructed molecular features",
        },
        "immutable_candidate_id": registry["candidate_id"],
        "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
        "model_sha256": sha256(args.frozen / "model.joblib"),
        "candidate_registry_sha256": sha256(args.frozen / "candidate_registry.json"),
        "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
        "splits_complete_sha256": sha256(args.splits / "complete.json"),
        "recovery_v1_complete_sha256": sha256(args.recovery_v1 / "complete.json"),
        "recovery_evaluator_sha256": sha256(evaluator),
        "recovery_common_sha256": sha256(common),
        "allowed_change": "validate reconstructed test identity against frozen split parent_id instead of source molecule_id",
        "prohibitions": [
            "model_refit", "candidate_change", "feature_family_change", "hyperparameter_change",
            "calibration", "threshold_change", "third_completion_attempt",
        ],
        "required_confirmation_flag": "--confirm-final-technical-recovery-test",
        "required_new_output": str((ROOT / "results/final/CLint_gate5_stageA_et_c01_test_recovery_v2").resolve()),
    }
    with stage_output(args.output) as out:
        dump_json(out / "recovery_manifest.json", manifest)
        dump_json(out / "train_feature_equivalence_audit.json", feature_audit)
        (out / "README.md").write_text(
            "# Gate 5 CLint final technical recovery amendment\n\n"
            "Two confirmed commands reached the test table but both stopped before prediction, metrics, bootstrap, or output. "
            "The second failure proved that source molecule_id and canonical parent_id were incorrectly conflated. "
            "This final amendment preserves the frozen model and every scoring rule, validates features against the frozen split parent_id, and forbids any further completion attempt.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate5_clint_technical_recovery_protocol_v2", inputs={
            "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "splits_complete_sha256": sha256(args.splits / "complete.json"),
            "recovery_v1_complete_sha256": sha256(args.recovery_v1 / "complete.json"),
        }, candidate_id=registry["candidate_id"], programmatic_test_table_loaded=True,
        prediction_or_metric_output_published=False, model_refit=False, test_evaluated=False,
        final_completion_attempt=True, partial=False)
    print(f"Gate 5 CLint final technical recovery protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
