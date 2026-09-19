"""Verify a separately executed Gate 7 clean-environment technical replay.

This verifier is deliberately limited to the frozen label-free R4 replay
input/output structure, provenance and numerical tolerance checks.  CSV byte
hashes are retained for audit, but are not a passing condition: independent
processes may serialize otherwise equivalent IEEE-754 values differently in
their last decimal places.  It never accesses labels, test metrics, models, or
candidate-selection logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


MANIFEST_STAGE = "gate7_reproducibility_delivery_manifest"
SMOKE_STAGE = "gate7_dual_lane_batch_inference"
STAGE = "gate7_clean_environment_reload_verification"


def compare_technical_predictions(reference_path: Path, candidate_path: Path, tolerance: float) -> dict:
    """Compare label-free replay outputs without interpreting them as scores."""
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    required = {"prediction_status", "predicted_physical"}
    if not required.issubset(reference.columns) or not required.issubset(candidate.columns):
        raise ValueError("Technical prediction files lack required output columns")
    same_schema = reference.columns.tolist() == candidate.columns.tolist()
    same_row_count = len(reference) == len(candidate)
    if not same_schema or not same_row_count:
        return {
            "prediction_schema_matches": same_schema,
            "prediction_row_count_matches": same_row_count,
            "non_prediction_fields_match": False,
            "numeric_output_count": None,
            "max_abs_cross_process_prediction_difference": None,
            "numeric_predictions_within_tolerance": False,
        }
    non_prediction = [column for column in reference.columns if column != "predicted_physical"]
    non_prediction_fields_match = reference[non_prediction].equals(candidate[non_prediction])
    numeric_reference = pd.to_numeric(reference.loc[reference["predicted_physical"].notna(), "predicted_physical"], errors="raise").to_numpy(dtype=float)
    numeric_candidate = pd.to_numeric(candidate.loc[candidate["predicted_physical"].notna(), "predicted_physical"], errors="raise").to_numpy(dtype=float)
    if len(numeric_reference) != len(numeric_candidate) or not np.isfinite(numeric_candidate).all():
        max_difference = None
        within_tolerance = False
    else:
        max_difference = float(np.max(np.abs(numeric_reference - numeric_candidate))) if len(numeric_reference) else 0.0
        within_tolerance = bool(max_difference <= tolerance)
    return {
        "prediction_schema_matches": same_schema,
        "prediction_row_count_matches": same_row_count,
        "non_prediction_fields_match": non_prediction_fields_match,
        "numeric_output_count": int(len(numeric_reference)),
        "max_abs_cross_process_prediction_difference": max_difference,
        "numeric_predictions_within_tolerance": within_tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-clean-reload-verification", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest_dir = args.manifest.resolve()
    candidate = args.candidate_smoke.resolve()
    manifest_meta = verify_stage(manifest_dir, MANIFEST_STAGE)
    candidate_meta = verify_stage(candidate, SMOKE_STAGE)
    if candidate == Path(manifest_dir.parent.parent.parent / "results/analysis/gate7_release_cli_smoke_v1").resolve():
        raise ValueError("The clean-reload candidate must be a new output, not the original R4 smoke")
    payload = json.loads((manifest_dir / "delivery_manifest.json").read_text(encoding="utf-8"))
    reference = payload["reference_replay"]
    candidate_input = candidate / "submitted_label_free_input.csv"
    candidate_predictions = candidate / "dual_lane_predictions.csv"
    tolerance = float(reference["cross_process_prediction_tolerance"])
    prediction_comparison = compare_technical_predictions(
        Path(reference["reference_r4_output"]), candidate_predictions, tolerance
    )
    checks = {
        "candidate_is_new_output": True,
        "candidate_nonpartial": candidate_meta.get("partial") is False,
        "candidate_no_labels_accessed": bool(candidate_meta.get("no_labels_accessed")),
        "candidate_technical_smoke": bool(candidate_meta.get("technical_smoke")),
        "candidate_input_rows": candidate_meta.get("input_rows") == reference["expected_input_rows"],
        "candidate_output_rows": candidate_meta.get("output_rows") == reference["expected_output_rows"],
        "candidate_input_hash_matches": sha256(candidate_input) == reference["input_sha256"],
        **prediction_comparison,
        "candidate_repeat_difference_within_tolerance": float(candidate_meta.get("max_abs_repeat_prediction_difference")) <= float(reference["in_process_repeat_tolerance"]),
        "candidate_contract_hash_matches": payload["predecessors"]["r1"]["complete_sha256"] in candidate_meta.get("inputs", {}).values(),
        "candidate_card_hash_matches": payload["predecessors"]["r3"]["complete_sha256"] in candidate_meta.get("inputs", {}).values(),
        "manifest_no_labels_accessed": bool(manifest_meta.get("no_labels_accessed")),
    }
    audit = {
        "reference_prediction_sha256": reference["reference_prediction_sha256"],
        "candidate_prediction_sha256": sha256(candidate_predictions),
        "prediction_file_byte_hash_matches": sha256(candidate_predictions) == reference["reference_prediction_sha256"],
        "cross_process_prediction_tolerance": tolerance,
        "in_process_repeat_tolerance": float(reference["in_process_repeat_tolerance"]),
    }
    boolean_checks = {name: value for name, value in checks.items() if isinstance(value, bool)}
    if not all(boolean_checks.values()):
        failed = [name for name, passed in boolean_checks.items() if not passed]
        raise ValueError(f"Clean-environment technical replay does not match the frozen R6 contract: {failed}")
    if args.check_only:
        print("Gate 7 clean-reload verification preflight: structure/provenance and numerical tolerance checks pass; no labels accessed.")
        return
    if not args.confirm_clean_reload_verification:
        raise ValueError("Clean-reload verification requires --confirm-clean-reload-verification")
    with stage_output(args.output) as folder:
        (folder / "verification_checks.json").write_text(json.dumps({"checks": checks, "boolean_check_names": list(boolean_checks), "audit": audit}, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "README.md").write_text(
            "# Gate 7 clean-environment technical replay verification\n\n"
            "This stage compares only label-free technical output structure, provenance and pre-registered numerical "
            "tolerance against the R6 manifest. CSV byte hashes are recorded as audit data, not used as a cross-process "
            "pass/fail criterion. It is not a model performance evaluation and reads no labels, model, or candidate-selection artifact.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE,
            inputs={str((manifest_dir / "complete.json")): sha256(manifest_dir / "complete.json"), str((candidate / "complete.json")): sha256(candidate / "complete.json")},
            partial=False, no_model_loaded=True, no_labels_accessed=True, technical_prediction_values_compared=True,
            no_performance_metrics_computed=True, no_model_fitted=True, no_calibration=True,
            no_candidate_selection_changed=True, clean_environment_replay_verified=True,
        )
    print(f"Gate 7 clean-environment reload verification: {args.output}")


if __name__ == "__main__":
    run_cli(main)
