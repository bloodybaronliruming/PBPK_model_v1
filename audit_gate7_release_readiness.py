"""Audit Gate 7 release assets without fitting, predicting, or reading labels.

This is a metadata/integrity gate only.  It deliberately distinguishes the
Gate-4 train-CV research references from older pre-test historical assets so
that a convenient inference wrapper cannot silently convert one into the
other.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check


GATE4_STAGE = "cross_gate_endpoint_candidate_freeze"
LEGACY_STAGE = "frozen_candidate_registry"
CLINT_STAGE = "gate5_clint_stagea_et_candidate"
THALF_STAGE = "thalf_v15_frozen_candidate"
GATE5_EVALUATION_STAGE = "gate5_clint_final_technical_recovery_test_evaluation"
GATE5_CLOSEOUT_STAGE = "gate5_clint_final_diagnostic_closeout"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_complete(folder: Path, expected_stage: str) -> dict:
    """Verify a completed immutable asset, including all declared artifacts."""
    complete = folder / "complete.json"
    startup_self_check([complete])
    meta = read_json(complete)
    if meta.get("stage") != expected_stage or meta.get("partial", False):
        raise ValueError(f"Invalid completed asset: {folder}")
    artifacts = meta.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"Missing artifact manifest: {folder}")
    for relative, digest in artifacts.items():
        artifact = folder / relative
        if not artifact.is_file() or sha256(artifact) != digest:
            raise ValueError(f"Artifact integrity failure: {artifact}")
    return meta


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Expected model file is missing: {path}")


def verify_legacy_candidate(root: Path, candidate: dict) -> tuple[str, int]:
    """Verify a legacy pre-test asset registered in final_candidates_v1."""
    kind = candidate["model_kind"]
    if kind == "multitask_seed_ensemble":
        runs = [Path(item) for item in candidate["run_dirs"]]
        for run in runs:
            verify_complete(run, "multitask_training")
            require_file(run / candidate["architecture"] / "model.pt")
            require_file(run / candidate["architecture"] / "preprocessing.joblib")
        return "multitask_seed_ensemble", len(runs)
    if kind == "stl":
        run = Path(candidate["run_dir"])
        verify_complete(run, "stl_training")
        require_file(run / candidate["algorithm"] / "model.joblib")
        return f"stl::{candidate['algorithm']}", 1
    raise ValueError(f"Unsupported legacy model kind: {kind}")


def build_rows(root: Path) -> tuple[pd.DataFrame, dict]:
    gate4 = root / "data/public_development/cross_gate_candidate_freeze_v1"
    gate4_meta = verify_complete(gate4, GATE4_STAGE)
    registry_path = gate4 / "endpoint_candidate_freeze_registry.csv"
    startup_self_check([registry_path])
    registry = pd.read_csv(registry_path)
    required = {
        "task_id", "endpoint", "canonical_unit", "unified_research_reference",
        "publication_asset", "publication_asset_policy", "test_lifecycle", "inferential_status",
    }
    if not required <= set(registry.columns) or len(registry) != 7 or registry.task_id.duplicated().any():
        raise ValueError("Gate-4 endpoint registry does not meet the seven-task release contract")

    legacy_dir = root / "models/frozen/final_candidates_v1"
    legacy_meta = verify_complete(legacy_dir, LEGACY_STAGE)
    legacy = read_json(legacy_dir / "candidate_registry.json")
    legacy_by_task = {item["task_id"]: item for item in legacy.get("candidates", [])}

    clint_dir = root / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2"
    clint_meta = verify_complete(clint_dir, CLINT_STAGE)
    for name in ["model.joblib", "model_contract.json", "evaluation_protocol.json", "train_membership.csv"]:
        require_file(clint_dir / name)
    clint_evaluation_dir = root / "results/final/CLint_gate5_stageA_et_c01_test_recovery_v2"
    clint_evaluation_meta = verify_complete(clint_evaluation_dir, GATE5_EVALUATION_STAGE)
    clint_closeout_dir = root / "results/analysis/gate5_clint_final_diagnostic_closeout_v1"
    clint_closeout_meta = verify_complete(clint_closeout_dir, GATE5_CLOSEOUT_STAGE)
    if (not clint_evaluation_meta.get("test_evaluated") or clint_evaluation_meta.get("refit")
            or clint_evaluation_meta.get("calibration") or clint_evaluation_meta.get("selection_changed")
            or not clint_closeout_meta.get("test_lifecycle_closed")):
        raise ValueError("Gate 5 CLint lifecycle closure is incomplete or changed")

    thalf_dir = root / "models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1"
    thalf_meta = verify_complete(thalf_dir, THALF_STAGE)
    thalf = read_json(thalf_dir / "candidate_registry.json")
    if thalf.get("task_id") != "Thalf__human__terminal_iv" or len(thalf.get("members", [])) != 3:
        raise ValueError("Frozen terminal-half-life ensemble registry is incomplete")
    for member in thalf["members"]:
        run = Path(member["run_dir"])
        verify_complete(run, "stl_training")
        require_file(run / "extratrees" / "model.joblib")
        if sha256(run / "complete.json") != member["run_complete_sha256"]:
            raise ValueError(f"Frozen terminal-half-life member hash differs: {run}")
        if sha256(run / "extratrees" / "model.joblib") != member["model_sha256"]:
            raise ValueError(f"Frozen terminal-half-life model hash differs: {run}")

    rows = []
    historical_tasks = {
        "fu__human__plasma", "Papp__human__caco2_ab", "CL__human__systemic_iv", "VDss__human__steady_state_iv",
    }
    for item in registry.itertuples(index=False):
        row = item._asdict()
        task = row["task_id"]
        if task in historical_tasks:
            candidate = legacy_by_task.get(task)
            if candidate is None:
                raise ValueError(f"Historical Gate-4 publication asset absent from legacy registry: {task}")
            model_kind, member_count = verify_legacy_candidate(root, candidate)
            asset_source = str(legacy_dir.relative_to(root))
            asset_status = "verified_historical_pretest_asset"
            release_lane = "historical_test_locked"
            next_operation = "preserve historical asset; do not relabel it as the Gate-4 research reference"
        elif task == "CLint__human__microsome":
            model_kind, member_count = "stl::extra_trees", 1
            asset_source = str(clint_dir.relative_to(root))
            asset_status = "verified_gate5_final_asset"
            release_lane = "current_frozen_final"
            next_operation = "eligible for unified inference-contract implementation"
            row["gate4_registry_test_lifecycle_original"] = row["test_lifecycle"]
            row["test_lifecycle"] = "test_consumed_no_reselection"
            row["test_lifecycle_evidence_source"] = "Gate5_final_evaluation_and_closeout"
        elif task == "Thalf__human__terminal_iv":
            model_kind, member_count = "stl_seed_ensemble::extratrees", 3
            asset_source = str(thalf_dir.relative_to(root))
            asset_status = "verified_frozen_ensemble"
            release_lane = "historical_test_locked_with_source_overlap_limit"
            next_operation = "eligible for contract implementation; expose TDC/Obach overlap limitation"
        elif task == "F__human__absolute_oral":
            model_kind, member_count = "none", 0
            asset_source = "none"
            asset_status = "intentional_no_model"
            release_lane = "exploratory_interface_only"
            next_operation = "return status only; no numeric prediction or performance claim"
        else:
            raise ValueError(f"Unknown Gate-4 task: {task}")
        row.setdefault("gate4_registry_test_lifecycle_original", row["test_lifecycle"])
        row.setdefault("test_lifecycle_evidence_source", "Gate4_test_lifecycle_registry")
        rows.append({
            "task_id": task,
            "endpoint": row["endpoint"],
            "canonical_unit": row["canonical_unit"],
            "gate4_unified_research_reference": row["unified_research_reference"],
            "gate4_publication_asset_policy": row["publication_asset_policy"],
            "test_lifecycle": row["test_lifecycle"],
            "inferential_status": row["inferential_status"],
            "verified_asset_source": asset_source,
            "verified_asset_status": asset_status,
            "release_lane": release_lane,
            "model_kind": model_kind,
            "verified_member_count": member_count,
            "gate7_next_operation": next_operation,
        })
    frame = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    inputs = {
        str((gate4 / "complete.json").resolve()): sha256(gate4 / "complete.json"),
        str(registry_path.resolve()): sha256(registry_path),
        str((legacy_dir / "complete.json").resolve()): sha256(legacy_dir / "complete.json"),
        str((clint_dir / "complete.json").resolve()): sha256(clint_dir / "complete.json"),
        str((clint_evaluation_dir / "complete.json").resolve()): sha256(clint_evaluation_dir / "complete.json"),
        str((clint_closeout_dir / "complete.json").resolve()): sha256(clint_closeout_dir / "complete.json"),
        str((thalf_dir / "complete.json").resolve()): sha256(thalf_dir / "complete.json"),
    }
    return frame, {
        "inputs": inputs,
        "gate4_stage": gate4_meta["stage"],
        "legacy_stage": legacy_meta["stage"],
        "clint_stage": clint_meta["stage"],
        "thalf_stage": thalf_meta["stage"],
    }


def write_readme(folder: Path, rows: pd.DataFrame) -> None:
    current = int(rows.release_lane.eq("current_frozen_final").sum())
    historical = int(rows.release_lane.str.startswith("historical_test_locked").sum())
    exploratory = int(rows.release_lane.eq("exploratory_interface_only").sum())
    folder.joinpath("README.md").write_text(
        "# Gate 7 R0 release-readiness audit\n\n"
        "This is a metadata-only integrity audit. It does not fit a model, generate a prediction, "
        "read any target label, validation/test label, or external label, or modify a candidate.\n\n"
        "## Result\n\n"
        f"All seven Gate-4 task entries have an auditable state: {current} current frozen-final asset, "
        f"{historical} historical pre-test/test-locked asset(s), and {exploratory} intentional no-model interface.\n\n"
        "CLint's Gate-4 pre-evaluation status is replaced by its verified Gate-5 consumed-and-closed lifecycle.\n\n"
        "The audit **does not authorize a single undifferentiated release model**. The four historical assets "
        "(fu, Papp, CL, VDss) are not the same objects as the later Gate-4 train-CV research references. "
        "Any subsequent unified interface must preserve these two lanes, report their different evidence status, "
        "and never use a historical consumed test result to select or validate a later reference model.\n\n"
        "F remains an exploratory status-only interface: it must not return a numeric prediction or claim a test result.\n\n"
        "## Next allowed action\n\n"
        "Freeze a machine-readable Gate-7 input/output contract and a no-label batch-inference smoke protocol. "
        "Before a new full-train research-reference package can be fitted for an endpoint with a consumed historical test, "
        "its release lane and non-comparability to that historical test must be pre-registered.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "results/analysis/gate7_release_readiness_audit_v3"
    rows, metadata = build_rows(root)
    if args.check_only:
        print(f"Gate 7 release-readiness audit check: endpoints={len(rows)}")
        print(rows[["task_id", "release_lane", "verified_asset_status"]].to_string(index=False))
        return
    with stage_output(output) as folder:
        rows.to_csv(folder / "release_readiness_by_endpoint.csv", index=False)
        write_readme(folder, rows)
        finish_stage(
            folder,
            "gate7_release_readiness_audit",
            inputs=metadata["inputs"],
            no_model_fitted=True,
            no_predictions_generated=True,
            no_labels_accessed=True,
            no_candidate_selection_changed=True,
            partial=False,
            gate7_r0_audit_passed=True,
            unified_numeric_release_authorized=False,
            endpoint_count=int(len(rows)),
            current_frozen_final_assets=int(rows.release_lane.eq("current_frozen_final").sum()),
            historical_test_locked_assets=int(rows.release_lane.str.startswith("historical_test_locked").sum()),
            exploratory_status_only_interfaces=int(rows.release_lane.eq("exploratory_interface_only").sum()),
            **{key: value for key, value in metadata.items() if key != "inputs"},
        )
    print(f"Gate 7 release-readiness audit: {output}")


if __name__ == "__main__":
    run_cli(main)
