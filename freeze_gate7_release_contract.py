"""Freeze the dual-lane Gate 7 input/output and no-label smoke contract.

This stage is intentionally declarative: it does not load a model, make a
prediction, or access labels.  It prevents a later implementation from
silently mixing historical test-locked assets with Gate-4 train-CV references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


READINESS_STAGE = "gate7_release_readiness_audit"


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_contract(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    readiness_dir = root / "results/analysis/gate7_release_readiness_audit_v3"
    readiness_meta = verify_stage(readiness_dir, READINESS_STAGE)
    if not readiness_meta.get("gate7_r0_audit_passed") or readiness_meta.get("unified_numeric_release_authorized"):
        raise ValueError("Gate 7 R0 does not permit this contract state")
    readiness_path = readiness_dir / "release_readiness_by_endpoint.csv"
    startup_self_check([readiness_path])
    rows = pd.read_csv(readiness_path)
    if len(rows) != 7 or rows.task_id.duplicated().any():
        raise ValueError("Gate 7 readiness table must contain exactly seven unique tasks")
    expected_lanes = {
        "current_frozen_final": 1,
        "historical_test_locked": 4,
        "historical_test_locked_with_source_overlap_limit": 1,
        "exploratory_interface_only": 1,
    }
    counts = rows.release_lane.value_counts().to_dict()
    if counts != expected_lanes:
        raise ValueError(f"Unexpected Gate 7 release-lane counts: {counts}")

    output_rows = []
    tasks = []
    for row in rows.sort_values("task_id").itertuples(index=False):
        item = row._asdict()
        lane = item["release_lane"]
        if lane == "exploratory_interface_only":
            numeric_status = "not_available_exploratory"
            smoke_eligible = False
            limitation = "Insufficient independent validation; numeric F prediction and test-performance claims are prohibited."
        elif lane == "current_frozen_final":
            numeric_status = "pending_no_label_technical_smoke"
            smoke_eligible = True
            limitation = "Frozen final asset; technical smoke is not an efficacy evaluation."
        elif lane == "historical_test_locked":
            numeric_status = "pending_no_label_technical_smoke"
            smoke_eligible = True
            limitation = "Historical pre-test asset only; its consumed test may not select, validate, or replace any later reference."
        elif lane == "historical_test_locked_with_source_overlap_limit":
            numeric_status = "pending_no_label_technical_smoke"
            smoke_eligible = True
            limitation = "Historical asset; disclose TDC/Obach source overlap and do not claim independent external validation."
        else:
            raise ValueError(f"Unsupported release lane: {lane}")
        item.update({
            "numeric_prediction_status": numeric_status,
            "no_label_smoke_eligible": smoke_eligible,
            "release_limitation": limitation,
        })
        output_rows.append(item)
        tasks.append({
            "task_id": item["task_id"],
            "endpoint": item["endpoint"],
            "unit": item["canonical_unit"],
            "model_lane": lane,
            "model_kind": item["model_kind"],
            "asset_source": item["verified_asset_source"],
            "test_lifecycle": item["test_lifecycle"],
            "inferential_status": item["inferential_status"],
            "numeric_prediction_status": numeric_status,
            "no_label_smoke_eligible": smoke_eligible,
            "limitation": limitation,
        })

    contract = {
        "contract_version": 1,
        "release_mode": "dual_lane_pre_smoke",
        "numeric_release_authorized": False,
        "input": {
            "format": "CSV",
            "required_columns": ["smiles"],
            "optional_columns": ["input_id"],
            "allowed_columns": ["smiles", "input_id"],
            "smiles_requirements": ["non-empty", "RDKit-parseable", "input-row order preserved"],
            "forbidden_label_column_tokens": ["label", "target", "response", "measurement", "observed", "actual", "y_true"],
        },
        "output": {
            "one_row_per_input_per_task": True,
            "required_columns": [
                "input_row", "input_id", "smiles", "task_id", "endpoint", "model_lane", "model_version",
                "prediction_status", "predicted_physical", "unit", "nearest_train_tanimoto",
                "applicability_domain", "evidence_status", "test_lifecycle", "limitation",
            ],
            "f_rule": "F must have prediction_status=not_available_exploratory and predicted_physical missing.",
            "dual_lane_rule": "Historical test-locked predictions must never be merged or directly ranked with later Gate-4 train-CV research references.",
        },
        "tasks": tasks,
    }
    smoke = {
        "protocol_version": 1,
        "stage": "gate7_no_label_batch_inference_smoke",
        "confirmation_flag": "--confirm-no-label-smoke",
        "input_policy": {
            "rows_min": 2,
            "rows_max": 8,
            "allowed_columns": ["smiles", "input_id"],
            "forbidden_label_column_tokens": contract["input"]["forbidden_label_column_tokens"],
            "fixed_label_free_probe_smiles": ["CCO", "c1ccccc1"],
        },
        "permitted_operations": [
            "parse input SMILES", "load only contract-bound assets", "produce technical predictions", "compute nearest-training similarity",
            "assert deterministic repeat equality", "write status and model metadata",
        ],
        "prohibited_operations": [
            "read labels", "compute performance metrics", "fit or calibrate models", "select or replace candidates",
            "access validation/test/external labels", "compare historical assets against Gate-4 research references",
        ],
        "required_assertions": [
            "output rows equal input rows multiplied by seven tasks", "all output schema columns present",
            "F is status-only with no numeric prediction", "all non-F task lanes match the frozen contract",
            "two deterministic in-process predictions agree exactly within 1e-12", "no label-like input columns accepted",
        ],
        "success_does_not_mean": [
            "model performance is evaluated", "a unified numeric release is authorized", "historical tests can be reused",
            "an endpoint has independent external validation",
        ],
    }
    inputs = {
        str((readiness_dir / "complete.json").resolve()): sha256(readiness_dir / "complete.json"),
        str(readiness_path.resolve()): sha256(readiness_path),
    }
    return pd.DataFrame(output_rows), contract, {"smoke": smoke, "inputs": inputs}


def write_readme(folder: Path) -> None:
    folder.joinpath("README.md").write_text(
        "# Gate 7 R1 dual-lane release contract\n\n"
        "This is a frozen, declarative input/output contract and no-label technical smoke protocol. "
        "It does not load a model, generate a prediction, read labels, or change a candidate.\n\n"
        "The contract has two non-interchangeable numeric lanes: the current frozen-final CLint asset and "
        "historical pre-test/test-locked assets. The historical lane cannot be relabelled as the later Gate-4 "
        "train-CV research-reference lane. Terminal t½ also retains its source-overlap limitation.\n\n"
        "F is status-only: it must return `not_available_exploratory`, not a numeric prediction.\n\n"
        "The next permitted engineering step is a separate explicit-confirmation, label-free technical smoke. "
        "Its success establishes only interface integrity, never performance or a unified numeric-release claim.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "data/public_development/gate7_release_contract_v2"
    rows, contract, extra = build_contract(root)
    if args.check_only:
        print(f"Gate 7 R1 contract check: tasks={len(rows)} smoke_eligible={int(rows.no_label_smoke_eligible.sum())}")
        return
    with stage_output(output) as folder:
        rows.to_csv(folder / "endpoint_release_contract.csv", index=False)
        dump_json(folder / "release_contract.json", contract)
        dump_json(folder / "no_label_smoke_protocol.json", extra["smoke"])
        write_readme(folder)
        finish_stage(
            folder,
            "gate7_release_contract_freeze",
            inputs=extra["inputs"],
            partial=False,
            no_model_fitted=True,
            no_predictions_generated=True,
            no_labels_accessed=True,
            no_candidate_selection_changed=True,
            dual_lane_contract_frozen=True,
            unified_numeric_release_authorized=False,
            no_label_smoke_authorized=True,
            endpoint_count=int(len(rows)),
            numeric_smoke_eligible_endpoints=int(rows.no_label_smoke_eligible.sum()),
        )
    print(f"Gate 7 R1 release contract: {output}")


if __name__ == "__main__":
    run_cli(main)
