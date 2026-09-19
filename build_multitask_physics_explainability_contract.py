#!/usr/bin/env python3
"""Publish the label-safe physical-meaning and explainability contract for seven PK endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


ENDPOINT_CONTRACTS = (
    ("fu__human__plasma", "Human plasma unbound fraction", "fraction", "logit",
     "Plasma protein binding under the documented assay conditions.",
     "Do not equate plasma fu with blood fu or treat it as a universal in-vivo constant."),
    ("CLint__human__microsome", "Human microsomal intrinsic clearance", "uL/min/mg protein", "log10",
     "In-vitro microsomal intrinsic clearance normalized by protein.",
     "Do not use it as systemic human CL without explicit scaling assumptions and parameters."),
    ("Papp__human__caco2_ab", "Human Caco-2 A-to-B apparent permeability", "1e-6 cm/s", "log10",
     "Specified Caco-2 A-to-B apparent permeability measurement.",
     "Do not equate Papp with fraction absorbed or oral bioavailability."),
    ("F__human__absolute_oral", "Human absolute oral bioavailability", "fraction", "identity",
     "Absolute oral bioavailability with an IV reference under documented conditions.",
     "Do not pool relative F or infer a value without route, dose and reference semantics."),
    ("CL__human__systemic_iv", "Human systemic IV clearance", "L/h/kg", "log10",
     "Systemic clearance following IV administration in the documented matrix and population.",
     "Do not silently mix blood and plasma clearance or use a mechanistic formula with missing inputs."),
    ("VDss__human__steady_state_iv", "Human IV steady-state distribution volume", "L/kg", "log10",
     "Steady-state distribution volume after IV administration.",
     "Do not substitute Vz, Vc, unspecified Vd, or oral apparent volumes."),
    ("Thalf__human__terminal_iv", "Human terminal IV half-life", "h", "log10",
     "Terminal elimination half-life after IV administration and terminal-phase estimation.",
     "Do not force it to equal ln(2) times VDss divided by CL; terminal phase can reflect Vz, multicompartment behavior and sampling."),
)

RELATIONSHIPS = (
    ("micro_to_systemic_cl", "fu__human__plasma + CLint__human__microsome", "CL__human__systemic_iv",
     "OOF cascade candidate and conditional well-stirred sensitivity diagnostic.",
     "Predicted inputs must be strictly OOF; any mechanistic diagnostic requires explicit blood/plasma convention, MPPGL, protein binding, blood-to-plasma ratio and hepatic-flow assumptions.",
     "No hard target, hard prediction constraint or imputation when required physiological parameters are absent."),
    ("permeability_to_bioavailability", "Papp__human__caco2_ab", "F__human__absolute_oral",
     "OOF cascade candidate and absorption-related ablation signal.",
     "Predicted input must be strictly OOF and F must retain an absolute-oral definition.",
     "No conversion of Papp into Fa, and no hard F = Fa x Fg x Fh closure."),
    ("cl_vdss_to_time_scale", "CL__human__systemic_iv + VDss__human__steady_state_iv", "Thalf__human__terminal_iv",
     "Conditional time-scale diagnostic and OOF cascade candidate with a direct structure path retained.",
     "Predicted CL and VDss must be strictly OOF and all reported units must be compatible.",
     "No universal t1/2 = ln(2) x VDss / CL hard constraint and no replacement of the direct terminal-half-life head."),
)

EVIDENCE_PROTOCOL = (
    ("global_stability", "train/validation/OOF only", "Three seeds, folds and source-holdout sensitivity; report feature-group direction/rank stability.",
     "Feature attribution is descriptive model behavior, not causal evidence."),
    ("cascade_ablation", "train/validation/OOF only", "Compare direct structure, plus micro OOF, then optional animal OOF candidates under identical endpoint scopes.",
     "A better score does not establish a biological mechanism."),
    ("local_prediction_card", "new inputs or held-out predictions", "Prediction, unit, uncertainty, nearest-neighbor similarity, AD flag and evidence grade.",
     "A local explanation is not an individualized clinical or causal claim."),
    ("physics_diagnostic", "train/validation/OOF or new external evidence", "Report time-scale and mechanistic-assumption diagnostics separately from primary performance.",
     "Diagnostics must not alter post-test predictions, calibration, training or selection."),
)


def build_contract_tables(registry: dict[str, dict[str, object]]):
    endpoint_rows = []
    for task_id, name, unit, transform, scope, boundary in ENDPOINT_CONTRACTS:
        actual = registry.get(task_id)
        if actual is None:
            raise ValueError(f"Task registry lacks endpoint required by physics contract: {task_id}")
        # The machine registry preserves ASCII units and uses underscores in
        # compound unit tokens; the publication-facing contract uses spaces and
        # a micro sign. They are semantically equivalent spellings only here.
        normalized_expected = unit.replace("µ", "u").replace("_", " ")
        normalized_actual = str(actual.get("unit", "")).replace("µ", "u").replace("_", " ")
        if normalized_actual != normalized_expected:
            raise ValueError(f"Unit mismatch for {task_id}: expected {unit}, got {actual.get('unit')}")
        if actual.get("transform") != transform:
            raise ValueError(f"Transform mismatch for {task_id}: expected {transform}, got {actual.get('transform')}")
        endpoint_rows.append({
            "task_id": task_id, "endpoint_name": name, "canonical_unit": unit,
            "transform": transform, "scientific_scope": scope,
            "interpretation_boundary": boundary, "hard_physics_constraint_allowed": False,
        })
    relationship_rows = [dict(zip((
        "relationship_id", "input_tasks", "target_task", "permitted_use", "required_conditions", "prohibited_use"), row))
                         for row in RELATIONSHIPS]
    evidence_rows = [dict(zip(("evidence_id", "allowed_data_scope", "required_artifact", "claim_boundary"), row))
                     for row in EVIDENCE_PROTOCOL]
    return endpoint_rows, relationship_rows, evidence_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=ROOT / "data/processed_v15/datasets")
    parser.add_argument("--readiness", type=Path, default=ROOT / "results/analysis/multitask_endpoint_readiness_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/multitask_physics_explainability_contract_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    registry_path = args.datasets / "task_registry.json"
    startup_self_check([registry_path, args.datasets / "complete.json", args.readiness / "complete.json"],
                       output=None if args.check_only else args.output)
    verify_stage(args.datasets, "datasets")
    verify_stage(args.readiness, "multitask_endpoint_readiness_audit")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    endpoint_rows, relationship_rows, evidence_rows = build_contract_tables(registry)
    if args.check_only:
        print(f"Seven-endpoint physics/explainability contract valid: endpoints={len(endpoint_rows)} relations={len(relationship_rows)}")
        return
    with stage_output(args.output) as out:
        write_csv(out / "endpoint_physics_contract.csv", endpoint_rows)
        write_csv(out / "relationship_contract.csv", relationship_rows)
        write_csv(out / "explainability_evidence_protocol.csv", evidence_rows)
        (out / "README.md").write_text(
            "# Seven-endpoint physics and explainability contract\n\n"
            "This planning artifact fixes endpoint semantics, permitted conditional relationships, prohibited hard constraints, "
            "and the evidence required for explainability claims. It reads task metadata and readiness-stage metadata only: "
            "no endpoint values, predictions, test labels, model weights, selections or calibration are read or changed. "
            "Physical diagnostics must remain separate from model selection and post-test evaluation.\n",
            encoding="utf-8")
        finish_stage(out, "multitask_physics_explainability_contract", inputs={
            "datasets_complete_sha256": sha256(args.datasets / "complete.json"),
            "task_registry_sha256": sha256(registry_path),
            "readiness_complete_sha256": sha256(args.readiness / "complete.json"),
        }, endpoints=len(endpoint_rows), relationships=len(relationship_rows), evidence_protocols=len(evidence_rows),
           label_safe=True, partial=False)
    print(f"Seven-endpoint physics/explainability contract: {args.output}")


if __name__ == "__main__":
    run_cli(main)
