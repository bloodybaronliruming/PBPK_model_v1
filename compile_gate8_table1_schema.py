"""Freeze and compile Gate 8 Table 1 from R0-approved summary assets only.

This manuscript-assembly stage reads only the Gate 8 R0 registry plus the two
approved endpoint-level summary CSVs.  It never opens raw labels, prediction
rows, models, folds, or test data, and it does not compute performance metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_table1_schema"
R0_STAGE = "gate8_manuscript_evidence_registry"
ORDER = [
    "fu__human__plasma",
    "CLint__human__microsome",
    "Papp__human__caco2_ab",
    "CL__human__systemic_iv",
    "VDss__human__steady_state_iv",
    "Thalf__human__terminal_iv",
    "F__human__absolute_oral",
]
R0_OUTPUT = "data/public_development/gate8_manuscript_evidence_registry_v2"
OUTPUT = "data/public_development/gate8_table1_schema_v1"


def required_asset(manifest: pd.DataFrame, source_id: str, artifact: str) -> pd.Series:
    match = manifest.loc[manifest.source_id.eq(source_id) & manifest.artifact.eq(artifact)]
    if len(match) != 1:
        raise ValueError(f"R0 must contain exactly one allowed asset: {source_id}/{artifact}")
    return match.iloc[0]


def approved_sources(root: Path) -> tuple[pd.DataFrame, Path, Path, dict[str, str]]:
    r0 = root / R0_OUTPUT
    r0_meta = verify_stage(r0, R0_STAGE)
    required_flags = (
        "no_model_loaded", "no_labels_accessed", "no_prediction_rows_accessed",
        "no_performance_metrics_computed", "no_model_fitted", "no_calibration",
        "no_candidate_selection_changed",
    )
    if not all(r0_meta.get(flag) is True for flag in required_flags):
        raise ValueError("R0 does not satisfy the required no-label manuscript-assembly boundary")
    manifest_path = r0 / "source_artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    needed = {"source_id", "source_path", "source_stage", "artifact", "artifact_sha256", "evidence_class"}
    if not needed <= set(manifest.columns):
        raise ValueError(f"R0 source manifest lacks columns: {sorted(needed - set(manifest.columns))}")

    gate4_row = required_asset(manifest, "gate4_candidate_freeze", "endpoint_candidate_freeze_registry.csv")
    gate7_row = required_asset(manifest, "gate7_evidence_matrix", "endpoint_diagnostic_evidence_matrix.csv")
    gate4_folder = Path(gate4_row.source_path)
    gate7_folder = Path(gate7_row.source_path)
    verify_stage(gate4_folder, gate4_row.source_stage)
    verify_stage(gate7_folder, gate7_row.source_stage)
    gate4_path = gate4_folder / gate4_row.artifact
    gate7_path = gate7_folder / gate7_row.artifact
    if sha256(gate4_path) != gate4_row.artifact_sha256 or sha256(gate7_path) != gate7_row.artifact_sha256:
        raise ValueError("An R0-approved Table 1 summary asset changed after registration")
    return manifest, gate4_path, gate7_path, {
        str((r0 / "complete.json").resolve()): sha256(r0 / "complete.json"),
        str(manifest_path.resolve()): sha256(manifest_path),
        str((gate4_folder / "complete.json").resolve()): sha256(gate4_folder / "complete.json"),
        str(gate4_path.resolve()): sha256(gate4_path),
        str((gate7_folder / "complete.json").resolve()): sha256(gate7_folder / "complete.json"),
        str(gate7_path.resolve()): sha256(gate7_path),
    }


def table1_schema() -> pd.DataFrame:
    rows = [
        ("task_id", "Task ID", "string", "yes", "both approved endpoint summaries", "exact one-to-one task key", "Internal reproducibility key; not a human-readable endpoint definition."),
        ("endpoint_definition", "Endpoint definition", "string", "yes", "gate7 evidence matrix: endpoint_label", "copied", "Do not merge assay systems, route definitions, or terminal/microsomal half-lives."),
        ("canonical_unit", "Canonical unit", "string", "yes", "both approved endpoint summaries: canonical_unit", "must agree exactly; copied", "Unit is part of the endpoint definition."),
        ("training_records", "Training records (n)", "integer", "yes", "both approved endpoint summaries: train_records", "must agree exactly; copied", "Record count is descriptive, not an independent-study count."),
        ("training_parents", "Unique canonical parents (n)", "integer", "yes", "both approved endpoint summaries: train_parents", "must agree exactly; copied", "Parent count is descriptive and may be lower than record count."),
        ("research_reference", "Frozen research reference", "string", "yes", "Gate 4 candidate freeze: unified_research_reference", "copied", "This is a strict internal research reference, not a claim of universal model superiority."),
        ("input_feature_view", "Input feature view", "string", "yes", "Gate 4 candidate freeze: reference_feature_set", "copied", "No new feature availability or modality claim is introduced."),
        ("evidence_lane", "Evidence lane", "string", "yes", "Gate 7 evidence matrix: release_lane", "copied", "Keep current frozen-final and historical test-locked lanes distinct."),
        ("test_lifecycle", "Test lifecycle", "string", "yes", "Gate 7 evidence matrix: test_lifecycle", "copied", "A consumed/prohibited test cannot be reused for selection or calibration."),
        ("split_grouping_context", "Split and grouping context", "string", "yes", "R0-approved summary assets only", "fixed disclosure; no fold members re-derived", "Exact parent/scaffold/source/study grouping and fold membership remain in the frozen protocol/registry and Supplement; this table does not reconstruct them."),
        ("source_definition_scope", "Source/definition context", "string", "yes", "Gate 7 evidence matrix: diagnostic_source_scope", "copied", "Do not infer source counts or external independence beyond this text."),
        ("key_limitation", "Key limitation", "string", "yes", "Gate 7 evidence matrix: source_or_definition_limitation", "copied", "Must remain visible for F, terminal t½, and historical/test-locked lanes."),
    ]
    return pd.DataFrame(rows, columns=[
        "column_id", "display_header", "dtype", "required", "allowed_source", "assembly_rule", "claim_guardrail",
    ])


def compile_table(gate4_path: Path, gate7_path: Path) -> pd.DataFrame:
    gate4 = pd.read_csv(gate4_path)
    gate7 = pd.read_csv(gate7_path)
    gate4_columns = {
        "task_id", "endpoint", "canonical_unit", "train_records", "train_parents",
        "unified_research_reference", "reference_feature_set",
    }
    gate7_columns = {
        "task_id", "endpoint_label", "canonical_unit", "train_records", "train_parents",
        "release_lane", "test_lifecycle", "diagnostic_source_scope", "source_or_definition_limitation",
    }
    if not gate4_columns <= set(gate4.columns) or not gate7_columns <= set(gate7.columns):
        raise ValueError("Approved summary schema no longer satisfies the frozen Table 1 contract")
    if set(gate4.task_id) != set(ORDER) or set(gate7.task_id) != set(ORDER):
        raise ValueError("Table 1 requires exactly the frozen seven-endpoint membership")
    if gate4.task_id.duplicated().any() or gate7.task_id.duplicated().any():
        raise ValueError("Table 1 source summaries must be one row per endpoint")

    merged = gate4.loc[:, sorted(gate4_columns)].merge(
        gate7.loc[:, sorted(gate7_columns)], on="task_id", suffixes=("_gate4", "_gate7"), validate="one_to_one"
    )
    for field in ("canonical_unit", "train_records", "train_parents"):
        if not merged[f"{field}_gate4"].equals(merged[f"{field}_gate7"]):
            raise ValueError(f"Gate 4 and Gate 7 disagree for Table 1 field: {field}")
    result = pd.DataFrame({
        "task_id": merged.task_id,
        "endpoint_definition": merged.endpoint_label,
        "canonical_unit": merged.canonical_unit_gate4,
        "training_records": merged.train_records_gate4.astype(int),
        "training_parents": merged.train_parents_gate4.astype(int),
        "research_reference": merged.unified_research_reference,
        "input_feature_view": merged.reference_feature_set,
        "evidence_lane": merged.release_lane,
        "test_lifecycle": merged.test_lifecycle,
        "split_grouping_context": "Frozen endpoint-specific registry/protocol; exact group and fold membership reported separately in Supplement.",
        "source_definition_scope": merged.diagnostic_source_scope,
        "key_limitation": merged.source_or_definition_limitation,
    })
    result["endpoint_order"] = result.task_id.map({task_id: index + 1 for index, task_id in enumerate(ORDER)})
    result = result.sort_values("endpoint_order").drop(columns="endpoint_order")
    if len(result) != 7 or result.isna().any().any() or result.task_id.tolist() != ORDER:
        raise ValueError("Compiled Table 1 violates its frozen seven-row/non-null contract")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT

    manifest, gate4_path, gate7_path, inputs = approved_sources(root)
    schema = table1_schema()
    table = compile_table(gate4_path, gate7_path)
    source_contract = manifest.loc[
        manifest.source_id.isin(["gate4_candidate_freeze", "gate7_evidence_matrix"])
        & manifest.artifact.isin(["endpoint_candidate_freeze_registry.csv", "endpoint_diagnostic_evidence_matrix.csv"])
    ].copy()
    if len(source_contract) != 2:
        raise ValueError("Table 1 must bind exactly two R0-approved summary assets")
    if args.check_only:
        print("Gate 8 R1 preflight: table_rows=7 schema_fields=12 approved_assets=2; no labels, prediction rows, models, folds, or metrics read.")
        return

    with stage_output(output) as folder:
        schema.to_csv(folder / "table_1_schema.csv", index=False)
        source_contract.to_csv(folder / "table_1_source_contract.csv", index=False)
        table.to_csv(folder / "Table_1_dataset_task_context.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R1 Table 1 schema and compiled context\n\n"
            "This directory freezes the schema and reproducible assembly of manuscript Table 1. It reads only two summary CSVs "
            "approved by Gate 8 R0 and verifies their hashes against the R0 manifest. It does not read raw labels, prediction rows, "
            "models, fold members, or tests; it does not compute performance metrics, fit, calibrate, or select models. Exact split/group "
            "membership remains cited in the frozen registries/protocols and Supplement rather than being reconstructed here.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True,
            no_labels_accessed=True, no_prediction_rows_accessed=True, no_fold_members_accessed=True,
            no_performance_metrics_computed=True, no_model_fitted=True, no_calibration=True,
            no_candidate_selection_changed=True, table_row_count=len(table), schema_field_count=len(schema),
            approved_asset_count=len(source_contract), source_registry_version="gate8_manuscript_evidence_registry_v2",
        )
    print(f"Gate 8 R1 Table 1 schema and context: {output}")


if __name__ == "__main__":
    run_cli(main)
