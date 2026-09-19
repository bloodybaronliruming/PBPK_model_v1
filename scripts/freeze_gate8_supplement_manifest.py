"""Freeze Gate 8 Supplement asset/publication-rights manifest.

The registry verifies completed Gate 8 tables and R0-approved aggregate assets
only. It does not open labels, prediction rows, models, foldwise records or
metric tables. Unregistered foldwise/bootstrap detail and MMPK content are
recorded as gaps/restrictions rather than silently assembled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_supplement_manifest"
R0_STAGE = "gate8_manuscript_evidence_registry"
R0_OUTPUT = "data/public_development/gate8_manuscript_evidence_registry_v2"
OUTPUT = "data/public_development/gate8_supplement_manifest_v1"
GATE8_STAGES = {
    "table1": ("data/public_development/gate8_table1_schema_v1", "gate8_table1_schema", [
        "Table_1_dataset_task_context.csv", "table_1_schema.csv", "table_1_source_contract.csv",
    ]),
    "table2": ("data/public_development/gate8_table2_literature_boundaries_v1", "gate8_table2_literature_boundaries", [
        "Table_2_Jia_literature_and_author_like_context.csv", "table_2_public_schema.csv",
        "table_2_internal_only_boundary.csv", "table_2_source_contract.csv",
    ]),
    "table3": ("data/public_development/gate8_table3_strict_internal_v1", "gate8_table3_strict_internal", [
        "Table_3_strict_internal_evidence.csv", "table_3_schema.csv", "table_3_source_contract.csv",
    ]),
    "table4": ("data/public_development/gate8_table4_frozen_evidence_v1", "gate8_table4_frozen_evidence", [
        "Table_4_frozen_diagnostic_evidence.csv", "Table_4_engineering_reproducibility_evidence.csv",
        "table_4_endpoint_schema.csv", "table_4_engineering_schema.csv", "table_4_exclusion_lineage.csv",
        "table_4_source_contract.csv",
    ]),
}


def r0_asset(manifest: pd.DataFrame, source_id: str, artifact: str) -> pd.Series:
    found = manifest.loc[manifest.source_id.eq(source_id) & manifest.artifact.eq(artifact)]
    if len(found) != 1:
        raise ValueError(f"R0 must register exactly one supplement asset: {source_id}/{artifact}")
    return found.iloc[0]


def verified_r0_asset(manifest: pd.DataFrame, source_id: str, artifact: str, inputs: dict[str, str]) -> Path:
    row = r0_asset(manifest, source_id, artifact)
    folder = Path(row.source_path)
    verify_stage(folder, row.source_stage)
    complete = folder / "complete.json"
    inputs[str(complete.resolve())] = sha256(complete)
    path = folder / artifact
    if not path.is_file() or sha256(path) != row.artifact_sha256:
        raise ValueError(f"R0-approved supplement asset changed after registration: {path}")
    inputs[str(path.resolve())] = sha256(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT
    r0 = root / R0_OUTPUT
    r0_meta = verify_stage(r0, R0_STAGE)
    no_label_flags = (
        "no_model_loaded", "no_labels_accessed", "no_prediction_rows_accessed",
        "no_performance_metrics_computed", "no_model_fitted", "no_calibration",
        "no_candidate_selection_changed",
    )
    if not all(r0_meta.get(flag) is True for flag in no_label_flags):
        raise ValueError("R0 manuscript registry does not satisfy the no-label boundary")
    r0_manifest_path = r0 / "source_artifact_manifest.csv"
    manifest = pd.read_csv(r0_manifest_path)
    needed_manifest_columns = {"source_id", "source_path", "source_stage", "artifact", "artifact_sha256", "evidence_class", "allowed_usage"}
    if not needed_manifest_columns <= set(manifest.columns):
        raise ValueError("R0 source manifest schema is incomplete")
    inputs = {
        str((r0 / "complete.json").resolve()): sha256(r0 / "complete.json"),
        str(r0_manifest_path.resolve()): sha256(r0_manifest_path),
    }
    public_rows = []
    for key, (relative, stage, artifacts) in GATE8_STAGES.items():
        folder = root / relative
        verify_stage(folder, stage)
        inputs[str((folder / "complete.json").resolve())] = sha256(folder / "complete.json")
        for artifact in artifacts:
            path = folder / artifact
            if not path.is_file():
                raise FileNotFoundError(f"Gate 8 supplement asset missing: {path}")
            inputs[str(path.resolve())] = sha256(path)
            section = {"table1": "S1 Dataset/task context", "table2": "S2 Literature and evidence boundaries", "table3": "S3 Strict internal evidence", "table4": "S4 Frozen diagnostics and reproducibility"}[key]
            public_rows.append({
                "asset_id": f"gate8_{key}_{artifact}", "supplement_section": section,
                "asset_path": str(path.resolve()), "asset_sha256": sha256(path),
                "evidence_class": "derived_from_declared_gate8_boundary", "release_status": "include_in_manuscript_package",
                "purpose": "Versioned Gate 8 table/schema/contract asset.",
                "claim_guardrail": "Use only with the evidence and lifecycle boundary stated in the corresponding Gate 8 record.",
            })
    approved_public = [
        ("p1_matched_stl", "overall_matched_comparison.csv", "S5 Matched STL comparison", "B", "Preserved matched strong-STL comparison summary."),
        ("p1_matched_stl", "Figure_P1_vs_matched_control.png", "S5 Matched STL comparison", "B", "Existing matched-comparison figure; no universal champion claim."),
        ("gate3_b6_formal", "decision.json", "S6 Negative architecture result", "B", "Formal 0/6 B6 stop decision metadata."),
        ("gate3_b6_formal", "table_7_endpoint_final_decisions.csv", "S6 Negative architecture result", "B", "Per-endpoint B6 failed-gate decision summary."),
        ("gate3_b6_formal", "figure_2_paired_bootstrap_forest.png", "S6 Negative architecture result", "B", "Existing B6 paired-bootstrap forest; retain the stop decision."),
        ("gate7_reproducibility", "delivery_manifest.json", "S7 Reproducibility", "not_comparison", "Code/data/environment provenance manifest."),
        ("gate7_reproducibility", "runtime_versions.json", "S7 Reproducibility", "not_comparison", "Runtime version record."),
        ("gate7_reproducibility", "clean_environment_reload_protocol.md", "S7 Reproducibility", "not_comparison", "Clean-environment reload protocol."),
        ("gate7_clean_reload", "verification_checks.json", "S7 Reproducibility", "not_comparison", "Technical replay verification; not predictive performance."),
    ]
    for source_id, artifact, section, evidence_class, purpose in approved_public:
        path = verified_r0_asset(manifest, source_id, artifact, inputs)
        public_rows.append({
            "asset_id": f"{source_id}_{artifact}", "supplement_section": section,
            "asset_path": str(path.resolve()), "asset_sha256": sha256(path),
            "evidence_class": evidence_class, "release_status": "include_in_manuscript_package",
            "purpose": purpose,
            "claim_guardrail": "Retain original protocol/evidence class; do not create a cross-protocol rank or post-test selection claim.",
        })
    restricted = []
    for artifact in ("sensitivity_signal_decision.csv", "pooled_paired_metrics.csv", "Figure_1_MMPK_N2c_paired_bootstrap.png"):
        row = r0_asset(manifest, "mmpk_n2c_internal", artifact)
        # Hash and stage verification are allowed provenance checks; no content is parsed.
        verified_r0_asset(manifest, "mmpk_n2c_internal", artifact, inputs)
        restricted.append({
            "source_family": "MMPK N2c paired sensitivity", "artifact_name": artifact,
            "evidence_class": row.evidence_class, "release_status": "exclude_from_public_supplement",
            "reason": "Data rights are not established for public redistribution; do not publish comparison rows, row-level data, predictions, figures, or redistributable models.",
            "permitted_use": "Internal research provenance only.",
        })
    gaps = pd.DataFrame([
        {
            "supplement_component": "Foldwise values and detailed paired bootstrap draws",
            "status": "not_assembled_unregistered_in_R0_v2",
            "reason": "R0 v2 registers aggregate P1/B6 summaries and figures, but not a foldwise table or bootstrap-draw asset for manuscript assembly.",
            "required_action": "Do not access alternate files. A future registry amendment requires explicit rationale, scope review and user approval before any new source is added.",
        },
        {
            "supplement_component": "MMPK N2c numerical sensitivity outputs",
            "status": "withheld_internal_only",
            "reason": "MMPK data rights for public redistribution are not established.",
            "required_action": "Keep excluded until explicit data rights permit a separately reviewed public release path.",
        },
        {
            "supplement_component": "Raw labels, prediction rows, model binaries and fold memberships",
            "status": "not_permitted_for_this_supplement_manifest",
            "reason": "Gate 8 R0–R5 assembly is summary/provenance only and must preserve label/test/fold boundaries.",
            "required_action": "Do not add these assets to the manuscript package through this workflow.",
        },
    ])
    plan = pd.DataFrame([
        ("S1", "Dataset/task context", "Gate 8 R1 table/schema/contract", "include_in_manuscript_package"),
        ("S2", "Literature/evidence boundaries", "Gate 8 R4 public table/schema/contract; MMPK boundary only", "include_in_manuscript_package"),
        ("S3", "Strict internal matched and negative evidence", "Gate 8 R2 table/schema/contract", "include_in_manuscript_package"),
        ("S4", "Frozen diagnostic, AD, calibration and engineering evidence", "Gate 8 R3 panels/schema/exclusion/contract", "include_in_manuscript_package"),
        ("S5", "Matched P1 strong-STL evidence", "R0-approved P1 aggregate table and figure", "include_in_manuscript_package"),
        ("S6", "B6 negative ablation", "R0-approved decision, per-endpoint decision table and figure", "include_in_manuscript_package"),
        ("S7", "Reproducibility", "R0-approved delivery/runtime/reload protocol and verification", "include_in_manuscript_package"),
        ("S8", "Foldwise/detailed bootstrap material", "No R0-approved asset", "not_assembled_unregistered_in_R0_v2"),
        ("S9", "MMPK sensitivity numerical material", "Internal-only source", "exclude_from_public_supplement"),
    ], columns=["supplement_id", "title", "bound_assets", "assembly_status"])
    public_manifest = pd.DataFrame(public_rows)
    restricted_manifest = pd.DataFrame(restricted)
    if len(public_manifest) != 25 or len(restricted_manifest) != 3 or public_manifest.asset_path.duplicated().any():
        raise ValueError("Supplement manifest row count or uniqueness contract failed")
    if args.check_only:
        print("Gate 8 R5 preflight: public_assets=25 restricted_assets=3 gaps=3 supplement_sections=9; no labels, prediction rows, models, foldwise values, or metrics read.")
        return
    with stage_output(output) as folder:
        public_manifest.to_csv(folder / "supplement_public_asset_manifest.csv", index=False)
        restricted_manifest.to_csv(folder / "supplement_restricted_asset_manifest.csv", index=False)
        gaps.to_csv(folder / "supplement_gap_register.csv", index=False)
        plan.to_csv(folder / "supplement_assembly_plan.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R5 Supplement manifest\n\n"
            "This manifest freezes 25 public manuscript-package assets, three MMPK internal-only exclusions, and three assembly gaps. "
            "It does not read labels, predictions, models, foldwise values or metrics. S8 is deliberately not assembled because R0 v2 does not approve "
            "a foldwise/bootstrap-detail asset. S9 is excluded because MMPK numerical material is internal-only pending explicit data rights.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_fold_members_accessed=True, no_foldwise_values_accessed=True,
            no_performance_metrics_computed=True, no_model_fitted=True, no_calibration=True,
            no_candidate_selection_changed=True, no_mmpk_numeric_values_accessed=True,
            no_mmpk_numeric_values_exported=True, public_asset_count=len(public_manifest),
            restricted_asset_count=len(restricted_manifest), gap_count=len(gaps), supplement_section_count=len(plan),
            source_registry_version="gate8_manuscript_evidence_registry_v2",
        )
    print(f"Gate 8 R5 Supplement manifest: {output}")


if __name__ == "__main__":
    run_cli(main)
