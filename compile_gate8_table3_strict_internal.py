"""Freeze and compile Gate 8 Table 3 strict internal B-level evidence only.

The script reads only assets explicitly registered by Gate 8 R0: Gate 4
endpoint references, P1 matched strong-STL comparisons, and the formal B6
negative ablation decision.  It neither accesses raw labels/predictions/models
nor recomputes any metric, bootstrap, ranking, calibration, or model choice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_table3_strict_internal"
R0_STAGE = "gate8_manuscript_evidence_registry"
R0_OUTPUT = "data/public_development/gate8_manuscript_evidence_registry_v2"
OUTPUT = "data/public_development/gate8_table3_strict_internal_v1"
ORDER = [
    "fu__human__plasma",
    "CLint__human__microsome",
    "Papp__human__caco2_ab",
    "CL__human__systemic_iv",
    "VDss__human__steady_state_iv",
    "Thalf__human__terminal_iv",
]


def asset(manifest: pd.DataFrame, source_id: str, artifact: str) -> pd.Series:
    result = manifest.loc[manifest.source_id.eq(source_id) & manifest.artifact.eq(artifact)]
    if len(result) != 1:
        raise ValueError(f"R0 must register exactly one Table 3 asset: {source_id}/{artifact}")
    return result.iloc[0]


def load_approved_assets(root: Path) -> tuple[dict[str, Path], pd.DataFrame, dict[str, str]]:
    r0 = root / R0_OUTPUT
    r0_meta = verify_stage(r0, R0_STAGE)
    flags = (
        "no_model_loaded", "no_labels_accessed", "no_prediction_rows_accessed",
        "no_performance_metrics_computed", "no_model_fitted", "no_calibration",
        "no_candidate_selection_changed",
    )
    if not all(r0_meta.get(flag) is True for flag in flags):
        raise ValueError("R0 does not meet the no-label/no-reselection boundary required by Table 3")
    manifest_path = r0 / "source_artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    required = {
        "gate4": asset(manifest, "gate4_candidate_freeze", "endpoint_candidate_freeze_registry.csv"),
        "p1": asset(manifest, "p1_matched_stl", "overall_matched_comparison.csv"),
        "b6": asset(manifest, "gate3_b6_formal", "table_7_endpoint_final_decisions.csv"),
        "b6_decision": asset(manifest, "gate3_b6_formal", "decision.json"),
    }
    paths: dict[str, Path] = {}
    inputs = {
        str((r0 / "complete.json").resolve()): sha256(r0 / "complete.json"),
        str(manifest_path.resolve()): sha256(manifest_path),
    }
    checked_folders: set[Path] = set()
    for key, row in required.items():
        folder = Path(row.source_path)
        if folder not in checked_folders:
            verify_stage(folder, row.source_stage)
            inputs[str((folder / "complete.json").resolve())] = sha256(folder / "complete.json")
            checked_folders.add(folder)
        path = folder / row.artifact
        if not path.is_file() or sha256(path) != row.artifact_sha256:
            raise ValueError(f"R0-approved Table 3 asset changed after registration: {path}")
        paths[key] = path
        inputs[str(path.resolve())] = sha256(path)
    return paths, manifest, inputs


def table_schema() -> pd.DataFrame:
    rows = [
        ("evidence_row_id", "Evidence row ID", "string", "yes", "compiler", "deterministic section/task key", "Traceability key only."),
        ("table_section", "Table section", "string", "yes", "compiler", "frozen section label", "Do not intermix reference, matched comparison, and B6 decision rows."),
        ("task_id", "Task ID", "string", "yes", "all approved assets", "must be one of six B-level tasks", "F is excluded because its project level is C/exploratory."),
        ("endpoint", "Endpoint", "string", "yes", "Gate 4/P1/B6", "source label copied", "Endpoint definitions are those of Table 1."),
        ("comparison_level", "Comparison level", "string", "yes", "compiler + Gate 4", "literal B", "Only within-project strict internal evidence is admitted."),
        ("comparison_id", "Comparison ID", "string", "yes", "compiler", "frozen protocol family key", "No row may compare different protocol families numerically."),
        ("analysis_scope", "Analysis scope", "string", "yes", "approved source type", "fixed disclosure", "A frozen reference is descriptive; a P1 pair is matched; B6 is a pre-registered stop decision."),
        ("primary_metric", "Primary metric", "string", "yes", "Gate 4/P1", "copied or not_applicable", "No cross-endpoint metric ranking."),
        ("reference_value", "Reference value", "string", "yes", "Gate 4/P1", "copied or not_applicable", "Numeric values may only be compared inside the same comparison_id."),
        ("candidate_or_observed_value", "Candidate/observed value", "string", "yes", "Gate 4/P1", "copied or not_applicable", "No new metric is computed."),
        ("relative_change_pct", "Relative change (%)", "string", "yes", "P1", "copied or not_applicable", "Descriptive matched result only; not a substitute decision by itself."),
        ("paired_bootstrap_95ci", "Paired bootstrap 95% CI", "string", "yes", "P1", "copied or not_applicable", "Only the source's matched comparison CI is reported."),
        ("advance_or_reporting_decision", "Advance/reporting decision", "string", "yes", "Gate 4/B6/compiler", "copied or fixed reporting boundary", "B6 must remain a negative ablation; no stopped family is reopened."),
        ("source_artifact", "Source artifact", "string", "yes", "R0 manifest", "exact approved artifact", "No asset outside the R0 manifest is permitted."),
        ("claim_guardrail", "Claim guardrail", "string", "yes", "compiler", "fixed", "No cross-protocol numerical ranking, test selection, or causal inference."),
    ]
    return pd.DataFrame(rows, columns=[
        "column_id", "display_header", "dtype", "required", "allowed_source", "assembly_rule", "claim_guardrail",
    ])


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def compile_table(paths: dict[str, Path]) -> pd.DataFrame:
    gate4 = pd.read_csv(paths["gate4"])
    p1 = pd.read_csv(paths["p1"])
    b6 = pd.read_csv(paths["b6"])
    decision = json.loads(paths["b6_decision"].read_text(encoding="utf-8"))
    gate4_fields = {
        "task_id", "endpoint", "primary_metric", "train_cv_score", "unified_research_reference",
        "comparison_level_current_project", "next_action",
    }
    p1_fields = {
        "task_id", "endpoint", "primary_metric_name", "p1_primary_metric", "matched_control_primary_metric",
        "relative_change_pct", "paired_bootstrap_difference_95ci_low", "paired_bootstrap_difference_95ci_high",
    }
    b6_fields = {"task_id", "endpoint_label", "A1_to_A6_all_passed", "B6_advanced", "failed_required_gates", "decision"}
    if not gate4_fields <= set(gate4.columns) or not p1_fields <= set(p1.columns) or not b6_fields <= set(b6.columns):
        raise ValueError("An approved Table 3 summary no longer satisfies its frozen schema")
    gate4_b = gate4.loc[gate4.task_id.isin(ORDER)].copy()
    if set(gate4_b.task_id) != set(ORDER) or set(p1.task_id) != set(ORDER) or set(b6.task_id) != set(ORDER):
        raise ValueError("Table 3 requires the same six B-level endpoint tasks in all source summaries")
    if any(frame.task_id.duplicated().any() for frame in (gate4_b, p1, b6)):
        raise ValueError("Table 3 sources must each contain one row per strict internal endpoint")
    if set(gate4.loc[gate4.task_id.eq("F__human__absolute_oral"), "comparison_level_current_project"]) != {"C"}:
        raise ValueError("F exclusion requires its frozen exploratory C-level designation")
    if set(gate4_b.comparison_level_current_project) != {"B"}:
        raise ValueError("All Table 3 endpoint references must be B-level")
    if decision.get("endpoints_advancing_B6") != 0 or decision.get("B6_family_advanced") is not False:
        raise ValueError("Table 3 requires the frozen B6 stop decision")
    if b6.B6_advanced.astype(bool).any() or b6.A1_to_A6_all_passed.astype(bool).any():
        raise ValueError("B6 endpoint table conflicts with its frozen 0/6 advancement decision")
    if set(b6.decision) != {"stop_B6_retain_as_negative_ablation"}:
        raise ValueError("B6 decision wording is not the frozen negative-ablation decision")

    p1_by_task = p1.set_index("task_id")
    b6_by_task = b6.set_index("task_id")
    rows = []
    for position, task_id in enumerate(ORDER, start=1):
        candidate = gate4_b.loc[gate4_b.task_id.eq(task_id)].iloc[0]
        p1_row = p1_by_task.loc[task_id]
        b6_row = b6_by_task.loc[task_id]
        if candidate.endpoint != p1_row.endpoint or candidate.endpoint != b6_row.endpoint_label.replace("Terminal t½", "Thalf") and task_id != "Thalf__human__terminal_iv":
            raise ValueError(f"Endpoint label mismatch in strict internal summaries: {task_id}")
        if candidate.primary_metric != p1_row.primary_metric_name:
            raise ValueError(f"P1 metric differs from Gate 4 metric for {task_id}")
        rows.extend([
            {
                "evidence_row_id": f"A{position}", "table_section": "A. Frozen endpoint-specific research reference",
                "task_id": task_id, "endpoint": candidate.endpoint, "comparison_level": "B",
                "comparison_id": "gate4_frozen_endpoint_reference",
                "analysis_scope": "Strict internal train-CV research reference; descriptive endpoint-specific candidate freeze.",
                "primary_metric": candidate.primary_metric, "reference_value": "not_applicable",
                "candidate_or_observed_value": fmt(candidate.train_cv_score), "relative_change_pct": "not_applicable",
                "paired_bootstrap_95ci": "not_applicable", "advance_or_reporting_decision": candidate.next_action,
                "source_artifact": "gate4_candidate_freeze/endpoint_candidate_freeze_registry.csv",
                "claim_guardrail": "Do not rank this value against a different endpoint, split, or historical test lane.",
            },
            {
                "evidence_row_id": f"B{position}", "table_section": "B. P1 matched strong-STL comparison",
                "task_id": task_id, "endpoint": p1_row.endpoint, "comparison_level": "B",
                "comparison_id": "p1_vs_matched_control",
                "analysis_scope": "Within-project matched strict internal comparison; separate from the Gate 4 frozen reference row.",
                "primary_metric": p1_row.primary_metric_name, "reference_value": fmt(p1_row.matched_control_primary_metric),
                "candidate_or_observed_value": fmt(p1_row.p1_primary_metric), "relative_change_pct": fmt(p1_row.relative_change_pct),
                "paired_bootstrap_95ci": f"[{fmt(p1_row.paired_bootstrap_difference_95ci_low)}, {fmt(p1_row.paired_bootstrap_difference_95ci_high)}]",
                "advance_or_reporting_decision": "Report the preserved matched result only; no cross-protocol candidate replacement claim.",
                "source_artifact": "p1_matched_stl/overall_matched_comparison.csv",
                "claim_guardrail": "Interpret only against its matched control under this comparison_id; no test use or universal champion claim.",
            },
            {
                "evidence_row_id": f"C{position}", "table_section": "C. B6 nested experimental-physchem ablation",
                "task_id": task_id, "endpoint": b6_row.endpoint_label, "comparison_level": "B",
                "comparison_id": "B6_preregistered_advance_gates",
                "analysis_scope": "Formal nested experimental-physchem ablation; endpoint advance gates A1–A6.",
                "primary_metric": "not_applicable: multi-gate advance decision", "reference_value": "not_applicable",
                "candidate_or_observed_value": "not_applicable", "relative_change_pct": "not_applicable",
                "paired_bootstrap_95ci": "not_applicable",
                "advance_or_reporting_decision": f"{b6_row.decision}; failed_required_gates={b6_row.failed_required_gates}",
                "source_artifact": "gate3_b6_formal/table_7_endpoint_final_decisions.csv; decision.json",
                "claim_guardrail": "Report the 0/6 B6 stop decision; do not reopen the stopped architecture family.",
            },
        ])
    output = pd.DataFrame(rows)
    if len(output) != 18 or set(output.comparison_level) != {"B"} or output.isna().any().any():
        raise ValueError("Table 3 must contain 18 complete, B-level-only evidence rows")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT
    paths, manifest, inputs = load_approved_assets(root)
    schema = table_schema()
    table = compile_table(paths)
    source_contract = manifest.loc[
        ((manifest.source_id.eq("gate4_candidate_freeze")) & manifest.artifact.eq("endpoint_candidate_freeze_registry.csv"))
        | ((manifest.source_id.eq("p1_matched_stl")) & manifest.artifact.eq("overall_matched_comparison.csv"))
        | ((manifest.source_id.eq("gate3_b6_formal")) & manifest.artifact.isin(["table_7_endpoint_final_decisions.csv", "decision.json"]))
    ].copy()
    if len(source_contract) != 4:
        raise ValueError("Table 3 must bind exactly four R0-approved assets")
    if args.check_only:
        print("Gate 8 R2 preflight: table_rows=18 schema_fields=15 approved_assets=4 B_level_only=yes; no labels, prediction rows, models, or metrics read.")
        return
    with stage_output(output) as folder:
        schema.to_csv(folder / "table_3_schema.csv", index=False)
        source_contract.to_csv(folder / "table_3_source_contract.csv", index=False)
        table.to_csv(folder / "Table_3_strict_internal_evidence.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R2 Table 3 strict internal evidence\n\n"
            "This directory compiles only B-level, within-project strict internal evidence already registered by Gate 8 R0: "
            "frozen endpoint references, P1 matched strong-STL comparisons, and the B6 formal negative ablation. F is excluded "
            "because it remains C-level exploratory. The compiler verifies R0 and source hashes but does not read labels, prediction "
            "rows, models, fold members or tests, and does not recompute metrics, bootstrap intervals, rankings, fit, calibrate or select. "
            "Rows from different comparison_id values must never be numerically ranked against one another.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_fold_members_accessed=True, no_test_labels_accessed=True,
            no_performance_metrics_computed=True, no_bootstrap_recomputed=True, no_model_fitted=True,
            no_calibration=True, no_candidate_selection_changed=True, table_row_count=len(table),
            schema_field_count=len(schema), approved_asset_count=len(source_contract), comparison_level="B_only",
            excluded_task_ids=["F__human__absolute_oral"], source_registry_version="gate8_manuscript_evidence_registry_v2",
        )
    print(f"Gate 8 R2 Table 3 strict internal evidence: {output}")


if __name__ == "__main__":
    run_cli(main)
