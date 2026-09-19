"""Freeze the read-only decision boundary for a second DMPK model iteration.

This audit uses only completed Gate 4/7/8 summary artifacts.  It neither
loads labels, prediction rows, fitted models, nor re-computes performance.
Its purpose is to decide whether a previously stopped model family can be
reopened.  A closed/prohibited test or an all-negative architecture record is
an explicit block: the output then specifies the new evidence required before
any future reconstruction protocol can be authorized.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate9_second_iteration_readiness"
OUTPUT = "data/public_development/gate9_second_iteration_readiness_v2"
SOURCES = [
    ("Gate 4 candidate registry", "data/public_development/cross_gate_candidate_freeze_v1", "cross_gate_endpoint_candidate_freeze"),
    ("Gate 7 evidence matrix", "results/analysis/gate7_endpoint_evidence_matrix_v3", "gate7_endpoint_diagnostic_evidence_matrix"),
    ("Gate 8 evidence registry", "data/public_development/gate8_manuscript_evidence_registry_v2", "gate8_manuscript_evidence_registry"),
    ("Gate 8 Table 1", "data/public_development/gate8_table1_schema_v1", "gate8_table1_schema"),
    ("Gate 8 Table 2", "data/public_development/gate8_table2_literature_boundaries_v1", "gate8_table2_literature_boundaries"),
    ("Gate 8 Table 3", "data/public_development/gate8_table3_strict_internal_v1", "gate8_table3_strict_internal"),
    ("Gate 8 Table 4", "data/public_development/gate8_table4_frozen_evidence_v1", "gate8_table4_frozen_evidence"),
    ("Gate 8 supplement", "data/public_development/gate8_supplement_manifest_v1", "gate8_supplement_manifest"),
    ("Gate 8 figure contract", "data/public_development/gate8_figure_contract_v1", "gate8_figure_contract"),
    ("Gate 8 publication figures", "results/analysis/gate8_figure_composition_v1", "gate8_figure_composition"),
    ("Gate 8 manuscript outline", "进度/投稿草稿/gate8_manuscript_outline_v1", "gate8_manuscript_outline"),
]


NEXT_EVIDENCE = {
    "fu__human__plasma": (
        "new independent source-family holdout and pre-registered uncertainty-calibration development protocol",
        "Do not reopen model families; historical frozen test is closed and the current gap is validation/uncertainty evidence.",
    ),
    "CLint__human__microsome": (
        "new independent document/source-family cohort, kept outside all current train/validation/test registries",
        "Do not use the closed n=28 test; freeze source-definition and source-holdout protocol before any new train-CV reconstruction.",
    ),
    "Papp__human__caco2_ab": (
        "source/units re-audit plus a new independent source-family holdout cohort",
        "Do not reopen stopped architectures; resolve Caco-2 A→B source-definition variation before authorizing a matched reconstruction.",
    ),
    "CL__human__systemic_iv": (
        "new independent source-family holdout cohort with low-similarity coverage registered in advance",
        "Cross-species replacement remains stopped; a future protocol must retain the frozen human-only strong STL comparator.",
    ),
    "VDss__human__steady_state_iv": (
        "new independent source-family holdout cohort with low-similarity coverage registered in advance",
        "Cross-species replacement remains stopped; a future protocol must retain the frozen human-only strong STL comparator.",
    ),
    "Thalf__human__terminal_iv": (
        "new non-Obach/non-TDC source-family cohort and independent-source holdout protocol",
        "Do not treat the existing frozen result as independent external validation or use it for reconstruction selection.",
    ),
    "F__human__absolute_oral": (
        "independent human absolute-F oral+IV data acquisition and definition audit before any numeric development cohort",
        "No model reconstruction, numeric prediction, or test access is authorized while validation remains insufficient.",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT

    inputs: dict[str, str] = {}
    for _, relative, expected_stage in SOURCES:
        folder = root / relative
        verify_stage(folder, expected_stage)
        complete = folder / "complete.json"
        inputs[str(complete.resolve())] = sha256(complete)

    candidate_path = root / "data/public_development/cross_gate_candidate_freeze_v1/endpoint_candidate_freeze_registry.csv"
    family_path = root / "data/public_development/cross_gate_candidate_freeze_v1/architecture_family_decisions.csv"
    lifecycle_path = root / "data/public_development/cross_gate_candidate_freeze_v1/test_lifecycle_registry.csv"
    diagnostic_path = root / "results/analysis/gate7_endpoint_evidence_matrix_v3/endpoint_diagnostic_evidence_matrix.csv"
    candidates = pd.read_csv(candidate_path)
    families = pd.read_csv(family_path)
    lifecycle = pd.read_csv(lifecycle_path)
    diagnostics = pd.read_csv(diagnostic_path)
    if set(candidates.task_id) != set(lifecycle.task_id) or set(candidates.task_id) != set(diagnostics.task_id):
        raise ValueError("Gate 9 endpoint membership must agree across Gate 4 and Gate 7 summaries")
    if candidates.task_id.duplicated().any() or lifecycle.task_id.duplicated().any() or diagnostics.task_id.duplicated().any():
        raise ValueError("Gate 9 requires one summary row per task")
    if not lifecycle.further_selection_allowed.eq(False).all():
        raise ValueError("Gate 9 must not proceed if an existing test allows further selection")
    if not diagnostics.test_lifecycle.isin({"test_consumed_no_reselection", "test_prohibited_insufficient_validation"}).all():
        raise ValueError("Gate 7 current lifecycle must close or prohibit every endpoint")
    if not families.advanced.eq(False).all():
        raise ValueError("A previously advanced architecture requires a separate protocol, not this blocked-family audit")

    family_map = families.groupby("endpoint").family.agg(lambda x: " | ".join(sorted(x))).to_dict()
    joined = candidates.merge(
        lifecycle[["task_id", "test_status", "further_selection_allowed", "allowed_next_action"]].rename(
            columns={"test_status": "gate4_historical_test_status"}
        ), on="task_id", validate="one_to_one"
    ).merge(
        diagnostics[["task_id", "test_lifecycle", "source_or_definition_limitation", "uncertainty_release_status"]].rename(
            columns={"test_lifecycle": "gate7_current_test_lifecycle"}
        ), on="task_id", validate="one_to_one"
    )
    rows = []
    for row in joined.itertuples(index=False):
        required_evidence, guardrail = NEXT_EVIDENCE[row.task_id]
        endpoint_families = family_map.get(row.endpoint, "no numeric model family authorized")
        rows.append({
            "task_id": row.task_id,
            "endpoint": row.endpoint,
            "current_reference": row.unified_research_reference,
            "gate4_historical_test_status": row.gate4_historical_test_status,
            "current_test_lifecycle": row.gate7_current_test_lifecycle,
            "lifecycle_reconciliation": "Gate 7 current matrix supersedes Gate 4 historical status where they differ.",
            "further_selection_allowed": bool(row.further_selection_allowed),
            "previously_stopped_or_unavailable_families": endpoint_families,
            "current_limitation": row.source_or_definition_limitation,
            "second_iteration_status": "blocked_pending_new_independent_evidence",
            "required_new_evidence_before_protocol": required_evidence,
            "matched_baseline_if_authorized": "Frozen endpoint-specific strong STL; retain same task definition and group-isolation contract.",
            "maximum_initial_budget_if_authorized": "Protocol + one train-only smoke; no formal reconstruction until new cohort/split and leakage audit pass.",
            "advance_threshold_if_authorized": "Pre-register matched >=2% primary-metric improvement, >=3/5 folds improved, paired parent bootstrap support, and no unacceptable source/low-similarity regression.",
            "hard_stop": guardrail,
        })
    audit = pd.DataFrame(rows)
    if len(audit) != 7 or not audit.second_iteration_status.eq("blocked_pending_new_independent_evidence").all():
        raise ValueError("Gate 9 audit must preserve all seven endpoints as evidence-gated")

    if args.check_only:
        print("Gate 9 R0 preflight: verified_stages=11 endpoints=7 stopped_architecture_rows=" f"{len(families)}; no labels, predictions, models, test values, or performance recomputation.")
        return

    with stage_output(output) as folder:
        audit.to_csv(folder / "second_iteration_decision_register.csv", index=False)
        pd.DataFrame(SOURCES, columns=["source_role", "source_path", "expected_stage"]).to_csv(
            folder / "verified_source_stages.csv", index=False
        )
        summary = """# Gate 9 R0: second-iteration readiness audit

## Decision

No endpoint is authorized for immediate model reconstruction. Every existing test is consumed or prohibited for further selection, and all tested complex architecture families in the Gate 4 decision registry are non-advanced. Re-running them against existing folds or historical frozen tests would not create independent evidence.

## Permitted path

For each endpoint, first obtain the new independent evidence named in `second_iteration_decision_register.csv`, freeze a new source-aware cohort/split protocol, and complete a train-only leakage smoke. Only then may a bounded, matched reconstruction protocol be proposed. The frozen endpoint-specific strong STL remains the required comparator.

## Scope

This is a decision register, not a performance analysis. It reads completed summary artifacts only and does not access labels, prediction rows, fitted models, test values, or MMPK numerical material. It does not recompute any metric or authorize training. Gate 7 is the current authority for test lifecycle; the register retains the older Gate 4 field solely to make the CLint timeline reconciliation explicit.
"""
        (folder / "README.md").write_text(summary, encoding="utf-8")
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_models_loaded=True, no_test_values_accessed=True,
            no_performance_metrics_computed=True, no_model_fitted=True, no_calibration=True,
            no_candidate_selection_changed=True, no_mmpk_numeric_values_accessed=True,
            verified_stage_count=len(SOURCES), endpoint_count=len(audit), stopped_architecture_row_count=len(families),
            immediate_reconstruction_authorized=False,
        )
    print(f"Gate 9 R0 second-iteration readiness: {output}")


if __name__ == "__main__":
    run_cli(main)
