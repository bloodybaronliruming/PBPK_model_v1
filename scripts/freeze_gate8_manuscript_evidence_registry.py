"""Freeze the allowed evidence inputs and assembly order for Gate 8 manuscript assets.

This is a registry-only stage.  It validates completed upstream stages and
hashes their approved summary/figure artifacts, but never reads labels,
prediction rows, models, or recomputes metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_manuscript_evidence_registry"

SOURCES = [
    {
        "source_id": "gate4_candidate_freeze",
        "relative": "data/public_development/cross_gate_candidate_freeze_v1",
        "stage": "cross_gate_endpoint_candidate_freeze",
        "artifacts": ["endpoint_candidate_freeze_registry.csv", "Figure_gate4_endpoint_decisions.png"],
        "evidence_class": "B",
        "usage": "Frozen endpoint-specific candidate and decision lineage; not a new performance comparison.",
    },
    {
        "source_id": "p1_matched_stl",
        "relative": "results/analysis/stl_p1_matched_comparison_v1",
        "stage": "stl_p1_matched_comparison",
        "artifacts": ["overall_matched_comparison.csv", "Figure_P1_vs_matched_control.png"],
        "evidence_class": "B",
        "usage": "Matched strict internal strong-STL comparison for main Table 3/Figure 2.",
    },
    {
        "source_id": "multitask_gradient_report",
        "relative": "results/analysis/multitask_gradient_affinity_report_v2",
        "stage": "multitask_gradient_affinity_report",
        "artifacts": ["endpoint_pair_affinity_summary.csv", "Figure_multitask_gradient_affinity_report.png"],
        "evidence_class": "B",
        "usage": "Mechanistic task-conflict diagnostic; association only, not causal evidence.",
    },
    {
        "source_id": "cross_species_cl",
        "relative": "results/analysis/cl_cross_species_transfer_traincv_analysis_v2",
        "stage": "cross_species_transfer_traincv_analysis",
        "artifacts": ["decision_registry.csv", "Figure_CL_transfer_trainCV.png"],
        "evidence_class": "B",
        "usage": "Matched human-only versus transfer ablation for CL.",
    },
    {
        "source_id": "cross_species_fu",
        "relative": "results/analysis/fu_cross_species_transfer_traincv_analysis_v2",
        "stage": "cross_species_transfer_traincv_analysis",
        "artifacts": ["decision_registry.csv", "Figure_fu_transfer_trainCV.png"],
        "evidence_class": "B",
        "usage": "Matched human-only versus transfer ablation for fu.",
    },
    {
        "source_id": "cross_species_vdss",
        "relative": "results/analysis/vdss_cross_species_transfer_traincv_analysis_v1",
        "stage": "vdss_cross_species_transfer_traincv_analysis",
        "artifacts": ["decision_registry.csv", "Figure_VDss_transfer_trainCV.png"],
        "evidence_class": "B",
        "usage": "Matched human-only versus transfer ablation for VDss.",
    },
    {
        "source_id": "cross_species_thalf",
        "relative": "results/analysis/thalf_cross_species_transfer_traincv_analysis_v2",
        "stage": "cross_species_transfer_traincv_analysis",
        "artifacts": ["decision_registry.csv", "Figure_Thalf_transfer_trainCV.png"],
        "evidence_class": "B",
        "usage": "Matched human-only versus transfer ablation for terminal t½.",
    },
    {
        "source_id": "gate3_b6_formal",
        "relative": "results/analysis/gate3_b6_physchem_formal_analysis_v2",
        "stage": "gate3_b6_physchem_formal_analysis",
        "artifacts": ["decision.json", "table_7_endpoint_final_decisions.csv", "figure_2_paired_bootstrap_forest.png"],
        "evidence_class": "B",
        "usage": "Formal negative nested experimental-physchem ablation; 0/6 advancement is reportable.",
    },
    {
        "source_id": "clint_gate5_closeout",
        "relative": "results/analysis/gate5_clint_final_diagnostic_closeout_v1",
        "stage": "gate5_clint_final_diagnostic_closeout",
        "artifacts": ["traincv_test_comparison.csv", "calibration_summary.csv", "Figure_2_CLint_AD_and_source_diagnostics.png"],
        "evidence_class": "not_comparison",
        "usage": "One-time frozen CLint test diagnostic; no post-test selection or calibration.",
    },
    {
        "source_id": "frozen_test_diagnostics",
        "relative": "results/final/frozen_test_diagnostics_v1",
        "stage": "frozen_test_diagnostics",
        "artifacts": ["table_1_bootstrap_primary_ci.csv", "table_3_applicability_domain.csv", "figure_3_applicability_domain.png"],
        "evidence_class": "not_comparison",
        "usage": "Frozen-test diagnostics and applicability-domain limitations; do not reuse for selection.",
    },
    {
        "source_id": "jia_author_like_closeout",
        "relative": "results/analysis/jia2025_author_like_r3c_closeout_v2",
        "stage": "jia2025_author_like_r3c_read_only_closeout",
        "artifacts": ["published_vs_author_like_primary.csv", "project_strict_protocol_context.csv", "Figure_2_Jia2025_evidence_boundaries.png"],
        "evidence_class": "multiple_levels_require_separate_rows",
        "usage": "Author-like replication must occupy B-level rows; published paper values must occupy separate C-level background rows unless an exact A-level protocol is demonstrated.",
    },
    {
        "source_id": "mmpk_n2c_internal",
        "relative": "results/analysis/mmpk_n2c_paired_sensitivity_analysis_v1",
        "stage": "mmpk_n2c_paired_sensitivity_readonly_analysis",
        "artifacts": ["sensitivity_signal_decision.csv", "pooled_paired_metrics.csv", "Figure_1_MMPK_N2c_paired_bootstrap.png"],
        "evidence_class": "internal_only_not_public_comparison",
        "usage": "Internal MMPK strict sensitivity evidence; do not publish a comparison row, row-level data, predictions, or a redistributable model without data rights.",
    },
    {
        "source_id": "gate7_evidence_matrix",
        "relative": "results/analysis/gate7_endpoint_evidence_matrix_v3",
        "stage": "gate7_endpoint_diagnostic_evidence_matrix",
        "artifacts": ["endpoint_diagnostic_evidence_matrix.csv", "manuscript_figure_plan.csv", "Figure_1_gate7_evidence_matrix.png"],
        "evidence_class": "not_comparison",
        "usage": "Cross-endpoint lifecycle, AD, uncertainty and claim-boundary registry.",
    },
    {
        "source_id": "gate7_reproducibility",
        "relative": "data/public_development/gate7_reproducibility_delivery_manifest_v8",
        "stage": "gate7_reproducibility_delivery_manifest",
        "artifacts": ["delivery_manifest.json", "runtime_versions.json", "clean_environment_reload_protocol.md"],
        "evidence_class": "not_comparison",
        "usage": "Environment and code/data provenance for Supplementary reproducibility materials.",
    },
    {
        "source_id": "gate7_clean_reload",
        "relative": "results/analysis/gate7_clean_environment_reload_verification_v3",
        "stage": "gate7_clean_environment_reload_verification",
        "artifacts": ["verification_checks.json"],
        "evidence_class": "not_comparison",
        "usage": "Independent clean-environment technical replay only; not model performance evidence.",
    },
]

MANUSCRIPT_ITEMS = [
    ("Table 1", "Dataset/task definitions and split/source context", "gate7_evidence_matrix;gate4_candidate_freeze", "not_comparison", "Do not infer sample counts from filenames or mix assay definitions."),
    ("Table 2", "Jia/MMPK literature context and author-like/internal replication boundaries", "jia_author_like_closeout;mmpk_n2c_internal", "not_comparison", "Each future quantitative row must be labelled exactly A, B, or C; keep author-like B rows and published C rows separate, while MMPK remains internal-only."),
    ("Table 3", "Strict internal strong-STL, candidate and negative-ablation evidence", "p1_matched_stl;gate3_b6_formal;gate4_candidate_freeze", "B", "No cross-protocol numeric ranking or reopening stopped architectures."),
    ("Table 4", "Frozen test, AD, calibration and engineering practicality", "clint_gate5_closeout;frozen_test_diagnostics;gate7_evidence_matrix;gate7_clean_reload", "not_comparison", "Describe lifecycle and limitations; do not select or calibrate with consumed tests."),
    ("Figure 1", "Workflow, governance, split and evidence lanes", "gate4_candidate_freeze;gate7_evidence_matrix", "not_comparison", "Separate historical test-locked assets from strict train-CV reference lanes."),
    ("Figure 2", "Strong STL matched baseline and endpoint-specific decisions", "p1_matched_stl;gate4_candidate_freeze", "B", "Show paired/internal evidence, not a universal algorithm champion."),
    ("Figure 3", "Multitask conflict/negative-transfer diagnostic", "multitask_gradient_report", "B", "Gradient affinity is associative, not causal."),
    ("Figure 4", "Controlled cross-species transfer ablations", "cross_species_cl;cross_species_fu;cross_species_vdss;cross_species_thalf", "B", "Always compare against matched human-only and strong STL controls."),
    ("Figure 5", "Nested experimental-physchem (B6) negative result", "gate3_b6_formal", "B", "Report the stop decision and no advancement."),
    ("Figure 6", "Frozen performance/AD limitations and release evidence status", "clint_gate5_closeout;frozen_test_diagnostics;gate7_evidence_matrix;gate7_reproducibility;gate7_clean_reload", "not_comparison", "Do not state independent external validation where overlap is documented."),
    ("Supplement", "Foldwise values, bootstrap, exclusions, provenance and environment", "p1_matched_stl;gate3_b6_formal;jia_author_like_closeout;mmpk_n2c_internal;gate7_reproducibility", "not_comparison", "Each comparison table row must be labelled exactly A, B, or C; MMPK row-level outputs remain excluded pending data rights."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "data/public_development/gate8_manuscript_evidence_registry_v2"

    source_rows = []
    inputs = {}
    for source in SOURCES:
        folder = root / source["relative"]
        verify_stage(folder, source["stage"])
        complete = folder / "complete.json"
        inputs[str(complete.resolve())] = sha256(complete)
        for artifact in source["artifacts"]:
            path = folder / artifact
            if not path.is_file():
                raise FileNotFoundError(f"Gate 8 evidence artifact missing: {path}")
            source_rows.append({
                "source_id": source["source_id"], "source_path": str(folder.resolve()), "source_stage": source["stage"],
                "artifact": artifact, "artifact_sha256": sha256(path), "evidence_class": source["evidence_class"],
                "allowed_usage": source["usage"],
            })
    if args.check_only:
        print(f"Gate 8 R0 preflight: sources={len(SOURCES)} artifacts={len(source_rows)}; no labels, prediction rows, models, or metrics read.")
        return
    items = pd.DataFrame(MANUSCRIPT_ITEMS, columns=["manuscript_item", "purpose", "source_ids", "comparison_level", "claim_guardrail"])
    with stage_output(output) as folder:
        pd.DataFrame(source_rows).to_csv(folder / "source_artifact_manifest.csv", index=False)
        items.to_csv(folder / "manuscript_item_plan.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R0 manuscript evidence registry\n\n"
            "This registry freezes the allowed upstream summary/figure artifacts and assembly order for manuscript tables, figures "
            "and supplement. It hashes files but does not read labels, prediction rows or models, recompute metrics, train, calibrate "
            "or select candidates. Comparison classes must remain visible in all later outputs.\n",
            encoding="utf-8",
        )
        finish_stage(folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
                     no_prediction_rows_accessed=True, no_performance_metrics_computed=True, no_model_fitted=True,
                     no_calibration=True, no_candidate_selection_changed=True, frozen_source_count=len(SOURCES),
                     frozen_artifact_count=len(source_rows), manuscript_item_count=len(items))
    print(f"Gate 8 R0 manuscript evidence registry: {output}")


if __name__ == "__main__":
    run_cli(main)
