"""Assemble a read-only, publication-oriented performance benchmark scorecard.

This stage does not score a model.  It uses frozen summary assets to separate
direct public benchmarking, author-like context, strict internal comparisons,
and closed frozen-test diagnostics.  It never reads labels, prediction rows,
or models; it never recomputes metrics or changes candidate/test lifecycles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_performance_benchmark_scorecard"
OUTPUT = "results/analysis/gate8_performance_benchmark_scorecard_v2"
SOURCES = [
    ("Gate 4 candidate freeze", "data/public_development/cross_gate_candidate_freeze_v1", "cross_gate_endpoint_candidate_freeze"),
    ("Gate 8 Table 2", "data/public_development/gate8_table2_literature_boundaries_v1", "gate8_table2_literature_boundaries"),
    ("Gate 8 Table 3", "data/public_development/gate8_table3_strict_internal_v1", "gate8_table3_strict_internal"),
    ("Gate 8 Table 4", "data/public_development/gate8_table4_frozen_evidence_v1", "gate8_table4_frozen_evidence"),
    ("TDC Half-Life Obach", "results/benchmarks/tdc_half_life_obach_rdkit2d_et_v2", "tdc_half_life_obach_benchmark"),
]


def output_figure(path: Path, figure: plt.Figure) -> None:
    figure.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0, 0))
        if min(dpi) < 599 or image.width < 2000 or image.height < 1200:
            raise ValueError(f"Publication PNG validation failed for {path.name}: size={image.size}, dpi={dpi}")


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

    table2_path = root / "data/public_development/gate8_table2_literature_boundaries_v1/Table_2_Jia_literature_and_author_like_context.csv"
    table3_path = root / "data/public_development/gate8_table3_strict_internal_v1/Table_3_strict_internal_evidence.csv"
    table4_path = root / "data/public_development/gate8_table4_frozen_evidence_v1/Table_4_frozen_diagnostic_evidence.csv"
    tdc_path = root / "results/benchmarks/tdc_half_life_obach_rdkit2d_et_v2/aggregate_metrics.json"
    for path in (table2_path, table3_path, table4_path, tdc_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[str(path.resolve())] = sha256(path)
    table2 = pd.read_csv(table2_path)
    table3 = pd.read_csv(table3_path)
    table4 = pd.read_csv(table4_path)
    tdc = json.loads(tdc_path.read_text(encoding="utf-8"))

    expected_endpoints = {"Fu", "CL", "VDss"}
    metrics = {"GMFE", "within_2fold"}
    published = table2.loc[
        table2.comparison_id.eq("jia2025_published_literature_background")
        & table2.endpoint.isin(expected_endpoints) & table2.metric.isin(metrics),
        ["endpoint", "metric", "reported_value", "evidence_level", "comparison_id", "claim_guardrail"],
    ].rename(columns={"reported_value": "published_value", "evidence_level": "published_evidence_level"})
    author_like = table2.loc[
        table2.comparison_id.eq("jia2025_author_like_same_workbook_author_membership")
        & table2.endpoint.isin(expected_endpoints) & table2.metric.isin(metrics),
        ["endpoint", "metric", "reported_value", "evidence_level", "comparison_id", "claim_guardrail"],
    ].rename(columns={"reported_value": "author_like_value", "evidence_level": "author_like_evidence_level"})
    jia = published.merge(author_like, on=["endpoint", "metric"], validate="one_to_one", suffixes=("_published", "_author_like"))
    if len(jia) != 6 or set(jia.metric) != metrics or set(jia.endpoint) != expected_endpoints:
        raise ValueError("Jia scorecard must contain three endpoints and two frozen metrics")
    jia["comparison_statement"] = "B author-like context; published C-level value shown side-by-side, not ranked."

    tdc_rows = pd.DataFrame([
        ("Terminal t½", "TDC Half-Life_Obach", "Project RDKit2D ExtraTrees", "Spearman", float(tdc["mean_test_spearman"]), float(tdc["sd_test_spearman"]), "A", "Same official TDC data and split; direct public benchmark."),
        ("Terminal t½", "TDC Half-Life_Obach", "Published CFA reference", "Spearman", 0.576, 0.025, "A", "Published leaderboard reference under the same benchmark protocol."),
        ("Terminal t½", "TDC Half-Life_Obach", "Published MapLight reference", "Spearman", 0.562, 0.008, "A", "Published leaderboard reference under the same benchmark protocol."),
    ], columns=["endpoint", "benchmark", "model_or_reference", "metric", "estimate", "sd_or_reported_uncertainty", "comparison_level", "claim_guardrail"])

    strict_refs = table3.loc[table3.table_section.str.startswith("A."), [
        "task_id", "endpoint", "comparison_level", "analysis_scope", "primary_metric", "candidate_or_observed_value", "advance_or_reporting_decision", "claim_guardrail",
    ]].rename(columns={"candidate_or_observed_value": "strict_internal_reference_value"})
    if len(strict_refs) != 6 or not strict_refs.comparison_level.eq("B").all():
        raise ValueError("Strict internal scorecard must retain six B-level endpoint references")
    diagnostics = table4[[
        "task_id", "endpoint", "evidence_class", "evidence_lane", "test_lifecycle", "test_molecules", "primary_metric", "primary_estimate", "frozen_95ci", "ad_or_similarity", "calibration_status", "source_definition_limitation", "claim_guardrail",
    ]]
    endpoint_scorecard = diagnostics.merge(strict_refs, on="task_id", how="left", suffixes=("_frozen", "_strict"), validate="one_to_one")
    endpoint_scorecard["strict_internal_available"] = endpoint_scorecard.strict_internal_reference_value.notna()
    endpoint_scorecard["performance_interpretation"] = np.where(
        endpoint_scorecard.task_id.eq("F__human__absolute_oral"),
        "Exploratory status only; no numeric performance claim.",
        "Report frozen diagnostic separately from strict internal reference; no cross-lane ranking or post-test selection.",
    )
    if len(endpoint_scorecard) != 7 or endpoint_scorecard.task_id.duplicated().any():
        raise ValueError("Endpoint scorecard must contain exactly seven endpoint rows")

    architecture = table3.loc[table3.table_section.str.startswith(("B.", "C.")), [
        "task_id", "endpoint", "table_section", "comparison_level", "comparison_id", "primary_metric", "reference_value", "candidate_or_observed_value", "relative_change_pct", "paired_bootstrap_95ci", "advance_or_reporting_decision", "claim_guardrail",
    ]].copy()
    if len(architecture) != 12 or not architecture.comparison_level.eq("B").all():
        raise ValueError("Architecture scorecard must preserve six P1 and six B6 B-level rows")

    if args.check_only:
        print("Performance scorecard preflight: endpoints=7 strict_internal=6 Jia_cells=6 TDC_rows=3 architecture_rows=12; read-only summaries, no labels/predictions/models/metric recomputation.")
        return

    with stage_output(output) as folder:
        jia.to_csv(folder / "Table_1_Jia_author_like_context.csv", index=False)
        tdc_rows.to_csv(folder / "Table_2_TDC_direct_benchmark.csv", index=False)
        endpoint_scorecard.to_csv(folder / "Table_3_endpoint_evaluation_scope.csv", index=False)
        architecture.to_csv(folder / "Table_4_architecture_decision_scope.csv", index=False)

        plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "axes.labelweight": "normal"})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        colors = {"Project": "#2A6FBB", "Published": "#8A8A8A"}
        labels = ["Project\nExtraTrees", "CFA", "MapLight"]
        tdc_colors = [colors["Project"], colors["Published"], colors["Published"]]
        axes[0].bar(labels, tdc_rows.estimate, yerr=tdc_rows.sd_or_reported_uncertainty, capsize=4, color=tdc_colors, edgecolor="black", linewidth=0.6)
        axes[0].set_ylim(0, 0.70)
        axes[0].set_ylabel("Test Spearman")
        axes[0].set_title("A  Direct TDC benchmark")
        axes[0].grid(axis="y", alpha=0.25)
        for x, value in enumerate(tdc_rows.estimate):
            axes[0].text(x, value + 0.035, f"{value:.3f}", ha="center", va="bottom")

        order = ["Fu", "CL", "VDss"]
        width = 0.36
        x = np.arange(len(order))
        for axis, metric, ylabel, formatter, top in [
            (axes[1], "GMFE", "GMFE", lambda value: f"{value:.2f}", 2.6),
            (axes[2], "within_2fold", "Within two-fold (%)", lambda value: f"{100*value:.1f}", 80),
        ]:
            values = jia.loc[jia.metric.eq(metric)].set_index("endpoint").loc[order]
            left = values.published_value.to_numpy() * (100 if metric == "within_2fold" else 1)
            right = values.author_like_value.to_numpy() * (100 if metric == "within_2fold" else 1)
            axis.bar(x - width / 2, left, width, label="Published Jia (C)", color="#8A8A8A", edgecolor="black", linewidth=0.6)
            axis.bar(x + width / 2, right, width, label="Project author-like (B)", color="#2A6FBB", edgecolor="black", linewidth=0.6)
            axis.set_xticks(x, order)
            axis.set_ylim(0, top)
            axis.set_ylabel(ylabel)
            axis.set_title("B  Jia context: GMFE" if metric == "GMFE" else "C  Jia context: two-fold")
            axis.grid(axis="y", alpha=0.25)
            for offset, values_array in [(-width / 2, left), (width / 2, right)]:
                for position, value in zip(x + offset, values_array):
                    axis.text(position, value + top * 0.025, formatter(value / 100 if metric == "within_2fold" else value), ha="center", va="bottom", fontsize=8)
        axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
        fig.text(0.5, -0.08, "Panels B–C show C-level published values beside B-level author-like reruns for context only; they are not a direct rank.", ha="center", fontsize=9)
        output_figure(folder / "Figure_1_direct_and_author_like_benchmarks.png", fig)

        columns = ["Strict internal\nB reference", "Closed frozen\ndiagnostic", "Direct TDC\nA benchmark", "Jia author-like\nB context", "Exploratory\nonly"]
        endpoint_order = ["Plasma fu", "Microsomal CLint", "Caco-2 Papp A→B", "Systemic CL (IV)", "VDss (IV)", "Terminal t½ (IV)", "Absolute oral F"]
        display = endpoint_scorecard.set_index("endpoint_frozen").reindex(endpoint_order)
        matrix = np.zeros((len(display), len(columns)), dtype=int)
        matrix[:, 0] = display.strict_internal_available.astype(int)
        matrix[:, 1] = display.evidence_class.eq("not_comparison").astype(int)
        matrix[endpoint_order.index("Terminal t½ (IV)"), 2] = 1
        for endpoint in ["Plasma fu", "Systemic CL (IV)", "VDss (IV)"]:
            matrix[endpoint_order.index(endpoint), 3] = 1
        matrix[endpoint_order.index("Absolute oral F"), 4] = 1
        fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
        cmap = ListedColormap(["#F2F2F2", "#2A6FBB"])
        axis.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(columns)), columns)
        axis.set_yticks(range(len(endpoint_order)), endpoint_order)
        axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(endpoint_order), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, "Yes" if matrix[row, col] else "—", ha="center", va="center", color="white" if matrix[row, col] else "#4C4C4C", fontsize=10)
        axis.set_title("Endpoint-level evaluation evidence and boundaries")
        axis.set_xlabel("Evidence lane; columns are not performance ranks", labelpad=12)
        for spine in axis.spines.values():
            spine.set_visible(False)
        output_figure(folder / "Figure_2_endpoint_evidence_status.png", fig)

        report = """# Performance evaluation and benchmark scorecard

## Scope

This read-only package reports already frozen summary evidence. It does not load labels, prediction rows, model files, or test values, and it does not recompute any metric. Consequently it cannot alter candidate selection or a closed test lifecycle.

## Main interpretation

- **A-level direct public benchmark:** on the official TDC Half-Life_Obach protocol, the project RDKit2D ExtraTrees result is `0.587 ± 0.004` test Spearman. It can be compared directly with the frozen CFA (`0.576 ± 0.025`) and MapLight (`0.562 ± 0.008`) references for that benchmark only.
- **B-level Jia author-like context:** fu, CL and VDss values are shown adjacent to published Jia values only to document proximity under the author-like workbook membership. They are not an A-level replication, strict external validation, or a cross-protocol rank.
- **B-level strict internal evidence:** each numeric endpoint retains an endpoint-specific strong STL reference. P1 and B6 rows remain matched decision evidence, not a universal architecture tournament.
- **Closed diagnostics:** frozen test, AD and calibration summaries are descriptive and must not be reused for selection. Terminal t½ retains the TDC/Obach source-overlap limitation; F remains status-only exploratory.

## Files

- `Table_1_Jia_author_like_context.csv`: published C-level and author-like B-level context.
- `Table_2_TDC_direct_benchmark.csv`: direct TDC comparison rows.
- `Table_3_endpoint_evaluation_scope.csv`: strict-internal and frozen-diagnostic lanes kept separate.
- `Table_4_architecture_decision_scope.csv`: P1 and B6 matched decision evidence.
- `Figure_1_direct_and_author_like_benchmarks.png` and `Figure_2_endpoint_evidence_status.png`: English 600-dpi publication figures.
"""
        (folder / "README.md").write_text(report, encoding="utf-8")
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_models_loaded=True, no_test_labels_accessed=True,
            no_performance_metrics_computed=True, no_model_fitted=True, no_calibration=True,
            no_candidate_selection_changed=True, read_only_frozen_summary_evaluation=True,
            endpoint_count=len(endpoint_scorecard), strict_internal_endpoint_count=len(strict_refs),
            jia_context_cell_count=len(jia), tdc_direct_benchmark_row_count=len(tdc_rows),
            figure_count=2, figure_dpi=600,
        )
    print(f"Performance benchmark scorecard: {output}")


if __name__ == "__main__":
    run_cli(main)
