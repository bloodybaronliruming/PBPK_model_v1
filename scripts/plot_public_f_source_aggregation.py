#!/usr/bin/env python3
"""Render a publication-ready label-blinded public-F source-aggregation PNG."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def draw(summary: dict) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 11, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 1, figsize=(7.25, 5.65), constrained_layout=True, gridspec_kw={"height_ratios": [0.85, 1.15]})
    ax = axes[0]
    stages = ["Registered\nChEMBL records", "Quarantined:\nstrict-F structure", "Suppressed after\nprior definition audit", "Remaining records for\nsource retrieval"]
    values = [summary["chembl_input_records"], summary["chembl_quarantined_existing_strict_f_structure_records"],
              summary["chembl_previously_rejected_document_assay_records"], summary["chembl_eligible_for_new_source_metadata_retrieval_records"]]
    colors = ["#4C78A8", "#E15759", "#BDBDBD", "#59A14F"]
    bars = ax.bar(range(4), values, color=colors, width=0.58, edgecolor="none")
    ax.set_xticks(range(4), stages)
    ax.set_ylabel("Records")
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title("A. Label-blinded source-retrieval flow")
    ax.yaxis.grid(True, linewidth=0.5, color="#D9D9D9")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.035, f"n = {value}", ha="center", va="bottom")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    priority = [summary["p1_document_assay_units"], summary["p2_document_assay_units"], summary["p3_document_assay_units"], summary["p4_document_assay_units"]]
    labels = ["P1: IV+PO metadata", "P2: PO metadata", "P3: route unknown", "P4: incomplete metadata"]
    colors = ["#F28E2B", "#59A14F", "#4C78A8", "#BDBDBD"]
    positions = list(range(len(priority)))[::-1]
    bars = ax.barh(positions, priority, color=colors, height=0.62, edgecolor="none")
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, max(priority) * 1.18)
    ax.set_xlabel("Document × assay units; metadata priority does not prove compatible human absolute F")
    ax.set_title(f"B. Metadata retrieval priorities across {summary['chembl_source_documents']} source documents")
    ax.xaxis.grid(True, linewidth=0.5, color="#D9D9D9")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, priority):
        ax.text(value + max(priority) * 0.02, bar.get_y() + bar.get_height() / 2, f"n = {value}", ha="left", va="center")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation", type=Path, default=ROOT / "results/analysis/public_f_source_aggregation_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/public_f_source_aggregation_figure_v3")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    summary_file = args.aggregation / "summary.json"
    startup_self_check([summary_file, args.aggregation / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.aggregation, "public_f_source_aggregation")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary["label_blinded"] is not True or summary["numeric_values_read"] is not False:
        raise ValueError("Figure requires a label-blinded aggregation stage")
    if args.check_only:
        print(f"Public-F source-aggregation figure valid: units={summary['chembl_document_assay_units']}")
        return
    with stage_output(args.output) as out:
        figure = draw(summary)
        figure.savefig(out / "Figure_S_public_F_source_aggregation.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# Public-F source-aggregation figure\\n\\n"
            "Publication-ready English PNG at 600 dpi. The figure displays label-blinded record, document-assay, and source-document counts "
            "only. Priority labels are explicitly metadata-retrieval priorities and do not assert a compatible human absolute-F endpoint.\\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_source_aggregation_figure", inputs={"aggregation_complete_sha256": sha256(args.aggregation / "complete.json")},
                     dpi=600, language="English", label_blinded=True, endpoint_values_plotted=False, partial=False)
    print(f"Public-F source-aggregation figure: {args.output}")


if __name__ == "__main__":
    run_cli(main)
