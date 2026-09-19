#!/usr/bin/env python3
"""Create a publication-ready, label-blinded PKSmart candidate-triage PNG.

This is an evidence-flow figure, not a model-performance plot. It uses no
PKSmart endpoint values: all counts come from released stage metadata,
candidate status summaries, and the audited IV-product lead manifest.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from render_pksmart_candidate_funnel import counts


STAGES = [
    ("Public external CSV rows", "public_rows", "#4C78A8"),
    ("Human half-life rows (label-blinded)", "human_rows", "#4C78A8"),
    ("After v15/TDC structure exclusion", "independent", "#4C78A8"),
    ("Named retrieval candidates", "named", "#59A14F"),
    ("Documented IV product leads", "iv_product_leads", "#F28E2B"),
    ("Formal external labels", "formal_labels", "#BDBDBD"),
]
GATES = [
    "Structure and source independence",
    "Row-level primary study locator",
    "Human direct intravenous administration",
    "Systemic parent-analyte measurement",
    "Terminal phase and an auditable point estimate",
]


def figure_table(values: dict[str, int]) -> pd.DataFrame:
    total = values["public_rows"]
    return pd.DataFrame([
        {"stage": label, "count": values[key], "fraction_of_public_rows": values[key] / total,
         "color": color}
        for label, key, color in STAGES
    ])


def draw(table: pd.DataFrame):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 11, "axes.titleweight": "bold",
    })
    figure = plt.figure(figsize=(7.25, 6.35), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    ax = figure.add_subplot(grid[0])
    rows = table.iloc[::-1].reset_index(drop=True)
    ax.barh(rows.index, rows["count"], color=rows["color"], height=0.62, edgecolor="none")
    ax.set_yticks(rows.index, rows["stage"])
    ax.set_xlim(0, max(330, int(table["count"].max() * 1.16)))
    ax.set_xlabel("Candidate records")
    ax.set_title("A. Structure-isolated candidate triage")
    ax.xaxis.grid(True, linewidth=0.5, color="#D9D9D9")
    ax.set_axisbelow(True)
    for row in rows.itertuples(index=True):
        ax.text(row.count + 5, row.Index, f"n = {row.count}", va="center", ha="left", fontsize=9)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    gate_ax = figure.add_subplot(grid[1])
    gate_ax.set_title("B. Evidence gates before formal-registry entry", loc="left")
    gate_ax.set_xlim(0, 1)
    gate_ax.set_ylim(0, len(GATES) + 0.6)
    gate_ax.axis("off")
    for index, gate in enumerate(GATES):
        y = len(GATES) - index - 0.12
        patch = FancyBboxPatch((0.08, y - 0.34), 0.84, 0.56, boxstyle="round,pad=0.012,rounding_size=0.025",
                               facecolor="#F2F2F2", edgecolor="#BDBDBD", linewidth=0.7)
        gate_ax.add_patch(patch)
        gate_ax.text(0.115, y - 0.05, f"{index + 1}", va="center", ha="center", fontsize=9, fontweight="bold",
                     color="#FFFFFF", bbox={"boxstyle": "circle,pad=0.30", "facecolor": "#4C78A8", "edgecolor": "none"})
        gate_ax.text(0.17, y - 0.05, gate, va="center", ha="left", fontsize=9)
        if index < len(GATES) - 1:
            gate_ax.annotate("", xy=(0.50, y - 0.55), xytext=(0.50, y - 0.39),
                             arrowprops={"arrowstyle": "-|>", "color": "#7F7F7F", "lw": 0.7})
    figure.text(0.5, 0.005,
                "Public-database records and regulatory labels are discovery aids under the current protocol; "
                "they do not become formal labels without all five gates.",
                ha="center", va="bottom", fontsize=8)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/raw/External_test_315.csv")
    parser.add_argument("--queue", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_candidate_queue_v1")
    parser.add_argument("--mapping", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_chembl_mapping_v2")
    parser.add_argument("--iv-product-leads", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/iv_product_leads_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_candidate_funnel_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.queue, "pksmart_external_candidate_queue")
    verify_stage(args.mapping, "pksmart_candidate_chembl_mapping")
    startup_self_check([args.raw, args.queue / "screening_summary.csv", args.mapping / "candidate_mapping_blinded.csv",
                        args.iv_product_leads], output=None if args.check_only else args.output)
    values = counts(args.raw, args.queue, args.mapping, args.iv_product_leads)
    table = figure_table(values)
    if args.check_only:
        print(table[["stage", "count"]].to_string(index=False))
        return
    with stage_output(args.output) as out:
        table.to_csv(out / "candidate_funnel_counts.csv", index=False)
        figure = draw(table)
        figure.savefig(out / "Figure_S1_pksmart_candidate_triage.png", dpi=600,
                       bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# PKSmart candidate-triage figure\n\n"
            "All labels are English and directly attached to their data; no legend can obscure a mark. "
            "The PNG is 600 dpi. The figure is a candidate/evidence-flow summary only, and no PKSmart "
            "endpoint values are read or displayed.\n", encoding="utf-8")
        finish_stage(out, "pksmart_candidate_funnel_figure", inputs={
            "raw_csv": sha256(args.raw), "candidate_queue_complete": sha256(args.queue / "complete.json"),
            "mapping_complete": sha256(args.mapping / "complete.json"), "iv_product_leads": sha256(args.iv_product_leads),
        }, label_blinded=True, formal_external_labels=0, candidates=values["independent"], partial=False)


if __name__ == "__main__":
    run_cli(main)
