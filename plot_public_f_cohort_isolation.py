#!/usr/bin/env python3
"""Render a publication-ready public-F cohort isolation audit PNG."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


GATES = [
    ("overlap_strict_f_parent", "Strict-F\nparent"), ("overlap_strict_f_scaffold", "Strict-F\nscaffold"),
    ("overlap_strict_f_source", "Strict-F\nsource"), ("overlap_boulton_parent", "Boulton\nparent"),
    ("overlap_boulton_scaffold", "Boulton\nscaffold"), ("overlap_boulton_source", "Boulton\nsource"),
]


def draw(records: pd.DataFrame, summary: dict) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 11, "axes.titleweight": "bold"})
    figure = plt.figure(figsize=(7.25, 5.85), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[0.76, 1.24])
    top = figure.add_subplot(grid[0])
    counts = [summary["strict_f_molecules_checked"], summary["boulton_reserved_molecules_checked"], summary["molecules"]]
    labels = ["Frozen strict-F\ncohort", "Reserved Boulton\nincrement", "New public-F\ncandidates"]
    colors = ["#4C78A8", "#F28E2B", "#59A14F"]
    bars = top.bar(np.arange(3), counts, color=colors, width=0.58, edgecolor="none")
    top.set_xticks(np.arange(3), labels)
    top.set_ylabel("Molecules")
    top.set_ylim(0, max(counts) * 1.28)
    top.set_title("A. Evidence layers kept separate")
    top.yaxis.grid(True, linewidth=0.5, color="#D9D9D9")
    top.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        top.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.04, f"n = {count}", ha="center", va="bottom")
    for spine in ["top", "right"]:
        top.spines[spine].set_visible(False)

    matrix = figure.add_subplot(grid[1])
    values = records[[column for column, _ in GATES]].astype(bool).to_numpy()
    matrix.imshow(values, cmap=matplotlib.colors.ListedColormap(["#59A14F", "#D62728"]), vmin=0, vmax=1, aspect="auto")
    matrix.set_xticks(np.arange(len(GATES)), [label for _, label in GATES])
    matrix.set_yticks(np.arange(len(records)), records["analyte"].str.capitalize())
    matrix.set_title("B. Triple-isolation audit: parent, scaffold, and source-cluster overlap")
    matrix.set_xticks(np.arange(-0.5, len(GATES), 1), minor=True)
    matrix.set_yticks(np.arange(-0.5, len(records), 1), minor=True)
    matrix.grid(which="minor", color="white", linewidth=2)
    matrix.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            matrix.text(column, row, "None" if not values[row, column] else "Overlap", ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/literature/public_f_cohort_isolation_view_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/public_f_cohort_isolation_figure_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    records_file, summary_file = args.cohort / "cohort_records.csv", args.cohort / "summary.json"
    startup_self_check([records_file, summary_file, args.cohort / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.cohort, "public_f_cohort_isolation_view")
    records, summary = pd.read_csv(records_file), json.loads(summary_file.read_text(encoding="utf-8"))
    if records[[column for column, _ in GATES]].any(axis=None):
        raise ValueError("Isolation figure refuses to render a cohort with unresolved overlap")
    if args.check_only:
        print(f"Public-F isolation figure valid: records={len(records)}")
        return
    with stage_output(args.output) as out:
        figure = draw(records, summary)
        figure.savefig(out / "Figure_S_public_F_triple_isolation.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# Public-F triple-isolation audit figure\\n\\n"
            "Publication-ready English PNG at 600 dpi. Labels are placed in the matrix cells and no floating legend covers data. "
            "It visualizes evidence-layer counts and six pre-specified overlap gates only; it contains no model prediction or performance claim.\\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_cohort_isolation_figure", inputs={"cohort_complete_sha256": sha256(args.cohort / "complete.json")},
                     records=len(records), dpi=600, language="English", endpoint_values_plotted=False, partial=False)
    print(f"Public-F triple-isolation figure: {args.output}")


if __name__ == "__main__":
    run_cli(main)
