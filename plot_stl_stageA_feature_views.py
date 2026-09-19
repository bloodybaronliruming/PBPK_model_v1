#!/usr/bin/env python3
"""Plot publication-ready Stage-A feature-view changes relative to RDKit2D."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def draw(table: pd.DataFrame) -> plt.Figure:
    labels = {"CL__human__systemic_iv": "CL", "CLint__human__microsome": "CLint",
              "Papp__human__caco2_ab": "Papp", "Thalf__human__terminal_iv": "Terminal t½",
              "VDss__human__steady_state_iv": "VDss", "fu__human__plasma": "fu"}
    order = list(labels)
    pivot = table.pivot(index="task_id", columns="feature_set", values="relative_change_vs_rdkit2d_percent").loc[order]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.78)
    y = np.arange(len(order)); offset = 0.16
    series = [("ecfp4", "ECFP4", "#F28E2B", -offset), ("ecfp4_rdkit2d", "ECFP4 + RDKit2D", "#4C78A8", offset)]
    for key, name, color, shift in series:
        values = pivot[key].to_numpy(float)
        ax.scatter(values, y + shift, s=45, color=color, label=name, zorder=3)
        for value, ypos in zip(values, y + shift):
            ax.text(value + (0.22 if value >= 0 else -0.22), ypos, f"{value:+.1f}%",
                    ha="left" if value >= 0 else "right", va="center", fontsize=8, color=color)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, [labels[x] for x in order])
    ax.invert_yaxis()
    ax.set_xlabel("Change in the endpoint primary error metric relative to RDKit2D (%)\nNegative values indicate lower error")
    fig.suptitle("Frozen-train Stage-A feature-view comparison", y=0.97, fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/stl_stageA_feature_view_comparison_v2")
    args = parser.parse_args()
    startup_self_check([args.audit / "complete.json", args.audit / "feature_view_best.csv"], output=args.output)
    meta = verify_stage(args.audit, "stl_stageA_three_view_audit")
    if meta.get("validation_labels_read") or meta.get("test_labels_read"):
        raise ValueError("Figure only accepts blinded Stage-A results")
    table = pd.read_csv(args.audit / "feature_view_best.csv")
    with stage_output(args.output) as out:
        fig = draw(table)
        fig.savefig(out / "Figure_STL_StageA_feature_views.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        (out / "README.md").write_text(
            "# STL Stage-A feature-view figure\n\nEnglish 600 dpi PNG. Values are frozen-train OOF comparisons, not fixed-validation or test results.\n",
            encoding="utf-8")
        finish_stage(out, "stl_stageA_feature_view_figure", inputs={
            "audit_complete_sha256": sha256(args.audit / "complete.json")
        }, dpi=600, language="English", validation_labels_read=False, test_labels_read=False, partial=False)
    print(f"STL Stage-A feature-view figure: {args.output}")


if __name__ == "__main__":
    run_cli(main)
