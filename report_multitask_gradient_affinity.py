#!/usr/bin/env python3
"""Summarize a locked multitask gradient-affinity run without refitting models."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


CLASS_COLORS = {"supportive": "#3B82A0", "unstable": "#B8B8B8", "conflicting": "#C44E52"}
CLASS_ORDER = ["supportive", "unstable", "conflicting"]


def short_label(task: str) -> str:
    if task.startswith("OneADMET__"):
        value = task.split("__", 1)[1].replace("_Public", "")
        value = value.replace("Clearance-", "CL ").replace("Half-Life_", "t½ ")
        value = value.replace("-pmL%min%kg", "").replace("-pHours", "")
        return "OA | " + value.replace("_", " ")
    parts = task.split("__")
    return " | ".join(parts).replace("systemic_iv", "systemic IV").replace(
        "steady_state_iv", "steady-state IV").replace("terminal_iv", "terminal IV"
    ).replace("absolute_oral", "absolute oral").replace("caco2_ab", "Caco-2 A→B")


def build_figure(human: pd.DataFrame, tasks: list[str], human_tasks: list[str], counts: pd.DataFrame) -> tuple[plt.Figure, float]:
    matrix = human.pivot(index="human_task", columns="partner_task", values="median_cosine").reindex(
        index=human_tasks, columns=tasks
    )
    values = matrix.to_numpy(float)
    for row_index, task in enumerate(human_tasks):
        if task in tasks:
            values[row_index, tasks.index(task)] = np.nan
    finite = np.abs(values[np.isfinite(values)])
    color_limit = max(0.05, math.ceil(float(finite.max()) * 100.0) / 100.0)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    figure = plt.figure(figsize=(15.8, 8.6))
    grid = figure.add_gridspec(2, 1, height_ratios=[4.7, 1.7], hspace=1.18)
    heat = figure.add_subplot(grid[0, 0])
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#F2F2F2")
    image = heat.imshow(values, vmin=-color_limit, vmax=color_limit, cmap=cmap, aspect="auto")
    row_labels = ["Terminal t½" if x.startswith("Thalf__") else x.split("__", 1)[0] for x in human_tasks]
    heat.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    heat.set_xticks(range(len(tasks)), [short_label(x) for x in tasks], rotation=90, fontsize=5.3)
    heat.set_xlabel("Partner task")
    heat.set_ylabel("Human evaluation task")
    heat.set_title("A  Off-diagonal shared-encoder task-gradient affinity", loc="left", fontweight="bold", fontsize=10)
    colorbar = figure.colorbar(image, ax=heat, fraction=0.018, pad=0.012)
    colorbar.set_label("Median gradient cosine across 15 fold–seed contexts")
    for row_index, task in enumerate(human_tasks):
        if task in tasks:
            heat.text(tasks.index(task), row_index, "×", ha="center", va="center", fontsize=9, color="#555555")

    bars = figure.add_subplot(grid[1, 0])
    draw = counts.set_index("human_task").reindex(human_tasks).fillna(0)
    x = np.arange(len(human_tasks))
    bottom = np.zeros(len(human_tasks))
    for classification in CLASS_ORDER:
        height = draw[classification].to_numpy(int)
        bars.bar(x, height, bottom=bottom, color=CLASS_COLORS[classification], width=0.68, label=classification.capitalize())
        for index, value in enumerate(height):
            if value:
                bars.text(index, bottom[index] + value / 2, str(value), ha="center", va="center", fontsize=7,
                          color="white" if classification != "unstable" else "#222222")
        bottom += height
    bars.set_xticks(x, row_labels)
    bars.set_ylabel("Partner tasks (n)")
    bars.set_title("B  Stable-sign classification counts (self-pairs excluded)", loc="left", fontweight="bold", fontsize=10)
    bars.set_ylim(0, max(bottom) * 1.18)
    bars.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    bars.set_axisbelow(True)
    bars.spines[["top", "right"]].set_visible(False)
    bars.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.03), frameon=False)
    figure.suptitle("Train-only multitask gradient-affinity diagnostic", fontweight="bold", fontsize=12, y=0.995)
    return figure, color_limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinity", type=Path, default=ROOT / "results/analysis/multitask_gradient_affinity_v1")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/multitask_gradient_affinity_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/multitask_gradient_affinity_report_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.affinity / "complete.json", args.affinity / "human_task_affinity_summary.csv",
                args.affinity / "pairwise_affinity_summary.csv", args.affinity / "task_gradient_norm_summary.csv",
                args.protocol / "complete.json", args.protocol / "protocol.json", args.protocol / "task_registry.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    affinity_meta = verify_stage(args.affinity, "multitask_gradient_affinity")
    protocol_meta = verify_stage(args.protocol, "multitask_gradient_affinity_protocol")
    if not affinity_meta.get("full_configuration") or affinity_meta.get("partial"):
        raise ValueError("Only a full formal gradient-affinity run may be reported")
    prohibited = ["evaluation_labels_read", "validation_target_file_opened", "test_labels_read", "model_fitted"]
    if any(affinity_meta.get(key, False) for key in prohibited) or protocol_meta.get("test_labels_read", False):
        raise ValueError("Gradient-affinity input violates the frozen label lifecycle")
    contract = json.loads((args.protocol / "protocol.json").read_text(encoding="utf-8"))
    tasks, human_tasks = list(map(str, contract["tasks"])), list(map(str, contract["human_tasks"]))
    if args.check_only:
        print(f"Gradient-affinity report ready: tasks={len(tasks)} human_tasks={len(human_tasks)}")
        return

    human = pd.read_csv(args.affinity / "human_task_affinity_summary.csv")
    pairs = pd.read_csv(args.affinity / "pairwise_affinity_summary.csv")
    norms = pd.read_csv(args.affinity / "task_gradient_norm_summary.csv")
    registry = pd.read_csv(args.protocol / "task_registry.csv")
    if human.duplicated(["human_task", "partner_task"]).any() or len(pairs) != len(tasks) * (len(tasks) + 1) // 2:
        raise ValueError("Affinity table key coverage is invalid")

    human_off = human.loc[human.human_task.ne(human.partner_task)].copy()
    count_table = (human_off.groupby(["human_task", "classification"]).size().unstack(fill_value=0)
                   .reindex(columns=CLASS_ORDER, fill_value=0).reset_index())
    endpoint = dict(zip(registry.task_id, registry.endpoint, strict=True))
    pair_off = pairs.loc[pairs.task_a.ne(pairs.task_b)].copy()
    pair_off["endpoint_a"] = pair_off.task_a.map(endpoint)
    pair_off["endpoint_b"] = pair_off.task_b.map(endpoint)
    pair_off["endpoint_pair"] = pair_off.apply(lambda row: " × ".join(sorted([row.endpoint_a, row.endpoint_b])), axis=1)
    endpoint_summary = (pair_off.groupby("endpoint_pair", as_index=False)
                        .agg(task_pairs=("classification", "size"), median_cosine=("median_cosine", "median"),
                             supportive=("classification", lambda x: int((x == "supportive").sum())),
                             unstable=("classification", lambda x: int((x == "unstable").sum())),
                             conflicting=("classification", lambda x: int((x == "conflicting").sum())))
                        .sort_values(["conflicting", "median_cosine"], ascending=[False, True]))
    norm_max = float(norms.median_norm_ratio.max())
    if int(norms.dominance_contexts.sum()) != 0:
        raise ValueError("Unexpected >10× gradient norm dominance requires a separate decision")
    figure, color_limit = build_figure(human, tasks, human_tasks, count_table)

    fu_row = count_table.loc[count_table.human_task.eq("fu__human__plasma")].iloc[0]
    report = (
        "# Formal multitask gradient-affinity result\n\n"
        "This report is a read-only post-analysis of the locked train-only diagnostic. It does not refit a model, "
        "read validation/test labels, or authorize architecture selection.\n\n"
        "The shared encoder shows broadly stable positive directions for CL, CLint, Papp, terminal half-life and VDss. "
        f"Human fu is the exception: {int(fu_row.supportive)} supportive, {int(fu_row.unstable)} unstable and "
        f"{int(fu_row.conflicting)} conflicting relationships among 44 partner tasks. Its supportive relationships are "
        "concentrated within the cross-species fu family, whereas many CL/CLint/Thalf/VDss/F-related tasks are conflicting. "
        "No task exhibits the pre-registered >10× norm-dominance flag "
        f"(largest median norm ratio {norm_max:.3f}).\n\n"
        "The single finite next candidate is a two-group encoder: one encoder shared by fu-family tasks and one shared by "
        "all non-fu tasks, with private task heads retained. F remains representation-only and cannot select an architecture. "
        "PCGrad/MMoE and broader searches remain deferred until this finite candidate is pre-registered and compared with "
        "the locked full-shared, fully-private and corrected Stage-A controls.\n\n"
        f"The figure masks self-pairs and uses a symmetric ±{color_limit:.2f} off-diagonal scale; the original locked formal "
        "artifacts remain unchanged.\n"
    )
    with stage_output(args.output) as out:
        human_off.to_csv(out / "human_task_offdiagonal_affinity.csv", index=False)
        count_table.to_csv(out / "human_task_classification_counts.csv", index=False)
        endpoint_summary.to_csv(out / "endpoint_pair_affinity_summary.csv", index=False)
        pd.DataFrame([{
            "candidate": "two_group_shared_encoder_private_heads",
            "group_1": "all fu endpoint tasks across species and plasma/serum systems",
            "group_2": "all non-fu tasks",
            "status": "recommended_for_preregistered_protocol_and_smoke",
            "architecture_training_authorized": False,
            "fixed_validation_authorized": False,
            "test_authorized": False,
            "reason": "human fu has 24/44 stable conflicts and prior full sharing worsened fu in all five folds; cross-species fu gradients remain supportive",
        }]).to_csv(out / "next_architecture_candidate.csv", index=False)
        figure.savefig(out / "Figure_multitask_gradient_affinity_report.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(report, encoding="utf-8")
        finish_stage(
            out, "multitask_gradient_affinity_report",
            inputs={"affinity_complete_sha256": sha256(args.affinity / "complete.json"),
                    "protocol_complete_sha256": sha256(args.protocol / "complete.json")},
            task_pairs=len(pair_off), human_offdiagonal_pairs=len(human_off),
            max_median_norm_ratio=norm_max, gradient_norm_dominance_flags=0,
            figure_offdiagonal_color_limit=color_limit, model_fitted=False,
            evaluation_labels_read=False, validation_target_file_opened=False, test_labels_read=False,
            architecture_training_authorized=False, fixed_validation_authorized=False,
            test_authorized=False, partial=False,
        )
    print(f"Multitask gradient-affinity report: {args.output}")


if __name__ == "__main__":
    run_cli(main)
