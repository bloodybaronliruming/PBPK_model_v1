#!/usr/bin/env python3
"""Read-only source and target-range sensitivity for the Stage-A structure control.

Every outer-fold evaluation record is classified by whether its document ID was
present in that fold's outer training rows. Parent results are then assigned to
exclusive all-seen versus any-unseen source groups. This is exploratory source
sensitivity, not a new independent test set.
"""
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
from scipy.special import expit

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


CONTROL = "structure_control__stageA_leader"
PAPP_TASK = "Papp__human__caco2_ab"
LABELS = {"CL__human__systemic_iv": "CL", "CLint__human__microsome": "CLint",
          PAPP_TASK: "Papp", "Thalf__human__terminal_iv": "Terminal t½",
          "VDss__human__steady_state_iv": "VDss", "fu__human__plasma": "fu"}


def parent_metric(frame: pd.DataFrame, task: str) -> float:
    if task == "fu__human__plasma":
        observed = expit(frame.observed_target.to_numpy(float))
        predicted = expit(frame.predicted_target.to_numpy(float))
        return float(np.mean(np.abs(predicted - observed)))
    return float(np.sqrt(np.mean((frame.predicted_target.to_numpy(float) - frame.observed_target.to_numpy(float)) ** 2)))


def source_figure(metrics: pd.DataFrame) -> plt.Figure:
    order = list(LABELS)
    table = metrics.loc[metrics.source_subset.isin(["all_sources", "all_seen_documents", "any_unseen_document"])]
    table = table.pivot(index="task_id", columns="source_subset", values="primary_metric").reindex(order)
    ratio = table["any_unseen_document"] / table["all_seen_documents"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    y = np.arange(len(order))
    ax.scatter(ratio, y, s=52, color="#C44E52", zorder=3)
    for value, ypos in zip(ratio, y):
        ax.text(value + 0.025, ypos, f"{value:.2f}×", va="center", fontsize=8)
    ax.axvline(1.0, color="#333333", linewidth=1)
    ax.set_yticks(y, [LABELS[key] for key in order])
    ax.invert_yaxis()
    ax.set_xlabel("Error ratio: any-unseen document / all-seen documents")
    ax.set_title("Exploratory document-source sensitivity of Stage-A structure controls", fontweight="bold", pad=12)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def papp_figure(metrics: pd.DataFrame) -> plt.Figure:
    order = ["≤100", "100–10,000", ">10,000"]
    table = metrics.loc[metrics.source_subset.str.startswith("papp_value_")].copy()
    table["label"] = table.source_subset.str.replace("papp_value_", "", regex=False)
    table = table.set_index("label").reindex(order)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    bars = ax.bar(np.arange(len(order)), table.primary_metric, color=["#4C78A8", "#F28E2B", "#C44E52"], width=0.65)
    for bar, count, value in zip(bars, table.parents, table.primary_metric):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"n={int(count)}\n{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_ylabel("Parent-level transformed RMSE")
    ax.set_xlabel("Papp value range (10⁻⁶ cm/s)")
    ax.set_title("Papp OOF error by label range", fontweight="bold", pad=12)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--adaptation", type=Path, default=ROOT / "results/benchmarks/gate1b_frozen_smiles_adaptation_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_source_error_sensitivity_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
                args.adaptation / "complete.json", args.adaptation / "predictions.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    adaptation_meta = verify_stage(args.adaptation, "gate1b_frozen_smiles_adaptation")
    if train_meta.get("fixed_validation_targets_published") or adaptation_meta.get("validation_target_file_opened") or adaptation_meta.get("test_labels_read"):
        raise ValueError("Source sensitivity requires train-only, test-closed inputs")
    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records["inner_fold_id"] = records.inner_fold_id.astype(int)
    predictions = pd.read_csv(args.adaptation / "predictions.csv", dtype=str, keep_default_na=False)
    predictions = predictions.loc[predictions.candidate_id.eq(CONTROL)].copy()
    for column in ["target_value", "train_mean", "train_std_population", "predicted_interface_target", "inner_fold_id"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    predictions["inner_fold_id"] = predictions.inner_fold_id.astype(int)
    if predictions.empty or set(predictions.row_id) != set(records.row_id):
        raise ValueError("Structure-control OOF predictions do not cover exactly the train-only records")
    if args.check_only:
        print(f"STL source sensitivity ready: records={len(records)} tasks={records.task_id.nunique()} control={CONTROL}")
        return

    source_rows, parent_rows, metrics = [], [], []
    joined = predictions.merge(records[["row_id", "doc_id"]], on="row_id", how="left", validate="one_to_one")
    if joined.doc_id.isna().any():
        raise ValueError("A prediction row lacks a source document ID")
    for (task, fold), evaluation in joined.groupby(["task_id", "inner_fold_id"], sort=True):
        training_docs = set(records.loc[records.task_id.eq(task) & records.inner_fold_id.ne(fold), "doc_id"].astype(str))
        evaluation = evaluation.copy()
        evaluation["source_document_seen_in_outer_train"] = evaluation.doc_id.astype(str).isin(training_docs)
        evaluation["observed_target"] = evaluation.target_value
        evaluation["predicted_target"] = evaluation.predicted_interface_target * evaluation.train_std_population + evaluation.train_mean
        source_rows.append(evaluation[["row_id", "molecule_id", "task_id", "endpoint", "inner_fold_id", "doc_id",
                                       "source_document_seen_in_outer_train", "observed_target", "predicted_target"]])
    source_frame = pd.concat(source_rows, ignore_index=True)
    for (task, molecule), group in source_frame.groupby(["task_id", "molecule_id"], sort=True):
        if group.inner_fold_id.nunique() != 1:
            raise ValueError("A parent appears in more than one fixed evaluation fold")
        parent_rows.append({
            "task_id": task, "endpoint": group.endpoint.iloc[0], "molecule_id": molecule,
            "inner_fold_id": int(group.inner_fold_id.iloc[0]), "source_documents": "|".join(sorted(set(group.doc_id.astype(str)))),
            "source_document_count": int(group.doc_id.nunique()),
            "source_all_seen_in_outer_train": bool(group.source_document_seen_in_outer_train.all()),
            "source_any_unseen_in_outer_train": bool((~group.source_document_seen_in_outer_train).any()),
            "observed_target": float(group.observed_target.mean()), "predicted_target": float(group.predicted_target.mean()),
        })
    parent_frame = pd.DataFrame(parent_rows)
    for task, group in parent_frame.groupby("task_id", sort=True):
        for subset, subset_frame in [("all_sources", group),
                                     ("all_seen_documents", group.loc[group.source_all_seen_in_outer_train]),
                                     ("any_unseen_document", group.loc[group.source_any_unseen_in_outer_train])]:
            if subset_frame.empty:
                continue
            metrics.append({"task_id": task, "endpoint": group.endpoint.iloc[0], "source_subset": subset,
                            "parents": len(subset_frame), "primary_metric": parent_metric(subset_frame, task),
                            "primary_metric_name": "physical_MAE" if task == "fu__human__plasma" else "transformed_RMSE"})
    papp = parent_frame.loc[parent_frame.task_id.eq(PAPP_TASK)].copy()
    papp["papp_raw_1e_minus_6_cm_s"] = np.power(10.0, papp.observed_target)
    papp["papp_range"] = pd.cut(papp.papp_raw_1e_minus_6_cm_s, bins=[0, 100, 10000, np.inf], labels=["≤100", "100–10,000", ">10,000"], include_lowest=True)
    for label, group in papp.groupby("papp_range", observed=False):
        if not group.empty:
            metrics.append({"task_id": PAPP_TASK, "endpoint": "Papp", "source_subset": f"papp_value_{label}",
                            "parents": len(group), "primary_metric": parent_metric(group, PAPP_TASK),
                            "primary_metric_name": "transformed_RMSE"})
    metric_frame = pd.DataFrame(metrics)
    if args.check_only:
        return
    with stage_output(args.output) as out:
        source_frame.to_csv(out / "record_source_membership.csv", index=False)
        parent_frame.to_csv(out / "parent_source_sensitivity.csv", index=False)
        metric_frame.to_csv(out / "source_and_label_range_metrics.csv", index=False)
        fig = source_figure(metric_frame)
        fig.savefig(out / "Figure_source_document_sensitivity.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        fig = papp_figure(metric_frame)
        fig.savefig(out / "Figure_Papp_label_range_error.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        summary = {"tasks": int(parent_frame.task_id.nunique()), "parents": int(len(parent_frame)),
                   "control_candidate": CONTROL, "source_definition": "outer-fold document-ID seen in outer training rows",
                   "parent_source_groups_exclusive": True, "papp_value_ranges": ["≤100", "100–10,000", ">10,000"],
                   "dpi": 600, "language": "English", "validation_target_file_opened": False,
                   "validation_rows_evaluated": False, "test_labels_read": False, "data_modified": False,
                   "model_fitted": False, "interpretation": "exploratory_sensitivity_not_independent_external_evaluation"}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Source and label-range sensitivity\n\nUses frozen train-CV structure-control OOF predictions. "
            "Document seen/unseen categories are exclusive at the parent level but remain exploratory: they do not isolate document conditions from chemical-space shift and are not an external test.\n",
            encoding="utf-8")
        finish_stage(out, "stl_source_error_sensitivity", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "adaptation_complete_sha256": sha256(args.adaptation / "complete.json"),
        }, partial=False, **summary)
    print(f"STL source sensitivity: {args.output}")


if __name__ == "__main__":
    run_cli(main)
