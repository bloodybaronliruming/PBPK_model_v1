#!/usr/bin/env python3
"""Register source-connected Papp/fu holdout folds without fitting models.

Within each endpoint, a source cluster is a connected component of the
parent--document bipartite graph. Components are indivisible: a document or a
parent cannot occur in both source-train and source-evaluation. For every
proposed source-cluster fold, all evaluation scaffolds are additionally removed
from training. The output is a feasibility audit, not a performance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASKS = ("Papp__human__caco2_ab", "fu__human__plasma")
DISPLAY = {"Papp__human__caco2_ab": "Papp", "fu__human__plasma": "fu"}


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def source_components(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_docs = records.groupby("molecule_id", sort=True).doc_id.agg(lambda x: sorted(set(x.astype(str))))
    document_parents = records.groupby("doc_id", sort=True).molecule_id.agg(lambda x: sorted(set(x.astype(str))))
    union = UnionFind(parent_docs.index.astype(str).tolist())
    for parents in document_parents:
        for parent in parents[1:]:
            union.union(parents[0], parent)
    buckets: dict[str, list[str]] = defaultdict(list)
    for parent in parent_docs.index.astype(str):
        buckets[union.find(parent)].append(parent)
    rows, doc_rows = [], []
    for members in buckets.values():
        members = sorted(members)
        docs = sorted({doc for parent in members for doc in parent_docs.loc[parent]})
        key = "|".join(members) + "||" + "|".join(docs)
        component = "src_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        for parent in members:
            rows.append({"molecule_id": parent, "source_component_id": component,
                         "component_parent_count": len(members), "component_document_count": len(docs)})
        for doc in docs:
            doc_rows.append({"doc_id": doc, "source_component_id": component,
                             "component_parent_count": len(members), "component_document_count": len(docs)})
    return pd.DataFrame(rows), pd.DataFrame(doc_rows)


def balanced_component_folds(components: pd.DataFrame, folds: int) -> pd.DataFrame:
    """Greedily balance parent counts; ties use stable component IDs."""
    unique = components[["source_component_id", "component_parent_count", "component_document_count"]].drop_duplicates()
    ordered = unique.sort_values(["component_parent_count", "component_document_count", "source_component_id"],
                                 ascending=[False, False, True])
    loads = [0] * folds
    assignments = []
    for row in ordered.itertuples(index=False):
        fold = min(range(folds), key=lambda value: (loads[value], value))
        assignments.append({"source_component_id": row.source_component_id, "source_fold": fold,
                            "component_parent_count": row.component_parent_count,
                            "component_document_count": row.component_document_count})
        loads[fold] += int(row.component_parent_count)
    return pd.DataFrame(assignments)


def audit_task(records: pd.DataFrame, folds: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parent_component, document_component = source_components(records)
    assignment = balanced_component_folds(parent_component, folds)
    parent = records[["molecule_id", "scaffold_group"]].drop_duplicates()
    if parent.molecule_id.duplicated().any():
        raise ValueError("A parent maps to more than one scaffold group")
    parent = parent.merge(parent_component, on="molecule_id", validate="one_to_one")
    record_component = records.merge(parent_component[["molecule_id", "source_component_id"]], on="molecule_id", validate="many_to_one")
    record_component = record_component.merge(assignment[["source_component_id", "source_fold"]], on="source_component_id", validate="many_to_one")
    audits = []
    for fold in range(folds):
        evaluation = parent.loc[parent.source_component_id.isin(
            assignment.loc[assignment.source_fold.eq(fold), "source_component_id"]
        )].copy()
        evaluation_parents = set(evaluation.molecule_id)
        evaluation_docs = set(record_component.loc[record_component.molecule_id.isin(evaluation_parents), "doc_id"].astype(str))
        evaluation_scaffolds = set(evaluation.scaffold_group.astype(str))
        training = parent.loc[~parent.molecule_id.isin(evaluation_parents)].copy()
        source_training = training.copy()
        training = training.loc[~training.scaffold_group.astype(str).isin(evaluation_scaffolds)].copy()
        train_parents = set(training.molecule_id)
        train_docs = set(record_component.loc[record_component.molecule_id.isin(train_parents), "doc_id"].astype(str))
        parent_overlap = len(train_parents & evaluation_parents)
        source_overlap = len(train_docs & evaluation_docs)
        scaffold_overlap = len(set(training.scaffold_group.astype(str)) & evaluation_scaffolds)
        excluded_by_scaffold = set(source_training.molecule_id) - train_parents
        audits.append({
            "task_id": records.task_id.iloc[0], "endpoint": records.endpoint.iloc[0], "source_fold": fold,
            "evaluation_source_components": int(assignment.source_fold.eq(fold).sum()),
            "evaluation_documents": len(evaluation_docs), "evaluation_parents": len(evaluation_parents),
            "evaluation_records": int(records.molecule_id.isin(evaluation_parents).sum()),
            "source_train_parents_before_scaffold_exclusion": len(source_training),
            "scaffold_collision_parents_removed_from_train": len(excluded_by_scaffold),
            "train_parents_after_scaffold_exclusion": len(train_parents),
            "train_records_after_scaffold_exclusion": int(records.molecule_id.isin(train_parents).sum()),
            "retained_train_parent_fraction": len(train_parents) / len(parent),
            "parent_overlap": parent_overlap, "document_overlap": source_overlap, "scaffold_overlap": scaffold_overlap,
            "feasible": bool(len(train_parents) > 0 and len(evaluation_parents) > 0 and not parent_overlap and not source_overlap and not scaffold_overlap),
        })
    component_summary = assignment.merge(
        parent_component.groupby("source_component_id", as_index=False).molecule_id.nunique().rename(columns={"molecule_id": "verified_parent_count"}),
        on="source_component_id", validate="one_to_one",
    )
    return parent_component, document_component, component_summary, pd.DataFrame(audits)


def figure(component_summary: pd.DataFrame, fold_audit: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for task, color in zip(TASKS, ("#4C78A8", "#C44E52"), strict=True):
        values = component_summary.loc[component_summary.task_id.eq(task), "component_parent_count"].sort_values(ascending=False).to_numpy()
        axes[0].step(np.arange(1, len(values) + 1), values, where="mid", label=DISPLAY[task], color=color, linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Source-connected component rank")
    axes[0].set_ylabel("Parents per component (log scale)")
    axes[0].set_title("Source-cluster size distribution", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].spines[["top", "right"]].set_visible(False)
    x = np.arange(5)
    width = 0.34
    for offset, task, color in [(-width / 2, TASKS[0], "#4C78A8"), (width / 2, TASKS[1], "#C44E52")]:
        rows = fold_audit.loc[fold_audit.task_id.eq(task)].sort_values("source_fold")
        axes[1].bar(x + offset, rows.retained_train_parent_fraction * 100, width, label=DISPLAY[task], color=color)
    axes[1].set_xticks(x, [f"Fold {value + 1}" for value in x])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Training parents retained after scaffold exclusion (%)")
    axes[1].set_title("Feasibility of source-cluster holdout", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_papp_fu_source_cluster_feasibility_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.folds != 5:
        raise ValueError("This pre-registered diagnostic requires exactly five source folds")
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    metadata = verify_stage(args.train_protocol, "stl_train_only_protocol")
    if metadata.get("fixed_validation_targets_published") or metadata.get("test_labels_read"):
        raise ValueError("Source-cluster feasibility requires train-only, test-closed inputs")
    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    records = records.loc[records.task_id.isin(TASKS)].copy()
    if set(records.task_id.unique()) != set(TASKS):
        raise ValueError("Required Papp/fu tasks are absent")
    if args.check_only:
        print(f"Source-cluster feasibility ready: tasks={len(TASKS)} records={len(records)} folds={args.folds}")
        return
    components, documents, component_summary, audits = [], [], [], []
    for _, task_records in records.groupby("task_id", sort=True):
        parent_component, document_component, component_table, audit_table = audit_task(task_records, args.folds)
        task = task_records.task_id.iloc[0]
        parent_component["task_id"], document_component["task_id"], component_table["task_id"] = task, task, task
        components.append(parent_component); documents.append(document_component); component_summary.append(component_table); audits.append(audit_table)
    parent_components = pd.concat(components, ignore_index=True)
    document_components = pd.concat(documents, ignore_index=True)
    component_frame = pd.concat(component_summary, ignore_index=True)
    audit_frame = pd.concat(audits, ignore_index=True)
    if not audit_frame.feasible.all():
        raise ValueError("At least one source-cluster fold is infeasible")
    if audit_frame[["parent_overlap", "document_overlap", "scaffold_overlap"]].to_numpy().any():
        raise ValueError("Source-cluster audit found overlap")
    with stage_output(args.output) as out:
        parent_components.to_csv(out / "parent_source_components.csv", index=False)
        document_components.to_csv(out / "document_source_components.csv", index=False)
        component_frame.to_csv(out / "source_component_fold_assignment.csv", index=False)
        audit_frame.to_csv(out / "source_cluster_fold_feasibility.csv", index=False)
        fig = figure(component_frame, audit_frame)
        fig.savefig(out / "Figure_source_cluster_feasibility.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        summary = {
            "tasks": len(TASKS), "folds": args.folds, "source_cluster_definition": "parent_document_bipartite_connected_component",
            "evaluation_unit": "source_cluster_fold", "training_exclusion": "evaluation_source_components_plus_evaluation_scaffolds",
            "all_folds_feasible": True, "validation_target_file_opened": False, "validation_rows_evaluated": False,
            "test_labels_read": False, "data_modified": False, "model_fitted": False, "model_selection_authorized": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Papp/fu source-cluster feasibility\n\n"
            "This read-only audit forms source clusters from parent--document connected components and verifies a five-fold source holdout with evaluation scaffolds excluded from training. It does not fit a model and does not estimate performance.\n",
            encoding="utf-8",
        )
        finish_stage(out, "stl_papp_fu_source_cluster_feasibility", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
        }, partial=False, **summary)
    print(f"Papp/fu source-cluster feasibility: {args.output}")


if __name__ == "__main__":
    run_cli(main)
