#!/usr/bin/env python3
"""Freeze VDss source-generalization feasibility and a purged document stress test."""
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


HUMAN_TASK = "VDss__human__steady_state_iv"
TASKS = [
    HUMAN_TASK,
    "VDss__dog__steady_state_iv",
    "VDss__monkey__steady_state_iv",
    "VDss__mouse__steady_state_iv",
    "VDss__rat__steady_state_iv",
]


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            following = self.parent[item]
            self.parent[item] = root
            item = following
        return root

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def connected_components(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_docs = records.groupby("parent_id", sort=True).doc_id.agg(lambda values: sorted(set(values.astype(str))))
    document_parents = records.groupby("doc_id", sort=True).parent_id.agg(lambda values: sorted(set(values.astype(str))))
    union = UnionFind(parent_docs.index.astype(str).tolist())
    for parents in document_parents:
        for parent in parents[1:]:
            union.union(parents[0], parent)
    buckets: dict[str, list[str]] = defaultdict(list)
    for parent in parent_docs.index.astype(str):
        buckets[union.find(parent)].append(parent)
    parent_rows, component_rows = [], []
    for members in buckets.values():
        members = sorted(members)
        documents = sorted({doc for parent in members for doc in parent_docs.loc[parent]})
        key = "|".join(members) + "||" + "|".join(documents)
        component = "src_" + hashlib.sha256(key.encode()).hexdigest()[:16]
        component_rows.append({
            "source_component_id": component,
            "component_parent_count": len(members),
            "component_document_count": len(documents),
            "document_ids": ";".join(documents),
        })
        parent_rows.extend({"parent_id": parent, "source_component_id": component} for parent in members)
    return pd.DataFrame(parent_rows), pd.DataFrame(component_rows)


def greedy_assign(frame: pd.DataFrame, id_column: str, weight_column: str, folds: int = 5) -> pd.DataFrame:
    ordered = frame.sort_values([weight_column, id_column], ascending=[False, True])
    loads = [0] * folds
    rows = []
    for row in ordered.itertuples(index=False):
        fold = min(range(folds), key=lambda value: (loads[value], value))
        rows.append({id_column: getattr(row, id_column), "source_fold": fold,
                     weight_column: int(getattr(row, weight_column))})
        loads[fold] += int(getattr(row, weight_column))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path,
                        default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path,
                        default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--transfer-protocol", type=Path,
                        default=ROOT / "data/public_development/vdss_cross_species_transfer_protocol_v4")
    parser.add_argument("--formal-analysis", type=Path,
                        default=ROOT / "results/analysis/vdss_cross_species_transfer_traincv_analysis_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/public_development/vdss_source_generalization_protocol_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    interface_records = args.interface / "training_records.csv"
    stl_records = args.stl / "benchmark_train_records.csv"
    required = [args.interface / "complete.json", interface_records, args.stl / "complete.json", stl_records,
                args.transfer_protocol / "complete.json", args.formal_analysis / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    metas = [
        verify_stage(args.interface, "multitask_pk_training_interface"),
        verify_stage(args.stl, "stl_train_only_protocol"),
        verify_stage(args.transfer_protocol, "vdss_cross_species_transfer_protocol"),
        verify_stage(args.formal_analysis, "vdss_cross_species_transfer_traincv_analysis"),
    ]
    if any(meta.get("test_labels_read", False) for meta in metas):
        raise ValueError("Source-generalization protocol inputs must remain test-closed")

    human = pd.read_csv(
        stl_records, dtype=str, keep_default_na=False,
        usecols=["row_id", "molecule_id", "task_id", "doc_id", "scaffold_group"],
    )
    human = human.loc[human.task_id.eq(HUMAN_TASK)].rename(columns={"molecule_id": "parent_id"}).copy()
    metadata = pd.read_csv(
        interface_records, dtype=str, keep_default_na=False,
        usecols=["row_id", "task_id", "split", "parent_id", "scaffold_group", "doc_id"],
    )
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(TASKS)].copy()
    if human.parent_id.nunique() != 562 or human.doc_id.nunique() != 22 or set(metadata.task_id) != set(TASKS):
        raise ValueError("Frozen VDss source membership differs from the expected input audit")

    parent_components, components = connected_components(human)
    component_assignment = greedy_assign(components, "source_component_id", "component_parent_count")
    components = components.merge(component_assignment, on=["source_component_id", "component_parent_count"],
                                  validate="one_to_one")
    largest_fraction = float(components.component_parent_count.max() / human.parent_id.nunique())
    connected_fivefold_feasible = bool(
        len(components) >= 5
        and components.groupby("source_fold").component_parent_count.sum().min() >= 30
        and largest_fraction <= 0.80
    )

    documents = human.groupby("doc_id", as_index=False).agg(
        evaluation_records=("row_id", "size"),
        evaluation_parent_incidence=("parent_id", "nunique"),
        evaluation_scaffolds=("scaffold_group", "nunique"),
    )
    document_assignment = greedy_assign(documents, "doc_id", "evaluation_parent_incidence")
    documents = documents.merge(document_assignment, on=["doc_id", "evaluation_parent_incidence"],
                                validate="one_to_one")
    fold_rows, task_rows = [], []
    for fold in range(5):
        evaluation_documents = set(documents.loc[documents.source_fold.eq(fold), "doc_id"].astype(str))
        evaluation = human.loc[human.doc_id.isin(evaluation_documents)].copy()
        evaluation_parents = set(evaluation.parent_id)
        evaluation_scaffolds = set(evaluation.scaffold_group)
        retained = metadata.loc[
            ~metadata.doc_id.isin(evaluation_documents)
            & ~metadata.parent_id.isin(evaluation_parents)
            & ~metadata.scaffold_group.isin(evaluation_scaffolds)
        ].copy()
        doc_overlap = len(set(retained.doc_id) & evaluation_documents)
        parent_overlap = len(set(retained.parent_id) & evaluation_parents)
        scaffold_overlap = len(set(retained.scaffold_group) & evaluation_scaffolds)
        task_counts = retained.groupby("task_id").agg(
            retained_records=("row_id", "size"), retained_parents=("parent_id", "nunique")
        )
        for task in TASKS:
            values = task_counts.loc[task]
            task_rows.append({"source_fold": fold, "task_id": task,
                              "retained_records": int(values.retained_records),
                              "retained_parents": int(values.retained_parents)})
        human_counts = task_counts.loc[HUMAN_TASK]
        fold_rows.append({
            "source_fold": fold,
            "evaluation_documents": len(evaluation_documents),
            "evaluation_document_ids": ";".join(sorted(evaluation_documents)),
            "evaluation_records": len(evaluation),
            "evaluation_parents": len(evaluation_parents),
            "evaluation_scaffolds": len(evaluation_scaffolds),
            "retained_human_records": int(human_counts.retained_records),
            "retained_human_parents": int(human_counts.retained_parents),
            "document_overlap": doc_overlap,
            "parent_overlap": parent_overlap,
            "scaffold_overlap": scaffold_overlap,
        })
    fold_audit = pd.DataFrame(fold_rows)
    task_retention = pd.DataFrame(task_rows)
    document_stress_feasible = bool(
        not fold_audit[["document_overlap", "parent_overlap", "scaffold_overlap"]].to_numpy().any()
        and fold_audit.evaluation_parents.min() > 0
        and fold_audit.retained_human_parents.min() >= 30
        and task_retention.retained_parents.min() > 0
    )
    if not document_stress_feasible:
        raise ValueError("Purged document-group sensitivity is not feasible")
    if args.check_only:
        print(
            f"VDss source protocol ready: components={len(components)} largest_fraction={largest_fraction:.3f} "
            f"component_5fold_feasible={connected_fivefold_feasible} document_stress_feasible=True"
        )
        return

    with stage_output(args.output) as out:
        parent_components.to_csv(out / "parent_source_components.csv", index=False)
        components.to_csv(out / "connected_component_assignment.csv", index=False)
        documents.to_csv(out / "document_group_assignment.csv", index=False)
        fold_audit.to_csv(out / "document_group_fold_audit.csv", index=False)
        task_retention.to_csv(out / "document_group_task_retention.csv", index=False)
        contract = {
            "schema_version": 1,
            "connected_component_holdout": "infeasible_for_balanced_fivefold",
            "connected_component_reason": "largest parent-document component contains 554/562 human parents",
            "registered_sensitivity": "five purged document groups",
            "training_exclusion": "evaluation documents plus every evaluation parent and scaffold across all five species tasks",
            "evaluation_target": "aggregate only records belonging to the held-out document group within source_fold and parent",
            "repeated_parent_policy": "a parent may be evaluated in multiple source folds through distinct documents; report fold-level paired sensitivity, not unique-parent OOF",
            "routes": ["corrected_stageA_STL", "human_only_neural_control",
                       "species_conditioned_joint_shared_encoder"],
            "interpretation": "source perturbation stress test only; no model selection or fixed-validation authorization",
            "validation_status": "closed",
            "test_status": "closed",
        }
        (out / "protocol.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
        ordered = components.sort_values("component_parent_count", ascending=False)
        axes[0].bar(np.arange(1, len(ordered) + 1), ordered.component_parent_count, color="#4C78A8")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Connected source-component rank")
        axes[0].set_ylabel("Human VDss parents (log scale)")
        axes[0].set_title("Source graph concentration", fontweight="bold")
        axes[0].spines[["top", "right"]].set_visible(False)
        x = np.arange(5)
        width = 0.36
        axes[1].bar(x - width / 2, fold_audit.evaluation_parents, width, label="Evaluation", color="#F58518")
        axes[1].bar(x + width / 2, fold_audit.retained_human_parents, width, label="Retained human train",
                    color="#54A24B")
        axes[1].set_xticks(x, [f"Fold {value + 1}" for value in x])
        axes[1].set_ylabel("Parents")
        axes[1].set_title("Purged document-group stress test", fontweight="bold")
        axes[1].legend(frameon=False)
        axes[1].spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "Figure_VDss_source_feasibility.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        (out / "README.md").write_text(
            "# VDss source-generalization protocol\n\n"
            "A balanced five-fold holdout of indivisible parent--document components is infeasible because 554 of "
            "562 human parents form one connected component. The registered fallback is a deliberately unbalanced "
            "five-group document perturbation with document, parent, and scaffold purging across every species task. "
            "It is a stress test only and cannot authorize model selection or fixed validation.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "vdss_source_generalization_protocol",
            inputs={
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_train_only_complete_sha256": sha256(args.stl / "complete.json"),
                "transfer_protocol_complete_sha256": sha256(args.transfer_protocol / "complete.json"),
                "formal_analysis_complete_sha256": sha256(args.formal_analysis / "complete.json"),
            },
            human_parents=int(human.parent_id.nunique()), human_documents=int(human.doc_id.nunique()),
            connected_components=len(components), largest_component_parents=int(components.component_parent_count.max()),
            largest_component_fraction=largest_fraction,
            connected_component_fivefold_feasible=connected_fivefold_feasible,
            document_group_stress_test_feasible=document_stress_feasible,
            document_groups=5, model_fitted=False, model_selection_authorized=False,
            validation_target_file_opened=False, test_labels_read=False, partial=False,
        )
    print(f"VDss source-generalization protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
