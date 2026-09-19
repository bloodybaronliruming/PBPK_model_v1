#!/usr/bin/env python3
"""Freeze internal-only MMPK N1b label tiers and author/strict split memberships."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage

ENDPOINTS = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]", "CL/F [L/h]", "Vz/F [L]", "MRT [h]", "F [%]")
LOG_TARGETS = {endpoint: f"Log {endpoint}" for endpoint in ENDPOINTS}
CORE = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]", "CL/F [L/h]", "Vz/F [L]")


def balanced_component_folds(records: pd.DataFrame, labels: pd.DataFrame, folds: int = 5) -> dict[str, int]:
    """Deterministic greedy allocation using only label presence, never label values."""
    direct = labels[labels.label_origin_proxy.eq("direct_raw_candidate_available")]
    augmented = labels[labels.model_label_present]
    summary = records[["strict_component_id"]].drop_duplicates().set_index("strict_component_id")
    summary["records"] = records.groupby("strict_component_id").size()
    for endpoint in ENDPOINTS:
        summary[f"aug::{endpoint}"] = augmented[augmented.endpoint.eq(endpoint)].groupby("strict_component_id").size()
        summary[f"direct::{endpoint}"] = direct[direct.endpoint.eq(endpoint)].groupby("strict_component_id").size()
    summary = summary.fillna(0).astype(int)
    columns = ["records"] + [f"aug::{x}" for x in ENDPOINTS] + [f"direct::{x}" for x in CORE]
    weights = np.asarray([1.0] + [1.0] * len(ENDPOINTS) + [1.5] * len(CORE))
    target = summary[columns].sum().to_numpy(float) / folds
    counts = np.zeros((folds, len(columns)), dtype=float)
    assignment: dict[str, int] = {}
    ordered = summary.assign(_size=summary[columns].mul(weights, axis=1).sum(axis=1)).sort_values(
        ["_size", "records"], ascending=[False, False], kind="stable")
    for rank, (component, row) in enumerate(ordered.iterrows()):
        vector = row[columns].to_numpy(float)
        if rank < folds:
            # Guarantee every declared outer fold has one nonempty component before balancing the remainder.
            chosen = rank
        else:
            scores = []
            for fold in range(folds):
                proposed = counts.copy()
                proposed[fold] += vector
                scale = np.where(target > 0, target, 1.0)
                scores.append(float(np.sum(weights * ((proposed - target) / scale) ** 2)))
            chosen = min(range(folds), key=lambda fold: (scores[fold], counts[fold, 0], fold))
        counts[chosen] += vector
        assignment[component] = chosen
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "基准研究参考")
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--n1a", type=Path, default=ROOT / "results/analysis/mmpk_n1a_lineage_feasibility_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    args = parser.parse_args()
    model_path = args.reference / "mmpk_human_oral_pk_parameters_prediction_v3/model_dataset/approved_model.csv"
    startup_self_check([args.n0 / "complete.json", args.n1a / "complete.json", model_path], output=args.output)
    n0 = verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    n1a = verify_stage(args.n1a, "mmpk_n1a_approved_raw_model_lineage_feasibility")
    if n1a["inputs"].get(str(model_path.resolve())) != sha256(model_path):
        raise ValueError("Approved model table differs from the N1a-hashed lineage input")
    model = pd.read_csv(model_path, usecols=["SMILES", *LOG_TARGETS.values()])
    registry = pd.read_csv(args.n1a / "approved_model_lineage_registry_hashed.csv")
    labels = pd.read_csv(args.n1a / "approved_model_target_lineage_hashed.csv")
    if len(registry) != len(model) or registry.model_record_id.duplicated().any():
        raise ValueError("N1a registry does not provide one lineage row per approved modeling row")
    expected = {(record, endpoint) for record in registry.model_record_id for endpoint in ENDPOINTS}
    actual = set(labels[["model_record_id", "endpoint"]].itertuples(index=False, name=None))
    if actual != expected or len(labels) != len(expected):
        raise ValueError("N1a target registry lacks a complete approved-record by endpoint matrix")
    registry = registry.copy()
    registry["author_exact_smiles_id"] = model.SMILES.map(lambda x: stable_id(f"author_exact_smiles:{x}"))
    registry["strict_component_id"] = registry.parent_source_scaffold_component_id
    labels = labels.merge(registry[["model_record_id", "strict_component_id"]], on="model_record_id", validate="many_to_one")
    # N1a records raw-observation provenance.  Model supervision, however,
    # requires the declared transformed target.  In particular, F=130% is a
    # valid raw boundary observation but has no finite log10(F/(130-F)) value.
    transformed_available = pd.DataFrame({"model_record_id": registry.model_record_id})
    for endpoint, column in LOG_TARGETS.items():
        transformed_available[endpoint] = pd.to_numeric(model[column], errors="coerce").notna().to_numpy()
    transformed_available = transformed_available.melt(id_vars="model_record_id", var_name="endpoint", value_name="transformed_target_available")
    labels = labels.merge(transformed_available, on=["model_record_id", "endpoint"], validate="one_to_one")
    if (labels.transformed_target_available & ~labels.model_label_present).any():
        raise ValueError("A transformed target is present where N1a reports no raw model label")
    # Exact implementation of the author repository split: original SMILES order, KFold shuffle seed 42,
    # then validation sampled from the fold's training SMILES with the same seed and 1/9 holdout.
    unique_smiles = model.SMILES.drop_duplicates().tolist()
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    author_membership = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(unique_smiles), start=1):
        train_values, test_values = [unique_smiles[i] for i in train_idx], [unique_smiles[i] for i in test_idx]
        train_values, val_values = train_test_split(train_values, test_size=1 / 9, random_state=42)
        train_keys = {stable_id(f"author_exact_smiles:{x}") for x in train_values}
        val_keys = {stable_id(f"author_exact_smiles:{x}") for x in val_values}
        test_keys = {stable_id(f"author_exact_smiles:{x}") for x in test_values}
        if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
            raise ValueError("Author exact-SMILES split partition overlap")
        for row in registry[["model_record_id", "author_exact_smiles_id"]].itertuples(index=False):
            role = "train" if row.author_exact_smiles_id in train_keys else "validation" if row.author_exact_smiles_id in val_keys else "test"
            author_membership.append({"fold": fold, "model_record_id": row.model_record_id, "role": role})
    author_membership = pd.DataFrame(author_membership)
    if len(author_membership) != 10 * len(registry) or author_membership.groupby("fold").size().ne(len(registry)).any():
        raise ValueError("Incomplete author split membership")
    assignments = balanced_component_folds(registry, labels, folds=5)
    registry["strict_outer_fold"] = registry.strict_component_id.map(assignments).astype(int)
    if registry.strict_outer_fold.isna().any():
        raise ValueError("Unassigned strict component")
    labels = labels.merge(registry[["model_record_id", "parent_id", "scaffold_id", "dose_arm_id", "source_family_id", "strict_outer_fold", "author_exact_smiles_id"]], on="model_record_id", validate="many_to_one")
    labels["reliability_tier"] = np.where(labels.label_origin_proxy.eq("direct_raw_candidate_available"), "R1_direct_observation", "R2_author_augmented")
    labels["direct_observation_eligible"] = labels.reliability_tier.eq("R1_direct_observation")
    labels["author_augmented_eligible"] = labels.transformed_target_available
    boundary_exclusions = labels[labels.model_label_present & ~labels.transformed_target_available].copy()
    labels = labels[labels.transformed_target_available].copy()
    author_audit = []
    for fold in range(1, 11):
        test_records = author_membership.loc[(author_membership.fold.eq(fold)) & (author_membership.role.eq("test")), "model_record_id"]
        test = registry[registry.model_record_id.isin(test_records)]
        other = registry[~registry.model_record_id.isin(test_records)]
        author_audit.append({"fold": fold, "test_records": int(len(test)), "test_exact_smiles": int(test.author_exact_smiles_id.nunique()),
                             "test_parents": int(test.parent_id.nunique()), "test_scaffolds": int(test.scaffold_id.nunique()),
                             "parent_overlap_with_remaining": int(len(set(test.parent_id) & set(other.parent_id))),
                             "scaffold_overlap_with_remaining": int(len(set(test.scaffold_id) & set(other.scaffold_id)))})
    strict_audit, strict_coverage = [], []
    for fold in range(5):
        test = registry[registry.strict_outer_fold.eq(fold)]
        other = registry[~registry.strict_outer_fold.eq(fold)]
        strict_audit.append({"outer_fold": fold, "test_records": int(len(test)), "test_parents": int(test.parent_id.nunique()),
                             "test_scaffolds": int(test.scaffold_id.nunique()), "test_components": int(test.strict_component_id.nunique()),
                             "component_overlap_with_remaining": int(len(set(test.strict_component_id) & set(other.strict_component_id))),
                             "parent_overlap_with_remaining": int(len(set(test.parent_id) & set(other.parent_id))),
                             "scaffold_overlap_with_remaining": int(len(set(test.scaffold_id) & set(other.scaffold_id)))})
        for endpoint in ENDPOINTS:
            subset = labels[(labels.strict_outer_fold.eq(fold)) & (labels.endpoint.eq(endpoint))]
            strict_coverage.append({"outer_fold": fold, "endpoint": endpoint,
                                    "R1_direct_observation": int(subset.direct_observation_eligible.sum()),
                                    "R2_author_augmented_only": int((subset.reliability_tier.eq("R2_author_augmented")).sum()),
                                    "author_augmented_total": int(subset.author_augmented_eligible.sum())})
    strict_audit = pd.DataFrame(strict_audit)
    if (strict_audit[["component_overlap_with_remaining", "parent_overlap_with_remaining", "scaffold_overlap_with_remaining"]] != 0).any().any():
        raise ValueError("Strict component split leaks parent or scaffold across an outer fold")
    if (strict_audit.test_records <= 0).any():
        raise ValueError("Strict component allocation contains an empty outer fold")
    with stage_output(args.output) as out:
        registry.to_csv(out / "approved_record_registry_hashed.csv", index=False)
        labels.to_csv(out / "approved_label_tier_registry_hashed.csv", index=False)
        author_membership.to_csv(out / "author_exact_smiles_split_membership_hashed.csv", index=False)
        pd.DataFrame(author_audit).to_csv(out / "author_split_leakage_audit.csv", index=False)
        strict_audit.to_csv(out / "strict_component_split_leakage_audit.csv", index=False)
        pd.DataFrame(strict_coverage).to_csv(out / "strict_fold_label_coverage.csv", index=False)
        tier_summary = labels.groupby(["endpoint", "reliability_tier"]).size().rename("labels").reset_index()
        tier_summary.to_csv(out / "label_tier_summary.csv", index=False)
        boundary_summary = boundary_exclusions.groupby("endpoint").size().rename("excluded_records").reset_index()
        boundary_summary.to_csv(out / "transformed_target_boundary_exclusion_summary.csv", index=False)
        protocol = {
            "scope": "internal_only_mmpk_approved_development",
            "external_lifecycle": "investigational_and_2024_labels_unread_and_sealed",
            "author_track": "B-grade exact-SMILES KFold(10, shuffle=True, random_state=42), fold-train validation fraction=1/9 random_state=42",
            "strict_track": "five deterministic outer folds over premerged parent+source+scaffold components",
            "tiers": {"R1_direct_observation": "finite transformed model label has a same-condition raw candidate contributor",
                      "R2_author_augmented": "model label lacks a same-condition raw candidate; derived/imputed/aggregate proxy"},
            "transformed_target_rule": "only a finite author-declared transformed target may enter a supervised mask; raw boundary observations without such a target are retained as exclusions, never imputed",
            "primary_first_batch": ["AUC", "Cmax", "Tmax", "t1/2"],
            "secondary_direct": ["CL/F", "Vz/F"], "exploratory_sparse": ["MRT", "F"],
            "forbidden": ["external scoring", "post-fold split modification", "treating R2 as independent direct evidence", "data redistribution without clarified terms"],
        }
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "README.md").write_text("# MMPK N1b internal cohort and split protocol\n\n"
            "This package freezes only hashed membership and label-origin tiers for approved development data. It contains no raw target values and does not read external cohort labels. The author exact-SMILES and project strict component tracks are separate and must never be directly ranked.\n", encoding="utf-8")
        finish_stage(out, "mmpk_n1b_internal_cohort_and_dual_split_protocol", inputs={
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
            str((args.n1a / "complete.json").resolve()): sha256(args.n1a / "complete.json"),
            str(model_path.resolve()): sha256(model_path)}, no_training=True, external_labels_accessed=False,
            raw_labels_exported=False, author_track_grade="B_exact_smiles_author_code", strict_outer_folds=5,
            author_folds=10, strict_constraints="parent+source+scaffold_connected_component", partial=False)
    print(f"MMPK N1b internal cohort/split protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
