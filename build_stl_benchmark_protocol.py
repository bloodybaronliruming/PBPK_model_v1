#!/usr/bin/env python3
"""Build the leakage-guarded strong-STL benchmark manifest and feature cache.

This stage does not fit a model.  It binds the six authorized human endpoints
to frozen inner folds, canonical-parent structures, common deterministic
features, algorithm availability, and a predeclared screening protocol.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from dmpk_toolkit import featurize
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


ALGORITHMS = [
    ("dummy_mean", "implemented", "reference", False, "all", "none", "yes"),
    ("ridge", "implemented", "broad_screen", True, "all", "alpha:3", "yes"),
    ("elasticnet", "implemented", "broad_screen", True, "rdkit2d,ecfp4_rdkit2d", "alpha_l1:4", "yes"),
    ("knn", "implemented", "broad_screen", True, "all", "neighbors:3", "no"),
    ("random_forest", "implemented", "broad_screen", False, "all", "trees_depth_leaf:4", "yes"),
    ("extra_trees", "implemented", "broad_screen", False, "all", "trees_depth_leaf:4", "yes"),
    ("hist_gradient_boosting", "implemented", "broad_screen", False, "rdkit2d", "depth_leaf_l2:4", "yes"),
    ("svr_rbf", "implemented", "broad_screen_small_medium", True, "rdkit2d", "C_gamma_epsilon:4", "yes"),
    ("kernel_ridge_rbf", "implemented", "broad_screen_small", True, "rdkit2d", "alpha_gamma:4", "yes"),
    ("xgboost", "implemented_optional", "broad_screen", False, "all", "depth_rate_leaf:4", "yes"),
    ("lightgbm", "implemented_optional", "broad_screen", False, "all", "leaves_rate_child:4", "yes"),
    ("catboost", "implemented_optional", "broad_screen", False, "all", "depth_rate_l2:4", "yes"),
    ("mlp", "implemented", "broad_screen", True, "rdkit2d,ecfp4_rdkit2d", "width_alpha:3", "no"),
    ("pls", "deferred_adapter", "small_task_followup", True, "rdkit2d", "components:3", "no"),
    ("gaussian_process", "deferred_adapter", "small_task_followup", True, "rdkit2d", "bounded_small_n", "yes"),
    ("resmlp", "deferred_gate1b", "neural_followup", True, "all", "multi_seed", "yes"),
    ("dmpnn", "deferred_gate1b", "graph_followup", False, "graph_plus_rdkit2d", "multi_seed", "yes"),
    ("gin_gine", "optional_deferred", "graph_followup", False, "graph", "multi_seed", "pending"),
    ("frozen_pretrained_encoder", "optional_deferred", "representation_followup", False, "embedding", "linear_or_mlp_head", "pending"),
]


def canonical_parent_smiles(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in approved interface: {smiles}")
    parent = rdMolStandardize.FragmentParent(mol)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    parent = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
    Chem.RemoveStereochemistry(parent)
    identity = Chem.MolToSmiles(parent, isomericSmiles=False)
    return identity, stable_id(identity)


def algorithm_registry() -> pd.DataFrame:
    modules = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}
    rows = []
    for algorithm, status, stage, scale, features, budget, weights in ALGORITHMS:
        module = modules.get(algorithm)
        available = bool(importlib.util.find_spec(module)) if module else status.startswith("implemented")
        rows.append({
            "algorithm": algorithm, "implementation_status": status, "runtime_available": available,
            "screening_stage": stage, "requires_train_only_scaling": scale,
            "allowed_feature_views": features, "small_budget_space": budget,
            "sample_weight_support": weights,
        })
    return pd.DataFrame(rows)


def build_manifests(interface: Path, splits: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(interface / "training_records.csv", dtype=str, keep_default_na=False)
    transforms = pd.read_csv(interface / "task_target_transforms_train_only.csv")
    sensitivity = pd.read_csv(interface / "source_cluster_sensitivity_validation_manifest.csv", dtype=str, keep_default_na=False)
    authorized = set(transforms.loc[transforms.head_selection_authorized.astype(bool), "task_id"])
    records = records.loc[records.task_id.isin(authorized)].copy()
    if len(authorized) != 6 or records.task_id.nunique() != 6 or records.endpoint.eq("F").any():
        raise ValueError("STL benchmark must contain exactly six authorized non-F human heads")
    transform_columns = ["task_id", "target_transform", "reporting_inverse", "train_mean",
                         "train_std_population", "model_output", "head_selection_authorized"]
    records = records.merge(transforms[transform_columns], on="task_id", validate="many_to_one")
    frozen = pd.read_csv(splits / "split_manifest.csv", dtype=str, keep_default_na=False)
    fold_map = frozen[["molecule_id", "fold_id", "eligible"]].rename(columns={"molecule_id": "source_molecule_id"})
    records = records.merge(fold_map, on="source_molecule_id", how="left", validate="many_to_one")
    if records.fold_id.isna().any() or not records.eligible_y.str.lower().eq("true").all():
        raise ValueError("Authorized records are not fully covered by eligible frozen folds")
    records = records.drop(columns=["eligible_y"]).rename(columns={"eligible_x": "eligible", "fold_id": "inner_fold_id"})
    records["inner_fold_id"] = pd.to_numeric(records.inner_fold_id, errors="raise").astype(int)
    if not records.loc[records.split.eq("train"), "inner_fold_id"].between(0, 4).all():
        raise ValueError("Training rows require frozen 0--4 inner folds")
    if not records.loc[records.split.eq("val"), "inner_fold_id"].eq(-1).all():
        raise ValueError("Fixed validation rows must remain outside inner CV")
    if (records.groupby("molecule_id").inner_fold_id.nunique() > 1).any() or (records.groupby("scaffold_group").inner_fold_id.nunique() > 1).any():
        raise ValueError("Parent or scaffold crosses frozen inner folds")
    source_view = sensitivity[["row_id", "source_cluster_seen_in_train", "sensitivity_subset"]]
    records = records.merge(source_view, on="row_id", how="left", validate="one_to_one")
    records.loc[records.split.eq("train"), "sensitivity_subset"] = "train_not_applicable"
    records.loc[records.split.eq("train"), "source_cluster_seen_in_train"] = ""

    parent_rows = []
    for parent_id, group in records.groupby("molecule_id", sort=True):
        identities = {canonical_parent_smiles(smi) for smi in group.smiles.unique()}
        hashes = {item[1] for item in identities}
        smiles_set = {item[0] for item in identities}
        if hashes != {parent_id} or len(smiles_set) != 1:
            raise ValueError(f"Canonical parent reconstruction disagrees for {parent_id}")
        parent_rows.append({"molecule_id": parent_id, "canonical_parent_smiles": next(iter(smiles_set))})
    parents = pd.DataFrame(parent_rows)
    parents["feature_index"] = np.arange(len(parents), dtype=int)
    records = records.merge(parents, on="molecule_id", validate="many_to_one")

    task_counts = (records.groupby(["task_id", "endpoint", "canonical_unit", "target_transform", "reporting_inverse"], dropna=False)
                   .agg(records=("row_id", "size"), parents=("molecule_id", "nunique"),
                        scaffolds=("scaffold_group", "nunique"), sources=("doc_id", "nunique"))
                   .reset_index())
    split_counts = records.groupby(["task_id", "split"]).size().unstack("split", fill_value=0).reset_index()
    fold_counts = records.loc[records.split.eq("train")].groupby("task_id").inner_fold_id.nunique().rename("inner_folds").reset_index()
    source_disjoint = (records.loc[records.sensitivity_subset.eq("validation_source_disjoint")]
                       .groupby("task_id").size().rename("source_disjoint_validation_records").reset_index())
    tasks = task_counts.merge(split_counts, on="task_id").merge(fold_counts, on="task_id").merge(source_disjoint, on="task_id", how="left")
    tasks["source_disjoint_validation_records"] = tasks.source_disjoint_validation_records.fillna(0).astype(int)
    tasks["primary_metric"] = np.where(tasks.endpoint.eq("fu"), "physical_MAE", "transformed_RMSE")
    tasks["selection_rule"] = "rank_on_frozen_train_inner_CV; fixed_validation_confirmation_only_after_shortlist"
    return records.sort_values(["task_id", "split", "inner_fold_id", "row_id"]), parents, tasks.sort_values("task_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--project-splits", type=Path, default=ROOT / "data/processed_v15/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    # Approved v6 structures have already passed an audited RDKit
    # standardization stage; suppress repeated tautomer/kekulization chatter
    # while retaining explicit Python exceptions and hash checks below.
    RDLogger.DisableLog("rdApp.*")
    startup_self_check([args.interface / "complete.json", args.project_splits / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.interface, "multitask_pk_training_interface")
    verify_stage(args.project_splits, "splits")
    if args.check_only and (args.output / "complete.json").exists():
        meta = verify_stage(args.output, "stl_benchmark_protocol")
        expected = {
            "interface_complete_sha256": sha256(args.interface / "complete.json"),
            "project_splits_complete_sha256": sha256(args.project_splits / "complete.json"),
        }
        if meta.get("inputs") != expected:
            raise ValueError("Published STL protocol no longer matches its current frozen inputs")
        print(f"STL benchmark protocol valid: records={meta['records']} parents={meta['parents']} tasks={meta['tasks']}")
        return
    records, parents, tasks = build_manifests(args.interface, args.project_splits)
    algorithms = algorithm_registry()
    if args.check_only:
        print(f"STL benchmark protocol valid: records={len(records)} parents={len(parents)} tasks={len(tasks)}")
        return
    with stage_output(args.output) as out:
        x, descriptor_names = featurize(parents.canonical_parent_smiles.tolist(), feature_set="ecfp4_rdkit2d", progress=True)
        np.savez_compressed(out / "canonical_parent_features_float32.npz",
                            ecfp4=x[:, :2048].astype(np.float32), rdkit2d=x[:, 2048:].astype(np.float32))
        records.to_csv(out / "benchmark_records.csv", index=False)
        parents.to_csv(out / "feature_row_manifest.csv", index=False)
        tasks.to_csv(out / "task_manifest.csv", index=False)
        algorithms.to_csv(out / "algorithm_registry.csv", index=False)
        (out / "feature_registry.json").write_text(json.dumps({
            "ecfp4": {"array": "ecfp4", "dimensions": 2048, "scaling": "algorithm_specific_train_fold_only"},
            "rdkit2d": {"array": "rdkit2d", "dimensions": len(descriptor_names), "descriptor_names": descriptor_names,
                         "imputation": "median_fit_on_training_fold_only", "scaling": "algorithm_specific_train_fold_only"},
            "ecfp4_rdkit2d": {"arrays": ["ecfp4", "rdkit2d"], "dimensions": int(x.shape[1]),
                               "imputation_and_scaling": "fit_on_training_fold_only"},
            "canonicalization": "FragmentParent + Uncharger + canonical tautomer + nonstereo; parent hash verified",
            "rdkit_version": rdBase.rdkitVersion,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "protocol.json").write_text(json.dumps({
            "stage_A": "small-budget broad screening ranked only by five frozen training inner folds",
            "stage_B": "top 2-3 configurations per endpoint refit on all train and evaluated on fixed validation",
            "stage_C": "multi-seed confirmation and optional nonnegative/simple OOF stacking",
            "test_policy": "no test input, label, metric, calibration, or selection",
            "human_F": "excluded because no authorized validation cohort",
            "feature_policy": "one shared deterministic canonical-parent cache; preprocessing fit inside each training fold",
            "hyperparameter_policy": "bounded spaces in algorithm_registry; no uncontrolled Cartesian grid",
            "source_policy": "report fixed validation overall and source-disjoint sensitivity after shortlist only",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Strong STL benchmark protocol\n\nSix authorized human PK endpoints, frozen five-fold training CV, fixed validation confirmation, canonical-parent features, and a staged algorithm registry. No F or test labels are present.\n",
            encoding="utf-8")
        finish_stage(out, "stl_benchmark_protocol", inputs={
            "interface_complete_sha256": sha256(args.interface / "complete.json"),
            "project_splits_complete_sha256": sha256(args.project_splits / "complete.json"),
        }, records=len(records), parents=len(parents), tasks=len(tasks), feature_dimensions=int(x.shape[1]),
            authorized_endpoints=sorted(tasks.endpoint.unique()), human_f_included=False,
            test_labels_read=False, model_fitted=False, partial=False)
    print(f"STL benchmark protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
