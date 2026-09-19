#!/usr/bin/env python3
"""Freeze train-only target transforms, tiered loss schedule, and source sensitivity views.

No model is fitted here.  This interface is the only approved input contract for
the next architecture experiments and deliberately excludes test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {"row_id", "task_id", "split", "target_value", "endpoint", "canonical_unit", "doc_id",
            "evidence_role", "reliability_tier", "evidence_quality_tier", "translation_role",
            "record_role", "mask", "molecule_id", "parent_id", "scaffold_group", "eligible"}
TIER_WEIGHTS = {
    "strict_A_curated_development": 0.50,
    "auxiliary_C_cross_species": 0.25,
    "public_B_published_auxiliary": 0.25,
}


def task_transform(task_id: str, endpoint: str, unit: str, project_registry: dict) -> dict:
    if task_id in project_registry:
        source = project_registry[task_id]
        transform = source["transform"]
    elif task_id.startswith("OneADMET__"):
        transform = "log10_as_published" if unit == "log10" else "identity_as_published"
    else:
        raise ValueError(f"No explicit target transform registered for {task_id}")
    output = {
        "target_transform": transform,
        "standardization": "zscore_fit_on_train_only",
        "target_space": "transformed_unstandardized_input",
        "inverse_transform_required_for_reporting": True,
    }
    if endpoint == "F":
        if transform != "identity" or unit != "fraction":
            raise ValueError(f"F task must be physical fraction identity: {task_id}")
        output.update({
            "standardization": "none_physical_fraction",
            "model_output": "sigmoid_physical_fraction",
            "loss": "physical_fraction_huber_masked",
            "reporting_space": "physical_fraction",
            "boundary_label_policy": "retain_exact_0_and_1; never_clip_or_logit_labels; use_stable_sigmoid_only_for_predictions",
            "reporting_inverse": "identity",
        })
    else:
        base_transform = transform.replace("_as_published", "")
        inverse = {"log10": "power10", "logit": "expit", "identity": "identity"}.get(base_transform)
        if inverse is None:
            raise ValueError(f"No reporting inverse registered for {task_id}: {transform}")
        output.update({"model_output": "unbounded_standardized_target", "loss": "huber_masked",
                       "reporting_space": transform, "reporting_inverse": inverse})
    return output


def build_interface(records: pd.DataFrame, readiness: pd.DataFrame, project_registry: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not REQUIRED <= set(records.columns):
        raise ValueError(f"Cohort schema missing {sorted(REQUIRED - set(records.columns))}")
    if set(records.split) - {"train", "val"} or not pd.to_numeric(records["mask"], errors="coerce").eq(1).all():
        raise ValueError("Only active train/validation rows are allowed")
    records = records.copy()
    records["target_value"] = pd.to_numeric(records.target_value, errors="coerce")
    if records.target_value.isna().any() or not np.isfinite(records.target_value).all():
        raise ValueError("Targets must be finite")
    if not records.molecule_id.eq(records.parent_id).all() or not records.eligible.astype(str).str.lower().eq("true").all():
        raise ValueError("Training interface requires eligible canonical-parent identities")
    for key in ["molecule_id", "parent_id", "scaffold_group"]:
        if (records.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Training interface contains cross-split {key} overlap")
    rows = []
    for task_id, group in records.groupby("task_id", sort=True):
        train = group.loc[group.split.eq("train"), "target_value"]
        val = group.loc[group.split.eq("val"), "target_value"]
        if train.empty:
            raise ValueError(f"Task has no train rows: {task_id}")
        std = float(train.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Task has non-identifiable train standard deviation: {task_id}")
        first = group.iloc[0]
        spec = task_transform(task_id, first.endpoint, first.canonical_unit, project_registry)
        f_train = train[(train == 0) | (train == 1)] if first.endpoint == "F" else train.iloc[0:0]
        f_val = val[(val == 0) | (val == 1)] if first.endpoint == "F" else val.iloc[0:0]
        auth = readiness.loc[readiness.task_id.eq(task_id), "human_head_selection_authorized"]
        if len(auth) != 1:
            raise ValueError(f"Readiness does not uniquely cover {task_id}")
        f_blocked = task_id == "F__human__absolute_oral"
        rows.append({
            "task_id": task_id, "endpoint": first.endpoint, "canonical_unit": first.canonical_unit,
            "evidence_role": first.evidence_role, "reliability_tier": first.reliability_tier,
            "evidence_quality_tier": first.evidence_quality_tier, "translation_role": first.translation_role,
            "train_records": len(train), "validation_records": len(val),
            "train_mean": float(train.mean()), "train_std_population": std,
            "train_min": float(train.min()), "train_max": float(train.max()),
            "validation_min_not_fit": float(val.min()) if len(val) else np.nan,
            "validation_max_not_fit": float(val.max()) if len(val) else np.nan,
            "boundary_train_records": len(f_train), "boundary_validation_records": len(f_val),
            "head_selection_authorized": bool(auth.iloc[0]) and not f_blocked,
            "human_F_model_selection_prohibited": f_blocked,
            **spec,
        })
    transforms = pd.DataFrame(rows)
    if not transforms.loc[transforms.task_id.eq("F__human__absolute_oral"), "human_F_model_selection_prohibited"].all():
        raise ValueError("Human F model-selection block was not preserved")

    schedule = transforms[["task_id", "evidence_role", "reliability_tier", "evidence_quality_tier",
                           "translation_role", "head_selection_authorized"]].copy()
    schedule["phase"] = "phase1_all_tiers_shared_representation"
    schedule["tier_loss_weight"] = schedule.evidence_role.map(TIER_WEIGHTS)
    if schedule.tier_loss_weight.isna().any():
        raise ValueError("Unknown evidence role in tiered loss schedule")
    schedule["tasks_in_tier"] = schedule.groupby("evidence_role").task_id.transform("size")
    schedule["within_tier_task_weight"] = 1.0 / schedule.tasks_in_tier
    schedule["record_reduction"] = "masked_mean_per_task_batch"
    schedule["optimizer_sampling"] = "tier_weight_then_uniform_task_with_replacement; optimizer_step_each_batch"
    schedule["phase2_human_refinement"] = schedule.evidence_role.eq("strict_A_curated_development")
    schedule["phase2_task_weight"] = np.where(schedule.phase2_human_refinement, 1.0 / int(schedule.phase2_human_refinement.sum()), 0.0)
    schedule["validation_selection"] = np.where(schedule.head_selection_authorized,
                                                 "fixed_human_validation", "not_used_for_human_head_selection")

    sensitivity = records.loc[records.split.eq("val") & records.evidence_role.eq("strict_A_curated_development"),
                              ["row_id", "task_id", "endpoint", "doc_id", "record_role"]].copy()
    train_sources = records.loc[records.split.eq("train") & records.evidence_role.eq("strict_A_curated_development"),
                                ["task_id", "doc_id"]].drop_duplicates()
    train_sources["source_cluster_seen_in_train"] = True
    sensitivity = sensitivity.merge(train_sources, on=["task_id", "doc_id"], how="left", validate="many_to_one")
    sensitivity["source_cluster_seen_in_train"] = sensitivity["source_cluster_seen_in_train"].eq(True)
    sensitivity["sensitivity_subset"] = np.where(sensitivity.source_cluster_seen_in_train,
                                                   "validation_source_seen_in_train", "validation_source_disjoint")
    summary = {
        "tasks": len(transforms), "records": len(records), "train_records": int(records.split.eq("train").sum()),
        "validation_records": int(records.split.eq("val").sum()),
        "human_head_selection_authorized": int(transforms.head_selection_authorized.sum()),
        "human_F_model_selection_prohibited": True,
        "all_records_available_phase1": True, "all_tasks_sampled_at_least_once_per_epoch": True,
        "test_labels_read": False,
        "source_sensitivity_validation_records": len(sensitivity),
        "cross_split_parent_overlap": 0, "cross_split_scaffold_overlap": 0,
        "training_records_embedded": True,
    }
    return transforms, schedule, sensitivity, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/public_development/multitask_pk_24h_v6")
    parser.add_argument("--readiness", type=Path, default=ROOT / "results/analysis/public_multitask_pk_training_readiness_v4")
    parser.add_argument("--project-task-registry", type=Path, default=ROOT / "data/processed_v15/datasets/task_registry.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    records = args.cohort / "development_records_long.csv"
    readiness = args.readiness / "task_training_quality.csv"
    startup_self_check([records, readiness, args.cohort / "complete.json", args.readiness / "complete.json", args.project_task_registry], output=None if args.check_only else args.output)
    verify_stage(args.cohort, "public_multitask_pk_development_cohort")
    verify_stage(args.readiness, "public_multitask_pk_training_readiness")
    transforms, schedule, sensitivity, summary = build_interface(
        pd.read_csv(records, dtype=str, keep_default_na=False), pd.read_csv(readiness),
        json.loads(args.project_task_registry.read_text(encoding="utf-8")),
    )
    if args.check_only:
        print(f"Multitask PK training interface valid: tasks={summary['tasks']} records={summary['records']}")
        return
    with stage_output(args.output) as out:
        # Embed the exact approved rows so consumers cannot silently bind to a
        # neighbouring historical cohort directory.
        pd.read_csv(records, dtype=str, keep_default_na=False).to_csv(out / "training_records.csv", index=False)
        transforms.to_csv(out / "task_target_transforms_train_only.csv", index=False)
        schedule.to_csv(out / "tiered_task_loss_schedule.csv", index=False)
        sensitivity.to_csv(out / "source_cluster_sensitivity_validation_manifest.csv", index=False)
        (out / "training_contract.json").write_text(json.dumps({
            "phase1": "all R1-R3 tasks; sample every task at least once per epoch, then choose tier using 0.50/0.25/0.25 weights and task uniformly within tier; cycle task records without replacement before reshuffle; masked batch mean and optimizer step every batch",
            "phase2": "R1 human tasks only; sample every task at least once per epoch, then uniformly task-balanced batches; cycle records before reshuffle; optimizer step per batch; validation selection only for six authorized heads",
            "human_F": "physical fraction sigmoid + Huber; exact boundary labels retained; no model selection or performance claim",
            "source_sensitivity": "report validation metrics separately for source clusters seen vs unseen in train",
            "records": "training_records.csv is the immutable interface-bound copy of the v6 cohort; no sibling-path inference",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Multitask PK training interface\n\n"
            "This stage freezes target statistics fitted from train rows only, the R1--R3 two-phase masked loss schedule, and a no-label source-cluster validation membership manifest. "
            "It does not train a model or read any test label.\n", encoding="utf-8")
        finish_stage(out, "multitask_pk_training_interface", inputs={
            "cohort_complete_sha256": sha256(args.cohort / "complete.json"),
            "readiness_complete_sha256": sha256(args.readiness / "complete.json"),
            "project_task_registry_sha256": sha256(args.project_task_registry),
        }, **summary, partial=False)
    print(f"Multitask PK training interface: {args.output}")


if __name__ == "__main__":
    run_cli(main)
