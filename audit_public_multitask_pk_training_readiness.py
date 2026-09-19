#!/usr/bin/env python3
"""Audit the training authority and tiered-use policy of the public PK cohort.

This is a label-preserving readiness audit.  It never changes targets or
splits; its output says how every retained R1--R3 record may enter training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {
    "split", "doc_id", "molecule_id", "task_id", "canonical_unit", "endpoint", "target_value",
    "smiles", "mask", "row_id", "source_task_name", "cohort_source", "record_role", "evidence_role",
    "reliability_tier", "evidence_quality_tier", "translation_role", "source_molecule_id",
    "parent_id", "scaffold_group", "eligible",
}
ROLE_POLICY = {
    "strict_A_curated_development": "human_supervision_and_validation",
    "auxiliary_C_cross_species": "cross_species_representation_and_sensitivity",
    "public_B_published_auxiliary": "published_auxiliary_representation_and_sensitivity",
}


def audit_records(records: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not REQUIRED <= set(records.columns):
        raise ValueError(f"Cohort schema missing: {sorted(REQUIRED - set(records.columns))}")
    if records.row_id.duplicated().any() or set(records.split) - {"train", "val"}:
        raise ValueError("Cohort has duplicate IDs or a forbidden split")
    records = records.copy()
    records["target_value"] = pd.to_numeric(records.target_value, errors="coerce")
    records["mask"] = pd.to_numeric(records["mask"], errors="coerce")
    if records.target_value.isna().any() or not np.isfinite(records.target_value).all() or not records["mask"].eq(1).all():
        raise ValueError("Cohort has missing/non-finite targets or an inactive mask")
    if not set(records.evidence_role) <= set(ROLE_POLICY):
        raise ValueError("Unknown evidence role")
    if not records.loc[records.task_id.str.startswith("F__"), "target_value"].between(0, 1).all():
        raise ValueError("F labels must remain in physical fraction space [0, 1]")
    if not records.reliability_tier.eq(records.evidence_quality_tier).all():
        raise ValueError("Legacy reliability tier must mirror the explicit evidence-quality tier")
    if not records.eligible.astype(str).str.lower().eq("true").all():
        raise ValueError("Ineligible project records entered the development cohort")
    if not records.molecule_id.eq(records.parent_id).all():
        raise ValueError("Global molecule identity must be the canonical parent ID")
    for key in ["molecule_id", "parent_id", "scaffold_group"]:
        if (records.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Cross-split {key} overlap remains")

    keys = ["task_id", "endpoint", "canonical_unit", "cohort_source", "evidence_role", "reliability_tier"]
    summary = (records.groupby(keys, dropna=False)
               .agg(records=("row_id", "size"), molecules=("molecule_id", "nunique"), sources=("doc_id", "nunique"))
               .reset_index())
    split_counts = records.groupby([*keys, "split"], dropna=False).size().unstack("split", fill_value=0)
    split_counts = split_counts.reindex(columns=["train", "val"], fill_value=0).reset_index()
    molecule_overlap = (records.groupby([*keys, "molecule_id"], dropna=False).split.nunique()
                        .rename("cross_split_molecules").reset_index())
    molecule_overlap = (molecule_overlap.groupby(keys, dropna=False).cross_split_molecules
                        .apply(lambda values: int((values > 1).sum())).reset_index())
    source_overlap = (records.groupby([*keys, "doc_id"], dropna=False).split.nunique()
                      .rename("cross_split_sources").reset_index())
    source_overlap = (source_overlap.groupby(keys, dropna=False).cross_split_sources
                      .apply(lambda values: int((values > 1).sum())).reset_index())
    task_quality = summary.merge(split_counts, on=keys).merge(molecule_overlap, on=keys).merge(source_overlap, on=keys)
    task_quality["tier"] = task_quality.reliability_tier.str.extract(r"^(R[1-4])", expand=False)
    task_quality["training_policy"] = task_quality.evidence_role.map(ROLE_POLICY)
    task_quality["cross_split_source_interpretation"] = np.where(
        task_quality.evidence_role.eq("public_B_published_auxiliary"),
        "expected_published_aggregate_source; never a human validation claim",
        np.where(task_quality.cross_split_sources.gt(0), "report_as_source_cluster_sensitivity", "none"),
    )
    task_quality["human_head_selection_authorized"] = (
        task_quality.evidence_role.eq("strict_A_curated_development")
        & task_quality.train.ge(30) & task_quality.val.ge(10)
    )
    f_human = task_quality.loc[task_quality.task_id.eq("F__human__absolute_oral")]
    if len(f_human) != 1 or int(f_human.iloc[0].val) != 0 or bool(f_human.iloc[0].human_head_selection_authorized):
        raise ValueError("Human F must remain explicitly blocked without validation")
    auth = registry[["task_id", "head_selection_authorized"]].copy()
    task_quality = task_quality.merge(auth, on="task_id", validate="one_to_one")
    if not task_quality.head_selection_authorized.eq(task_quality.human_head_selection_authorized).all():
        raise ValueError("Cohort registry and readiness audit disagree on head authority")

    policy = task_quality[["task_id", "tier", "training_policy", "human_head_selection_authorized"]].copy()
    policy["phase_1_all_data"] = True
    policy["phase_2_human_refinement"] = policy.training_policy.eq("human_supervision_and_validation")
    policy["validation_decision_metric"] = np.where(
        policy.human_head_selection_authorized, "fixed_human_validation_only", "not_authorized_for_human_model_selection"
    )
    policy["loss_rule"] = np.where(
        policy.phase_2_human_refinement,
        "masked_loss; task-balanced auxiliary pretraining then human-only refinement",
        "masked_loss; task-balanced auxiliary pretraining and human-validation sensitivity only",
    )
    overall = {
        "records": len(records), "tasks": int(records.task_id.nunique()), "molecules": int(records.molecule_id.nunique()),
        "cross_split_molecule_overlap": int((records.groupby("molecule_id").split.nunique() > 1).sum()),
        "cross_split_parent_overlap": int((records.groupby("parent_id").split.nunique() > 1).sum()),
        "cross_split_scaffold_overlap": int((records.groupby("scaffold_group").split.nunique() > 1).sum()),
        "authorized_human_heads": int(task_quality.human_head_selection_authorized.sum()),
        "all_records_retained_for_tiered_training": True,
        "human_f_head_selection_authorized": False,
        "test_labels_read": False, "formal_external_validation_allowed": False,
    }
    return task_quality.sort_values(keys, ignore_index=True), policy.sort_values("task_id", ignore_index=True), overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/public_development/multitask_pk_24h_v6")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_multitask_pk_training_readiness_v4")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    records_path = args.cohort / "development_records_long.csv"
    registry_path = args.cohort / "task_registry.csv"
    startup_self_check([records_path, registry_path, args.cohort / "complete.json"], output=None if args.check_only else args.output)
    verify_stage(args.cohort, "public_multitask_pk_development_cohort")
    task_quality, policy, overall = audit_records(pd.read_csv(records_path, dtype=str, keep_default_na=False), pd.read_csv(registry_path))
    if args.check_only:
        print(f"Public multitask PK readiness valid: records={overall['records']} authorized_heads={overall['authorized_human_heads']}")
        return
    with stage_output(args.output) as out:
        task_quality.to_csv(out / "task_training_quality.csv", index=False)
        policy.to_csv(out / "tiered_training_policy.csv", index=False)
        (out / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public multitask PK training readiness\n\n"
            "All retained records remain available under a tiered policy. R1 human tasks support supervised training and fixed-human-validation selection when their registry authorizes it. "
            "R3 cross-species and published aggregate tasks are used in task-balanced masked auxiliary pretraining and sensitivity analyses, never relabelled as human outcomes. "
            "Human F remains blocked from head selection because it has no validation records.\n",
            encoding="utf-8")
        finish_stage(out, "public_multitask_pk_training_readiness", inputs={
            "cohort_complete_sha256": sha256(args.cohort / "complete.json"),
        }, **overall, partial=False)
    print(f"Public multitask PK readiness: {args.output}")


if __name__ == "__main__":
    run_cli(main)
