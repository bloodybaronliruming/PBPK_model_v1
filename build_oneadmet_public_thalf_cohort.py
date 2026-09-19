#!/usr/bin/env python3
"""Publish a protected public human-plasma half-life modelling cohort.

The cohort is intentionally a broader public human-plasma endpoint, not the
strict human direct-IV terminal-half-life endpoint.  It preserves all formal
external and already-scored PKSmart structures as protected references before
any public-model training can begin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_split_manifest import structure_groups
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "Half-Life_Human-Plasma-pHours_Public.csv"
RECORD_REQUIRED = {
    "SMILES", "Set", "task_name", "target_value", "source_record_id", "parent_id", "scaffold_group",
    "strict_aux_split", "fold_id", "allowed_use", "source_cross_split_scaffold_overlap",
}
FORMAL_REQUIRED = {"molecule_id", "eligibility_decision"}


def reference_structures(formal_registry: pd.DataFrame, sources: list[tuple[Path, str, str]]) -> pd.DataFrame:
    """Resolve every accepted formal molecule to a local, label-free structure source."""
    accepted = set(formal_registry.loc[formal_registry.eligibility_decision.eq("accepted"), "molecule_id"].astype(str))
    if not accepted:
        raise ValueError("Formal registry has no accepted molecules to protect")
    pieces = []
    for path, identifier, smiles_column in sources:
        frame = pd.read_csv(path, keep_default_na=False)
        if identifier not in frame or smiles_column not in frame:
            raise ValueError(f"Formal structure source schema invalid: {path}")
        rows = frame.loc[frame[identifier].astype(str).isin(accepted), [identifier, smiles_column]].copy()
        rows = rows.rename(columns={identifier: "molecule_id", smiles_column: "smiles"})
        rows["reference_source"] = str(path.relative_to(ROOT))
        pieces.append(rows)
    resolved = pd.concat(pieces, ignore_index=True)
    resolved = resolved.loc[resolved.smiles.astype(str).str.strip().ne("")].copy()
    grouped = resolved.groupby("molecule_id", sort=True).smiles.nunique()
    if (grouped > 1).any():
        raise ValueError(f"Conflicting formal structures: {grouped[grouped > 1].index.tolist()}")
    missing = accepted - set(resolved.molecule_id)
    if missing:
        raise ValueError(f"Cannot resolve all accepted formal molecules to structure: {sorted(missing)}")
    resolved = resolved.sort_values(["molecule_id", "reference_source"]).drop_duplicates("molecule_id")
    parents, scaffolds = [], []
    for smiles in resolved.smiles:
        parent, scaffold = structure_groups(smiles)
        parents.append(parent)
        scaffolds.append(scaffold)
    resolved["parent_id"] = parents
    resolved["scaffold_group"] = scaffolds
    return resolved.reset_index(drop=True)


def assign_cohort_usage(records: pd.DataFrame, formal: pd.DataFrame, pksmart_members: pd.DataFrame) -> pd.DataFrame:
    missing = RECORD_REQUIRED - set(records.columns)
    if missing:
        raise ValueError(f"OneADMET split records missing columns: {sorted(missing)}")
    pksmart_required = {"parent_id", "scaffold_group", "endpoint_values_hidden"}
    if pksmart_required - set(pksmart_members.columns) or not pksmart_members.endpoint_values_hidden.astype(bool).all():
        raise ValueError("PKSmart protected membership must remain label-blinded")
    work = records.loc[records.task_name.eq(TASK)].copy()
    if work.empty or not np.isfinite(work.target_value.to_numpy(float)).all():
        raise ValueError("No finite OneADMET human-plasma half-life labels")
    work["formal_external_parent_overlap"] = work.parent_id.isin(set(formal.parent_id))
    work["formal_external_scaffold_overlap"] = work.scaffold_group.isin(set(formal.scaffold_group))
    work["pksmart_evaluated_parent_overlap"] = work.parent_id.isin(set(pksmart_members.parent_id))
    work["pksmart_evaluated_scaffold_overlap"] = work.scaffold_group.isin(set(pksmart_members.scaffold_group))
    work["cohort_usage"] = "quarantine_preexisting_overlap_or_source_conflict"
    train = work.strict_aux_split.eq("aux_train") & work.allowed_use.eq("auxiliary_training")
    validation = work.strict_aux_split.eq("aux_val") & work.allowed_use.eq("auxiliary_training")
    source_test = (work.strict_aux_split.eq("source_test_reserved")
                   & work.allowed_use.eq("oneadmet_reproduction_test_only")
                   & ~work.source_cross_split_scaffold_overlap.astype(bool))
    work.loc[train, "cohort_usage"] = "public_train"
    work.loc[validation, "cohort_usage"] = "public_validation"
    work.loc[source_test, "cohort_usage"] = "public_source_test_scaffold_isolated"
    protected = (work.formal_external_parent_overlap | work.formal_external_scaffold_overlap
                 | work.pksmart_evaluated_parent_overlap | work.pksmart_evaluated_scaffold_overlap)
    work.loc[protected, "cohort_usage"] = "quarantine_protected_evaluation_structure"
    work["raw_value_h"] = np.power(10.0, work.target_value.to_numpy(float))
    if not np.isfinite(work.raw_value_h).all() or (work.raw_value_h <= 0).any():
        raise ValueError("OneADMET log10 half-life cannot be converted to positive finite hours")
    return work.sort_values(["cohort_usage", "source_record_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oneadmet-splits", type=Path,
                        default=ROOT / "data/external/oneadmet_pk_splits_v2")
    parser.add_argument("--formal-registry", type=Path,
                        default=ROOT / "results/analysis/thalf_external_validation_decisions_v23.csv")
    parser.add_argument("--formal-queue", type=Path,
                        default=ROOT / "results/analysis/thalf_external_validation_queue_v4/candidate_records_blinded.csv")
    parser.add_argument("--pkdb-queue", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_review_batch_v2/candidate_records_blinded.csv")
    parser.add_argument("--pubmed-queue", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_independent_batch_v3/candidate_records_blinded.csv")
    parser.add_argument("--pksmart-registration", type=Path,
                        default=ROOT / "results/benchmarks/pksmart_public_thalf_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/public_benchmarks/oneadmet_human_plasma_thalf_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.oneadmet_splits, "oneadmet_pk_safe_splits")
    verify_stage(args.pksmart_registration, "pksmart_public_benchmark_registry")
    needed = [args.oneadmet_splits / "pk_records_split.csv", args.formal_registry, args.formal_queue,
              args.pkdb_queue, args.pubmed_queue, args.pksmart_registration / "cohort_membership_blinded.csv"]
    startup_self_check(needed, output=None if args.check_only else args.output)
    formal_registry = pd.read_csv(args.formal_registry, keep_default_na=False)
    if FORMAL_REQUIRED - set(formal_registry.columns):
        raise ValueError("Formal registry schema is unsupported")
    formal = reference_structures(formal_registry, [
        (args.formal_queue, "molecule_id", "smiles"),
        (args.pkdb_queue, "parent_id", "parent_canonical_smiles"),
        (args.pubmed_queue, "molecule_id", "parent_canonical_smiles"),
    ])
    records = pd.read_csv(args.oneadmet_splits / "pk_records_split.csv", low_memory=False)
    pksmart = pd.read_csv(args.pksmart_registration / "cohort_membership_blinded.csv", keep_default_na=False)
    cohort = assign_cohort_usage(records, formal, pksmart)
    required_uses = {"public_train", "public_validation", "public_source_test_scaffold_isolated"}
    if not required_uses <= set(cohort.cohort_usage):
        raise ValueError("OneADMET public cohort lacks a required development or test partition")
    usable = cohort.cohort_usage.isin(required_uses)
    if cohort.loc[usable].source_record_id.duplicated().any() or cohort.loc[usable].SMILES.duplicated().any():
        raise ValueError("OneADMET usable public cohort contains duplicate source records or SMILES")
    group_splits = cohort.loc[usable].groupby("scaffold_group").cohort_usage.nunique()
    if (group_splits > 1).any():
        raise ValueError("OneADMET public train/validation/test split has scaffold overlap")
    if args.check_only:
        counts = cohort.groupby("cohort_usage").source_record_id.size().to_dict()
        print(f"OneADMET public Thalf cohort contract valid: {counts}")
        return
    with stage_output(args.output) as out:
        audit_columns = ["source_record_id", "SMILES", "Set", "parent_id", "scaffold_group", "strict_aux_split", "fold_id",
                         "cohort_usage", "formal_external_parent_overlap", "formal_external_scaffold_overlap",
                         "pksmart_evaluated_parent_overlap", "pksmart_evaluated_scaffold_overlap"]
        development_columns = ["source_record_id", "SMILES", "parent_id", "scaffold_group", "fold_id",
                               "target_value", "raw_value_h", "cohort_usage"]
        audit = cohort[audit_columns]
        development = cohort.loc[cohort.cohort_usage.isin(["public_train", "public_validation"]), development_columns]
        source_test = cohort.loc[cohort.cohort_usage.eq("public_source_test_scaffold_isolated"), audit_columns]
        if len(source_test) and {"target_value", "raw_value_h"} & set(source_test.columns):
            raise RuntimeError("OneADMET source-test membership must remain label-blinded")
        audit.to_csv(out / "cohort_audit_blinded.csv", index=False)
        development.to_csv(out / "development_records.csv", index=False, float_format="%.8g")
        source_test.to_csv(out / "source_test_membership_blinded.csv", index=False)
        formal.to_csv(out / "protected_formal_external_structures.csv", index=False)
        (cohort.groupby("cohort_usage", dropna=False)
         .agg(records=("source_record_id", "size"), molecules=("parent_id", "nunique"), scaffolds=("scaffold_group", "nunique"))
         .reset_index().to_csv(out / "cohort_summary.csv", index=False))
        (out / "README.md").write_text(
            "# OneADMET public human-plasma half-life cohort\n\n"
            "This is a broad public human-plasma half-life modelling cohort. It is not a human direct-IV "
            "terminal-half-life cohort. `development_records.csv` is the only labelled input for public-model "
            "development. The source-test member file is label-blinded and can be opened only by a later evaluator. "
            "Public-train, public-validation, and source-test partitions are scaffold disjoint. Accepted formal "
            "external and all already-scored PKSmart benchmark parent/scaffold structures are protected and cannot "
            "be used for training or validation.\n",
            encoding="utf-8")
        finish_stage(out, "oneadmet_public_thalf_cohort", inputs={
            "oneadmet_splits_complete_sha256": sha256(args.oneadmet_splits / "complete.json"),
            "formal_registry_sha256": sha256(args.formal_registry),
            "formal_queue_sha256": sha256(args.formal_queue),
            "pkdb_queue_sha256": sha256(args.pkdb_queue),
            "pubmed_queue_sha256": sha256(args.pubmed_queue),
            "pksmart_registration_complete_sha256": sha256(args.pksmart_registration / "complete.json"),
        }, endpoint="public_human_plasma_half_life_hours", strict_iv_terminal=False,
           protected_pksmart_records=len(pksmart), protected_formal_external_molecules=len(formal),
           formal_external_validation_allowed=False, partial=False)
    print(f"OneADMET public human-plasma half-life cohort: {args.output}")


if __name__ == "__main__":
    run_cli(main)
