"""为 OneADMET PK 辅助层生成标签无关、按母体/骨架隔离的安全划分。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from pipeline_common import (ROOT, configure_logging, finish_stage, run_cli, sha256,
                             stage_output, startup_self_check, verify_stage)


def hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_safe_splits(records: pd.DataFrame, seed: int = 2026,
                       validation_fraction: float = 0.1, folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction 必须位于 (0, 0.5)")
    if folds < 2:
        raise ValueError("folds 必须至少为2")
    required = {"SMILES", "Set", "parent_id", "scaffold_group", "task_name",
                "overlap_v15_thalf", "overlap_tdc_obach", "overlap_chembl_external_queue"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"OneADMET PK 辅助表缺少列: {sorted(missing)}")
    structures = records[["SMILES", "Set", "parent_id", "scaffold_group"]].drop_duplicates()
    if structures.SMILES.duplicated().any():
        raise ValueError("同一 OneADMET SMILES 对应多个来源集合或结构键")
    cross_parent = set(structures.groupby("parent_id").Set.nunique().loc[lambda x: x > 1].index)
    cross_scaffold = set(structures.groupby("scaffold_group").Set.nunique().loc[lambda x: x > 1].index)
    structures["source_cross_split_parent_overlap"] = structures.parent_id.isin(cross_parent)
    structures["source_cross_split_scaffold_overlap"] = structures.scaffold_group.isin(cross_scaffold)
    structures["strict_aux_split"] = ""
    structures.loc[structures.parent_id.isin(cross_parent), "strict_aux_split"] = "source_cross_split_quarantine"
    structures.loc[structures.Set.eq("test") & structures.strict_aux_split.eq(""),
                   "strict_aux_split"] = "source_test_reserved"
    trainable = structures.Set.eq("train") & structures.strict_aux_split.eq("")
    fractions = structures.loc[trainable, "scaffold_group"].map(lambda x: hash_fraction(x, seed))
    structures.loc[trainable, "strict_aux_split"] = "aux_train"
    structures.loc[trainable & structures.index.isin(fractions[fractions < validation_fraction].index),
                   "strict_aux_split"] = "aux_val"
    structures["fold_id"] = -1
    train = structures.strict_aux_split.eq("aux_train")
    structures.loc[train, "fold_id"] = structures.loc[train, "scaffold_group"].map(
        lambda x: int(hash_fraction(x, seed + 1) * folds)).clip(upper=folds - 1).astype(int)
    records = records.merge(structures, on=["SMILES", "Set", "parent_id", "scaffold_group"],
                            validate="many_to_one")
    overlap = records[["overlap_v15_thalf", "overlap_tdc_obach",
                       "overlap_chembl_external_queue"]].fillna(False).any(axis=1)
    records["independence_status"] = "nonoverlap_structure"
    records.loc[overlap, "independence_status"] = "overlap_existing_thalf_evidence"
    records["allowed_use"] = "auxiliary_training"
    records.loc[records.strict_aux_split.eq("source_test_reserved"), "allowed_use"] = "oneadmet_reproduction_test_only"
    records.loc[records.strict_aux_split.eq("source_cross_split_quarantine"), "allowed_use"] = "quarantine"
    records.loc[overlap, "allowed_use"] = "overlap_audit_only"
    records["formal_external_validation_allowed"] = False
    return records, structures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_logging(args.root, "build_oneadmet_pk_splits")
    source = args.source or args.root / "data/external/oneadmet_auxiliary_v2"
    output = args.output or args.root / "data/external/oneadmet_pk_splits_v1"
    meta = verify_stage(source, "oneadmet_auxiliary")
    pk_path = source / "pk_auxiliary_long.csv"
    startup_self_check([pk_path], output=None if args.check_only else output)
    records = pd.read_csv(pk_path, low_memory=False)
    split, structures = assign_safe_splits(records, args.seed, args.validation_fraction, args.folds)
    if args.check_only:
        return
    with stage_output(output) as out:
        split.to_csv(out / "pk_records_split.csv", index=False)
        structures.to_csv(out / "pk_structure_manifest.csv", index=False)
        (split.groupby(["task_name", "strict_aux_split", "allowed_use"], dropna=False)
         .agg(records=("source_record_id", "size"), molecules=("parent_id", "nunique"),
              scaffolds=("scaffold_group", "nunique"))
         .reset_index().to_csv(out / "task_split_counts.csv", index=False))
        finish_stage(out, "oneadmet_pk_safe_splits", inputs={
            "oneadmet_complete_sha256": sha256(source / "complete.json"),
        }, source_rows=meta.get("rows"), records=len(split), structures=len(structures),
                     seed=args.seed, validation_fraction=args.validation_fraction, folds=args.folds,
                     cross_source_split_parents=int(structures.loc[
                         structures.source_cross_split_parent_overlap, "parent_id"].nunique()),
                     cross_source_split_scaffolds=int(structures.loc[
                         structures.source_cross_split_scaffold_overlap, "scaffold_group"].nunique()),
                     split_method="SHA256 scaffold-group assignment; author test reserved; cross-set parents quarantined",
                     formal_external_validation_allowed=False, partial=False)


if __name__ == "__main__":
    run_cli(main)
