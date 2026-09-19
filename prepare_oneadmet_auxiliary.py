"""审计 OneADMET 原始附件并发布隔离的多任务/PK 辅助数据层。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from build_split_manifest import structure_groups
from pipeline_common import (ROOT, configure_logging, finish_stage, run_cli, sha256,
                             stable_id, stage_output, startup_self_check)


SMILES_COLUMN = "SMILES"
SPLIT_COLUMN = "Set"
PK_PATTERN = re.compile(r"^(?:Clearance|Half-Life)[_-]", re.IGNORECASE)
HUMAN_PLASMA_THALF = "Half-Life_Human-Plasma-pHours_Public.csv"


def zip_member_sha256(path: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_family(name: str) -> str:
    if name.startswith("BindingDB___"):
        return "bindingdb_activity"
    if name.startswith("Half-Life_"):
        return "half_life"
    if name.startswith("Clearance-"):
        return "clearance"
    return "public_admet_or_activity"


def target_scale(name: str) -> str:
    # OneADMET names transformed continuous endpoints as p<unit>; observed values
    # are retained exactly. Back-transformation is allowed only for named p-units.
    return "log10" if re.search(r"-p(?:Hours|mL|mg|M|uM|nM)", name) else "as_published"


def load_reference_parents(root: Path) -> dict[str, set[str]]:
    specs = {
        "overlap_v15_thalf": [
            (root / "data/processed_v15/audit/source_records.csv", "smiles", True),
        ],
        "overlap_tdc_obach": [
            (root / "data/external/tdc_benchmark/admet_group/half_life_obach/train_val.csv", "Drug", False),
            (root / "data/external/tdc_benchmark/admet_group/half_life_obach/test.csv", "Drug", False),
        ],
        "overlap_chembl_external_queue": [
            (root / "results/analysis/thalf_external_validation_queue_v4/candidate_records_blinded.csv", "smiles", False),
        ],
    }
    result = {}
    for label, sources in specs.items():
        parents = set()
        for path, column, filter_thalf in sources:
            if not path.is_file():
                logging.warning("缺少可选重叠参照: %s", path)
                continue
            frame = pd.read_csv(path, low_memory=False)
            if filter_thalf:
                frame = frame.loc[(frame.task_id == "Thalf__human__terminal_iv")
                                  & (frame.status == "accepted")]
            for value in frame[column].dropna().astype(str):
                try:
                    parent_id, _ = structure_groups(value)
                    parents.add(parent_id)
                except (RuntimeError, ValueError):
                    logging.warning("参照结构规范化失败: %s", value)
        result[label] = parents
    return result


def audit_oneadmet(raw: Path, archive: Path, root: Path, output: Path,
                   chunk_size: int = 2000) -> None:
    startup_self_check([raw, archive], output=output)
    if zip_member_sha256(archive, raw.name) != sha256(raw):
        raise ValueError("ZIP 内 oneADMET.csv 与解压文件 SHA-256 不一致")
    with raw.open(newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream))
    if columns[:1] != [SMILES_COLUMN] or columns[-1:] != [SPLIT_COLUMN]:
        raise ValueError("OneADMET 列契约不符：必须以 SMILES 开始、Set 结束")
    tasks = columns[1:-1]
    pk_tasks = [name for name in tasks if PK_PATTERN.match(name)]
    if HUMAN_PLASMA_THALF not in pk_tasks:
        raise ValueError(f"缺少关键任务 {HUMAN_PLASMA_THALF}")

    counts = {split: np.zeros(len(tasks), dtype=np.int64) for split in ("train", "test")}
    invalid_numeric = np.zeros(len(tasks), dtype=np.int64)
    rows = Counter()
    measurements = 0
    seen_smiles = set()
    duplicate_smiles = 0
    pk_parts = []
    usecols = [SMILES_COLUMN, *pk_tasks, SPLIT_COLUMN]
    for chunk in pd.read_csv(raw, chunksize=chunk_size, low_memory=False):
        split = chunk[SPLIT_COLUMN].fillna("<NA>").astype(str)
        unexpected = set(split) - {"train", "test"}
        if unexpected:
            raise ValueError(f"OneADMET 出现未知 Set: {sorted(unexpected)}")
        for value in chunk[SMILES_COLUMN].astype(str):
            if value in seen_smiles:
                duplicate_smiles += 1
            else:
                seen_smiles.add(value)
        values = chunk[tasks]
        numeric = values.apply(pd.to_numeric, errors="coerce")
        invalid_numeric += (values.notna() & numeric.isna()).sum().to_numpy(np.int64)
        measurements += int(numeric.notna().sum().sum())
        for name in ("train", "test"):
            mask = split.eq(name)
            rows[name] += int(mask.sum())
            counts[name] += numeric.loc[mask].notna().sum().to_numpy(np.int64)
        selected = chunk[usecols]
        long = selected.melt(id_vars=[SMILES_COLUMN, SPLIT_COLUMN], var_name="task_name",
                             value_name="target_value").dropna(subset=["target_value"])
        pk_parts.append(long)

    registry = pd.DataFrame({
        "task_name": tasks,
        "task_family": [task_family(name) for name in tasks],
        "target_scale": [target_scale(name) for name in tasks],
        "train_measurements": counts["train"],
        "test_measurements": counts["test"],
        "invalid_numeric_cells": invalid_numeric,
    })
    registry["total_measurements"] = registry.train_measurements + registry.test_measurements
    if registry.invalid_numeric_cells.sum():
        raise ValueError("OneADMET 含非空但不可解析为数值的标签")

    pk = pd.concat(pk_parts, ignore_index=True)
    pk["source_record_id"] = [stable_id(f"oneadmet|{s}|{t}")
                              for s, t in zip(pk[SMILES_COLUMN], pk.task_name)]
    unique = pk[[SMILES_COLUMN]].drop_duplicates().copy()
    parents, scaffolds, errors = [], [], []
    for value in unique[SMILES_COLUMN]:
        try:
            parent, scaffold = structure_groups(value)
            parents.append(parent); scaffolds.append(scaffold); errors.append("")
        except (RuntimeError, ValueError) as exc:
            parents.append(""); scaffolds.append(""); errors.append(f"{type(exc).__name__}: {exc}")
    unique["parent_id"] = parents
    unique["scaffold_group"] = scaffolds
    unique["standardization_error"] = errors
    pk = pk.merge(unique, on=SMILES_COLUMN, validate="many_to_one")
    references = load_reference_parents(root)
    for label, parent_ids in references.items():
        pk[label] = pk.parent_id.isin(parent_ids)
    pk["target_scale"] = pk.task_name.map(target_scale)
    pk["target_value"] = pd.to_numeric(pk.target_value, errors="raise")
    pk["usage_class"] = "auxiliary_training_only"
    pk.loc[pk.standardization_error.ne(""), "usage_class"] = "quarantine_invalid_structure"

    candidate = pk.loc[pk.task_name.eq(HUMAN_PLASMA_THALF), [
        "source_record_id", SMILES_COLUMN, SPLIT_COLUMN, "parent_id", "scaffold_group",
        "standardization_error", *references.keys(),
    ]].copy()
    candidate["eligibility_status"] = "primary_provenance_required"
    candidate["endpoint_value_hidden"] = True
    if "target_value" in candidate.columns:
        raise RuntimeError("盲态候选意外包含标签")

    parent_split = (pk.drop_duplicates(["parent_id", SPLIT_COLUMN])
                    .groupby("parent_id")[SPLIT_COLUMN].nunique())
    cross_split_parents = set(parent_split[parent_split > 1].index) - {""}
    pk["source_split_parent_overlap"] = pk.parent_id.isin(cross_split_parents)

    with stage_output(output) as out:
        registry.sort_values(["task_family", "task_name"]).to_csv(out / "task_registry.csv", index=False)
        pk.to_csv(out / "pk_auxiliary_long.csv", index=False)
        candidate.to_csv(out / "human_plasma_half_life_candidates_blinded.csv", index=False)
        unique.loc[unique.standardization_error.ne("")].to_csv(out / "structure_failures.csv", index=False)
        finish_stage(out, "oneadmet_auxiliary", inputs={
            "raw_csv": str(raw.resolve()), "raw_sha256": sha256(raw),
            "source_zip": str(archive.resolve()), "zip_sha256": sha256(archive),
        }, source_doi="10.1021/acs.jmedchem.6c00049", rows=sum(rows.values()),
                     source_splits=dict(rows), tasks=len(tasks), measurements=measurements,
                     text_duplicate_smiles=duplicate_smiles, pk_tasks=len(pk_tasks),
                     pk_measurements=len(pk), human_plasma_half_life_candidates=len(candidate),
                     cross_source_split_pk_parents=len(cross_split_parents),
                     label_blinded_candidate_file=True, partial=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--zip", dest="archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_logging(args.root, "prepare_oneadmet_auxiliary")
    raw = args.input or args.root / "data/OneADMET/data/raw/oneADMET.csv"
    archive = args.archive or args.root / "data/OneADMET/data/raw/jm6c00049_si_002.zip"
    output = args.output or args.root / "data/external/oneadmet_auxiliary_v1"
    startup_self_check([raw, archive], output=None if args.check_only else output)
    digest = zip_member_sha256(archive, raw.name)
    if digest != sha256(raw):
        raise ValueError("ZIP 与 CSV 不一致")
    if args.check_only:
        logging.info("OneADMET 文件存在且 ZIP/CSV 哈希一致。")
        return
    audit_oneadmet(raw, archive, args.root, output, args.chunk_size)


if __name__ == "__main__":
    run_cli(main)
