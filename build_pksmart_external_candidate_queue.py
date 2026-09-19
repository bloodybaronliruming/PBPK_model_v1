#!/usr/bin/env python3
"""Create a label-blinded, structure-isolated candidate queue from PKSmart's public external CSV.

PKSmart's external CSV is a valuable published benchmark resource, but it does
not carry a primary source locator per half-life row.  This script therefore
never promotes its values to a formal external label.  It only creates a
source-provenance review queue after removing v15/TDC/current-registry
structural overlaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_split_manifest import structure_groups
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check
from preprocess_pkdb_candidates import reference_sets


REQUIRED_COLUMNS = {"smiles_r", "human_thalf"}
OUTPUT_COLUMNS = [
    "candidate_id", "source_row", "parent_canonical_smiles", "parent_id", "scaffold_group", "molecule_id",
    "overlap_reference_parent", "overlap_reference_scaffold", "overlap_existing_external_molecule",
    "screening_status", "provenance_requirement", "endpoint_values_hidden",
]


def accepted_external_molecules(registry: Path) -> set[str]:
    frame = pd.read_csv(registry, keep_default_na=False)
    required = {"molecule_id", "eligibility_decision"}
    if not required <= set(frame.columns):
        raise ValueError(f"外部注册表缺少列: {sorted(required - set(frame.columns))}")
    return set(frame.loc[frame.eligibility_decision.eq("accepted"), "molecule_id"].astype(str))


def build_queue(raw: pd.DataFrame, used_parents: set[str], used_scaffolds: set[str],
                existing_molecules: set[str]) -> pd.DataFrame:
    if not REQUIRED_COLUMNS <= set(raw.columns):
        raise ValueError(f"PKSmart CSV 缺少列: {sorted(REQUIRED_COLUMNS - set(raw.columns))}")
    rows = []
    # A nonempty public endpoint only establishes that a source lookup may be
    # worthwhile; its numerical value is intentionally never read or emitted.
    for source_row, item in raw.loc[raw.human_thalf.notna() & raw.human_thalf.astype(str).str.strip().ne("")].iterrows():
        smiles = str(item.smiles_r).strip()
        try:
            parent_id, scaffold_group = structure_groups(smiles)
        except (RuntimeError, ValueError):
            rows.append({
                "candidate_id": f"pksmart_external:{source_row}", "source_row": int(source_row),
                "parent_canonical_smiles": "", "parent_id": "", "scaffold_group": "", "molecule_id": "",
                "overlap_reference_parent": False, "overlap_reference_scaffold": False,
                "overlap_existing_external_molecule": False, "screening_status": "structure_standardization_required",
                "provenance_requirement": "primary_source_locator_required", "endpoint_values_hidden": True,
            })
            continue
        molecule_id = stable_id(parent_id)
        parent_overlap = parent_id in used_parents
        scaffold_overlap = scaffold_group in used_scaffolds
        registry_overlap = molecule_id in existing_molecules
        if parent_overlap or scaffold_overlap:
            status = "exclude_reference_structure_overlap"
        elif registry_overlap:
            status = "exclude_existing_external_molecule"
        else:
            status = "primary_source_provenance_required"
        rows.append({
            "candidate_id": f"pksmart_external:{source_row}", "source_row": int(source_row),
            "parent_canonical_smiles": smiles, "parent_id": parent_id, "scaffold_group": scaffold_group,
            "molecule_id": molecule_id, "overlap_reference_parent": parent_overlap,
            "overlap_reference_scaffold": scaffold_overlap,
            "overlap_existing_external_molecule": registry_overlap, "screening_status": status,
            "provenance_requirement": "primary_source_locator_required", "endpoint_values_hidden": True,
        })
    queue = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if queue.candidate_id.duplicated().any() or not queue.endpoint_values_hidden.all():
        raise ValueError("PKSmart 盲态候选唯一性/标签隐藏校验失败")
    return queue.sort_values(["screening_status", "candidate_id"], kind="stable", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/raw/External_test_315.csv")
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "results/analysis/thalf_external_validation_decisions_v23.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_candidate_queue_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    startup_self_check([args.raw, args.registry], output=None if args.check_only else args.output)
    if args.check_only:
        frame = pd.read_csv(args.raw, keep_default_na=False)
        if not REQUIRED_COLUMNS <= set(frame.columns):
            raise ValueError("PKSmart CSV schema is not supported")
        print(f"PKSmart candidate source check passed: rows={len(frame)}")
        return
    _, used_parents, used_scaffolds = reference_sets(args.root)
    queue = build_queue(pd.read_csv(args.raw, keep_default_na=False), used_parents, used_scaffolds,
                        accepted_external_molecules(args.registry))
    with stage_output(args.output) as out:
        queue.to_csv(out / "candidate_records_blinded.csv", index=False)
        (queue.groupby("screening_status").size().rename("records").reset_index()
         .to_csv(out / "screening_summary.csv", index=False))
        (out / "README.md").write_text(
            "# PKSmart external candidate queue\n\n"
            "This is not a formal external-validation label set. PKSmart's public CSV provides structures and "
            "half-life availability but no row-level primary study/FDA-label locator. All endpoint values remain "
            "hidden, and every non-overlapping row requires a recoverable primary source proving human direct IV, "
            "parent systemic analyte, terminal phase, and a locatable point estimate before it can enter the formal registry.\n",
            encoding="utf-8")
        finish_stage(out, "pksmart_external_candidate_queue", inputs={
            "raw_sha256": sha256(args.raw), "formal_registry_sha256": sha256(args.registry),
        }, source_rows=len(queue), label_blinded=True, formal_external_labels=0, partial=False)


if __name__ == "__main__":
    run_cli(main)
