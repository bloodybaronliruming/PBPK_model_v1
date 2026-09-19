#!/usr/bin/env python3
"""Map label-blinded PKSmart provenance candidates to local ChEMBL identities.

The output contains names and structure identifiers only. It never reads or
emits a PKSmart half-life, and a ChEMBL match is not evidence of eligibility.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def candidate_keys(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "parent_canonical_smiles", "screening_status", "endpoint_values_hidden"}
    if not required <= set(frame.columns):
        raise ValueError(f"候选队列缺少列: {sorted(required - set(frame.columns))}")
    candidates = frame.loc[frame.screening_status.eq("primary_source_provenance_required")].copy()
    if not candidates.endpoint_values_hidden.astype(bool).all():
        raise ValueError("输入候选未保持端点标签隐藏")
    keys = []
    neutralized_keys = []
    uncharger = rdMolStandardize.Uncharger()
    for row in candidates.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.parent_canonical_smiles)
        if mol is None:
            raise ValueError(f"候选结构无效: {row.candidate_id}")
        keys.append(Chem.MolToInchiKey(mol))
        # This secondary key is only a name-retrieval aid for records whose
        # supplied representation is protonated.  It is deliberately kept
        # distinct from the exact parent key and never establishes identity
        # for a formal external-validation label.
        neutralized_keys.append(Chem.MolToInchiKey(uncharger.uncharge(mol)))
    candidates["parent_inchikey"] = keys
    candidates["neutralized_parent_inchikey"] = neutralized_keys
    return candidates


def chembl_map(database: Path, keys: list[str]) -> pd.DataFrame:
    rows = []
    with sqlite3.connect(database) as connection:
        for start in range(0, len(keys), 500):
            block = keys[start:start + 500]
            placeholders = ",".join("?" for _ in block)
            rows.extend(connection.execute(f"""
                SELECT cs.standard_inchi_key, md.chembl_id, COALESCE(md.pref_name, ''), md.max_phase
                FROM compound_structures cs
                JOIN molecule_dictionary md ON md.molregno = cs.molregno
                WHERE cs.standard_inchi_key IN ({placeholders})
            """, block))
    result = pd.DataFrame(rows, columns=["parent_inchikey", "chembl_id", "chembl_pref_name", "chembl_max_phase"])
    # Exact standard InChIKey should map to at most one current ChEMBL parent.
    duplicates = result.duplicated("parent_inchikey", keep=False)
    if duplicates.any():
        raise ValueError("ChEMBL exact InChIKey has multiple parent mappings")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_candidate_queue_v1")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/raw/chembl_37/chembl_37_sqlite/chembl_37.db")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_chembl_mapping_v1")
    args = parser.parse_args()
    verify_stage(args.source, "pksmart_external_candidate_queue")
    startup_self_check([args.source / "candidate_records_blinded.csv", args.database], output=args.output)
    candidates = candidate_keys(pd.read_csv(args.source / "candidate_records_blinded.csv", keep_default_na=False))
    exact = chembl_map(args.database, candidates.parent_inchikey.tolist())
    mapped = candidates.merge(exact, on="parent_inchikey", how="left", validate="one_to_one")
    unresolved = mapped.chembl_id.isna()
    neutralized = chembl_map(args.database, mapped.loc[unresolved, "neutralized_parent_inchikey"].tolist())
    neutralized = neutralized.rename(columns={"parent_inchikey": "neutralized_parent_inchikey",
                                               "chembl_id": "neutralized_chembl_id",
                                               "chembl_pref_name": "neutralized_chembl_pref_name",
                                               "chembl_max_phase": "neutralized_chembl_max_phase"})
    mapped = mapped.merge(neutralized, on="neutralized_parent_inchikey", how="left", validate="one_to_one")
    normalized_match = mapped.chembl_id.isna() & mapped.neutralized_chembl_id.notna()
    for column in ["chembl_id", "chembl_pref_name", "chembl_max_phase"]:
        mapped.loc[normalized_match, column] = mapped.loc[normalized_match, f"neutralized_{column}"]
    mapped["mapping_status"] = "unmapped"
    mapped.loc[mapped.neutralized_chembl_id.notna(), "mapping_status"] = "neutralized_parent_chembl37"
    mapped.loc[mapped.parent_inchikey.isin(exact.parent_inchikey), "mapping_status"] = "exact_chembl37"
    mapped["source_retrieval_priority"] = "manual_primary_source_required"
    mapped.loc[mapped.chembl_max_phase.fillna(0).ge(4), "source_retrieval_priority"] = "regulatory_or_primary_source_priority"
    columns = ["candidate_id", "source_row", "chembl_id", "chembl_pref_name", "chembl_max_phase",
               "parent_inchikey", "neutralized_parent_inchikey", "parent_canonical_smiles", "parent_id", "scaffold_group", "molecule_id",
               "mapping_status", "source_retrieval_priority", "provenance_requirement", "endpoint_values_hidden"]
    with stage_output(args.output) as out:
        mapped[columns].sort_values(["source_retrieval_priority", "chembl_pref_name", "candidate_id"],
                                    kind="stable").to_csv(out / "candidate_mapping_blinded.csv", index=False)
        (mapped.groupby(["mapping_status", "source_retrieval_priority"], dropna=False).size().rename("records")
         .reset_index().to_csv(out / "mapping_summary.csv", index=False))
        (out / "README.md").write_text(
            "# PKSmart candidate ChEMBL mapping\n\n"
            "Exact ChEMBL 37 mappings are for source retrieval only. A regulatory/clinical ChEMBL entry is not "
            "evidence that the PKSmart row is human direct-IV terminal half-life, and no endpoint values are present.\n",
            encoding="utf-8")
        finish_stage(out, "pksmart_candidate_chembl_mapping", inputs={
            "candidate_queue_complete_sha256": sha256(args.source / "complete.json"),
            "chembl_database": str(args.database.resolve()),
        }, candidates=len(mapped), exact_chembl_mappings=int(mapped.mapping_status.eq("exact_chembl37").sum()),
                     neutralized_parent_mappings=int(mapped.mapping_status.eq("neutralized_parent_chembl37").sum()),
                     label_blinded=True, partial=False)


if __name__ == "__main__":
    run_cli(main)
