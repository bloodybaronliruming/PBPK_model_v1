#!/usr/bin/env python3
"""Map reviewed PubMed papers to ChEMBL structures and build a blinded external index."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from rdkit import Chem

from build_split_manifest import structure_groups
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check
from preprocess_pkdb_candidates import reference_sets


MAPPING_COLUMNS = [
    "candidate_id", "pubmed_id", "source_identifier", "parent_name", "parent_chembl_id", "fulltext_path",
    "mapping_basis", "reviewer", "review_date",
]


def chembl_mapping(database: Path, chembl_ids: set[str]) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    with sqlite3.connect(database) as connection:
        for chembl_id, inchikey, smiles in connection.execute(
            f"SELECT md.chembl_id, cs.standard_inchi_key, cs.canonical_smiles "
            f"FROM molecule_dictionary md JOIN compound_structures cs ON cs.molregno=md.molregno "
            f"WHERE md.chembl_id IN ({','.join('?' for _ in chembl_ids)})", sorted(chembl_ids)):
            rows[chembl_id] = (inchikey, smiles)
    if set(rows) != set(chembl_ids):
        raise ValueError(f"ChEMBL 37 未找到映射: {sorted(set(chembl_ids) - set(rows))}")
    return rows


def build_index(mapping: pd.DataFrame, documents: pd.DataFrame, database: Path,
                used_pmids: set[str], used_parents: set[str], used_scaffolds: set[str], root: Path = ROOT) -> pd.DataFrame:
    missing = set(MAPPING_COLUMNS) - set(mapping.columns)
    if missing or set(mapping.columns) != set(MAPPING_COLUMNS):
        raise ValueError(f"结构映射表列不一致 missing={sorted(missing)} extra={sorted(set(mapping.columns)-set(MAPPING_COLUMNS))}")
    mapping = mapping[MAPPING_COLUMNS].copy().fillna("")
    if mapping.candidate_id.eq("").any() or mapping.candidate_id.duplicated().any():
        raise ValueError("结构映射表含空或重复 candidate_id")
    if mapping.parent_chembl_id.eq("").any() or mapping.fulltext_path.eq("").any():
        raise ValueError("结构映射表缺少 ChEMBL ID 或全文路径")
    doc_columns = {"candidate_id", "pubmed_id", "doi", "title", "journal", "publication_date", "endpoint_values_hidden"}
    if not doc_columns <= set(documents.columns):
        raise ValueError(f"候选文献表缺少列: {sorted(doc_columns-set(documents.columns))}")
    docs = documents[list(doc_columns)].copy().fillna("").set_index("candidate_id")
    structures = chembl_mapping(database, set(mapping.parent_chembl_id))
    rows = []
    for row in mapping.itertuples(index=False):
        if row.candidate_id not in docs.index:
            raise ValueError(f"结构映射不在 PubMed 候选索引: {row.candidate_id}")
        document = docs.loc[row.candidate_id]
        if str(document.pubmed_id) != str(row.pubmed_id):
            raise ValueError(f"结构映射不得变更 PubMed 身份: {row.candidate_id}")
        source_identifier = str(row.source_identifier).strip().lower()
        if source_identifier.startswith("pmid:"):
            if source_identifier.removeprefix("pmid:") != str(row.pubmed_id):
                raise ValueError(f"PMID 来源标识不匹配: {row.candidate_id}")
            identifier_kind = "pmid_fallback"
        elif source_identifier.startswith("10."):
            discovered_doi = str(document.doi).strip().lower()
            # PubMed title-only discovery can legitimately omit a DOI.  An
            # independently checked DOI may then enrich the mapping, but it
            # may never contradict a DOI that the discovery record did have.
            if discovered_doi and source_identifier != discovered_doi:
                raise ValueError(f"结构映射 DOI 与题录 DOI 不一致: {row.candidate_id}")
            identifier_kind = "doi"
        else:
            raise ValueError(f"结构映射不得变更文献来源标识: {row.candidate_id}")
        fulltext = Path(row.fulltext_path)
        if not fulltext.is_absolute():
            fulltext = root / fulltext
        if not fulltext.is_file():
            raise FileNotFoundError(f"未找到已登记全文: {row.fulltext_path}")
        inchikey, smiles = structures[row.parent_chembl_id]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or Chem.MolToInchiKey(mol) != inchikey:
            raise ValueError(f"ChEMBL 映射结构无效: {row.parent_chembl_id}")
        parent_id, scaffold_group = structure_groups(smiles)
        document_overlap = str(row.pubmed_id) in used_pmids
        parent_overlap = parent_id in used_parents
        scaffold_overlap = scaffold_group in used_scaffolds
        if document_overlap or parent_overlap or scaffold_overlap:
            raise ValueError(f"候选不独立: {row.candidate_id} document={document_overlap} "
                             f"parent={parent_overlap} scaffold={scaffold_overlap}")
        rows.append({
            "candidate_id": row.candidate_id,
            "activity_id": f"pubmed:{row.pubmed_id}",
            "molecule_id": stable_id(parent_id),
            # Legacy merger expects a `doi` field.  It is a canonical document
            # identifier: DOI when available, otherwise an explicit PMID fallback.
            "doi": source_identifier,
            "source_identifier": source_identifier,
            "source_identifier_kind": identifier_kind,
            "pubmed_id": row.pubmed_id,
            "title": document.title,
            "journal": document.journal,
            "publication_date": document.publication_date,
            "parent_name": row.parent_name,
            "parent_chembl_id": row.parent_chembl_id,
            "parent_inchikey": inchikey,
            "parent_canonical_smiles": smiles,
            "parent_id": parent_id,
            "scaffold_group": scaffold_group,
            "fulltext_path": row.fulltext_path,
            "mapping_basis": row.mapping_basis,
            "mapping_reviewer": row.reviewer,
            "mapping_review_date": row.review_date,
            "endpoint_values_hidden": True,
        })
    return pd.DataFrame(rows).sort_values("candidate_id", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--documents", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_review_batch_v1/candidate_documents_blinded.csv")
    parser.add_argument("--mapping", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_structure_mapping_v1.csv")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/raw/chembl_37/chembl_37_sqlite/chembl_37.db")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_independent_batch_v3")
    args = parser.parse_args()
    startup_self_check([args.documents, args.mapping, args.database], output=args.output)
    used_pmids, used_parents, used_scaffolds = reference_sets(args.root)
    index = build_index(pd.read_csv(args.mapping, keep_default_na=False),
                        pd.read_csv(args.documents, keep_default_na=False), args.database,
                        used_pmids, used_parents, used_scaffolds, args.root)
    with stage_output(args.output) as out:
        index.to_csv(out / "candidate_records_blinded.csv", index=False)
        (out / "README.md").write_text(
            "# Independent PubMed external candidate index\n\n"
            "This index is label-blinded. Each source mapping was checked against local ChEMBL 37 and was excluded "
            "if it overlapped the v15/TDC/ChEMBL-queue parent or scaffold reference sets. Numerical values are absent. "
            "`doi` is a legacy adapter column holding a canonical source identifier: a normalized DOI, or `pmid:<id>` "
            "only when PubMed supplies no DOI. The explicit `source_identifier_kind` prevents a PMID fallback from "
            "being mistaken for a DOI.\n",
            encoding="utf-8")
        finish_stage(out, "pubmed_external_independent_candidate_index", inputs={
            "documents_sha256": sha256(args.documents), "mapping_sha256": sha256(args.mapping),
            "chembl_database": str(args.database.resolve()),
        }, candidates=len(index), label_blinded=True, partial=False)


if __name__ == "__main__":
    run_cli(main)
