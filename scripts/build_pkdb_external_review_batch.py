"""Build a label-blinded PK-DB primary-source review batch with audited structure overrides."""
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd
from rdkit import Chem

from build_split_manifest import structure_groups
from pipeline_common import (ROOT, configure_logging, finish_stage, run_cli, sha256,
                             stable_id, stage_output, startup_self_check, verify_stage)
from preprocess_pkdb_candidates import reference_sets


MAPPING_COLUMNS = {
    "substance_sid", "source_label", "mapping_decision", "parent_name",
    "parent_chembl_id", "parent_inchikey", "parent_canonical_smiles",
    "mapping_basis", "evidence_source", "reviewer", "review_date",
}
DOCUMENT_COLUMNS = {
    "study_sid", "reference_pmid", "primary_doi", "mapping_decision",
    "mapping_basis", "evidence_source", "reviewer", "review_date",
}
FORBIDDEN_CANDIDATE_COLUMNS = {
    "value", "mean", "median", "min", "max", "sd", "se", "cv",
    "extracted_value", "canonical_value", "raw_value", "target_value", "tdc_value", "y",
}
CANDIDATE_COLUMNS = [
    "candidate_id", "study_sid", "study_name", "reference_pmid", "reference_title",
    "reference_date", "primary_doi", "substance_sid", "label", "parent_name", "parent_chembl_id",
    "parent_inchikey", "parent_canonical_smiles", "parent_id", "scaffold_group",
    "iv_interventions", "source_access", "screening_flags", "endpoint_values_hidden",
    "screening_rank",
]


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _chembl_rows(database: Path, chembl_ids: set[str]) -> dict[str, tuple[str, str]]:
    if not chembl_ids:
        return {}
    rows: dict[str, tuple[str, str]] = {}
    with sqlite3.connect(database) as connection:
        ordered = sorted(chembl_ids)
        for start in range(0, len(ordered), 500):
            block = ordered[start:start + 500]
            placeholders = ",".join("?" for _ in block)
            result = connection.execute(f"""
                SELECT md.chembl_id, cs.standard_inchi_key, cs.canonical_smiles
                FROM molecule_dictionary md
                JOIN compound_structures cs ON cs.molregno=md.molregno
                WHERE md.chembl_id IN ({placeholders})
            """, block)
            for chembl_id, inchikey, smiles in result:
                if chembl_id in rows and rows[chembl_id] != (inchikey, smiles):
                    raise ValueError(f"ChEMBL ID 对应多个结构: {chembl_id}")
                rows[chembl_id] = (inchikey, smiles)
    return rows


def validate_mapping_registry(registry: pd.DataFrame, database: Path) -> pd.DataFrame:
    missing = MAPPING_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"结构映射注册表缺少列: {sorted(missing)}")
    registry = registry.copy().fillna("")
    if registry.substance_sid.astype(str).str.strip().eq("").any():
        raise ValueError("结构映射注册表含空 substance_sid")
    if registry.substance_sid.duplicated().any():
        raise ValueError("结构映射注册表含重复 substance_sid")
    unsupported = set(registry.mapping_decision) - {"accepted", "rejected"}
    if unsupported:
        raise ValueError(f"不支持的结构映射裁定: {sorted(unsupported)}")

    accepted = registry.loc[registry.mapping_decision.eq("accepted")]
    required = ["parent_name", "parent_chembl_id", "parent_inchikey",
                "parent_canonical_smiles", "mapping_basis", "evidence_source"]
    for column in required:
        if accepted[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"accepted 结构映射缺少 {column}")

    database_rows = _chembl_rows(database, set(accepted.parent_chembl_id))
    for row in accepted.itertuples(index=False):
        if row.parent_chembl_id not in database_rows:
            raise ValueError(f"ChEMBL 37 中不存在映射 ID: {row.parent_chembl_id}")
        db_key, db_smiles = database_rows[row.parent_chembl_id]
        if row.parent_inchikey != db_key or row.parent_canonical_smiles != db_smiles:
            raise ValueError(f"映射与 ChEMBL 37 结构不一致: {row.substance_sid}")
        mol = Chem.MolFromSmiles(row.parent_canonical_smiles)
        if mol is None or Chem.MolToInchiKey(mol) != row.parent_inchikey:
            raise ValueError(f"映射 SMILES/InChIKey 不一致: {row.substance_sid}")
    return registry


def validate_document_registry(registry: pd.DataFrame) -> pd.DataFrame:
    missing = DOCUMENT_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"文献映射注册表缺少列: {sorted(missing)}")
    registry = registry.copy().fillna("")
    if registry.study_sid.astype(str).str.strip().eq("").any() or registry.study_sid.duplicated().any():
        raise ValueError("文献映射注册表含空或重复 study_sid")
    unsupported = set(registry.mapping_decision) - {"accepted", "rejected"}
    if unsupported:
        raise ValueError(f"不支持的文献映射裁定: {sorted(unsupported)}")
    accepted = registry.loc[registry.mapping_decision.eq("accepted")]
    for column in ["reference_pmid", "primary_doi", "mapping_basis", "evidence_source"]:
        if accepted[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"accepted 文献映射缺少 {column}")
    if accepted.primary_doi.duplicated().any():
        raise ValueError("accepted 文献映射含重复 DOI")
    return registry


def apply_structure_overrides(candidates: pd.DataFrame, registry: pd.DataFrame,
                              used_parents: set[str], used_scaffolds: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply accepted mappings only to unresolved structures and recompute leakage flags."""
    frame = candidates.copy()
    accepted = registry.loc[registry.mapping_decision.eq("accepted")].set_index("substance_sid")
    audit_rows = []
    for index, row in frame.loc[frame.substance_sid.isin(accepted.index)].iterrows():
        decision = accepted.loc[row.substance_sid]
        existing = _text(row.get("canonical_smiles"))
        if existing:
            raise ValueError(f"注册表不得覆盖已有结构: {row.substance_sid}")
        smiles = decision.parent_canonical_smiles
        parent_id, scaffold = structure_groups(smiles)
        frame.at[index, "chembl_id"] = decision.parent_chembl_id
        frame.at[index, "canonical_smiles"] = smiles
        frame.at[index, "parent_id"] = parent_id
        frame.at[index, "scaffold_group"] = scaffold
        frame.at[index, "structure_status"] = ""
        frame.at[index, "overlap_existing_parent"] = parent_id in used_parents
        frame.at[index, "overlap_existing_scaffold"] = scaffold in used_scaffolds
        audit_rows.append({
            "substance_sid": row.substance_sid,
            "source_label": decision.source_label,
            "parent_name": decision.parent_name,
            "parent_chembl_id": decision.parent_chembl_id,
            "parent_inchikey": decision.parent_inchikey,
            "parent_canonical_smiles": smiles,
            "parent_id": parent_id,
            "scaffold_group": scaffold,
            "overlap_existing_parent": parent_id in used_parents,
            "overlap_existing_scaffold": scaffold in used_scaffolds,
            "mapping_basis": decision.mapping_basis,
            "evidence_source": decision.evidence_source,
        })
    mapped_sids = set(frame.loc[frame.canonical_smiles.fillna("").ne(""), "substance_sid"])
    missing_accepted = set(accepted.index) - set(frame.substance_sid)
    if missing_accepted:
        raise ValueError(f"结构映射注册表含不在候选中的物质: {sorted(missing_accepted)}")
    unresolved = set(registry.loc[registry.mapping_decision.eq("accepted"), "substance_sid"]) - mapped_sids
    if unresolved:
        raise ValueError(f"accepted 映射未生效: {sorted(unresolved)}")
    return frame, pd.DataFrame(audit_rows)


def screening_flags(row) -> str:
    flags = []
    title = _text(row.reference_title).lower()
    label = _text(row.label).lower()
    if label == "angiotensin i":
        flags.append("endogenous_peptide_review")
    if "absorption" in title or "gastric emptying" in title:
        flags.append("absorption_focused_study")
    if "renal excretion" in title:
        flags.append("renal_excretion_focused_study")
    if "pharmacokinetics" not in title and "bioavailability" not in title:
        flags.append("terminal_pk_not_explicit_in_title")
    if _text(row.get("access")).lower() == "closed":
        flags.append("pkdb_closed_record")
    return ";".join(flags)


def build_review_batch(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    if "study_sid" not in frame.columns:
        if "sid" not in frame.columns:
            raise ValueError("PK-DB 候选缺少 study_sid/sid")
        frame["study_sid"] = frame["sid"].astype(str)
    if "primary_doi" not in frame.columns:
        frame["primary_doi"] = ""
    frame["screening_status"] = "primary_source_required"
    frame.loc[frame.document_overlap_existing.fillna(False), "screening_status"] = "exclude_document_overlap"
    frame.loc[frame.overlap_existing_parent.fillna(False), "screening_status"] = "exclude_structure_overlap"
    frame.loc[frame.substance_role.ne("parent_candidate_unverified"), "screening_status"] = "exclude_nonparent_or_marker"
    frame.loc[frame.canonical_smiles.fillna("").eq("")
              & frame.screening_status.eq("primary_source_required"), "screening_status"] = "structure_mapping_required"
    review = frame.loc[frame.screening_status.eq("primary_source_required")].copy()
    review["candidate_id"] = review.apply(
        lambda row: stable_id(f"pkdb|{row.study_sid}|{row.substance_sid}|{row.parent_id}"), axis=1)
    review["parent_name"] = review["label"]
    review["parent_chembl_id"] = review["chembl_id"]
    review["parent_inchikey"] = review["inchikey"]
    review["parent_canonical_smiles"] = review["canonical_smiles"]
    review["source_access"] = review["access"]
    review["screening_flags"] = review.apply(screening_flags, axis=1)
    review = review.sort_values(
        ["screening_flags", "reference_date", "study_sid", "substance_sid"],
        ascending=[True, False, True, True], ignore_index=True)
    review["screening_rank"] = range(1, len(review) + 1)
    result = review[CANDIDATE_COLUMNS].copy()
    forbidden = FORBIDDEN_CANDIDATE_COLUMNS & {column.lower() for column in result.columns}
    if forbidden:
        raise ValueError(f"盲态 PK-DB 候选含禁止标签列: {sorted(forbidden)}")
    if not result.endpoint_values_hidden.astype(bool).all():
        raise ValueError("PK-DB 候选未保持标签隐藏")
    return result


def run(root: Path, source: Path, mapping_registry: Path, document_registry: Path,
        database: Path, output: Path) -> None:
    source_meta = verify_stage(source, "pkdb_candidate_preprocessing")
    audit_meta = verify_stage(root / "data/processed_v15/audit", "audit")
    startup_self_check([mapping_registry, document_registry, database], output=output)
    recorded_database = audit_meta.get("source_database", {})
    stat = database.stat()
    if (Path(recorded_database.get("path", "")).resolve() != database.resolve()
            or recorded_database.get("size") != stat.st_size
            or recorded_database.get("mtime_ns") != stat.st_mtime_ns):
        raise ValueError("ChEMBL 数据库与 v15 审计来源不一致")

    candidates = pd.read_csv(source / "candidate_substances_blinded.csv", dtype={"reference_pmid": str})
    if "study_sid" not in candidates.columns:
        if "sid" not in candidates.columns:
            raise ValueError("PK-DB 预处理候选缺少 study_sid/sid")
        candidates["study_sid"] = candidates["sid"].astype(str)
    registry = validate_mapping_registry(pd.read_csv(mapping_registry, keep_default_na=False), database)
    documents_registry = validate_document_registry(
        pd.read_csv(document_registry, keep_default_na=False, dtype={"reference_pmid": str}))
    accepted_documents = documents_registry.loc[documents_registry.mapping_decision.eq("accepted")].copy()
    unknown_studies = set(accepted_documents.study_sid) - set(candidates.study_sid)
    if unknown_studies:
        raise ValueError(f"文献映射注册表含不在候选中的研究: {sorted(unknown_studies)}")
    pmid_by_study = candidates.groupby("study_sid").reference_pmid.first().fillna("").astype(str)
    for row in accepted_documents.itertuples(index=False):
        if pmid_by_study.get(row.study_sid, "") != row.reference_pmid:
            raise ValueError(f"文献映射 PMID 与上游不一致: {row.study_sid}")
    candidates = candidates.merge(
        accepted_documents[["study_sid", "primary_doi"]], on="study_sid", how="left",
        validate="many_to_one")
    candidates["primary_doi"] = candidates.primary_doi.fillna("")
    _, used_parents, used_scaffolds = reference_sets(root)
    candidates, mapping_audit = apply_structure_overrides(
        candidates, registry, used_parents, used_scaffolds)
    review = build_review_batch(candidates)

    documents = (review.sort_values("screening_rank")
                 .groupby(["study_sid", "study_name", "reference_pmid", "primary_doi", "reference_title",
                           "reference_date", "source_access"], dropna=False, sort=False)
                 .agg(records=("candidate_id", "size"), substances=("substance_sid", "nunique"),
                      first_screening_rank=("screening_rank", "min"),
                      screening_flags=("screening_flags", lambda values: ";".join(
                          dict.fromkeys(flag for value in values for flag in str(value).split(";") if flag))))
                 .reset_index().sort_values("first_screening_rank", ignore_index=True))

    eligibility = review[["candidate_id", "study_sid", "substance_sid", "reference_pmid"]].copy()
    for column in ["fulltext_path", "eligibility_decision", "human_evidence",
                   "iv_route_evidence", "parent_systemic_evidence", "terminal_phase_evidence",
                   "source_table_or_page", "exclusion_reason", "evidence_note", "reviewer", "review_date"]:
        eligibility[column] = ""
    extraction = review[["candidate_id", "study_sid", "substance_sid", "reference_pmid"]].copy()
    for column in ["eligibility_registry_sha256", "extracted_value", "extracted_unit",
                   "aggregation_group", "source_table_or_page", "extraction_note", "reviewer", "review_date"]:
        extraction[column] = ""

    summary = (candidates.assign(final_screening_status=lambda x: "primary_source_required")
               .pipe(lambda x: x))
    # Reuse the exact screening logic so summary counts cannot diverge from the review table.
    summary["final_screening_status"] = "primary_source_required"
    summary.loc[summary.document_overlap_existing.fillna(False), "final_screening_status"] = "exclude_document_overlap"
    summary.loc[summary.overlap_existing_parent.fillna(False), "final_screening_status"] = "exclude_structure_overlap"
    summary.loc[summary.substance_role.ne("parent_candidate_unverified"), "final_screening_status"] = "exclude_nonparent_or_marker"
    summary.loc[summary.canonical_smiles.fillna("").eq("")
                & summary.final_screening_status.eq("primary_source_required"),
                "final_screening_status"] = "structure_mapping_required"
    summary = summary.groupby("final_screening_status").size().rename("records").reset_index()

    with stage_output(output) as out:
        review.to_csv(out / "candidate_records_blinded.csv", index=False)
        documents.to_csv(out / "candidate_documents.csv", index=False)
        eligibility.to_csv(out / "eligibility_template_blinded.csv", index=False)
        extraction.to_csv(out / "extraction_template_after_lock.csv", index=False)
        mapping_audit.to_csv(out / "structure_mapping_audit.csv", index=False)
        summary.to_csv(out / "screening_summary.csv", index=False)
        protocol = f"""# PK-DB independent Thalf primary-source review batch

This label-blinded batch contains {len(review)} study-substance records from {documents.study_sid.nunique()} studies. PK-DB endpoint outputs were unavailable when the upstream snapshot was created; no half-life value was used to map structures, rank studies, or decide overlap.

## Required review order

1. Complete `eligibility_template_blinded.csv` from the original human study without entering a numerical half-life.
2. Lock and hash the eligibility registry.
3. Only for accepted rows, enter a point estimate in `extraction_template_after_lock.csv` with an exact source location.
4. Do not run the frozen v15 model until the final external registry reaches its preregistered sample-size gate and is frozen.

## Inclusion criteria

- Original human study and intravenous administration of the mapped parent drug.
- Parent drug measured in systemic plasma or serum.
- Explicit terminal/elimination-phase half-life point estimate with a recoverable unit and source location.

## Exclusions

- Oral-only exposure, endogenous response marker, metabolite, total radioactivity, tissue fluid, absorption/distribution/effective half-life, range-only value, secondary citation, or unclear terminal method.
- Any parent structure or source already present in v15, TDC/Obach, or the existing ChEMBL external queue.
"""
        (out / "protocol.md").write_text(protocol, encoding="utf-8")
        finish_stage(out, "pkdb_external_review_batch", inputs={
            "pkdb_preprocessed_complete_sha256": sha256(source / "complete.json"),
            "mapping_registry_sha256": sha256(mapping_registry),
            "document_registry_sha256": sha256(document_registry),
            "audit_complete_sha256": sha256(root / "data/processed_v15/audit/complete.json"),
            "chembl_database": str(database.resolve()),
            "chembl_database_size": stat.st_size,
            "chembl_database_mtime_ns": stat.st_mtime_ns,
        }, source_api_version=source_meta.get("source_api_version"), label_blinded=True,
                     records=len(review), studies=documents.study_sid.nunique(),
                     substances=review.substance_sid.nunique(), structure_overrides=len(mapping_audit),
                     partial=False)
    logging.info("PK-DB 盲态原文审核批次完成: %s；%d 条、%d 项研究。",
                 output, len(review), documents.study_sid.nunique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mapping-registry", type=Path)
    parser.add_argument("--document-registry", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_logging(args.root, "build_pkdb_external_review_batch")
    source = args.source or args.root / "data/external/pkdb_preprocessed_v1"
    registry = args.mapping_registry or args.root / "results/analysis/pkdb_structure_mapping_decisions_v1.csv"
    documents = args.document_registry or args.root / "results/analysis/pkdb_document_mapping_decisions_v1.csv"
    database = args.database or args.root / "data/raw/chembl_37/chembl_37_sqlite/chembl_37.db"
    output = args.output or args.root / "results/analysis/pkdb_external_review_batch_v2"
    verify_stage(source, "pkdb_candidate_preprocessing")
    startup_self_check([registry, documents, database], output=None if args.check_only else output)
    if args.check_only:
        validate_mapping_registry(pd.read_csv(registry, keep_default_na=False), database)
        validate_document_registry(pd.read_csv(documents, keep_default_na=False, dtype={"reference_pmid": str}))
        logging.info("PK-DB 原文审核批次自检通过。")
        return
    run(args.root, source, registry, documents, database, output)


if __name__ == "__main__":
    run_cli(main)
