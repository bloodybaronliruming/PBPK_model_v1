"""规范化 PK-DB human+IV 候选的文献、物质和结构标识，保持标签盲态。"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from build_split_manifest import structure_groups
from pipeline_common import (ROOT, configure_logging, finish_stage, run_cli, sha256,
                             stage_output, startup_self_check, verify_stage)


BASE_URL = "https://pk-db.com/api/v1"
EXPLICIT_NONPARENT = {
    "placebo", "urea", "inulin", "iothalamate", "galactose", "glycocholic-acid",
    "p-aminohippurate", "pentagastrin",
}


def fetch_json(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "MTL-model-PKDB-audit/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def identifiers(node: dict) -> dict[str, str]:
    result = {"inchikey": "", "pubchem_cid": "", "chembl_xref": "", "drugbank_xref": ""}
    for item in node.get("annotations") or []:
        collection = str(item.get("collection") or "").lower()
        term = str(item.get("term") or "")
        if collection == "inchikey":
            result["inchikey"] = term
        elif collection == "pubchem.compound":
            result["pubchem_cid"] = term
    for item in node.get("xrefs") or []:
        name = str(item.get("name") or "").lower()
        accession = str(item.get("accession") or "")
        if name == "pubchem" and not result["pubchem_cid"]:
            result["pubchem_cid"] = accession
        elif name == "chembl":
            result["chembl_xref"] = accession
        elif name == "drugbank":
            result["drugbank_xref"] = accession
    return result


def chembl_structures(database: Path, keys: set[str]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame(columns=["inchikey", "chembl_id", "canonical_smiles"])
    rows = []
    with sqlite3.connect(database) as connection:
        ordered = sorted(keys)
        for start in range(0, len(ordered), 500):
            block = ordered[start:start + 500]
            placeholders = ",".join("?" for _ in block)
            rows.extend(connection.execute(f"""
                SELECT cs.standard_inchi_key, md.chembl_id, cs.canonical_smiles
                FROM compound_structures cs
                JOIN molecule_dictionary md ON md.molregno=cs.molregno
                WHERE cs.standard_inchi_key IN ({placeholders})
            """, block))
    frame = pd.DataFrame(rows, columns=["inchikey", "chembl_id", "canonical_smiles"])
    return frame.drop_duplicates("inchikey", keep=False)


def reference_sets(root: Path) -> tuple[set[str], set[str], set[str]]:
    source = pd.read_csv(root / "data/processed_v15/audit/source_records.csv", low_memory=False)
    thalf = source.loc[(source.task_id == "Thalf__human__terminal_iv") & (source.status == "accepted")]
    used_pmids = set(thalf.pubmed_id.dropna().astype(str)) - {"", "nan"}
    structures = list(thalf.smiles.dropna().astype(str))
    queue_path = root / "results/analysis/thalf_external_validation_queue_v4/candidate_records_blinded.csv"
    queue = pd.read_csv(queue_path, low_memory=False)
    queue_pmids = set(queue.pubmed_id.dropna().astype(str)) - {"", "nan"}
    structures.extend(queue.smiles.dropna().astype(str))
    for name in ("train_val.csv", "test.csv"):
        frame = pd.read_csv(root / "data/external/tdc_benchmark/admet_group/half_life_obach" / name)
        structures.extend(frame.Drug.dropna().astype(str))
    parents, scaffolds = set(), set()
    for smiles in structures:
        try:
            parent, scaffold = structure_groups(smiles)
            parents.add(parent); scaffolds.add(scaffold)
        except (RuntimeError, ValueError):
            logging.warning("参照结构规范化失败: %s", smiles)
    return used_pmids | queue_pmids, parents, scaffolds


def explicit_role(sid: str) -> str:
    text = sid.lower()
    if text in EXPLICIT_NONPARENT:
        return "explicit_nonparent_or_marker"
    if text.startswith("14c-") or text.startswith("c14-"):
        return "radiolabeled_material"
    if text in {"exp3174", "m6g", "hydroxyglimepiride"}:
        return "explicit_metabolite"
    return "parent_candidate_unverified"


def run(root: Path, snapshot: Path, database: Path, output: Path,
        base_url: str, request_delay: float) -> None:
    meta = verify_stage(snapshot, "pkdb_iv_candidate_snapshot")
    audit_complete = root / "data/processed_v15/audit/complete.json"
    audit_meta = verify_stage(audit_complete.parent, "audit")
    startup_self_check([database, audit_complete], output=output)
    recorded_database = audit_meta.get("source_database", {})
    database_stat = database.stat()
    if (Path(recorded_database.get("path", "")).resolve() != database.resolve()
            or recorded_database.get("size") != database_stat.st_size
            or recorded_database.get("mtime_ns") != database_stat.st_mtime_ns):
        raise ValueError("ChEMBL 数据库与 v15 已审计来源的路径/大小/时间戳不一致")
    studies = pd.read_csv(snapshot / "candidate_studies_blinded.csv", dtype={"reference_pmid": str})
    with zipfile.ZipFile(snapshot / "raw/pkdb_iv_download.zip") as archive:
        interventions = pd.read_csv(archive.open("interventions.csv"))
    study_ids = set(studies.sid.astype(str))
    interventions = interventions.loc[interventions.study_sid.astype(str).isin(study_ids)
                                      & interventions.route.fillna("").str.lower().eq("iv")].copy()
    substances = sorted(set(interventions.substance.dropna().astype(str)))
    used_pmids, used_parents, used_scaffolds = reference_sets(root)

    with stage_output(output) as out:
        raw_folder = out / "raw/info_nodes"
        raw_folder.mkdir(parents=True)
        node_rows = []
        for position, sid in enumerate(substances):
            url = f"{base_url}/info_nodes/{urllib.parse.quote(sid, safe='')}/"
            try:
                payload = fetch_json(url)
                node = json.loads(payload)
                (raw_folder / f"{sid}.json").write_bytes(payload)
                row = {"substance_sid": sid, "node_status": "ok", "label": node.get("label", ""),
                       "deprecated": node.get("deprecated", False), **identifiers(node)}
            except Exception as exc:
                logging.warning("PK-DB 物质节点获取失败 sid=%s error=%s", sid, exc)
                row = {"substance_sid": sid, "node_status": f"failed:{type(exc).__name__}",
                       "label": "", "deprecated": False, "inchikey": "", "pubchem_cid": "",
                       "chembl_xref": "", "drugbank_xref": ""}
            node_rows.append(row)
            if request_delay and position + 1 < len(substances):
                time.sleep(request_delay)
        nodes = pd.DataFrame(node_rows)
        mapped = chembl_structures(database, set(nodes.inchikey) - {""})
        # 多个 PK-DB 节点可代表同一母体的盐型/标记形式，因此允许多个
        # source node 映射到同一个 ChEMBL 规范结构；反向一键多结构仍被拒绝。
        nodes = nodes.merge(mapped, on="inchikey", how="left", validate="many_to_one")
        nodes["substance_role"] = nodes.substance_sid.map(explicit_role)
        parents, scaffolds, errors = [], [], []
        for smiles in nodes.canonical_smiles.fillna(""):
            if not smiles:
                parents.append(""); scaffolds.append(""); errors.append("structure_unavailable")
                continue
            try:
                parent, scaffold = structure_groups(smiles)
                parents.append(parent); scaffolds.append(scaffold); errors.append("")
            except (RuntimeError, ValueError) as exc:
                parents.append(""); scaffolds.append(""); errors.append(f"{type(exc).__name__}: {exc}")
        nodes["parent_id"] = parents
        nodes["scaffold_group"] = scaffolds
        nodes["structure_status"] = errors
        nodes["overlap_existing_parent"] = nodes.parent_id.isin(used_parents)
        nodes["overlap_existing_scaffold"] = nodes.scaffold_group.isin(used_scaffolds)

        links = (interventions[["study_sid", "study_name", "substance", "intervention_pk"]]
                 .drop_duplicates(["study_sid", "substance"])
                 .rename(columns={"substance": "substance_sid"}))
        result = studies.merge(links, left_on="sid", right_on="study_sid", how="inner")
        result = result.merge(nodes, on="substance_sid", how="left", validate="many_to_one")
        result["document_overlap_existing"] = result.reference_pmid.fillna("").astype(str).isin(used_pmids)
        result["screening_status"] = "primary_source_required"
        result.loc[result.document_overlap_existing, "screening_status"] = "exclude_document_overlap"
        result.loc[result.overlap_existing_parent.fillna(False), "screening_status"] = "exclude_structure_overlap"
        result.loc[result.substance_role.ne("parent_candidate_unverified"), "screening_status"] = "exclude_nonparent_or_marker"
        result.loc[result.structure_status.eq("structure_unavailable")
                   & result.screening_status.eq("primary_source_required"), "screening_status"] = "structure_mapping_required"
        result["endpoint_values_hidden"] = True
        prohibited = {"value", "mean", "median", "min", "max", "sd", "se", "cv"}
        if prohibited & set(result.columns):
            raise RuntimeError("预处理候选含禁止数值列")
        rank = {"primary_source_required": 0, "structure_mapping_required": 1,
                "exclude_document_overlap": 2, "exclude_structure_overlap": 3,
                "exclude_nonparent_or_marker": 4}
        result["screening_rank_group"] = result.screening_status.map(rank).fillna(9)
        result = result.sort_values(["screening_rank_group", "reference_date", "sid", "substance_sid"],
                                    ascending=[True, False, True, True], ignore_index=True)

        nodes.to_csv(out / "substance_registry.csv", index=False)
        result.to_csv(out / "candidate_substances_blinded.csv", index=False)
        (result.groupby("screening_status").size().rename("records").reset_index()
         .to_csv(out / "screening_summary.csv", index=False))
        finish_stage(out, "pkdb_candidate_preprocessing", inputs={
            "snapshot_complete_sha256": sha256(snapshot / "complete.json"),
            "audit_complete_sha256": sha256(audit_complete),
            "chembl_database": str(database.resolve()),
            "chembl_database_size": database_stat.st_size,
            "chembl_database_mtime_ns": database_stat.st_mtime_ns,
        }, source_api_version=meta.get("api_version"), studies=result.sid.nunique(),
                     substances=len(nodes), mapped_structures=int(nodes.canonical_smiles.notna().sum()),
                     primary_source_candidates=int(result.screening_status.eq("primary_source_required").sum()),
                     structure_mapping_required=int(result.screening_status.eq("structure_mapping_required").sum()),
                     label_blinded=True, partial=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_logging(args.root, "preprocess_pkdb_candidates")
    snapshot = args.snapshot or args.root / "data/external/pkdb_iv_candidates_v1"
    database = args.database or args.root / "data/raw/chembl_37/chembl_37_sqlite/chembl_37.db"
    output = args.output or args.root / "data/external/pkdb_preprocessed_v1"
    verify_stage(snapshot, "pkdb_iv_candidate_snapshot")
    startup_self_check([database], output=None if args.check_only else output)
    if args.check_only:
        logging.info("PK-DB 快照与 ChEMBL 映射库自检通过。")
        return
    run(args.root, snapshot, database, output, args.base_url, args.request_delay)


if __name__ == "__main__":
    run_cli(main)
