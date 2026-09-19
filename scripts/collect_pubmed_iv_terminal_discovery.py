#!/usr/bin/env python3
"""Build a blinded PubMed discovery queue for original human IV terminal-PK papers.

The queue deliberately stores bibliographic metadata only.  It does not fetch
abstracts, full texts, structures, or endpoint values; every document still
needs source, structure, parent-analyte, and terminal-phase review.
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = (
    '((pharmacokinetic*[Title/Abstract] OR pharmacokinetics[MeSH Terms]) '
    'AND (intravenous[Title/Abstract] OR "i.v."[Title/Abstract] OR intravenous[MeSH Terms]) '
    'AND ("half-life"[Title/Abstract] OR "half life"[Title/Abstract]) '
    'AND (terminal[Title/Abstract] OR elimination[Title/Abstract]) '
    'AND humans[MeSH Terms]) '
    'NOT (animals[MeSH Terms] NOT humans[MeSH Terms]) '
    'NOT review[Publication Type]'
)


def fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "MTL-model-PubMed-discovery/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def request_url(endpoint: str, parameters: dict[str, str]) -> str:
    return f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(parameters)}"


def parse_esearch(payload: bytes) -> tuple[int, list[str]]:
    root = ET.fromstring(payload)
    total = int(root.findtext("Count", default="0"))
    identifiers = [node.text.strip() for node in root.findall("./IdList/Id") if node.text]
    return total, identifiers


def parse_esummary(payload: bytes) -> pd.DataFrame:
    root = ET.fromstring(payload)
    records = []
    for document in root.findall("./DocSum"):
        values = {item.attrib.get("Name", ""): "".join(item.itertext()).strip()
                  for item in document.findall("./Item")}
        doi = ""
        for item in document.findall(".//Item"):
            if item.attrib.get("Name") == "ArticleId" and item.attrib.get("IdType", "").lower() == "doi":
                doi = "".join(item.itertext()).strip().lower()
                break
        records.append({
            "pubmed_id": document.findtext("Id", default="").strip(),
            "doi": doi,
            "title": values.get("Title", ""),
            "journal": values.get("FullJournalName", "") or values.get("Source", ""),
            "publication_date": values.get("PubDate", ""),
        })
    return pd.DataFrame(records, columns=["pubmed_id", "doi", "title", "journal", "publication_date"])


def known_source_ids(root: Path) -> tuple[set[str], set[str]]:
    """Return previously used/queued source identifiers to preserve independence."""
    pmids: set[str] = set()
    dois: set[str] = set()
    audit = root / "data/processed_v15/audit/source_records.csv"
    if audit.is_file():
        source = pd.read_csv(audit, usecols=["task_id", "status", "pubmed_id", "doi"],
                             keep_default_na=False, low_memory=False)
        source = source.loc[source.task_id.eq("Thalf__human__terminal_iv") & source.status.eq("accepted")]
        pmids.update(source.pubmed_id.astype(str).str.strip().replace("", pd.NA).dropna())
        dois.update(source.doi.astype(str).str.strip().str.lower().replace("", pd.NA).dropna())
    for path, pmid_column, doi_column in [
        (root / "results/analysis/thalf_external_validation_queue_v4/candidate_records_blinded.csv", "pubmed_id", "doi"),
        (root / "results/analysis/pkdb_external_review_batch_v2/candidate_documents.csv", "reference_pmid", "primary_doi"),
    ]:
        if path.is_file():
            frame = pd.read_csv(path, keep_default_na=False)
            pmids.update(frame[pmid_column].astype(str).str.strip().replace("", pd.NA).dropna())
            dois.update(frame[doi_column].astype(str).str.strip().str.lower().replace("", pd.NA).dropna())
    registry = root / "results/analysis/thalf_external_validation_decisions_v22.csv"
    if registry.is_file():
        frame = pd.read_csv(registry, usecols=["source_doi", "primary_source_doi"], keep_default_na=False)
        for column in frame.columns:
            dois.update(frame[column].astype(str).str.strip().str.lower().replace("", pd.NA).dropna())
    return pmids, dois


def filter_independent(records: pd.DataFrame, used_pmids: set[str], used_dois: set[str]) -> pd.DataFrame:
    result = records.copy()
    result["source_overlap_pmid"] = result.pubmed_id.astype(str).isin(used_pmids)
    result["source_overlap_doi"] = result.doi.astype(str).str.lower().isin(used_dois)
    result = result.loc[~result.source_overlap_pmid & ~result.source_overlap_doi].copy()
    result["discovery_status"] = "primary_source_and_structure_review_required"
    result["endpoint_values_hidden"] = True
    return result.sort_values(["publication_date", "pubmed_id"], ascending=[False, False], ignore_index=True)


def run(root: Path, output: Path, query: str, limit: int, batch_size: int,
        delay: float, email: str) -> None:
    startup_self_check(output=output)
    parameters = {"db": "pubmed", "term": query, "retmax": str(limit), "sort": "relevance", "retmode": "xml"}
    if email:
        parameters["email"] = email
    search_url = request_url("esearch.fcgi", parameters)
    search_raw = fetch(search_url)
    total, identifiers = parse_esearch(search_raw)
    summaries: list[pd.DataFrame] = []
    raw_batches: list[tuple[int, bytes]] = []
    for start in range(0, len(identifiers), batch_size):
        identifiers_block = identifiers[start:start + batch_size]
        summary_parameters = {"db": "pubmed", "id": ",".join(identifiers_block), "retmode": "xml"}
        if email:
            summary_parameters["email"] = email
        raw = fetch(request_url("esummary.fcgi", summary_parameters))
        raw_batches.append((start // batch_size + 1, raw))
        summaries.append(parse_esummary(raw))
        if delay and start + batch_size < len(identifiers):
            time.sleep(delay)
    metadata = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(
        columns=["pubmed_id", "doi", "title", "journal", "publication_date"])
    used_pmids, used_dois = known_source_ids(root)
    discovery = filter_independent(metadata, used_pmids, used_dois)
    discovery.insert(0, "discovery_id", discovery.pubmed_id.map(lambda value: f"pubmed:{value}"))
    discovery["query"] = query

    with stage_output(output) as out:
        raw_dir = out / "raw"
        raw_dir.mkdir()
        (raw_dir / "esearch.xml").write_bytes(search_raw)
        for number, raw in raw_batches:
            (raw_dir / f"esummary_{number:03d}.xml").write_bytes(raw)
        discovery.to_csv(out / "candidate_documents_blinded.csv", index=False)
        summary = pd.DataFrame([{
            "pubmed_matches": total,
            "retrieved_metadata": len(metadata),
            "excluded_known_pmid_or_doi": len(metadata) - len(discovery),
            "new_documents": len(discovery),
            "endpoint_values_hidden": True,
        }])
        summary.to_csv(out / "summary.csv", index=False)
        (out / "README.md").write_text(
            "# PubMed human-IV terminal-PK discovery queue\n\n"
            "This queue holds bibliographic discovery metadata only. It must not be treated as labels or used "
            "for model evaluation. Map each document to an exact parent structure and independently confirm "
            "human IV route, systemic parent analyte, terminal/elimination phase, and a source-located point estimate.\n",
            encoding="utf-8")
        finish_stage(out, "pubmed_iv_terminal_discovery", inputs={
            "query": query, "limit": limit, "known_v22_sha256": sha256(root / "results/analysis/thalf_external_validation_decisions_v22.csv"),
        }, pubmed_matches=total, retrieved_metadata=len(metadata), new_documents=len(discovery),
           label_blinded=True, partial=limit < total)
    logging.info("PubMed source-discovery queue: matched=%d retrieved=%d new=%d", total, len(metadata), len(discovery))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pubmed_iv_terminal_discovery_v1")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--email", default="", help="Optional contact email for NCBI etiquette.")
    args = parser.parse_args()
    if args.limit <= 0 or args.batch_size <= 0 or args.delay < 0:
        parser.error("--limit/--batch-size must be positive and --delay nonnegative")
    run(args.root, args.output, args.query, args.limit, args.batch_size, args.delay, args.email)


if __name__ == "__main__":
    run_cli(main)
