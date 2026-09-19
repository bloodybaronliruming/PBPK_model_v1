#!/usr/bin/env python3
"""Build small, high-information PubMed discovery subqueues for human IV terminal PK.

This is a source-discovery stage only: it saves bibliographic metadata, not
abstracts, full texts, structures, or endpoint values.  It never creates a
training or evaluation label.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from collect_pubmed_iv_terminal_discovery import (
    EUTILS,
    fetch,
    filter_independent,
    known_source_ids,
    parse_esearch,
    parse_esummary,
    request_url,
)
from pipeline_common import ROOT, configure_logging, finish_stage, run_cli, sha256, stage_output, startup_self_check


BASE_QUERY = (
    '(("terminal half-life"[Title/Abstract] OR "elimination half-life"[Title/Abstract]) '
    'AND (intravenous[Title/Abstract] OR "i.v."[Title/Abstract]) '
    'AND pharmacokinetic*[Title/Abstract] AND humans[MeSH Terms]) '
    'NOT (animals[MeSH Terms] NOT humans[MeSH Terms]) NOT review[Publication Type]'
)

# Signals are deliberately about original dosing-study design, rather than a
# disease or a particular chemical class, to avoid silently creating a narrow
# chemical-space benchmark.
QUERY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("phase1_or_healthy_volunteers", "Phase-I/healthy-volunteer dosing studies",
     f'({BASE_QUERY}) AND ("phase i"[Title/Abstract] OR "healthy volunteer"[Title/Abstract] '
     'OR "healthy subject"[Title/Abstract])'),
    ("single_or_multiple_dose", "Single- or multiple-dose PK studies",
     f'({BASE_QUERY}) AND ("single dose"[Title/Abstract] OR "multiple dose"[Title/Abstract])'),
    ("oral_iv_crossover", "Oral/IV crossover or absolute-bioavailability studies",
     f'({BASE_QUERY}) AND ("absolute bioavailability"[Title/Abstract] '
     'OR "oral and intravenous"[Title/Abstract] OR "intravenous and oral"[Title/Abstract])'),
)


def collect_one(name: str, description: str, query: str, limit: int, email: str) -> tuple[pd.DataFrame, dict, bytes, list[bytes]]:
    parameters = {"db": "pubmed", "term": query, "retmax": str(limit), "sort": "relevance", "retmode": "xml"}
    if email:
        parameters["email"] = email
    search_raw = fetch(request_url("esearch.fcgi", parameters))
    total, identifiers = parse_esearch(search_raw)
    batches: list[bytes] = []
    summaries: list[pd.DataFrame] = []
    for start in range(0, len(identifiers), 100):
        params = {"db": "pubmed", "id": ",".join(identifiers[start:start + 100]), "retmode": "xml"}
        if email:
            params["email"] = email
        raw = fetch(request_url("esummary.fcgi", params))
        batches.append(raw)
        summaries.append(parse_esummary(raw))
    records = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(
        columns=["pubmed_id", "doi", "title", "journal", "publication_date"])
    records.insert(0, "source_queue", name)
    records.insert(1, "source_queue_description", description)
    records.insert(2, "source_rank", range(1, len(records) + 1))
    return records, {"source_queue": name, "description": description, "query": query,
                     "pubmed_matches": total, "retrieved_metadata": len(records), "partial": limit < total}, search_raw, batches


def merge_subqueues(rows: list[pd.DataFrame], used_pmids: set[str], used_dois: set[str]) -> pd.DataFrame:
    """Filter formal-source overlap, then retain a deterministic first queue per PMID."""
    columns = ["source_queue", "source_queue_description", "source_rank", "pubmed_id", "doi", "title", "journal", "publication_date"]
    all_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    independent = filter_independent(all_rows, used_pmids, used_dois)
    independent = independent.sort_values(["source_queue", "source_rank", "pubmed_id"], kind="stable")
    independent = independent.drop_duplicates("pubmed_id", keep="first").copy()
    independent.insert(0, "discovery_id", independent.pubmed_id.map(lambda value: f"pubmed:{value}"))
    return independent.sort_values(["source_queue", "source_rank", "pubmed_id"], kind="stable", ignore_index=True)


def run(root: Path, output: Path, limit_per_queue: int, email: str, delay: float) -> None:
    startup_self_check(output=output)
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    raw: list[tuple[str, bytes, list[bytes]]] = []
    for position, (name, description, query) in enumerate(QUERY_SPECS):
        records, summary, search_raw, batches = collect_one(name, description, query, limit_per_queue, email)
        frames.append(records)
        summaries.append(summary)
        raw.append((name, search_raw, batches))
        if delay and position + 1 < len(QUERY_SPECS):
            time.sleep(delay)
    used_pmids, used_dois = known_source_ids(root)
    discovery = merge_subqueues(frames, used_pmids, used_dois)
    summary = pd.DataFrame(summaries)
    summary["all_queues_retrieved_metadata"] = sum(item["retrieved_metadata"] for item in summaries)
    summary["unique_documents_after_known_source_and_cross_queue_dedup"] = len(discovery)

    with stage_output(output) as out:
        raw_dir = out / "raw"
        raw_dir.mkdir()
        for name, search_raw, batches in raw:
            (raw_dir / f"{name}_esearch.xml").write_bytes(search_raw)
            for number, payload in enumerate(batches, start=1):
                (raw_dir / f"{name}_esummary_{number:03d}.xml").write_bytes(payload)
        discovery.to_csv(out / "candidate_documents_blinded.csv", index=False)
        summary.to_csv(out / "subqueue_summary.csv", index=False)
        (out / "README.md").write_text(
            "# Focused PubMed source-discovery subqueues\n\n"
            "This is bibliographic discovery metadata only. It is not a label set, and a PMID/DOI must not be "
            "used in modelling until an independent original-source review has confirmed human direct IV dosing, "
            "systemic parent analyte, terminal phase, structure identity and a locatable point estimate.\n",
            encoding="utf-8")
        finish_stage(out, "pubmed_iv_terminal_subqueues", inputs={
            "queue_count": len(QUERY_SPECS), "limit_per_queue": limit_per_queue,
            "known_v22_sha256": sha256(root / "results/analysis/thalf_external_validation_decisions_v22.csv"),
            "eutils_base": EUTILS,
        }, candidate_documents=len(discovery), label_blinded=True,
           partial=bool(summary.partial.any()))
    logging.info("PubMed focused subqueues: queues=%d candidates=%d", len(QUERY_SPECS), len(discovery))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pubmed_iv_terminal_subqueues_v1")
    parser.add_argument("--limit-per-queue", type=int, default=25)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--email", default="", help="Optional contact email for NCBI etiquette.")
    args = parser.parse_args()
    if args.limit_per_queue <= 0 or args.delay < 0:
        parser.error("--limit-per-queue must be positive and --delay nonnegative")
    configure_logging(args.root, "collect_pubmed_iv_terminal_subqueues")
    run(args.root, args.output, args.limit_per_queue, args.email, args.delay)


if __name__ == "__main__":
    run_cli(main)
