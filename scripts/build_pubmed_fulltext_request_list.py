#!/usr/bin/env python3
"""Prioritise a blinded PubMed discovery queue for original-fulltext requests.

Title-level signals can reduce unnecessary PDF requests, but are not evidence
of eligibility and must never be merged into the formal decision registry.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check


ANIMAL_PATTERN = re.compile(r"\b(rat|rats|mouse|mice|dog|dogs|hen|hens|animal|animals|veterinary)\b", re.I)
SECONDARY_PATTERN = re.compile(r"\b(population pharmacokinetic|pharmacometric|meta-analysis|review)\b", re.I)
BIOLOGIC_PATTERN = re.compile(
    r"\b(antibody|antibodies|bispecific|biologic|enzyme replacement|t-cell engager|oligomer|donanemab|abelacimab|empasiprubart)\b", re.I)
POSITIVE_SIGNALS = (
    (re.compile(r"\bhealthy volunteer|healthy subject\b", re.I), 4, "healthy-volunteer study signal"),
    (re.compile(r"\bphase i\b", re.I), 3, "phase-I study signal"),
    (re.compile(r"\bintravenous|\bi\.v\.\b", re.I), 3, "IV dosing named in title"),
    (re.compile(r"\bsingle dose|multiple dose\b", re.I), 2, "dose-study signal"),
    (re.compile(r"\boral and intravenous|intravenous and oral|absolute bioavailability\b", re.I), 2,
     "oral-IV crossover signal"),
    (re.compile(r"\bpharmacokinetic", re.I), 1, "PK study signal"),
)


def title_screen(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"discovery_id", "pubmed_id", "doi", "title", "journal", "publication_date", "endpoint_values_hidden"}
    if not required <= set(frame.columns):
        raise ValueError(f"候选表缺少列: {sorted(required - set(frame.columns))}")
    if not frame.endpoint_values_hidden.fillna(False).astype(bool).all():
        raise ValueError("来源发现表必须保持 endpoint_values_hidden=True")
    rows = []
    for row in frame.itertuples(index=False):
        title = str(row.title)
        signals: list[str] = []
        score = 0
        if ANIMAL_PATTERN.search(title):
            disposition = "deprioritized_title_nonhuman_signal"
            signals.append("animal term in title")
        elif SECONDARY_PATTERN.search(title):
            disposition = "deprioritized_title_secondary_analysis_signal"
            signals.append("secondary-analysis term in title")
        elif BIOLOGIC_PATTERN.search(title):
            disposition = "deprioritized_title_large_molecule_signal"
            signals.append("large-molecule/biologic term in title")
        else:
            disposition = "priority_fulltext_request"
            for pattern, points, reason in POSITIVE_SIGNALS:
                if pattern.search(title):
                    score += points
                    signals.append(reason)
            if not signals:
                signals.append("query match only; primary-study and structure check required")
        rows.append({
            "discovery_id": row.discovery_id,
            "pubmed_id": str(row.pubmed_id),
            "doi": str(row.doi),
            "title": title,
            "journal": str(row.journal),
            "publication_date": str(row.publication_date),
            "title_screen_disposition": disposition,
            "priority_score": score,
            "title_screen_rationale": "; ".join(signals),
            "formal_eligibility_decision": "not_reviewed",
            "endpoint_values_hidden": True,
        })
    return pd.DataFrame(rows).sort_values(
        ["title_screen_disposition", "priority_score", "publication_date", "pubmed_id"],
        ascending=[True, False, False, False], kind="stable", ignore_index=True)


def run(source: Path, output: Path, request_limit: int, exclude_pmids: set[str] | None = None) -> None:
    startup_self_check([source / "complete.json", source / "candidate_documents_blinded.csv"], output=output)
    candidates = pd.read_csv(source / "candidate_documents_blinded.csv", keep_default_na=False)
    exclude_pmids = exclude_pmids or set()
    candidates = candidates.loc[~candidates.pubmed_id.astype(str).isin(exclude_pmids)].copy()
    screened = title_screen(candidates)
    priority = screened.loc[screened.title_screen_disposition.eq("priority_fulltext_request")].head(request_limit).copy()
    deferred = screened.loc[~screened.title_screen_disposition.eq("priority_fulltext_request")].copy()
    with stage_output(output) as out:
        priority.to_csv(out / "priority_fulltext_request.csv", index=False)
        deferred.to_csv(out / "deprioritized_title_screen.csv", index=False)
        screened.to_csv(out / "title_screen_all.csv", index=False)
        (out / "README.md").write_text(
            "# Original-fulltext request list\n\n"
            "This list is a title-only triage of blinded PubMed bibliographic metadata. `priority_fulltext_request` "
            "means only that an original PDF could be useful to inspect; it is not an eligibility decision. "
            "`deprioritized` rows are not formal rejections. Every document remains unreviewed until the original "
            "source establishes human direct IV dosing, systemic parent analyte, terminal phase, structure identity "
            "and a locatable point estimate. No endpoint values are present in this stage.\n",
            encoding="utf-8")
        finish_stage(out, "pubmed_fulltext_request_list", inputs={
            "source_complete_sha256": sha256(source / "complete.json"), "request_limit": request_limit,
            "excluded_pubmed_ids": sorted(exclude_pmids),
        }, discovered_documents=len(candidates), priority_fulltext_requests=len(priority),
           deprioritized_title_screen=len(deferred), label_blinded=True, partial_source=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/analysis/pubmed_iv_terminal_subqueues_smoke_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pubmed_fulltext_request_list_v1")
    parser.add_argument("--request-limit", type=int, default=12)
    parser.add_argument("--exclude-pmids", default="",
                        help="comma-separated PubMed IDs already fully screened; exclusion only prevents duplicate requests")
    args = parser.parse_args()
    if args.request_limit <= 0:
        parser.error("--request-limit must be positive")
    excluded = {item.strip() for item in args.exclude_pmids.split(",") if item.strip()}
    run(args.source, args.output, args.request_limit, excluded)


if __name__ == "__main__":
    run_cli(main)
