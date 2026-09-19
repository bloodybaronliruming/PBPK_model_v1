#!/usr/bin/env python3
"""Create an eligibility-only review packet for requested PubMed full texts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check


ELIGIBILITY_COLUMNS = [
    "candidate_id", "pubmed_id", "doi", "title", "journal", "publication_date", "fulltext_path",
    "eligibility_decision", "human_evidence", "iv_route_evidence", "parent_systemic_evidence",
    "terminal_phase_evidence", "source_table_or_page", "exclusion_reason", "evidence_note", "reviewer",
    "review_date",
]


def build_review_batch(priority: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"discovery_id", "pubmed_id", "doi", "title", "journal", "publication_date",
                "endpoint_values_hidden", "formal_eligibility_decision"}
    if not required <= set(priority.columns):
        raise ValueError(f"全文索取表缺少列: {sorted(required - set(priority.columns))}")
    if not priority.endpoint_values_hidden.fillna(False).astype(bool).all():
        raise ValueError("全文索取表不可包含端点数值")
    if not priority.formal_eligibility_decision.eq("not_reviewed").all():
        raise ValueError("全文索取表不能预先填入正式资格裁定")
    docs = priority[["discovery_id", "pubmed_id", "doi", "title", "journal", "publication_date",
                     "title_screen_rationale"]].copy()
    docs.insert(0, "candidate_id", docs.pubmed_id.map(lambda value: f"pubmed_iv:{value}"))
    docs["endpoint_values_hidden"] = True
    template = docs[["candidate_id", "pubmed_id", "doi", "title", "journal", "publication_date"]].copy()
    for column in ELIGIBILITY_COLUMNS:
        if column not in template:
            template[column] = ""
    return docs, template[ELIGIBILITY_COLUMNS]


def run(source: Path, output: Path) -> None:
    startup_self_check([source / "complete.json", source / "priority_fulltext_request.csv"], output=output)
    priority = pd.read_csv(source / "priority_fulltext_request.csv", keep_default_na=False)
    documents, template = build_review_batch(priority)
    with stage_output(output) as out:
        documents.to_csv(out / "candidate_documents_blinded.csv", index=False)
        template.to_csv(out / "eligibility_template.csv", index=False)
        (out / "README.md").write_text(
            "# PubMed original-fulltext eligibility review batch\n\n"
            "Use `eligibility_template.csv` only after saving the exact original full text. First fill eligibility "
            "evidence without an endpoint value. `accepted` requires documented human direct IV dosing, systemic "
            "parent analyte and a terminal/elimination phase; numeric extraction is deliberately not present in this "
            "batch and must occur only after a separate eligibility lock. A title-screen priority is not evidence.\n",
            encoding="utf-8")
        finish_stage(out, "pubmed_external_eligibility_review_batch", inputs={
            "priority_source_complete_sha256": sha256(source / "complete.json"),
        }, candidate_documents=len(documents), label_blinded=True, eligibility_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/analysis/pubmed_fulltext_request_list_v2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pubmed_external_review_batch_v1")
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    run_cli(main)
