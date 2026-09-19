#!/usr/bin/env python3
"""Build a label-blinded primary-source acquisition queue from PKSmart IV leads."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from render_pksmart_candidate_funnel import validate_leads


def build_queue(leads: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    leads = validate_leads(leads, mapping)
    queue = leads.copy()
    queue["review_priority"] = queue.primary_source_status.map({
        "primary_article_located": "primary_fulltext_request", 
        "primary_article_not_located": "primary_source_locator_search",
    })
    queue["formal_eligibility_status"] = "not_reviewed"
    queue["endpoint_values_hidden"] = True
    columns = [
        "candidate_id", "chembl_id", "chembl_pref_name", "identity_mapping_basis", "primary_source_locator",
        "primary_source_status", "review_priority", "formal_eligibility_status", "official_label_url", "route_note",
        "endpoint_values_hidden", "reviewer", "review_date",
    ]
    return queue[columns].sort_values(["review_priority", "candidate_id"], kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_chembl_mapping_v2")
    parser.add_argument("--iv-product-leads", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/iv_product_leads_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_request_queue_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.mapping, "pksmart_candidate_chembl_mapping")
    startup_self_check([args.mapping / "candidate_mapping_blinded.csv", args.iv_product_leads],
                       output=None if args.check_only else args.output)
    queue = build_queue(pd.read_csv(args.iv_product_leads, keep_default_na=False),
                        pd.read_csv(args.mapping / "candidate_mapping_blinded.csv", keep_default_na=False))
    if not queue.endpoint_values_hidden.astype(bool).all():
        raise ValueError("Primary-source request queue must remain label-blinded")
    if args.check_only:
        print(queue[["candidate_id", "review_priority"]].to_string(index=False))
        return
    with stage_output(args.output) as out:
        queue.to_csv(out / "primary_source_request_queue_blinded.csv", index=False)
        (queue.groupby("review_priority").size().rename("records").reset_index()
         .to_csv(out / "queue_summary.csv", index=False))
        (out / "README.md").write_text(
            "# PKSmart primary-source request queue\n\n"
            "This queue requests or locates primary sources only. A label-listed IV product is not a formal "
            "external-validation label. Before entry into the formal registry, the primary source must still pass "
            "structure/source independence, human direct-IV, parent-analyte, terminal-phase, and point-estimate review.\n",
            encoding="utf-8")
        finish_stage(out, "pksmart_primary_source_request_queue", inputs={
            "mapping_complete_sha256": sha256(args.mapping / "complete.json"),
            "iv_product_leads_sha256": sha256(args.iv_product_leads),
        }, candidates=len(queue), label_blinded=True, formal_external_labels=0, partial=False)


if __name__ == "__main__":
    run_cli(main)
