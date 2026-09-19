#!/usr/bin/env python3
"""Audit source retrievability of label-blinded PKSmart external candidates.

This is a candidate-routing audit, not an eligibility registry and never reads
or emits PKSmart endpoint values.  In particular, an oral product label does
not prove that a compound never had an IV study; it only cannot substantiate a
row whose required original human-IV terminal-PK locator is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED_SCREEN_COLUMNS = [
    "candidate_id", "chembl_id", "chembl_pref_name", "official_label_url", "label_route_evidence",
    "screening_disposition", "evidence_note", "reviewer", "review_date",
]
ORAL_ONLY = "official_oral_label_only"


def validate_screen(screen: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_SCREEN_COLUMNS) - set(screen.columns)
    extra = set(screen.columns) - set(REQUIRED_SCREEN_COLUMNS)
    if missing or extra:
        raise ValueError(f"route-screen 列不一致 missing={sorted(missing)} extra={sorted(extra)}")
    screen = screen[REQUIRED_SCREEN_COLUMNS].copy().fillna("")
    if screen.candidate_id.eq("").any() or screen.candidate_id.duplicated().any():
        raise ValueError("route-screen candidate_id 为空或重复")
    if set(screen.screening_disposition) != {ORAL_ONLY}:
        raise ValueError(f"route-screen 只允许 {ORAL_ONLY}，避免把未审候选伪装为结论")
    for column in ["chembl_id", "chembl_pref_name", "official_label_url", "label_route_evidence",
                   "evidence_note", "reviewer", "review_date"]:
        if screen[column].eq("").any():
            raise ValueError(f"route-screen 缺少 {column}")
    if (~screen.official_label_url.str.startswith("https://dailymed.nlm.nih.gov/")).any():
        raise ValueError("route-screen 必须引用 DailyMed 官方标签链接")
    available = mapping.set_index("candidate_id")
    if not set(screen.candidate_id) <= set(available.index):
        raise ValueError("route-screen 含不在候选映射内的 candidate_id")
    for row in screen.itertuples(index=False):
        mapped = available.loc[row.candidate_id]
        if mapped.mapping_status != "exact_chembl37":
            raise ValueError(f"非精确 ChEMBL 映射不得进入官方标签 route-screen: {row.candidate_id}")
        if str(mapped.chembl_id) != row.chembl_id or str(mapped.chembl_pref_name) != row.chembl_pref_name:
            raise ValueError(f"route-screen 身份与 ChEMBL 映射不一致: {row.candidate_id}")
    return screen


def build_audit(mapping: pd.DataFrame, screen: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "source_row", "chembl_id", "chembl_pref_name", "mapping_status",
                "source_retrieval_priority", "provenance_requirement", "endpoint_values_hidden"}
    if not required <= set(mapping.columns):
        raise ValueError(f"候选映射缺少列: {sorted(required - set(mapping.columns))}")
    if not mapping.endpoint_values_hidden.astype(bool).all():
        raise ValueError("候选映射未保持端点标签隐藏")
    result = mapping.merge(screen, on=["candidate_id", "chembl_id", "chembl_pref_name"], how="left",
                           validate="one_to_one")
    screened = result.screening_disposition.eq(ORAL_ONLY)
    result["provenance_audit_status"] = "primary_source_locator_still_required"
    result.loc[screened, "provenance_audit_status"] = "oral_label_not_a_direct_iv_source"
    result["formal_external_label_allowed"] = False
    result["formal_external_label_reason"] = "row_level_primary_source_locator_required"
    result.loc[screened, "formal_external_label_reason"] = (
        "official_oral_label_not_evidence_of_human_direct_iv_terminal_pk")
    columns = [
        "candidate_id", "source_row", "chembl_id", "chembl_pref_name", "mapping_status",
        "source_retrieval_priority", "provenance_requirement", "provenance_audit_status",
        "formal_external_label_allowed", "formal_external_label_reason", "official_label_url",
        "label_route_evidence", "screening_disposition", "evidence_note", "reviewer", "review_date",
        "endpoint_values_hidden",
    ]
    return result[columns].sort_values(["provenance_audit_status", "candidate_id"], kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_chembl_mapping_v1")
    parser.add_argument("--route-screen", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/regulatory_route_screen_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_provenance_audit_v1")
    args = parser.parse_args()
    verify_stage(args.source, "pksmart_candidate_chembl_mapping")
    startup_self_check([args.source / "candidate_mapping_blinded.csv", args.route_screen], output=args.output)
    mapping = pd.read_csv(args.source / "candidate_mapping_blinded.csv", keep_default_na=False)
    screen = validate_screen(pd.read_csv(args.route_screen, keep_default_na=False), mapping)
    audit = build_audit(mapping, screen)
    if not audit.endpoint_values_hidden.astype(bool).all():
        raise ValueError("输出不得解除端点标签隐藏")
    with stage_output(args.output) as out:
        audit.to_csv(out / "candidate_provenance_audit_blinded.csv", index=False)
        (audit.groupby(["provenance_audit_status", "formal_external_label_reason"], dropna=False).size()
         .rename("records").reset_index().to_csv(out / "audit_summary.csv", index=False))
        (out / "README.md").write_text(
            "# PKSmart candidate provenance audit\n\n"
            "All records remain label-blinded and none is a formal external-validation label. "
            "An `oral_label_not_a_direct_iv_source` outcome means only that the cited official oral-product "
            "label cannot establish the required primary human direct-IV terminal-PK evidence for the unlinked "
            "PKSmart row; it is not a claim that no IV study exists.\n",
            encoding="utf-8")
        finish_stage(out, "pksmart_candidate_provenance_audit", inputs={
            "candidate_mapping_complete_sha256": sha256(args.source / "complete.json"),
            "route_screen_sha256": sha256(args.route_screen),
        }, candidates=len(audit), oral_label_only=int(audit.provenance_audit_status.eq(
            "oral_label_not_a_direct_iv_source").sum()), formal_external_labels=0,
            label_blinded=True, partial=False)


if __name__ == "__main__":
    run_cli(main)
