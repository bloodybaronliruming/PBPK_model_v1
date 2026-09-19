#!/usr/bin/env python3
"""Read-only N0 reproducibility audit for Jia et al. (2025).

This stage deliberately does not fit a model, calculate predictive metrics, or
write endpoint label values.  It records only source hashes, field-level
availability, author-split membership (hashed), and leakage-relevant overlap
counts needed to decide whether an author-protocol baseline is justified.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

from pipeline_common import (ROOT, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stable_id, stage_output,
                             startup_self_check)


WORKBOOK_SHEET = "Fu_VDss_CL_modeling_set"
CMP_SHEET = "106_cmp_test_set"
ENDPOINTS = {
    "Fu": {"value": "Fu_final", "log_value": "lgFu", "split": "Fu_train_test", "unit": "fraction (unitless)"},
    "CL": {"value": "CL_final(L/hour/kg)", "log_value": "lgCL", "split": "CL_train_test", "unit": "L/hour/kg"},
    "VDss": {"value": "VD_final(L/kg)", "log_value": "lgVD", "split": "VD_train_test", "unit": "L/kg"},
}


def canonical_parent_and_scaffold(smiles: str) -> tuple[str, str]:
    """Use the project-standard parent/scaffold identity without exporting SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid_smiles")
    parent = rdMolStandardize.FragmentParent(mol)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    parent = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
    Chem.RemoveStereochemistry(parent)
    parent_smiles = Chem.MolToSmiles(parent, isomericSmiles=False)
    scaffold = MurckoScaffold.GetScaffoldForMol(parent)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold.GetNumAtoms() else parent_smiles
    return stable_id(parent_smiles), stable_id(scaffold_smiles)


def no_label_field_dictionary(frame: pd.DataFrame, label_columns: set[str]) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        rows.append({
            "sheet": WORKBOOK_SHEET,
            "field": column,
            "role": "label_or_transformed_label" if column in label_columns else "metadata_or_structure",
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "nonmissing_rows": int(series.notna().sum()),
            "missing_rows": int(series.isna().sum()),
            "unique_nonmissing_values": int(series.nunique(dropna=True)),
            "values_exported": False,
        })
    return pd.DataFrame(rows)


def split_summary(records: pd.DataFrame, endpoint: str, spec: dict) -> tuple[list[dict], pd.DataFrame, dict]:
    usable = records.loc[records[spec["value"]].notna() & records[spec["split"]].notna()].copy()
    usable["author_split"] = usable[spec["split"]].astype(str).str.strip().str.lower()
    allowed = {"train", "test"}
    bad = sorted(set(usable.author_split) - allowed)
    if bad:
        raise ValueError(f"{endpoint} has unexpected author split values: {bad}")
    if usable.empty or set(usable.author_split) != allowed:
        raise ValueError(f"{endpoint} lacks a complete train/test author split")

    membership = usable[["parent_id", "scaffold_id", "author_split"]].drop_duplicates().copy()
    membership.insert(0, "endpoint", endpoint)
    train = usable.loc[usable.author_split.eq("train")]
    test = usable.loc[usable.author_split.eq("test")]
    train_parents, test_parents = set(train.parent_id), set(test.parent_id)
    train_scaffolds, test_scaffolds = set(train.scaffold_id), set(test.scaffold_id)
    rows = []
    for split, chunk in (("train", train), ("test", test)):
        rows.append({
            "endpoint": endpoint,
            "author_split": split,
            "records_with_observed_label": int(len(chunk)),
            "unique_parent_ids": int(chunk.parent_id.nunique()),
            "unique_scaffold_ids": int(chunk.scaffold_id.nunique()),
            "duplicate_records_by_parent": int(len(chunk) - chunk.parent_id.nunique()),
            "duplicate_records_by_parent_and_split": int(chunk.duplicated(["parent_id", "author_split"]).sum()),
            "raw_value_exported": False,
        })
    overlap = {
        "endpoint": endpoint,
        "train_test_parent_overlap": int(len(train_parents & test_parents)),
        "train_test_scaffold_overlap": int(len(train_scaffolds & test_scaffolds)),
        "split_is_parent_disjoint": bool(not (train_parents & test_parents)),
        "split_is_scaffold_disjoint": bool(not (train_scaffolds & test_scaffolds)),
        "author_split_designation": "author_train_test_field; random_or_scaffold_method_not_stated_in_workbook",
    }
    return rows, membership, overlap


def main():
    parser = argparse.ArgumentParser(description="Read-only Jia 2025 N0 audit; never trains a model.")
    parser.add_argument("--workbook", type=Path, default=ROOT / "基准研究参考/jm5c00340_si_001.xlsx")
    parser.add_argument("--article", type=Path, default=ROOT / "基准研究参考/Jia et al. - 2025 - Application of Machine Learning and Mechanistic Modeling to Predict Intravenous Pharmacokinetic Prof.pdf")
    parser.add_argument("--supporting-pdf", type=Path, default=ROOT / "基准研究参考/jm5c00340_si_002.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/jia2025_n0_audit_v1")
    args = parser.parse_args()
    startup_self_check([args.workbook, args.article, args.supporting_pdf], output=args.output)
    configure_logging(ROOT, "audit_jia2025_n0")
    RDLogger.DisableLog("rdApp.*")

    inputs = {str(p.resolve()): sha256(p) for p in (args.workbook, args.article, args.supporting_pdf)}
    with stage_output(args.output) as out:
        excel = pd.ExcelFile(args.workbook)
        if WORKBOOK_SHEET not in excel.sheet_names or CMP_SHEET not in excel.sheet_names:
            raise ValueError("Required Jia workbook sheets are absent")
        model = pd.read_excel(args.workbook, sheet_name=WORKBOOK_SHEET)
        cmp106 = pd.read_excel(args.workbook, sheet_name=CMP_SHEET)
        required = {"SMILES", "SMILES_parent", *[x for spec in ENDPOINTS.values() for x in (spec["value"], spec["log_value"], spec["split"])]}
        if not required <= set(model.columns):
            raise ValueError(f"Modeling sheet is missing fields: {sorted(required - set(model.columns))}")
        if "SMILES" not in cmp106.columns:
            raise ValueError("106 comparison sheet lacks SMILES")

        parent_scaffold = model.SMILES_parent.map(canonical_parent_and_scaffold)
        model["parent_id"] = parent_scaffold.map(lambda x: x[0])
        model["scaffold_id"] = parent_scaffold.map(lambda x: x[1])
        cmp_ids = cmp106.SMILES.map(canonical_parent_and_scaffold)
        cmp106["parent_id"] = cmp_ids.map(lambda x: x[0])
        cmp106["scaffold_id"] = cmp_ids.map(lambda x: x[1])

        label_columns = {x for spec in ENDPOINTS.values() for x in (spec["value"], spec["log_value"])}
        fields = no_label_field_dictionary(model, label_columns)
        cmp_fields = no_label_field_dictionary(cmp106, set(c for c in cmp106.columns if c.startswith("pred_") or c in {"fu", "CL_final(L/hour/kg)", "VD_final(L/kg)"}))
        cmp_fields["sheet"] = CMP_SHEET
        pd.concat([fields, cmp_fields], ignore_index=True).to_csv(out / "field_dictionary_no_labels.csv", index=False)

        sheet_inventory = []
        for sheet in excel.sheet_names:
            x = pd.read_excel(args.workbook, sheet_name=sheet)
            sheet_inventory.append({"sheet": sheet, "rows": int(len(x)), "columns": int(len(x.columns)), "column_names_exported": sheet in {WORKBOOK_SHEET, CMP_SHEET}})
        pd.DataFrame(sheet_inventory).to_csv(out / "workbook_sheet_inventory.csv", index=False)

        summaries, memberships, overlaps = [], [], []
        for endpoint, spec in ENDPOINTS.items():
            part_summary, part_membership, part_overlap = split_summary(model, endpoint, spec)
            summaries.extend(part_summary)
            memberships.append(part_membership)
            overlaps.append(part_overlap)
        pd.DataFrame(summaries).to_csv(out / "endpoint_author_split_summary.csv", index=False)
        pd.concat(memberships, ignore_index=True).sort_values(["endpoint", "author_split", "parent_id"]).to_csv(out / "author_split_membership_hashed.csv", index=False)
        pd.DataFrame(overlaps).to_csv(out / "author_split_overlap_audit.csv", index=False)

        cmp_rows = []
        all_parent_ids, all_scaffold_ids = set(model.parent_id), set(model.scaffold_id)
        cmp_parent_ids, cmp_scaffold_ids = set(cmp106.parent_id), set(cmp106.scaffold_id)
        for endpoint, spec in ENDPOINTS.items():
            usable = model.loc[model[spec["value"]].notna() & model[spec["split"]].notna()].copy()
            usable["author_split"] = usable[spec["split"]].astype(str).str.strip().str.lower()
            for split in ("all_modeling_rows", "train", "test"):
                member = usable if split == "all_modeling_rows" else usable.loc[usable.author_split.eq(split)]
                cmp_rows.append({
                    "endpoint": endpoint,
                    "modeling_member_set": split,
                    "cmp106_rows": int(len(cmp106)),
                    "cmp106_unique_parent_ids": int(cmp106.parent_id.nunique()),
                    "cmp106_parent_overlap": int(len(cmp_parent_ids & set(member.parent_id))),
                    "cmp106_scaffold_overlap": int(len(cmp_scaffold_ids & set(member.scaffold_id))),
                    "raw_labels_exported": False,
                })
        cmp_rows.insert(0, {"endpoint": "all", "modeling_member_set": "all_raw_rows", "cmp106_rows": int(len(cmp106)),
                            "cmp106_unique_parent_ids": int(cmp106.parent_id.nunique()),
                            "cmp106_parent_overlap": int(len(cmp_parent_ids & all_parent_ids)),
                            "cmp106_scaffold_overlap": int(len(cmp_scaffold_ids & all_scaffold_ids)), "raw_labels_exported": False})
        pd.DataFrame(cmp_rows).to_csv(out / "cmp106_relationship_audit.csv", index=False)

        source_registry = {
            "reference": "Jia et al., Journal of Medicinal Chemistry (2025), DOI 10.1021/acs.jmedchem.5c00340",
            "local_article_license_text": "CC BY-NC-ND 4.0 (as printed on local article first page)",
            "reuse_interpretation": "Local article indicates noncommercial attribution license; this audit does not provide legal advice and does not establish rights for downstream redistribution.",
            "local_author_code_repository": "not_found_in_project_workspace",
            "local_versioned_execution_environment": "not_found_in_project_workspace",
            "data_source_scope": "local article and supporting workbook/PDF only; no external retrieval performed",
            "training_performed": False,
            "predictive_metric_computed": False,
            "project_consumed_test_accessed": False,
        }
        dump_json(out / "source_reuse_registry.json", source_registry)
        task_definitions = [
            {"endpoint": "Fu", "species": "human", "route_context": "human IV PK study; parameter source compilation", "system": "fraction unbound", "unit": "fraction (unitless)", "modeling_target": "base-10 log (lgFu)", "author_reported_primary_metrics": "R2, MAE, RMSE; GMFE and percent within 2-fold after inverse transform", "paper_best_model": "SVM with merged descriptors"},
            {"endpoint": "CL", "species": "human", "route_context": "human IV PK study; in vivo clearance", "system": "systemic in vivo clearance", "unit": "L/hour/kg", "modeling_target": "base-10 log (lgCL)", "author_reported_primary_metrics": "R2, MAE, RMSE; GMFE and percent within 2-fold after inverse transform", "paper_best_model": "consensus model"},
            {"endpoint": "VDss", "species": "human", "route_context": "human IV PK study; volume parameter", "system": "steady-state volume of distribution", "unit": "L/kg", "modeling_target": "base-10 log (lgVD)", "author_reported_primary_metrics": "R2, MAE, RMSE; GMFE and percent within 2-fold after inverse transform", "paper_best_model": "consensus model"},
        ]
        pd.DataFrame(task_definitions).to_csv(out / "paper_task_definitions.csv", index=False)

        overlap_df = pd.DataFrame(overlaps)
        grade = "B"
        grade_reason = "Endpoint definitions, units, local data, and author train/test fields are available; the local sources do not state the split-generation algorithm or supply an author code environment."
        if overlap_df.train_test_parent_overlap.eq(0).all() and overlap_df.train_test_scaffold_overlap.eq(0).all():
            grade = "A"
            grade_reason = "Endpoint definitions, units, local data, and parent/scaffold-disjoint author splits are available; exact preprocessing/code still require separate verification."
        report = [
            "# Jia 2025 N0 reproducibility audit", "",
            "## Scope", "",
            "Read-only metadata, membership and leakage audit. No model was fit, no predictive metric was computed, and no project-consumed test set was read.", "",
            "## Decision", "",
            f"**Provisional reproduction grade: {grade}.** {grade_reason}", "",
            "The paper specifies human IV context; Fu is unitless, CL is L/hour/kg, VDss is L/kg, and the three targets are log10-transformed. The paper describes approximately 88/12 author train/test splits, 10-fold CV within training, and removal of training compounds sharing identical descriptor vectors with test compounds. The workbook does not itself document the split-generation random seed or code environment.", "",
            "## Audit outputs", "",
            "- `endpoint_author_split_summary.csv`: endpoint coverage and author split sizes, without numerical labels.",
            "- `author_split_overlap_audit.csv`: exact parent/scaffold overlap counts for each author split.",
            "- `author_split_membership_hashed.csv`: reproducible membership using hashed parent/scaffold identifiers only.",
            "- `cmp106_relationship_audit.csv`: relationship of the 106-compound comparison set to each parameter dataset.",
            "- `source_reuse_registry.json`: local license/code/source status.", "",
            "## Interpretation boundary", "",
            "Any later author-like baseline must be labelled author-like unless external code, exact preprocessing and split-generation details are recovered. It must be reported separately from this project’s predeclared scaffold/source-cluster protocols and must never use consumed project test results for selection.", "",
        ]
        (out / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
        finish_stage(out, "jia2025_n0_reproducibility_audit", inputs=inputs, audit_scope="read_only_no_training_no_metrics",
                     provisional_reproduction_grade=grade, paper_year=2025, article_license="CC BY-NC-ND 4.0", partial=False)
    print(f"Jia 2025 N0 reproducibility audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
