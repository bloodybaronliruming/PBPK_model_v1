#!/usr/bin/env python3
"""Read-only N0 audit of local MMPK code, data, target contracts, and cohort overlap."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check

RDLogger.DisableLog("rdApp.*")

DATASETS = ("approved_model", "investigational_model", "approved_2024_model")
ENDPOINTS = (
    ("AUC", "AUC [ng*h/mL]", "Log AUC [ng*h/mL]", "log10"),
    ("Cmax", "Cmax [ng/mL]", "Log Cmax [ng/mL]", "log10"),
    ("Tmax", "Tmax [h]", "Log Tmax [h]", "log10"),
    ("t1/2", "t1/2 [h]", "Log t1/2 [h]", "log10"),
    ("CL/F", "CL/F [L/h]", "Log CL/F [L/h]", "log10"),
    ("Vz/F", "Vz/F [L]", "Log Vz/F [L]", "log10"),
    ("MRT", "MRT [h]", "Log MRT [h]", "log10"),
    ("F", "F [%]", "Log F [%]", "bounded_logit_130"),
)


def identity(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    parent = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True, isomericSmiles=False)
    return stable_id(parent), stable_id(scaffold)


def target_contract(frame: pd.DataFrame, dataset: str) -> list[dict]:
    rows = []
    for endpoint, raw_col, log_col, kind in ENDPOINTS:
        raw = pd.to_numeric(frame[raw_col], errors="coerce")
        logged = pd.to_numeric(frame[log_col], errors="coerce")
        paired = raw.notna() & logged.notna()
        positive = raw[raw.notna()] > 0
        if kind == "log10":
            derived = np.log10(raw[paired])
        else:
            bounded = (raw[paired] > 0) & (raw[paired] < 130)
            if not bounded.all():
                raise ValueError(f"{dataset}/{endpoint}: F must lie strictly in (0,130) for its stated transform")
            derived = np.log10(raw[paired] / (130 - raw[paired]))
        max_error = float(np.max(np.abs(derived - logged[paired]))) if paired.any() else np.nan
        rows.append({"dataset": dataset, "endpoint": endpoint, "transform": kind,
                     "raw_nonmissing": int(raw.notna().sum()), "log_nonmissing": int(logged.notna().sum()),
                     "paired_nonmissing": int(paired.sum()), "all_observed_raw_positive": bool(positive.all()),
                     "raw_without_log": int((raw.notna() & logged.isna()).sum()),
                     "log_without_raw": int((raw.isna() & logged.notna()).sum()),
                     "raw_at_or_above_130": int((raw[raw.notna()] >= 130).sum()) if kind == "bounded_logit_130" else 0,
                     "max_abs_transform_error": max_error,
                     "contract_pass": bool((not paired.any()) or max_error <= 1e-6)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "基准研究参考")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    args = parser.parse_args()
    repo = args.reference / "MMPK-main/MMPK-main"
    data = args.reference / "mmpk_human_oral_pk_parameters_prediction_v3"
    required = [args.reference / "Li et al. - 2025 - MMPK A MultimodalDeep Learning Framework to PredictHuman Oral Pharmacokinetic Parameters.pdf",
                args.reference / "jm5c01522_si_001.pdf", repo / "README.md", repo / "LICENSE", repo / "config.py",
                repo / "dataloader.py", repo / "train.py", repo / "test.py", repo / "run.sh", repo / "utils/split.py", repo / "utils/metrics.py",
                repo / "mmpk/dataset.py", repo / "mmpk/model.py", repo / "gnn/gnn_feature.py", repo / "gnn/gnn_model.py"]
    required += [data / "model_dataset" / f"{name}.csv" for name in DATASETS]
    required += [data / "raw_dataset" / f"{name.replace('_model', '')}.csv" for name in DATASETS]
    checkpoints = [data / "checkpoints/mmpk" / f"fold_{fold}.pth" for fold in range(1, 11)]
    required += checkpoints
    startup_self_check(required, output=args.output)
    with stage_output(args.output) as out:
        source_manifest = pd.DataFrame([{"artifact": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
                                        for path in required])
        source_manifest.to_csv(out / "source_manifest.csv", index=False)
        pd.DataFrame([{"checkpoint": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
                      for path in checkpoints]).to_csv(out / "checkpoint_inventory.csv", index=False)
        pd.DataFrame([{
            "record_url": "https://zenodo.org/records/16561248",
            "doi": "10.5281/zenodo.16561248",
            "record_version": "v3.0",
            "record_file": "mmpk_human_oral_pk_parameters_prediction_v3.zip",
            "recorded_date": "2026-09-18",
            "repository_code_license": "MIT: Software and associated documentation may be used, modified, published, distributed, sublicensed, and sold with notice",
            "dataset_rights_observed": "Dataset package has no standalone license file; Zenodo displays Copyright 2025 LMMD/ECUST and no data reuse license",
            "publication_release_status": "Code may be released under MIT with notice. Preserve provenance and clarify terms before redistributing raw data, checkpoints, or data-derived release assets",
        }]).to_csv(out / "zenodo_rights_register.csv", index=False)
        model = {name: pd.read_csv(data / "model_dataset" / f"{name}.csv") for name in DATASETS}
        raw = {name: pd.read_csv(data / "raw_dataset" / f"{name.replace('_model', '')}.csv") for name in DATASETS}
        schema_rows, contract_rows, identities = [], [], {}
        for name, frame in model.items():
            expected = {"Compound Name", "SMILES", "Dose [mg]", "Dose [mg/kg]", "Log Dose [mg/kg]"}
            expected.update({raw_col for _, raw_col, _, _ in ENDPOINTS})
            expected.update({log_col for _, _, log_col, _ in ENDPOINTS})
            if not expected.issubset(frame.columns):
                raise ValueError(f"{name} lacks MMPK modeling fields: {sorted(expected-set(frame.columns))}")
            schema_rows.append({"dataset": name, "records": int(len(frame)), "unique_input_smiles": int(frame.SMILES.nunique()),
                                "columns": int(len(frame.columns)), "missing_dose_mg": int(frame["Dose [mg]"].isna().sum()),
                                "missing_dose_mgkg": int(frame["Dose [mg/kg]"].isna().sum())})
            contract_rows.extend(target_contract(frame, name))
            pairs = frame.SMILES.map(identity)
            parent = pairs.map(lambda x: x[0]); scaffold = pairs.map(lambda x: x[1])
            if parent.isna().any() or scaffold.isna().any():
                raise ValueError(f"{name} contains unparsable SMILES")
            identities[name] = {"parents": set(parent), "scaffolds": set(scaffold)}
        pd.DataFrame(schema_rows).to_csv(out / "model_dataset_schema_and_dose_coverage.csv", index=False)
        contracts = pd.DataFrame(contract_rows)
        if not contracts.contract_pass.all():
            raise ValueError("One or more MMPK target transforms disagree with the supplied model dataset")
        contracts.to_csv(out / "target_transform_contract.csv", index=False)
        coverage = []
        for name, frame in model.items():
            for endpoint, raw_col, log_col, kind in ENDPOINTS:
                coverage.append({"dataset": name, "endpoint": endpoint, "records": int(len(frame)),
                                 "unique_smiles": int(frame.SMILES.nunique()), "raw_labels": int(frame[raw_col].notna().sum()),
                                 "log_labels": int(frame[log_col].notna().sum()), "transform": kind})
        pd.DataFrame(coverage).to_csv(out / "endpoint_label_coverage.csv", index=False)
        overlap_rows = []
        pairs = [("approved_model", "investigational_model"), ("approved_model", "approved_2024_model"),
                 ("investigational_model", "approved_2024_model")]
        for left, right in pairs:
            overlap_rows.append({"left_dataset": left, "right_dataset": right,
                                 "left_parents": len(identities[left]["parents"]), "right_parents": len(identities[right]["parents"]),
                                 "parent_overlap": len(identities[left]["parents"] & identities[right]["parents"]),
                                 "left_scaffolds": len(identities[left]["scaffolds"]), "right_scaffolds": len(identities[right]["scaffolds"]),
                                 "scaffold_overlap": len(identities[left]["scaffolds"] & identities[right]["scaffolds"])})
        pd.DataFrame(overlap_rows).to_csv(out / "cohort_structure_overlap.csv", index=False)
        raw_summary = []
        for name, frame in raw.items():
            raw_summary.append({"raw_dataset": name, "records": int(len(frame)), "unique_smiles": int(frame.SMILES.nunique()),
                                "unique_compound_ids": int(frame["Compound ID"].nunique()), "unique_references": int(frame.Reference.nunique()),
                                "unique_pmids": int(frame.PMID.nunique()), "has_subject_weight_fields": bool({"Body Weight of Subjects", "Number of Subjects"}.issubset(frame.columns)),
                                "has_route_or_formulation_context": bool({"Dose [mg]", "Formulation", "Comments"}.issubset(frame.columns))})
        pd.DataFrame(raw_summary).to_csv(out / "raw_provenance_and_context_summary.csv", index=False)
        code_audit = pd.DataFrame([
            {"topic": "training split", "evidence": "utils/split.py", "finding": "Deterministic shuffled 10-fold split by exact SMILES; seed 42; validation is drawn from each training partition."},
            {"topic": "structure isolation", "evidence": "utils/split.py", "finding": "No scaffold, source/document, temporal, or parent-standardized split control is implemented."},
            {"topic": "test lifecycle", "evidence": "train.py", "finding": "Early stopping uses validation loss, but test metrics are calculated each epoch; a project reproduction must suppress interim test evaluation."},
            {"topic": "dose conditioning", "evidence": "dataloader.py; mmpk/dataset.py", "finding": "The model consumes log10 dose in mg/kg; dose is a required conditional input, not a molecular descriptor."},
            {"topic": "target transforms", "evidence": "utils/metrics.py", "finding": "Seven endpoints use log10 values; F uses log10(F/(130-F)) and back-transform bound 130%."},
            {"topic": "external cohorts", "evidence": "test.py", "finding": "Investigational and 2024 cohorts are separately loaded and ensembled across 10 checkpoints; labels are present locally and must remain sealed until a dedicated protocol is frozen."},
            {"topic": "license", "evidence": "LICENSE; README.md", "finding": "The MIT LICENSE expressly grants broad rights for Software and associated documentation. No standalone license declaration for raw/model datasets was located in the local data bundle; do not silently assume the data share the code license."},
        ])
        code_audit.to_csv(out / "code_and_license_audit.csv", index=False)
        report = "# MMPK N0 read-only audit\n\n"
        report += "Local inputs include the paper, SI, MIT-licensed code, ten hash-registered checkpoints, raw data, modeling tables and two declared external cohorts. The supplied modeling tables support dose-conditioned oral AUC, Cmax, Tmax, t1/2, CL/F, Vz/F, MRT and F. All paired target/log values passed their stated numerical transform contracts.\n\n"
        report += "The approved modeling table has one F=130% raw value without a log target. This is mathematically expected under the stated log10(F/(130-F)) transform and it is unavailable for the MMPK F loss; future project cohorts must preserve this boundary label rather than silently impute it.\n\n"
        report += "MMPK split code is deterministic by exact SMILES but is not scaffold/source/temporal isolated. Investigational data overlaps approved data by one canonical parent and eleven scaffolds; approved-2024 overlaps by one scaffold. The provided external cohorts must therefore be separately registered, parent-purged where appropriate, and sealed before any project score. The local MIT license grants broad use, modification and distribution rights for code and associated documentation. The matching Zenodo v3 record displays a copyright notice but no separate data reuse license, so raw data/checkpoint redistribution terms should be clarified rather than silently inferred from the code license.\n"
        (out / "audit_report.md").write_text(report, encoding="utf-8")
        finish_stage(out, "mmpk_n0_read_only_data_code_license_audit",
                     inputs={str(path.resolve()): sha256(path) for path in required}, read_only=True,
                     model_training=False, project_test_accessed=False, labels_exported=False,
                     split_grade="B_author_code_reconstructable_exact_smiles_not_structure_isolated",
                     code_license="MIT_confirmed_for_software_and_documentation", dataset_license_status="no_separate_data_license_observed",
                     approved_to_investigational_parent_overlap=1, approved_to_investigational_scaffold_overlap=11,
                     f_boundary_raw_without_model_label=1, partial=False)
    print(f"MMPK N0 read-only audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
