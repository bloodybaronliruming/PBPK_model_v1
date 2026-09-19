#!/usr/bin/env python3
"""Read-only lineage feasibility audit for MMPK approved compound-dose modeling rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage

RDLogger.DisableLog("rdApp.*")
ENDPOINTS = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]", "CL/F [L/h]", "Vz/F [L]", "MRT [h]", "F [%]")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def canonical_ids(smiles: str) -> tuple[str, str]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Unparsable SMILES in approved data")
    parent = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(molecule), canonical=True, isomericSmiles=False)
    return stable_id(parent), stable_id(scaffold)


def component_ids(n: int, parent_ids: list[str], scaffold_ids: list[str], document_sets: list[set[str]], include_scaffold: bool) -> list[str]:
    uf = UnionFind(n)
    seen: dict[str, int] = {}
    for idx, parent in enumerate(parent_ids):
        key = f"parent:{parent}"
        if key in seen: uf.union(idx, seen[key])
        else: seen[key] = idx
    for idx, documents in enumerate(document_sets):
        for document in documents:
            key = f"document:{document}"
            if key in seen: uf.union(idx, seen[key])
            else: seen[key] = idx
    if include_scaffold:
        for idx, scaffold in enumerate(scaffold_ids):
            key = f"scaffold:{scaffold}"
            if key in seen: uf.union(idx, seen[key])
            else: seen[key] = idx
    return [stable_id(f"component:{uf.find(idx)}") for idx in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "基准研究参考")
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/mmpk_n1a_lineage_feasibility_v2")
    args = parser.parse_args()
    data = args.reference / "mmpk_human_oral_pk_parameters_prediction_v3"
    raw_path = data / "raw_dataset/approved.csv"
    model_path = data / "model_dataset/approved_model.csv"
    startup_self_check([raw_path, model_path, args.n0 / "complete.json"], output=args.output)
    n0 = verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    for path in (raw_path, model_path):
        if n0["inputs"].get(str(path.resolve())) != sha256(path):
            raise ValueError(f"MMPK approved source differs from N0-hashed input: {path}")
    raw = pd.read_csv(raw_path)
    model = pd.read_csv(model_path)
    required_raw = {"SMILES", "Dose [mg]", "Body Weight of Subjects", "Reference", "PMID", *ENDPOINTS}
    required_model = {"SMILES", "Dose [mg]", "Dose [mg/kg]", *ENDPOINTS}
    if not required_raw.issubset(raw.columns) or not required_model.issubset(model.columns):
        raise ValueError("Approved raw/model fields do not satisfy the N1a lineage contract")
    raw["_ids"] = raw.SMILES.map(canonical_ids)
    raw["parent_id"] = raw._ids.map(lambda x: x[0])
    raw["raw_mgkg"] = raw["Dose [mg]"] / raw["Body Weight of Subjects"]
    raw["document_id"] = raw.apply(lambda x: stable_id(f"PMID:{x.PMID}" if pd.notna(x.PMID) else f"Reference:{x.Reference}"), axis=1)
    model["_ids"] = model.SMILES.map(canonical_ids)
    model["parent_id"] = model._ids.map(lambda x: x[0])
    model["scaffold_id"] = model._ids.map(lambda x: x[1])
    by_parent: dict[str, pd.DataFrame] = {parent: frame for parent, frame in raw.groupby("parent_id", sort=False)}
    registry_rows, target_rows, target_registry_rows, document_sets = [], [], [], []
    for index, row in model.iterrows():
        candidates = by_parent.get(row.parent_id)
        if candidates is None:
            raise ValueError("A modeling parent is absent from raw approved data")
        if pd.notna(row["Dose [mg]"]):
            mask = np.isclose(candidates["Dose [mg]"].to_numpy(float), float(row["Dose [mg]"]), atol=1e-9, rtol=0)
            mapping_rule = "canonical_parent_plus_dose_mg"
        else:
            # Model mg/kg values are stored to six decimal places; raw dose/body-weight ratios differ by <=4.74e-7.
            mask = np.isclose(candidates["raw_mgkg"].to_numpy(float), float(row["Dose [mg/kg]"]), atol=1e-6, rtol=0)
            mapping_rule = "canonical_parent_plus_dose_mgkg_rounded_1e6_when_model_mg_missing"
        matched = candidates.loc[mask].copy()
        if matched.empty:
            raise ValueError(f"No raw lineage candidate for approved model record {index}")
        documents = set(matched.document_id)
        document_sets.append(documents)
        dose_key = f"mg:{float(row['Dose [mg]']):.12g}" if pd.notna(row["Dose [mg]"]) else f"mgkg:{float(row['Dose [mg/kg]']):.12g}"
        record_id = stable_id(f"mmpk_approved_model_row:{index}")
        registry_rows.append({"model_record_id": record_id, "parent_id": row.parent_id, "scaffold_id": row.scaffold_id,
                              "dose_arm_id": stable_id(f"{row.parent_id}|{dose_key}"), "mapping_rule": mapping_rule,
                              "raw_candidate_records": int(len(matched)), "source_documents": int(len(documents)),
                              "source_family_id": stable_id("|".join(sorted(documents))),
                              "model_dose_mg_missing": bool(pd.isna(row["Dose [mg]"]))})
        for endpoint in ENDPOINTS:
            model_label = bool(pd.notna(row[endpoint]))
            raw_label = bool(matched[endpoint].notna().any())
            target_rows.append({"endpoint": endpoint, "model_label_present": model_label,
                                "raw_candidate_label_present": raw_label,
                                "mapping_rule": mapping_rule})
            target_registry_rows.append({"model_record_id": record_id, "endpoint": endpoint,
                                         "model_label_present": model_label, "raw_candidate_label_present": raw_label,
                                         "label_origin_proxy": "direct_raw_candidate_available" if model_label and raw_label else
                                                               "model_label_without_direct_raw_candidate" if model_label else
                                                               "raw_candidate_label_not_modeled" if raw_label else "label_absent",
                                         "mapping_rule": mapping_rule})
    registry = pd.DataFrame(registry_rows)
    registry["parent_source_component_id"] = component_ids(len(registry), registry.parent_id.tolist(), registry.scaffold_id.tolist(), document_sets, include_scaffold=False)
    registry["parent_source_scaffold_component_id"] = component_ids(len(registry), registry.parent_id.tolist(), registry.scaffold_id.tolist(), document_sets, include_scaffold=True)
    with stage_output(args.output) as out:
        registry.to_csv(out / "approved_model_lineage_registry_hashed.csv", index=False)
        target_frame = pd.DataFrame(target_rows)
        target_frame["model_label_with_raw_contributor"] = target_frame.model_label_present & target_frame.raw_candidate_label_present
        target_frame["model_label_without_raw_contributor"] = target_frame.model_label_present & ~target_frame.raw_candidate_label_present
        target_frame["raw_contributor_without_model_label"] = ~target_frame.model_label_present & target_frame.raw_candidate_label_present
        target = target_frame.groupby(["endpoint", "mapping_rule"], dropna=False).agg(
            records=("model_label_present", "size"), model_labels=("model_label_present", "sum"),
            model_labels_with_raw_contributor=("model_label_with_raw_contributor", "sum"),
            model_labels_without_raw_contributor=("model_label_without_raw_contributor", "sum"),
            raw_contributor_without_model_label=("raw_contributor_without_model_label", "sum")).reset_index()
        target.to_csv(out / "target_lineage_coverage.csv", index=False)
        pd.DataFrame(target_registry_rows).to_csv(out / "approved_model_target_lineage_hashed.csv", index=False)
        mapping = registry.groupby("mapping_rule").agg(records=("model_record_id", "size"),
            parents=("parent_id", "nunique"), dose_arms=("dose_arm_id", "nunique"),
            median_raw_candidates=("raw_candidate_records", "median"), max_raw_candidates=("raw_candidate_records", "max"),
            median_source_documents=("source_documents", "median"), max_source_documents=("source_documents", "max")).reset_index()
        mapping.to_csv(out / "mapping_coverage_summary.csv", index=False)
        components = []
        for column, label in (("parent_source_component_id", "parent_plus_source"),
                              ("parent_source_scaffold_component_id", "parent_plus_source_plus_scaffold")):
            sizes = registry.groupby(column).size()
            components.append({"constraint": label, "components": int(len(sizes)), "records": int(len(registry)),
                               "largest_component_records": int(sizes.max()), "median_component_records": float(sizes.median()),
                               "components_with_at_least_2_records": int((sizes >= 2).sum())})
        pd.DataFrame(components).to_csv(out / "constraint_component_feasibility.csv", index=False)
        report = "# MMPK N1a approved raw-model lineage feasibility\n\n"
        report += "Every approved modeling record has an exact canonical-parent-plus-dose raw provenance match: records with model mg use mg; records lacking model mg use raw mg/kg computed from dose/body weight and rounded in the model table. Source identity is hashed and no raw labels are exported.\n\n"
        report += "Target-level lineage is separately registered. A model label without a raw candidate label is marked as a model-only/derived-or-imputed-or-aggregate proxy rather than silently treated as a direct measurement. A future strict split must retain all dose arms for a parent together and must treat each parent/source (and, if selected, parent/source/scaffold) connected component as indivisible. Whether the resulting component size distribution supports five outer folds is reported in the machine-readable feasibility table.\n"
        (out / "audit_report.md").write_text(report, encoding="utf-8")
        finish_stage(out, "mmpk_n1a_approved_raw_model_lineage_feasibility", inputs={
            str(raw_path.resolve()): sha256(raw_path), str(model_path.resolve()): sha256(model_path),
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json")},
            read_only=True, model_training=False, external_labels_accessed=False, raw_labels_exported=False,
            mapping_rules=["canonical_parent_plus_dose_mg", "canonical_parent_plus_dose_mgkg_rounded_1e6_when_model_mg_missing"], partial=False)
    print(f"MMPK N1a lineage feasibility: {args.output}")


if __name__ == "__main__":
    run_cli(main)
