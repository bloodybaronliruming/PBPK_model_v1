#!/usr/bin/env python3
"""N1d: audit approved MMPK context and derivation availability without model fitting."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


ENDPOINTS = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]", "CL/F [L/h]", "Vz/F [L]", "MRT [h]", "F [%]")
FIRST_BATCH = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")
RDLogger.DisableLog("rdApp.*")


def parent_id(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Unparsable SMILES in approved MMPK data")
    return stable_id(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))


def context_state(values: pd.Series) -> tuple[str, str]:
    """Return a privacy-preserving completeness/consistency state and hash."""
    cleaned = sorted({str(x).strip() for x in values.dropna() if str(x).strip()})
    if not cleaned:
        return "missing", ""
    if len(cleaned) == 1:
        return "single_value", stable_id(cleaned[0])
    return "multiple_values", stable_id("\n".join(cleaned))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "基准研究参考")
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--n1a", type=Path, default=ROOT / "results/analysis/mmpk_n1a_lineage_feasibility_v2")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/mmpk_n1d_context_derivation_audit_v1")
    args = parser.parse_args()
    data = args.reference / "mmpk_human_oral_pk_parameters_prediction_v3"
    raw_path = data / "raw_dataset/approved.csv"
    model_path = data / "model_dataset/approved_model.csv"
    startup_self_check([raw_path, model_path, args.n0 / "complete.json", args.n1a / "complete.json", args.n1b / "complete.json"], output=args.output)
    n0 = verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    n1a = verify_stage(args.n1a, "mmpk_n1a_approved_raw_model_lineage_feasibility")
    n1b = verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    for source in (raw_path, model_path):
        if n0["inputs"].get(str(source.resolve())) != sha256(source):
            raise ValueError(f"Approved source differs from N0-hashed input: {source}")
    if n1a["inputs"].get(str(model_path.resolve())) != sha256(model_path):
        raise ValueError("N1a is not bound to the current approved model table")
    if n1b["inputs"].get(str(model_path.resolve())) != sha256(model_path):
        raise ValueError("N1b is not bound to the current approved model table")
    raw_columns = ["SMILES", "Dose [mg]", "Body Weight of Subjects", "Number of Subjects", "Formulation", "Salt", "Prodrug", "Comments", "AUC Type", "Reference", "PMID"]
    model_columns = ["SMILES", "Dose [mg]", "Dose [mg/kg]", "Log Dose [mg/kg]"]
    raw = pd.read_csv(raw_path, usecols=raw_columns)
    model = pd.read_csv(model_path, usecols=model_columns)
    raw["parent_id"] = raw.SMILES.map(parent_id)
    raw["raw_mgkg"] = pd.to_numeric(raw["Dose [mg]"], errors="coerce") / pd.to_numeric(raw["Body Weight of Subjects"], errors="coerce")
    model["parent_id"] = model.SMILES.map(parent_id)
    registry = pd.read_csv(args.n1a / "approved_model_lineage_registry_hashed.csv")
    records = pd.read_csv(args.n1b / "approved_record_registry_hashed.csv")
    labels = pd.read_csv(args.n1b / "approved_label_tier_registry_hashed.csv")
    expected_ids = [stable_id(f"mmpk_approved_model_row:{index}") for index in model.index]
    if len(model) != len(registry) or set(expected_ids) != set(registry.model_record_id) or set(expected_ids) != set(records.model_record_id):
        raise ValueError("Approved model identity disagrees with frozen N1a/N1b registries")
    model.insert(0, "model_record_id", expected_ids)
    raw_by_parent = {key: value for key, value in raw.groupby("parent_id", sort=False)}
    context_rows = []
    for _, model_row in model.iterrows():
        candidates = raw_by_parent.get(model_row.parent_id)
        if candidates is None:
            raise ValueError("A model row lacks a raw parent candidate")
        if pd.notna(model_row["Dose [mg]"]):
            selected = candidates[np.isclose(candidates["Dose [mg]"].to_numpy(float), float(model_row["Dose [mg]"]), rtol=0, atol=1e-9)]
            map_rule = "parent_plus_dose_mg"
        else:
            selected = candidates[np.isclose(candidates.raw_mgkg.to_numpy(float), float(model_row["Dose [mg/kg]"]), rtol=0, atol=1e-6)]
            map_rule = "parent_plus_dose_mgkg_rounded_1e6"
        if selected.empty:
            raise ValueError("N1d cannot reproduce the N1a raw mapping")
        formulation_state, formulation_hash = context_state(selected["Formulation"])
        salt_state, salt_hash = context_state(selected["Salt"])
        prodrug_state, prodrug_hash = context_state(selected["Prodrug"])
        auc_type_state, auc_type_hash = context_state(selected["AUC Type"])
        comments_nonempty = int(selected.Comments.notna().sum())
        body_weight = pd.to_numeric(selected["Body Weight of Subjects"], errors="coerce")
        subjects = pd.to_numeric(selected["Number of Subjects"], errors="coerce")
        context_rows.append({
            "model_record_id": model_row.model_record_id, "mapping_rule_reproduced": map_rule,
            "raw_candidate_records": int(len(selected)),
            "log_dose_available": bool(pd.notna(model_row["Log Dose [mg/kg]"])),
            "formulation_state": formulation_state, "formulation_value_hash": formulation_hash,
            "salt_state": salt_state, "salt_value_hash": salt_hash,
            "prodrug_state": prodrug_state, "prodrug_value_hash": prodrug_hash,
            "auc_type_state": auc_type_state, "auc_type_value_hash": auc_type_hash,
            "comment_present_any": bool(comments_nonempty > 0),
            "body_weight_complete_all_candidates": bool(body_weight.notna().all()),
            "subject_count_complete_all_candidates": bool(subjects.notna().all()),
            # These are deliberately scope limitations, not inferred values.
            "route_context": "cohort_level_human_oral_not_row_explicit",
            "population_context": "aggregate_body_weight_and_subject_count_only",
            "time_window_context": "not_row_explicit",
            "analyte_matrix_context": "not_row_explicit",
        })
    contexts = pd.DataFrame(context_rows).merge(records[["model_record_id", "strict_outer_fold", "source_family_id"]], on="model_record_id", validate="one_to_one")
    if contexts.model_record_id.duplicated().any() or len(contexts) != len(model):
        raise ValueError("Incomplete model context registry")
    field_rows = []
    for field in ["Formulation", "Salt", "Prodrug", "Comments", "AUC Type", "Body Weight of Subjects", "Number of Subjects"]:
        values = raw[field]
        field_rows.append({"field": field, "raw_records": len(raw), "nonmissing_records": int(values.notna().sum()),
                           "missing_records": int(values.isna().sum()), "unique_nonmissing_values": int(values.dropna().nunique()),
                           "role": "conditional_candidate" if field in {"Formulation", "AUC Type"} else "provenance_or_context_only"})
    readiness = []
    for endpoint in ENDPOINTS:
        subset = labels[labels.endpoint.eq(endpoint)]
        for tier, column in [("R1_direct", "direct_observation_eligible"), ("R1_plus_R2_augmented", "author_augmented_eligible")]:
            eligible_ids = set(subset.loc[subset[column].astype(bool), "model_record_id"])
            eligible_contexts = contexts[contexts.model_record_id.isin(eligible_ids)]
            readiness.append({"endpoint": endpoint, "tier_view": tier, "supervised_records": int(len(eligible_ids)),
                              "log_dose_available": int(eligible_contexts.log_dose_available.sum()),
                              "single_formulation_context": int(eligible_contexts.formulation_state.eq("single_value").sum()),
                              "complete_subject_metadata": int((eligible_contexts.body_weight_complete_all_candidates & eligible_contexts.subject_count_complete_all_candidates).sum()),
                              "route_statement": "cohort_level_oral_only", "time_window_available": 0,
                              "analyte_matrix_available": 0,
                              "first_batch_role": "primary" if endpoint in FIRST_BATCH else "secondary_or_exploratory"})
    field_summary = pd.DataFrame(field_rows)
    readiness = pd.DataFrame(readiness)
    if not contexts.log_dose_available.all() or not contexts.route_context.eq("cohort_level_human_oral_not_row_explicit").all():
        raise ValueError("N1d input scope contract unexpectedly failed")
    with stage_output(args.output) as out:
        field_summary.to_csv(out / "raw_context_field_coverage.csv", index=False)
        contexts.to_csv(out / "approved_model_context_registry_hashed.csv", index=False)
        readiness.to_csv(out / "endpoint_context_readiness.csv", index=False)
        (out / "README.md").write_text(
            "# MMPK N1d context and derivation audit\n\n"
            "This read-only audit maps approved model records back to their approved raw-context candidates and publishes only hashed metadata states, coverage counts, source-family IDs and frozen split IDs. It contains no target values, predictions or performance metrics. Route is a cohort-level human-oral scope assertion rather than a row-level variable; time-window and analyte/matrix fields are absent, so no model may claim conditioning on them.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n1d_approved_context_and_derivation_audit", inputs={
            str(raw_path.resolve()): sha256(raw_path), str(model_path.resolve()): sha256(model_path),
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
            str((args.n1a / "complete.json").resolve()): sha256(args.n1a / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json")},
            no_training=True, no_predictions=True, no_performance_metrics=True, external_labels_accessed=False,
            raw_labels_exported=False, context_fields_hashed=True, route_scope="cohort_level_human_oral_only",
            time_window_row_context_available=False, analyte_matrix_row_context_available=False, partial=False)
    print(f"MMPK N1d context/derivation audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
