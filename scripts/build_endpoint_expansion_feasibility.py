#!/usr/bin/env python3
"""Freeze the active F/logP/pKa endpoint-expansion feasibility contract.

This stage is an inventory and protocol decision only.  It does not merge raw
datasets, train models, or read protected endpoint values.  AUC and Cmax are
explicitly recorded as deferred scope rather than silently becoming molecular
structure-only labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, sha256, stage_output, startup_self_check


LOGP_TASK = "PUBLIC___LogPow_Public.csv"
LOGD_TASK = "PUBLIC___LogD74_Public.csv"
HUMAN_F_TASK = "F__human__absolute_oral"
ANIMAL_F_TASKS = {
    "F__dog__absolute_oral",
    "F__monkey__absolute_oral",
    "F__mouse__absolute_oral",
    "F__rat__absolute_oral",
}


def one_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame.loc[frame[column].eq(value)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {column}={value!r} row, found {len(rows)}")
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oneadmet-registry",
        type=Path,
        default=ROOT / "data/external/oneadmet_auxiliary_v2/task_registry.csv",
    )
    parser.add_argument(
        "--pk-registry",
        type=Path,
        default=ROOT / "data/public_development/multitask_pk_24h_v6/task_registry.csv",
    )
    parser.add_argument(
        "--feature-registry",
        type=Path,
        default=ROOT / "data/public_development/stl_benchmark_protocol_v1/feature_registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/analysis/endpoint_expansion_feasibility_v1",
    )
    args = parser.parse_args()
    startup_self_check(
        [args.oneadmet_registry, args.pk_registry, args.feature_registry], output=args.output
    )

    auxiliary = pd.read_csv(args.oneadmet_registry)
    pk = pd.read_csv(args.pk_registry)
    features = json.loads(args.feature_registry.read_text(encoding="utf-8"))
    logp = one_row(auxiliary, "task_name", LOGP_TASK)
    logd = one_row(auxiliary, "task_name", LOGD_TASK)
    human_f = one_row(pk, "task_id", HUMAN_F_TASK)
    animal_f = pk.loc[pk.task_id.isin(ANIMAL_F_TASKS)].copy()
    if set(animal_f.task_id) != ANIMAL_F_TASKS:
        raise ValueError("The four prespecified animal-F transfer tasks are not all present")

    active = pd.DataFrame(
        [
            {
                "endpoint_id": HUMAN_F_TASK,
                "endpoint_family": "F",
                "role": "core_PK_exploratory",
                "definition": "absolute oral bioavailability; parent analyte; oral plus IV reference",
                "canonical_unit": "fraction",
                "local_records": int(human_f.development_records),
                "local_molecules": int(human_f.development_molecules),
                "local_sources": int(human_f.sources),
                "train_records": int(human_f.train_records),
                "validation_records": int(human_f.validation_records),
                "readiness": "representation_and_sensitivity_only",
                "selection_authorized": False,
                "next_action": "retain strict/public/cross-species isolation; expand human validation evidence",
            },
            {
                "endpoint_id": "F__nonhuman__absolute_oral_transfer",
                "endpoint_family": "F",
                "role": "cross_species_auxiliary",
                "definition": "species-conditioned absolute oral bioavailability; never pooled as human truth",
                "canonical_unit": "fraction",
                "local_records": int(animal_f.development_records.sum()),
                "local_molecules": int(animal_f.development_molecules.sum()),
                "local_sources": int(animal_f.sources.sum()),
                "train_records": int(animal_f.train_records.sum()),
                "validation_records": int(animal_f.validation_records.sum()),
                "readiness": "available_for_transfer_ablation",
                "selection_authorized": False,
                "next_action": "fit animal heads and transfer only through protected OOF representation",
            },
            {
                "endpoint_id": "LogP__experimental__octanol_water",
                "endpoint_family": "logP",
                "role": "physchem_auxiliary",
                "definition": "experimental/as-published neutral-species octanol-water partition coefficient",
                "canonical_unit": "log10_partition_coefficient",
                "local_records": int(logp.total_measurements),
                "local_molecules": pd.NA,
                "local_sources": pd.NA,
                "train_records": int(logp.train_measurements),
                "validation_records": int(logp.test_measurements),
                "readiness": "protocol_and_provenance_audit_required",
                "selection_authorized": False,
                "next_action": "build an isolated auxiliary cohort; preserve source split; audit duplicates and leakage",
            },
            {
                "endpoint_id": "LogD__experimental__pH7_4",
                "endpoint_family": "logD",
                "role": "physchem_auxiliary_separate_from_logP",
                "definition": "experimental/as-published distribution coefficient at pH 7.4",
                "canonical_unit": "log10_distribution_coefficient",
                "local_records": int(logd.total_measurements),
                "local_molecules": pd.NA,
                "local_sources": pd.NA,
                "train_records": int(logd.train_measurements),
                "validation_records": int(logd.test_measurements),
                "readiness": "protocol_and_provenance_audit_required",
                "selection_authorized": False,
                "next_action": "keep separate from logP; require pH=7.4 semantics and source-split isolation",
            },
            {
                "endpoint_id": "pKa__acidic__experimental",
                "endpoint_family": "pKa",
                "role": "physchem_auxiliary_candidate",
                "definition": "experimental acidic pKa with macro/micro and site annotation",
                "canonical_unit": "pKa",
                "local_records": 0,
                "local_molecules": 0,
                "local_sources": 0,
                "train_records": 0,
                "validation_records": 0,
                "readiness": "data_acquisition_required",
                "selection_authorized": False,
                "next_action": "register a licensed/public source before constructing a task",
            },
            {
                "endpoint_id": "pKa__basic__experimental",
                "endpoint_family": "pKa",
                "role": "physchem_auxiliary_candidate",
                "definition": "experimental basic pKa with macro/micro and site annotation",
                "canonical_unit": "pKa",
                "local_records": 0,
                "local_molecules": 0,
                "local_sources": 0,
                "train_records": 0,
                "validation_records": 0,
                "readiness": "data_acquisition_required",
                "selection_authorized": False,
                "next_action": "register a licensed/public source before constructing a task",
            },
        ]
    )

    pka_schema = pd.DataFrame(
        [
            ("structure_id", "yes", "parent structure key"),
            ("value", "yes", "numeric experimental pKa"),
            ("acid_base_class", "yes", "acidic or basic; never pool without a task mask"),
            ("macro_micro_definition", "yes", "macroscopic or microscopic definition"),
            ("ionization_site", "conditional", "required for microscopic/site-specific labels"),
            ("rank_within_molecule", "yes", "supports multiple pKa values per molecule"),
            ("solvent", "yes", "aqueous or explicitly recorded alternative"),
            ("temperature_c", "conditional", "retain missingness; do not silently assume"),
            ("ionic_strength", "conditional", "retain missingness; do not silently assume"),
            ("measurement_method", "yes", "experimental method; predicted values are separate evidence"),
            ("source_id", "yes", "DOI/database/version/provenance"),
            ("reliability_tier", "yes", "controls weighting and sensitivity, not a model feature"),
        ],
        columns=["field", "required", "scientific_role"],
    )

    descriptor_names = features["rdkit2d"]["descriptor_names"]
    direct_logp = [
        name
        for name in descriptor_names
        if name == "MolLogP" or name.startswith("BCUT2D_LOGP") or name.startswith("SlogP_VSA")
    ]
    if "MolLogP" not in direct_logp or not direct_logp:
        raise ValueError("Expected computed logP-derived RDKit descriptors were not found")
    leakage = pd.DataFrame(
        [
            {
                "target_family": target,
                "feature_view": "rdkit2d_or_combined",
                "excluded_feature": name,
                "rule": "exclude_from_target_head_and_report_reduced-descriptor_ablation",
                "reason": "computed logP-derived descriptor would inflate experimental partition/distribution prediction",
            }
            for target in ("logP", "logD")
            for name in direct_logp
        ]
    )

    deferred = pd.DataFrame(
        [
            {
                "endpoint": "AUC",
                "decision": "deferred_out_of_current_scope",
                "reason": "requires dose, route, formulation, regimen, analyte and AUC time-window conditioning",
                "future_model_class": "conditional_exposure_model",
            },
            {
                "endpoint": "Cmax",
                "decision": "deferred_out_of_current_scope",
                "reason": "requires dose, route, formulation, regimen, sampling and population conditioning",
                "future_model_class": "conditional_exposure_model",
            },
        ]
    )

    sequence = pd.DataFrame(
        [
            (1, "F", "continue current strict/public/cross-species work; do not wait for new endpoints"),
            (2, "logP_logD", "build isolated auxiliary cohorts and leakage-safe STL baselines"),
            (3, "pKa", "acquire and audit a public/licensed experimental dataset, then build acid/basic heads"),
            (4, "multimodal_ablation", "compare structure-only, plus physchem auxiliary representation, plus species context"),
        ],
        columns=["priority", "workstream", "gate"],
    )

    summary = {
        "active_endpoint_families": ["F", "logP", "logD", "pKa"],
        "deferred_endpoint_families": ["AUC", "Cmax"],
        "oneadmet_logp_total": int(logp.total_measurements),
        "oneadmet_logd74_total": int(logd.total_measurements),
        "strict_human_f_records": int(human_f.development_records),
        "strict_human_f_validation_records": int(human_f.validation_records),
        "animal_f_records": int(animal_f.development_records.sum()),
        "local_pka_records": 0,
        "training_or_labels_modified": False,
        "validation_or_test_labels_read": False,
        "decision": "expand_auxiliary_physchem_contract_without_expanding_current_confirmatory_scope",
    }

    with stage_output(args.output) as out:
        active.to_csv(out / "active_endpoint_registry.csv", index=False)
        pka_schema.to_csv(out / "pka_data_contract.csv", index=False)
        leakage.to_csv(out / "physchem_feature_leakage_exclusions.csv", index=False)
        deferred.to_csv(out / "deferred_endpoint_registry.csv", index=False)
        sequence.to_csv(out / "development_sequence.csv", index=False)
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / "README.md").write_text(
            "# F/logP/pKa endpoint-expansion feasibility v1\n\n"
            "This is a data-contract and feasibility stage only; no model was trained and no protected labels were read. "
            "F remains an existing core but exploratory human endpoint. Experimental logP and logD 7.4 are separate "
            "auxiliary heads. pKa is split into acidic and basic experimental tasks and remains gated on auditable data. "
            "AUC and Cmax are deferred from the current scope. Computed RDKit logP-derived descriptors are explicitly "
            "excluded from experimental logP/logD target heads to prevent circular feature leakage.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "endpoint_expansion_feasibility",
            inputs={
                "oneadmet_registry_sha256": sha256(args.oneadmet_registry),
                "pk_registry_sha256": sha256(args.pk_registry),
                "feature_registry_sha256": sha256(args.feature_registry),
            },
            **summary,
            partial=False,
        )
    print(f"Endpoint-expansion feasibility: {args.output}")


if __name__ == "__main__":
    main()
