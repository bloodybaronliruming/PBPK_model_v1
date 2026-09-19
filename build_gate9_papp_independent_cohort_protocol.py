"""Register a metadata-only independent-cohort protocol for second-iteration Papp.

The protocol treats human Caco-2 A→B Papp as the first Gate 9 target because
the existing evidence identifies source/unit heterogeneity as its actionable
gap.  It reads only current membership, source-document and assay identifiers
needed to make an exclusion ledger.  It never reads Papp values, transformed
targets, prediction rows, models, or any historical test values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, stable_id, verify_stage


STAGE = "gate9_papp_independent_cohort_protocol"
OUTPUT = "data/public_development/gate9_papp_independent_cohort_protocol_v1"
TASK = "Papp__human__caco2_ab"
SOURCES = [
    ("Gate 9 readiness audit", "data/public_development/gate9_second_iteration_readiness_v2", "gate9_second_iteration_readiness"),
    ("processed v15 datasets", "data/processed_v15/datasets", "datasets"),
    ("processed v15 splits", "data/processed_v15/splits", "splits"),
]


def fingerprint_frame(values: pd.Series, kind: str) -> pd.DataFrame:
    unique = sorted({str(value) for value in values.dropna().tolist() if str(value)})
    return pd.DataFrame({"fingerprint_kind": kind, "fingerprint": [stable_id(value) for value in unique]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT

    inputs: dict[str, str] = {}
    for _, relative, expected_stage in SOURCES:
        folder = root / relative
        verify_stage(folder, expected_stage)
        complete = folder / "complete.json"
        inputs[str(complete.resolve())] = sha256(complete)

    task_path = root / f"data/processed_v15/datasets/tasks/{TASK}.csv"
    split_path = root / "data/processed_v15/splits/split_manifest.csv"
    registry_path = root / "data/processed_v15/datasets/task_registry.json"
    for path in (task_path, split_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[str(path.resolve())] = sha256(path)

    # Explicit metadata-only allow-list: no raw_value, target_value, mask,
    # SMILES, predictions, or model artifacts may enter this protocol.
    task = pd.read_csv(task_path, usecols=["row_id", "molecule_id", "split", "assay_id", "doc_id", "source_activity_ids", "source_n"])
    split = pd.read_csv(split_path, usecols=["molecule_id", "parent_id", "scaffold_group", "split", "eligible"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    spec = registry.get(TASK)
    if spec != {
        "endpoint": "Papp", "species": "human", "system": "caco2_ab", "unit": "1e-6_cm/s",
        "transform": "log10", "target_space": "transformed_unstandardized",
        "standardization": "fit_only_in_each_training_fold", "exact_only": True,
        "aggregation": "median_within_molecule_task_assay_document",
    }:
        raise ValueError("Papp task contract does not match the frozen Gate 9 protocol definition")
    if task.empty or task.molecule_id.isna().any() or task.doc_id.isna().any() or task.assay_id.isna().any():
        raise ValueError("Papp membership metadata is incomplete")
    if not task.split.isin({"train", "val", "test"}).all():
        raise ValueError("Papp task has an invalid historical split")
    split = split.loc[split.eligible].copy()
    if split.molecule_id.duplicated().any():
        raise ValueError("Eligible split registry must contain one row per molecule")
    joined = task.merge(split, on="molecule_id", how="left", suffixes=("_task", "_registry"), validate="many_to_one")
    mapped = joined.loc[joined.parent_id.notna()].copy()
    unmapped = joined.loc[joined.parent_id.isna()].copy()
    if not mapped.split_task.eq(mapped.split_registry).all():
        raise ValueError("Mapped Papp metadata must agree with the frozen eligible split registry")
    summary = (
        joined.assign(membership_status=joined.parent_id.notna().map({
            True: "mapped_to_current_eligible_registry",
            False: "unmapped_from_current_eligible_registry",
        })).groupby(["membership_status", "split_task"], sort=True)
        .agg(records=("row_id", "size"), molecules=("molecule_id", "nunique"), parents=("parent_id", "nunique"),
             scaffolds=("scaffold_group", "nunique"), source_documents=("doc_id", "nunique"), assays=("assay_id", "nunique"))
        .reset_index().rename(columns={"split_task": "historical_split"})
    )
    if set(summary.historical_split) != {"train", "val", "test"}:
        raise ValueError("All historical Papp partitions must be represented in the exclusion ledger")
    fingerprints = pd.concat([
        fingerprint_frame(joined.molecule_id, "molecule_id_all_historical_task_records"),
        fingerprint_frame(mapped.parent_id, "parent_id_current_eligible_registry"),
        fingerprint_frame(mapped.scaffold_group, "scaffold_group_current_eligible_registry"),
        fingerprint_frame(joined.doc_id, "source_document_id"),
        fingerprint_frame(joined.assay_id, "assay_id"),
    ], ignore_index=True)
    if fingerprints.duplicated().any():
        raise ValueError("Exclusion fingerprints must be unique within their type")

    candidates = pd.DataFrame([
        ("PAPP-R1-C01", "post-freeze public assay/document delta", "public ChEMBL or equivalent release after frozen snapshot", "metadata_only_pending_download", "Document/source identifier must be absent from the frozen ledger; no numerical label may be imported at registration."),
        ("PAPP-R1-C02", "primary Caco-2 A→B literature studies", "primary publication or author-released supplement", "metadata_only_pending_rights_and_screen", "Require retrievable primary source, explicit A→B direction, Caco-2 system, compatible unit/conversion evidence, and publication rights review."),
        ("PAPP-R1-C03", "public dataset with primary-source provenance", "curated public release", "metadata_only_pending_provenance_audit", "Accept only if each value links to a primary document/assay and does not silently mix Caco-2 directions or permeability systems."),
    ], columns=["candidate_id", "candidate_class", "expected_source", "registration_status", "entry_requirement"])

    protocol = """# Gate 9 R1: Papp independent-cohort eligibility protocol

## Scope

This protocol is metadata-only. It registers candidate sources for a *new* human Caco-2 A→B Papp cohort; it does not extract a numeric label, train a model, or read the historical test values. The frozen task definition is exact human Caco-2 A→B Papp in `1e-6 cm/s`; transformed target construction remains a future train-fold-only operation.

## Independence tiers

1. **Strict external-validation tier:** candidate records must have no overlap with the frozen Papp ledger in source document, molecule/parent, scaffold group, or derivation family. This tier is not yet eligible because 24 historical Papp molecule identities (27 records) are absent from the current eligible split registry and therefore lack current parent/scaffold fingerprints.
2. **Source-independent sensitivity tier:** source document, molecule/parent and derivation family must be disjoint. Scaffold overlap is allowed only as a separately reported sensitivity set and cannot be called strict external validation. No candidate may be admitted until the same 24-identity lineage gap is resolved or conservatively excluded by a separately frozen repair protocol.

## Admission requirements

- Primary-source provenance, license/reuse status, assay direction (A→B), Caco-2 system, unit and conversion evidence must be explicit.
- B→A, bidirectional aggregates, MDCK, PAMPA, qualitative/censored-only and undefined-system records are excluded from the numeric cohort.
- Candidate identifiers must be checked against every frozen train/validation/test partition through `papp_frozen_exclusion_fingerprints.csv`; historical test membership is an exclusion boundary, not an evaluation resource.
- Before any future model work, freeze the accepted cohort, reliability strata, source/parent/scaffold/derivation isolation policy, metrics, matched frozen-STL comparator and train-only smoke. No stopped architecture is reopened by this protocol.

## Current status

All three registered source classes are metadata-only candidates. No source has passed admission and no reconstruction is authorized.
"""

    if args.check_only:
        print(f"Gate 9 R1 Papp preflight: task_records={len(task)} molecules={task.molecule_id.nunique()} source_documents={task.doc_id.nunique()} candidate_classes={len(candidates)}; metadata-only, no labels/predictions/models/test values.")
        return
    with stage_output(output) as folder:
        summary.to_csv(folder / "papp_frozen_membership_summary.csv", index=False)
        fingerprints.to_csv(folder / "papp_frozen_exclusion_fingerprints.csv", index=False)
        unmapped.loc[:, ["row_id", "molecule_id", "split_task", "assay_id", "doc_id"]].assign(
            molecule_fingerprint=unmapped.molecule_id.map(stable_id),
            reason="absent_from_current_eligible_split_registry; no parent/scaffold fingerprint exported",
        ).drop(columns="molecule_id").to_csv(folder / "papp_unmapped_historical_membership_register.csv", index=False)
        candidates.to_csv(folder / "papp_independent_candidate_register.csv", index=False)
        (folder / "README.md").write_text(protocol, encoding="utf-8")
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, endpoint_task=TASK,
            no_labels_accessed=True, no_raw_or_transformed_target_accessed=True,
            no_prediction_rows_accessed=True, no_models_loaded=True, no_test_values_accessed=True,
            no_performance_metrics_computed=True, no_model_fitted=True, no_calibration=True,
            no_candidate_selection_changed=True, metadata_only=True, candidate_class_count=len(candidates),
            frozen_membership_record_count=len(task), frozen_source_document_count=task.doc_id.nunique(),
            strict_external_requires_source_parent_scaffold_derivation_disjoint=True,
            unmapped_historical_record_count=len(unmapped), unmapped_historical_molecule_count=unmapped.molecule_id.nunique(),
            strict_external_admission_authorized=False,
        )
    print(f"Gate 9 R1 Papp independent cohort protocol: {output}")


if __name__ == "__main__":
    run_cli(main)
