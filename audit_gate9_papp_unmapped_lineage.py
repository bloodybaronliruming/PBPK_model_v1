"""Resolve Gate 9 Papp exclusion-ledger lineage without reading Papp labels.

R1 found Papp task records absent from the *eligible* split registry.  This
audit checks those records against the full frozen split manifest, recomputes
their canonical parent/scaffold using the frozen split implementation, and
publishes only anonymous exclusion fingerprints.  The records remain excluded
from every historic split and cannot become training, validation, or test data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_split_manifest import structure_groups
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, verify_stage


STAGE = "gate9_papp_unmapped_lineage_audit"
OUTPUT = "data/public_development/gate9_papp_unmapped_lineage_audit_v1"
TASK = "Papp__human__caco2_ab"
SOURCES = [
    ("Gate 9 Papp cohort protocol", "data/public_development/gate9_papp_independent_cohort_protocol_v1", "gate9_papp_independent_cohort_protocol"),
    ("processed v15 datasets", "data/processed_v15/datasets", "datasets"),
    ("processed v15 splits", "data/processed_v15/splits", "splits"),
]


def fingerprint_frame(values: pd.Series, kind: str) -> pd.DataFrame:
    values = sorted({str(value) for value in values.dropna().tolist() if str(value)})
    return pd.DataFrame({"fingerprint_kind": kind, "fingerprint": [stable_id(value) for value in values]})


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
    register_path = root / "data/public_development/gate9_papp_independent_cohort_protocol_v1/papp_unmapped_historical_membership_register.csv"
    task_path = root / f"data/processed_v15/datasets/tasks/{TASK}.csv"
    split_path = root / "data/processed_v15/splits/split_manifest.csv"
    for path in (register_path, task_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[str(path.resolve())] = sha256(path)

    register = pd.read_csv(register_path, usecols=["row_id", "split_task"])
    task = pd.read_csv(task_path, usecols=["row_id", "molecule_id", "smiles", "split", "assay_id", "doc_id"])
    split = pd.read_csv(
        split_path,
        usecols=["molecule_id", "parent_id", "scaffold_group", "split", "standardization_status", "standardization_error", "eligible", "exclusion_reason"],
    ).rename(columns={"split": "registry_split"})
    records = register.merge(task, on="row_id", validate="one_to_one").merge(split, on="molecule_id", validate="many_to_one")
    if len(records) != 27 or records.molecule_id.nunique() != 24:
        raise ValueError("R2 must reconcile the exact R1 gap of 27 records / 24 molecules")
    if not records.split_task.eq(records.split).all() or not records.split_task.eq(records.registry_split).all():
        raise ValueError("R2 cannot repair a split-assignment disagreement")
    if not records.standardization_status.eq("ok").all() or records.standardization_error.fillna("").ne("").any():
        raise ValueError("R2 only supports successful frozen standardization")
    if records.eligible.any() or not records.exclusion_reason.eq("cross_split_parent_or_scaffold").all():
        raise ValueError("R2 must preserve the frozen cross-split exclusion reason")

    recomputed = [structure_groups(smiles) for smiles in records.smiles]
    records["recomputed_parent_id"] = [item[0] for item in recomputed]
    records["recomputed_scaffold_group"] = [item[1] for item in recomputed]
    if not records.parent_id.eq(records.recomputed_parent_id).all() or not records.scaffold_group.eq(records.recomputed_scaffold_group).all():
        raise ValueError("Recomputed canonical parent/scaffold disagrees with the frozen full registry")

    reconciliation = pd.DataFrame({
        "record_fingerprint": records.row_id.map(stable_id),
        "molecule_fingerprint": records.molecule_id.map(stable_id),
        "historical_split": records.split_task,
        "registry_eligible": records.eligible,
        "frozen_exclusion_reason": records.exclusion_reason,
        "parent_fingerprint": records.parent_id.map(stable_id),
        "scaffold_fingerprint": records.scaffold_group.map(stable_id),
        "recomputed_parent_matches_frozen_registry": True,
        "recomputed_scaffold_matches_frozen_registry": True,
        "repair_action": "retain_excluded_add_to_external_candidate_exclusion_ledger",
    })
    if reconciliation.record_fingerprint.duplicated().any():
        raise ValueError("R2 record reconciliation must remain one row per historical record")
    supplemental = pd.concat([
        fingerprint_frame(records.parent_id, "parent_id_cross_split_excluded"),
        fingerprint_frame(records.scaffold_group, "scaffold_group_cross_split_excluded"),
    ], ignore_index=True)
    if supplemental.duplicated().any():
        raise ValueError("Supplemental exclusion fingerprints must be unique")

    if args.check_only:
        print(f"Gate 9 R2 Papp preflight: records={len(records)} molecules={records.molecule_id.nunique()} parents={records.parent_id.nunique()} scaffolds={records.scaffold_group.nunique()}; no Papp values, targets, masks, predictions, or test values.")
        return
    with stage_output(output) as folder:
        reconciliation.to_csv(folder / "papp_cross_split_excluded_lineage_reconciliation.csv", index=False)
        supplemental.to_csv(folder / "papp_cross_split_excluded_supplemental_fingerprints.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 9 R2: Papp lineage reconciliation\n\n"
            "All 27 R1 records (24 molecule identities) are present in the full frozen split registry, standardize successfully, "
            "and are excluded solely because of cross-split parent/scaffold conflict. Recomputed canonical parent/scaffold identities "
            "match the frozen registry for every record. They remain ineligible for historical train/validation/test use. This audit only "
            "adds anonymous parent/scaffold fingerprints to the future external-candidate exclusion ledger; it does not repair the old split, "
            "restore any record to modelling, read a Papp label, or authorize a new cohort/model.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, endpoint_task=TASK,
            no_labels_accessed=True, no_raw_or_transformed_target_accessed=True, no_mask_accessed=True,
            no_prediction_rows_accessed=True, no_models_loaded=True, no_test_values_accessed=True,
            no_performance_metrics_computed=True, no_model_fitted=True, no_calibration=True,
            no_candidate_selection_changed=True, historical_records_reconciled=len(records),
            historical_molecules_reconciled=records.molecule_id.nunique(), records_restored_to_eligible=0,
            canonical_parent_match_count=len(records), canonical_scaffold_match_count=len(records),
            repair_mode="conservative_exclusion_ledger_only",
        )
    print(f"Gate 9 R2 Papp lineage reconciliation: {output}")


if __name__ == "__main__":
    run_cli(main)
