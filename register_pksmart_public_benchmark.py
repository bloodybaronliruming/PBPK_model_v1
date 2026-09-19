#!/usr/bin/env python3
"""Pre-register a PKSmart public-dataset benchmark isolated from the formal external set.

This registry is intentionally label-blinded.  It fixes the raw PKSmart file,
the frozen v15 ensemble, structure-only cohort membership, metrics, and
interpretation before any PKSmart label is loaded for scoring.  It neither
changes the formal external-validation registry nor permits model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED_RAW_COLUMNS = {"smiles_r", "human_thalf"}
REQUIRED_QUEUE_COLUMNS = {
    "candidate_id", "source_row", "parent_id", "scaffold_group", "molecule_id", "screening_status",
    "endpoint_values_hidden",
}
QUEUE_COLUMNS = [
    "candidate_id", "source_row", "parent_id", "scaffold_group", "molecule_id", "screening_status",
    "endpoint_values_hidden",
]
PRIMARY_STATUS = "primary_source_provenance_required"


def build_membership(queue: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_QUEUE_COLUMNS - set(queue.columns)
    if missing:
        raise ValueError(f"PKSmart candidate queue misses columns: {sorted(missing)}")
    if not queue.endpoint_values_hidden.astype(bool).all():
        raise ValueError("PKSmart benchmark registration requires a label-blinded candidate queue")
    members = queue[QUEUE_COLUMNS].copy()
    members["benchmark_cohort"] = "diagnostic_public_rows_with_reference_overlap"
    members.loc[members.screening_status.eq(PRIMARY_STATUS), "benchmark_cohort"] = (
        "primary_structure_isolated_public_rows")
    members["primary_metric_cohort"] = members.screening_status.eq(PRIMARY_STATUS)
    members["endpoint_values_hidden"] = True
    return members.sort_values(["benchmark_cohort", "candidate_id"], kind="stable")


def model_contract(frozen: Path) -> dict:
    meta = verify_stage(frozen, "thalf_v15_frozen_candidate")
    registry_path = frozen / "candidate_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("task_id") != "Thalf__human__terminal_iv" or registry.get("feature_set") != "rdkit2d":
        raise ValueError("Frozen model does not match the registered Thalf v15 contract")
    if not registry.get("selection_finalized_before_tdc_benchmark"):
        raise ValueError("Frozen model predates neither its selection benchmark nor its registry contract")
    return {
        "frozen_stage_sha256": sha256(frozen / "complete.json"),
        "candidate_registry_sha256": sha256(registry_path),
        "members": registry["members"],
        "aggregation": registry["aggregation"],
        "frozen_model_test_evaluated": bool(meta.get("v15_test_labels_evaluated")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/raw/External_test_315.csv")
    parser.add_argument("--candidate-queue", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_candidate_queue_v1")
    parser.add_argument("--frozen-model", type=Path,
                        default=ROOT / "models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/benchmarks/pksmart_public_thalf_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.candidate_queue, "pksmart_external_candidate_queue")
    startup_self_check([args.raw, args.candidate_queue / "candidate_records_blinded.csv",
                        args.frozen_model / "complete.json", args.frozen_model / "candidate_registry.json"],
                       output=None if args.check_only else args.output)
    header = pd.read_csv(args.raw, nrows=0)
    if not REQUIRED_RAW_COLUMNS <= set(header.columns):
        raise ValueError(f"Unsupported PKSmart raw schema: {sorted(REQUIRED_RAW_COLUMNS - set(header.columns))}")
    membership = build_membership(pd.read_csv(args.candidate_queue / "candidate_records_blinded.csv", keep_default_na=False))
    contract = model_contract(args.frozen_model)
    if args.check_only:
        print(f"PKSmart public benchmark contract valid: primary_structure_isolated={membership.primary_metric_cohort.sum()} "
              f"diagnostic_rows={len(membership)}; endpoint values not loaded")
        return
    policy = {
        "registry_version": 1,
        "benchmark_kind": "public_dataset_aligned_secondary_benchmark",
        "endpoint": "human plasma half-life in hours as supplied by the frozen public PKSmart CSV",
        "raw_source": str(args.raw.resolve()),
        "raw_sha256": sha256(args.raw),
        "raw_label_column": "human_thalf",
        "raw_smiles_column": "smiles_r",
        "label_visibility_at_registration": "hidden",
        "primary_cohort": "primary_structure_isolated_public_rows",
        "diagnostic_cohort": "diagnostic_public_rows_with_reference_overlap",
        "cohort_policy": (
            "Primary metrics use only rows pre-excluded from v15/TDC/current-formal-registry parent or scaffold overlap. "
            "The full public cohort is diagnostic only. Neither cohort is source-independent because PKSmart does not "
            "provide a row-level primary-study locator."),
        "frozen_model": contract,
        "pre_registered_metrics": ["log10_rmse", "log10_mae", "molecule_mean_spearman", "gmfe", "within_2fold", "within_3fold"],
        "uncertainty": {"method": "molecule_bootstrap", "replicates": 10000, "seed": 20260915},
        "prohibited_actions": [
            "model_or_hyperparameter_selection", "calibration", "retraining", "label_based_cohort_filtering",
            "merging_with_formal_external_registry", "claiming_source_independent_external_validation",
        ],
        "evaluation_status": "registered_not_scored",
    }
    with stage_output(args.output) as out:
        membership.to_csv(out / "cohort_membership_blinded.csv", index=False)
        (membership.groupby(["benchmark_cohort", "primary_metric_cohort"], dropna=False).size().rename("records")
         .reset_index().to_csv(out / "cohort_summary.csv", index=False))
        (out / "benchmark_registry.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "README.md").write_text(
            "# PKSmart public-dataset benchmark registry\n\n"
            "This is a separate, pre-registered public-dataset-aligned benchmark. It is not part of the formal "
            "primary-evidence external validation registry. The membership file is label-blinded; labels can be loaded "
            "only by a later evaluator that validates this immutable registry and the frozen-model hashes.\n",
            encoding="utf-8")
        finish_stage(out, "pksmart_public_benchmark_registry", inputs={
            "raw_sha256": sha256(args.raw),
            "candidate_queue_complete_sha256": sha256(args.candidate_queue / "complete.json"),
            "frozen_model_complete_sha256": sha256(args.frozen_model / "complete.json"),
        }, primary_structure_isolated_records=int(membership.primary_metric_cohort.sum()),
           diagnostic_public_records=len(membership), label_blinded=True, scored=False, partial=False)
    print(f"PKSmart public benchmark registered without loading labels: {args.output}")


if __name__ == "__main__":
    run_cli(main)
