#!/usr/bin/env python3
"""Freeze the Gate-3 B6 aggregate-analysis rules before sensitivity outcomes are read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, default=ROOT / "data/public_development/gate3_b6_formal_run_v4")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_protocol_v2")
    parser.add_argument("--batch", type=Path, default=ROOT / "results/benchmarks/gate3_b6_physchem_formal_cells_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/gate3_b6_aggregate_analysis_protocol_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.formal / "complete.json", args.formal / "formal_authorization.json",
        args.formal / "formal_cell_registry.csv", args.protocol / "complete.json",
        args.protocol / "protocol.json", args.protocol / "endpoint_advance_gates.csv",
        args.batch / "batch_complete.json",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    formal = verify_stage(args.formal, "gate3_b6_formal_run_freeze")
    protocol = verify_stage(args.protocol, "gate3_b6_physchem_protocol")
    batch = json.loads((args.batch / "batch_complete.json").read_text())
    if not formal.get("formal_train_cv_authorized") or formal.get("test_labels_read"):
        raise ValueError("Formal train-CV authorization is invalid")
    if protocol.get("test_labels_read") or batch.get("cells_verified") != 30:
        raise ValueError("Protocol/test isolation or batch coverage is invalid")
    for key in ["all_cells_full_configuration", "all_cells_parent_scaffold_overlap_zero",
                "all_cells_imputation_audits_complete"]:
        if batch.get(key) is not True:
            raise ValueError(f"Batch contract failed: {key}")

    contract = {
        "schema_version": 1,
        "analysis_id": "gate3_b6_preregistered_aggregate_v1",
        "primary_aggregation": "equal_weight_mean_of_five_outer_fold_primary_errors",
        "paired_bootstrap": {
            "unit": "parent",
            "stratification": "outer_fold",
            "replicates": 10000,
            "seed": 20260917,
            "confidence_interval": "percentile_95",
            "difference_direction": "B6_minus_comparator; negative_favors_B6",
        },
        "low_similarity_subset": {
            "fingerprint": "Morgan_radius2_2048bit_no_chirality",
            "reference": "nearest_parent_in_same_task_current_outer_training_folds",
            "cutoff": "within_outer_fold_lower_quartile_inclusive",
            "label_free_definition": True,
        },
        "repeated_parent_subset": "source_train_record_count_per_task_parent_at_least_2",
        "sufficient_power": {"minimum_parents": 50, "minimum_outer_folds": 3},
        "sensitivity_relative_harm_limit_pct": 2.0,
        "A6_rule": "every sufficiently powered subset must have B6_vs_C0 relative harm <=2%; low_similarity must be evaluable; underpowered repeated-parent is reported but does not fabricate a pass/fail claim",
        "figures": {"format": "PNG", "dpi": 600, "language": "English"},
        "model_refit": False,
        "route_reselection_before_gate_table": False,
        "fixed_validation_status": "closed",
        "test_status": "closed",
    }
    if args.check_only:
        print("Gate 3 B6 aggregate-analysis freeze ready: bootstrap=10000 sensitivity_n>=50 folds>=3")
        return
    inputs = {str(path.resolve()): sha256(path) for path in required}
    with stage_output(args.output) as out:
        (out / "analysis_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Gate 3 B6 aggregate-analysis protocol v1\n\n"
            "This label-independent analysis contract was frozen before low-similarity and repeated-parent "
            "sensitivity outcomes were computed. It binds the completed 30-cell v4 batch, uses 10,000 "
            "outer-fold-stratified paired-parent bootstrap replicates, and keeps fixed validation/test closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "gate3_b6_aggregate_analysis_protocol", inputs=inputs,
            bootstrap_replicates=10000, bootstrap_seed=20260917,
            sensitivity_minimum_parents=50, sensitivity_minimum_outer_folds=3,
            model_fitted=False, architecture_selection_authorized=False,
            fixed_validation_authorized=False, test_labels_read=False, partial=False,
        )
    print(f"Gate 3 B6 aggregate-analysis protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
