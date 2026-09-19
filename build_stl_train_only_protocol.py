#!/usr/bin/env python3
"""Publish a model-facing STL view that contains training rows only.

The upstream benchmark remains immutable.  This stage gives future train-CV
runners a file that cannot expose fixed-validation targets and freezes the
fold-local target-standardization contract.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.upstream / "complete.json", args.upstream / "benchmark_records.csv",
                args.upstream / "task_manifest.csv", args.upstream / "canonical_parent_features_float32.npz"]
    startup_self_check(required, output=None if args.check_only else args.output)
    upstream = verify_stage(args.upstream, "stl_benchmark_protocol")
    if upstream.get("test_labels_read"):
        raise ValueError("Upstream protocol reports test-label access")

    train_rows, validation_membership = [], []
    with (args.upstream / "benchmark_records.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["split"] == "train":
                train_rows.append(row)
            elif row["split"] == "val":
                validation_membership.append({
                    "row_id": row["row_id"], "molecule_id": row["molecule_id"],
                    "task_id": row["task_id"], "inner_fold_id": row["inner_fold_id"],
                })
            else:
                raise ValueError(f"Unexpected benchmark split: {row['split']}")
    train = pd.DataFrame(train_rows)
    membership = pd.DataFrame(validation_membership)
    train["inner_fold_id"] = pd.to_numeric(train.inner_fold_id, errors="raise").astype(int)
    if len(train) != 10093 or train.task_id.nunique() != 6 or not train.inner_fold_id.between(0, 4).all():
        raise ValueError("Unexpected train-only benchmark population")
    if len(membership) != 1179 or not membership.inner_fold_id.eq("-1").all():
        raise ValueError("Unexpected locked fixed-validation membership")
    if set(train.row_id) & set(membership.row_id):
        raise ValueError("Train and fixed-validation row membership overlap")
    if train.groupby("molecule_id").inner_fold_id.nunique().gt(1).any() or train.groupby("scaffold_group").inner_fold_id.nunique().gt(1).any():
        raise ValueError("Parent or scaffold crosses frozen train folds")

    inputs = {"upstream_complete_sha256": sha256(args.upstream / "complete.json")}
    if args.check_only:
        if (args.output / "complete.json").exists():
            published = verify_stage(args.output, "stl_train_only_protocol")
            if published.get("inputs") != inputs:
                raise ValueError("Published train-only protocol no longer matches upstream")
        print(f"STL train-only protocol valid: records={len(train)} tasks={train.task_id.nunique()} validation_membership={len(membership)}")
        return

    summary = {
        "train_records": len(train), "train_parents": int(train.molecule_id.nunique()),
        "tasks": int(train.task_id.nunique()), "folds": 5,
        "fixed_validation_membership_rows": len(membership),
        "fixed_validation_targets_published": False,
        "runner_validation_target_access_possible": False,
        "fold_local_target_standardization_required": True,
        "test_labels_read": False,
    }
    with stage_output(args.output) as out:
        train.to_csv(out / "benchmark_train_records.csv", index=False)
        membership.to_csv(out / "fixed_validation_membership_no_targets.csv", index=False)
        pd.read_csv(args.upstream / "task_manifest.csv").to_csv(out / "task_manifest.csv", index=False)
        (out / "target_preprocessing_contract.json").write_text(json.dumps({
            "fit_unit": "collapsed canonical parent within each task and training fold",
            "fit_scope": "for held-out fold k, mean and population SD use only parents in folds != k",
            "training_target": "(parent_mean_transformed_target - fold_train_mean) / fold_train_population_sd",
            "prediction_storage": "convert to transformed scale, then to the frozen global-interface scale for comparable metrics",
            "zero_sd_policy": "hard failure",
            "fixed_validation_policy": "no target-bearing validation file is available to train-CV runners",
        }, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# STL train-only protocol v1\n\nModel-facing train-CV input containing no fixed-validation targets. "
            "Every fold must fit target centering/scaling from collapsed parents in that fold's training partition.\n",
            encoding="utf-8",
        )
        finish_stage(out, "stl_train_only_protocol", inputs=inputs, partial=False, **summary)
    print(f"STL train-only protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
