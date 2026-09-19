#!/usr/bin/env python3
"""Read-only anomaly audit for the six-task STL training input.

Audits Papp high values and extreme RDKit2D descriptor values without opening
fixed-validation or test target files and without modifying any data/model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


PAPP_TASK = "Papp__human__caco2_ab"


def numeric_summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "finite_values": int(len(finite)), "missing_or_nonfinite": int(len(values) - len(finite)),
        "min": float(np.min(finite)), "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)), "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)), "abs_gt_1e15": int(np.count_nonzero(np.abs(finite) > 1e15)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--historical-papp-decisions", type=Path, default=ROOT / "results/analysis/papp_provenance_decisions_v2.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_data_anomaly_audit_v1")
    parser.add_argument("--papp-high-threshold", type=float, default=1e4,
                        help="Canonical Papp threshold in 1e-6 cm/s; auditing only")
    parser.add_argument("--descriptor-extreme-threshold", type=float, default=1e15)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
                args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
                args.stl_protocol / "feature_registry.json", args.stl_protocol / "feature_row_manifest.csv",
                args.historical_papp_decisions]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    if train_meta.get("fixed_validation_targets_published") or train_meta.get("test_labels_read") or stl_meta.get("test_labels_read"):
        raise ValueError("Audit requires the train-only, test-closed protocol")
    if not np.isfinite(args.papp_high_threshold) or args.papp_high_threshold <= 0:
        raise ValueError("Papp threshold must be finite and positive")
    if not np.isfinite(args.descriptor_extreme_threshold) or args.descriptor_extreme_threshold <= 0:
        raise ValueError("Descriptor threshold must be finite and positive")

    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    records["target_value"] = pd.to_numeric(records.target_value, errors="raise")
    records["feature_index"] = pd.to_numeric(records.feature_index, errors="raise").astype(int)
    if not records.split.eq("train").all():
        raise ValueError("Train-only protocol unexpectedly contains non-train rows")
    registry = json.loads((args.stl_protocol / "feature_registry.json").read_text(encoding="utf-8"))
    descriptor_names = registry["rdkit2d"]["descriptor_names"]
    features = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    rdkit = features["rdkit2d"].astype(np.float64)
    parents = pd.read_csv(args.stl_protocol / "feature_row_manifest.csv", dtype={"molecule_id": str})
    if rdkit.shape[0] != len(parents) or rdkit.shape[1] != len(descriptor_names):
        raise ValueError("RDKit cache, parent manifest, and descriptor registry do not agree")
    if parents.feature_index.to_numpy(int).tolist() != list(range(len(parents))):
        raise ValueError("Parent feature indices are not canonical contiguous rows")
    decisions = pd.read_csv(args.historical_papp_decisions, dtype=str, keep_default_na=False)
    decision_columns = ["source_doc_id", "source_doi", "source_pubmed_id", "reviewer_decision",
                        "source_table_or_page", "evidence_note", "reviewer", "review_date"]
    if not set(decision_columns).issubset(decisions.columns):
        raise ValueError("Historical Papp decision registry has an unexpected schema")
    decisions = decisions[decision_columns].drop_duplicates("source_doc_id")

    papp = records.loc[records.task_id.eq(PAPP_TASK)].copy()
    if papp.empty or not papp.canonical_unit.eq("1e-6_cm/s").all() or not papp.target_transform.eq("log10").all():
        raise ValueError("Papp training definition differs from the registered audit scope")
    papp["derived_raw_value_1e_minus_6_cm_s"] = np.power(10.0, papp.target_value)
    papp["is_high_value"] = papp.derived_raw_value_1e_minus_6_cm_s.ge(args.papp_high_threshold)
    papp = papp.merge(decisions, left_on="doc_id", right_on="source_doc_id", how="left", validate="many_to_one")
    papp["decision_status"] = np.where(papp.reviewer_decision.eq(""), "no_historical_decision", "historical_decision_present")
    papp["current_audit_action"] = np.where(
        papp.is_high_value,
        "verify original unit, table/page, and ChEMBL conversion before any dataset revision",
        "no high-value review triggered",
    )

    descriptor_rows = []
    for index, name in enumerate(descriptor_names):
        descriptor_rows.append({"descriptor_index": index, "descriptor_name": name,
                                **numeric_summary(rdkit[:, index])})
    descriptor_summary = pd.DataFrame(descriptor_rows).sort_values("max", ascending=False)
    extreme_mask = np.abs(rdkit) > args.descriptor_extreme_threshold
    parent_rows, feature_rows = np.where(extreme_mask)
    descriptor_extremes = pd.DataFrame({
        "molecule_id": parents.molecule_id.to_numpy()[parent_rows],
        "canonical_parent_smiles": parents.canonical_parent_smiles.to_numpy()[parent_rows],
        "feature_index": parent_rows, "descriptor_index": feature_rows,
        "descriptor_name": [descriptor_names[item] for item in feature_rows],
        "descriptor_value": rdkit[parent_rows, feature_rows],
        "abs_descriptor_value": np.abs(rdkit[parent_rows, feature_rows]),
    }).sort_values("abs_descriptor_value", ascending=False)
    task_membership = records.groupby("molecule_id", as_index=False).agg(
        tasks=("task_id", lambda values: "|".join(sorted(set(values))),),
        record_rows=("row_id", "size"),
    )
    descriptor_extremes = descriptor_extremes.merge(task_membership, on="molecule_id", how="left", validate="many_to_one")

    papp_high = papp.loc[papp.is_high_value].sort_values("derived_raw_value_1e_minus_6_cm_s", ascending=False)
    document_queue = (papp_high.groupby(["doc_id", "source_doc_id", "source_doi", "source_pubmed_id", "reviewer_decision",
                                         "source_table_or_page", "evidence_note", "decision_status"], dropna=False)
                      .agg(high_train_records=("row_id", "size"), high_parents=("molecule_id", "nunique"),
                           value_min=("derived_raw_value_1e_minus_6_cm_s", "min"),
                           value_max=("derived_raw_value_1e_minus_6_cm_s", "max"))
                      .reset_index().sort_values("value_max", ascending=False))
    document_queue["review_status"] = "pending_current_version_source_and_unit_reconciliation"
    document_queue["allowed_action"] = "read-only evidence check; no label change without versioned decision"

    if args.check_only:
        print(f"STL anomaly audit ready: train_records={len(records)} papp_records={len(papp)} "
              f"papp_high={len(papp_high)} descriptor_extremes={len(descriptor_extremes)}")
        return
    with stage_output(args.output) as out:
        papp.sort_values("derived_raw_value_1e_minus_6_cm_s", ascending=False).to_csv(out / "papp_train_record_audit.csv", index=False)
        papp_high.to_csv(out / "papp_high_value_train_records.csv", index=False)
        document_queue.to_csv(out / "papp_high_value_document_queue.csv", index=False)
        descriptor_summary.to_csv(out / "rdkit2d_descriptor_distribution.csv", index=False)
        descriptor_extremes.to_csv(out / "rdkit2d_extreme_parent_records.csv", index=False)
        summary = {
            "train_records": int(len(records)), "papp_train_records": int(len(papp)),
            "papp_train_parents": int(papp.molecule_id.nunique()), "papp_high_threshold_1e_minus_6_cm_s": args.papp_high_threshold,
            "papp_high_records": int(len(papp_high)), "papp_high_parents": int(papp_high.molecule_id.nunique()),
            "papp_high_documents": int(papp_high.doc_id.nunique()), "descriptor_dimensions": int(rdkit.shape[1]),
            "descriptor_extreme_threshold": args.descriptor_extreme_threshold,
            "descriptor_extreme_cells": int(len(descriptor_extremes)),
            "descriptor_extreme_parents": int(descriptor_extremes.molecule_id.nunique()),
            "validation_target_file_opened": False, "validation_rows_evaluated": False, "test_labels_read": False,
            "data_modified": False, "model_fitted": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# STL data anomaly audit\n\nRead-only audit of training-only Papp values and RDKit2D descriptor extremes. "
            "A high value or extreme descriptor is a review signal, not an automatic exclusion. "
            "The historical Papp registry is attached for provenance only and must be reconciled to the current dataset before any revision.\n",
            encoding="utf-8",
        )
        finish_stage(out, "stl_data_anomaly_audit", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "historical_papp_decisions_sha256": sha256(args.historical_papp_decisions),
        }, partial=False, **summary)
    print(f"STL data anomaly audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
