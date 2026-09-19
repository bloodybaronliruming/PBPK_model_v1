#!/usr/bin/env python3
"""N1c smoke: verify strict-train-only conditional MMPK loaders without fitting a model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mmpk_nca_common import ENDPOINTS, TIER_TO_COLUMN, approved_model_path, load_strict_train_fold
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "基准研究参考")
    parser.add_argument("--n0", type=Path, default=ROOT / "results/analysis/mmpk_n0_audit_v5")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/mmpk_n1c_train_fold_loader_smoke_v1")
    args = parser.parse_args()
    model_path = approved_model_path(args.reference)
    startup_self_check([args.n0 / "complete.json", args.n1b / "complete.json", model_path], output=args.output)
    verify_stage(args.n0, "mmpk_n0_read_only_data_code_license_audit")
    n1b_meta = verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    if n1b_meta["inputs"].get(str(model_path.resolve())) != sha256(model_path):
        raise ValueError("N1b source hash does not bind this approved model table")
    records = pd.read_csv(args.n1b / "approved_record_registry_hashed.csv")
    labels = pd.read_csv(args.n1b / "approved_label_tier_registry_hashed.csv")
    rows, sealing = [], []
    for fold in range(5):
        held = records[records.strict_outer_fold.eq(fold)]
        for tier_view in TIER_TO_COLUMN:
            loaded = load_strict_train_fold(fold, tier_view, reference=args.reference, n0=args.n0, n1b=args.n1b)
            train_ids = set(loaded.model_record_ids)
            held_ids = set(held.model_record_id)
            if train_ids & held_ids or len(train_ids) + len(held_ids) != len(records):
                raise ValueError("Held outer-fold records entered the returned train-only loader")
            if loaded.features.shape[0] != len(train_ids) or loaded.features.shape[1] != 2049:
                raise ValueError("Unexpected conditional feature layout")
            if not np.isfinite(loaded.features).all() or not np.isfinite(loaded.log_dose).all():
                raise ValueError("Non-finite training input features")
            if not np.array_equal(np.isfinite(loaded.targets), loaded.target_mask):
                raise ValueError("Masked target matrix contains an unmasked or non-finite label")
            for idx, (endpoint, _) in enumerate(ENDPOINTS):
                expected = labels[(labels.strict_outer_fold.ne(fold)) & (labels.endpoint.eq(endpoint))][TIER_TO_COLUMN[tier_view]].astype(bool).sum()
                observed = int(loaded.target_mask[:, idx].sum())
                if observed != int(expected):
                    raise ValueError(f"{endpoint}/{tier_view}: loaded label count differs from frozen N1b membership")
                rows.append({"outer_fold": fold, "tier_view": tier_view, "endpoint": endpoint,
                             "train_records": len(train_ids), "train_labels": observed,
                             "feature_dimension": int(loaded.features.shape[1]),
                             "training_membership_hash": loaded.input_state["training_membership_hash"],
                             "dose_scaler_state_hash": loaded.input_state["dose_scaler_state_hash"]})
            sealing.append({"outer_fold": fold, "tier_view": tier_view, "held_records": len(held_ids),
                            "returned_train_records": len(train_ids), "held_record_overlap": len(train_ids & held_ids),
                            "held_parent_overlap": len(set(held.parent_id) & set(records.loc[records.model_record_id.isin(train_ids), "parent_id"])),
                            "held_scaffold_overlap": len(set(held.scaffold_id) & set(records.loc[records.model_record_id.isin(train_ids), "scaffold_id"])),
                            "fit_scope": loaded.input_state["dose_scaler_fit_scope"],
                            "feature_layout": "ECFP4_2048_plus_log10_dose_mg_per_kg"})
    audit = pd.DataFrame(sealing)
    if (audit[["held_record_overlap", "held_parent_overlap", "held_scaffold_overlap"]] != 0).any().any():
        raise ValueError("Strict train-only loader has a record, parent, or scaffold holdout leak")
    counts = pd.DataFrame(rows)
    direct = counts[counts.tier_view.eq("R1_direct")].set_index(["outer_fold", "endpoint"]).train_labels
    augmented = counts[counts.tier_view.eq("R1_plus_R2_augmented")].set_index(["outer_fold", "endpoint"]).train_labels
    if not (direct <= augmented).all():
        raise ValueError("R1 direct labels are not a subset of the augmented view")
    with stage_output(args.output) as out:
        counts.to_csv(out / "strict_train_input_mask_summary.csv", index=False)
        audit.to_csv(out / "strict_train_sealing_audit.csv", index=False)
        (out / "README.md").write_text(
            "# MMPK N1c strict train-fold-only loader smoke\n\n"
            "This is an interface audit, not a model experiment. For every strict outer fold it materializes only approved-development outer-training records, ECFP4 plus log-dose input, and R1 or R1+R2 target masks. It neither fits a predictive model nor creates predictions or performance metrics. Held-out labels, investigational labels and 2024 labels are not returned or exported. Dose scaling state is fitted only on the returned outer-training inputs and only its hash is retained.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n1c_strict_train_fold_conditional_loader_smoke", inputs={
            str((args.n0 / "complete.json").resolve()): sha256(args.n0 / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json"),
            str(model_path.resolve()): sha256(model_path)}, no_training=True, no_predictions=True,
            no_performance_metrics=True, external_labels_accessed=False, heldout_labels_returned=False,
            raw_labels_exported=False, strict_outer_folds=5, feature_layout="ECFP4_2048_plus_log10_dose_mg_per_kg",
            tier_views=list(TIER_TO_COLUMN), partial=False)
    print(f"MMPK N1c strict train-fold loader smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
