#!/usr/bin/env python3
"""One-time descriptive Jia author-test scorer for the frozen R3a models."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_common import (ROOT, configure_logging, finish_stage, run_cli, sha256,
                             stage_output, startup_self_check, stable_id, verify_stage)
from smoke_jia2025_author_train_only import TASKS
from train_jia2025_author_like_r3a import feature_matrices
from audit_jia2025_n0 import canonical_parent_and_scaffold


def transformed_features(matrix, state):
    x = state["imputer"].transform(matrix[:, state["keep"]])
    return state["scaler"].transform(x) if state["scaler"] is not None else x


def metric_row(frame, prediction, endpoint, model_id, cohort):
    observed = frame["target"].to_numpy(float)
    error = np.abs(prediction - observed)
    return {"endpoint": endpoint, "model_id": model_id, "cohort": cohort, "records": int(len(frame)),
            "parents": int(frame.parent_hash.nunique()), "R2_log": float(r2_score(observed, prediction)),
            "MAE_log": float(mean_absolute_error(observed, prediction)),
            "RMSE_log": float(np.sqrt(mean_squared_error(observed, prediction))),
            "GMFE": float(10 ** error.mean()), "within_2fold": float(np.mean(error <= np.log10(2))),
            "selection_changed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/jia2025_author_like_r3b_protocol_v5")
    p.add_argument("--workbook", type=Path, default=ROOT / "基准研究参考/jm5c00340_si_001.xlsx")
    p.add_argument("--output", type=Path, default=ROOT / "results/final/jia2025_author_like_r3b_one_time_score_v5")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--membership-smoke", action="store_true", help="Read only SMILES_parent and split fields; never read labels or predict")
    p.add_argument("--confirm-one-time-author-test-score", action="store_true")
    args = p.parse_args()
    startup_self_check([args.protocol / "complete.json"], output=None if args.check_only else args.output)
    protocol = verify_stage(args.protocol, "jia2025_author_like_r3b_scoring_protocol")
    evaluator_path = Path(__file__).resolve()
    if protocol["inputs"].get(str(evaluator_path)) != sha256(evaluator_path):
        raise ValueError("Evaluator code differs from the protocol-bound scorer; register a new protocol version")
    models = pd.read_csv(args.protocol / "model_hashes.csv")
    root = args.protocol.parents[2]
    for _, row in models.iterrows():
        path = root / row.artifact
        if not path.is_file() or sha256(path) != row.sha256:
            raise ValueError(f"Frozen model missing or changed: {row.artifact}")
        bundle = joblib.load(path)
        if bundle["endpoint"] != row.endpoint or bundle["model_id"] != row.model_id:
            raise ValueError(f"Frozen model identity mismatch: {row.artifact}")
    if args.membership_smoke:
        startup_self_check([args.workbook])
        source = pd.read_excel(args.workbook, sheet_name="Fu_VDss_CL_modeling_set",
                               usecols=["SMILES_parent", *[spec[2] for spec in TASKS.values()]])
        expected = pd.read_csv(args.protocol / "author_test_membership_hashed.csv")
        for endpoint, (_, _, split_col) in TASKS.items():
            observed = set(source.loc[source[split_col].astype(str).str.lower().eq("test"), "SMILES_parent"].map(lambda s: canonical_parent_and_scaffold(s)[0]))
            frozen = set(expected.loc[expected.endpoint.eq(endpoint), "parent_hash"])
            if observed != frozen:
                raise ValueError(f"{endpoint} label-blind canonical membership smoke disagrees with frozen registry")
        print(f"Jia R3b canonical membership smoke: models={len(models)} all endpoint parent sets passed; no labels/predictions")
        return
    if args.check_only:
        print(f"Jia R3b scorer smoke: models={len(models)} hashes/reload passed; no author-test row read")
        return
    if not args.confirm_one_time_author_test_score:
        raise ValueError("Author test scoring requires --confirm-one-time-author-test-score")
    startup_self_check([args.workbook], output=args.output)
    configure_logging(ROOT, "evaluate_jia2025_author_like_r3b")
    membership = pd.read_csv(args.protocol / "author_test_membership_hashed.csv")
    expected = {e: set(membership.loc[membership.endpoint.eq(e), "parent_hash"]) for e in TASKS}
    with stage_output(args.output) as out:
        raw = pd.read_excel(args.workbook, sheet_name="Fu_VDss_CL_modeling_set")
        test, train_parent_sets = {}, {}
        for endpoint, (raw_col, log_col, split_col) in TASKS.items():
            table = raw.loc[raw[split_col].astype(str).str.lower().eq("test"), ["SMILES_parent", raw_col, log_col]].dropna().copy()
            table["raw"] = pd.to_numeric(table[raw_col], errors="raise")
            table["target"] = pd.to_numeric(table[log_col], errors="raise")
            if (table.raw <= 0).any() or not np.isclose(np.log10(table.raw), table.target, atol=1e-5, rtol=0).all():
                raise ValueError(f"{endpoint} author-test raw/log contract fails")
            table["parent_hash"] = table.SMILES_parent.map(lambda s: canonical_parent_and_scaffold(s)[0])
            if set(table.parent_hash) != expected[endpoint]:
                raise ValueError(f"{endpoint} author-test membership disagrees with frozen N0 registry")
            train = raw.loc[raw[split_col].astype(str).str.lower().eq("train"), "SMILES_parent"].dropna()
            train_parent_sets[endpoint] = set(train.map(lambda s: canonical_parent_and_scaffold(s)[0]))
            test[endpoint] = table.reset_index(drop=True)
        all_smiles = list(dict.fromkeys(s for table in test.values() for s in table.SMILES_parent.tolist()))
        features, _ = feature_matrices(all_smiles)
        index = {s: i for i, s in enumerate(all_smiles)}
        rows, predictions = [], []
        for endpoint, table in test.items():
            ids = np.asarray([index[s] for s in table.SMILES_parent], int)
            per_model = {}
            for _, row in models.loc[models.endpoint.eq(endpoint)].iterrows():
                bundle = joblib.load(root / row.artifact)
                matrix = np.zeros((len(ids), 1), dtype=np.float32) if row.representation == "none" else features[row.representation][ids]
                pred = np.asarray(bundle["estimator"].predict(transformed_features(matrix, bundle["feature_state"])), float)
                if not np.isfinite(pred).all(): raise ValueError(f"Nonfinite author-test prediction: {endpoint}/{row.model_id}")
                per_model[row.model_id] = pred
            if endpoint in {"CL", "VDss"}:
                paper_ids = sorted(k for k in per_model if k.startswith("paper_"))
                if len(paper_ids) != 15: raise ValueError(f"{endpoint} consensus lacks 15 paper models")
                per_model["paper_consensus_15"] = np.mean(np.stack([per_model[k] for k in paper_ids]), axis=0)
            purged = ~table.parent_hash.isin(train_parent_sets[endpoint]).to_numpy()
            for model_id, pred in per_model.items():
                for cohort, mask in (("author_native_record", np.ones(len(table), dtype=bool)), ("author_parent_purged_sensitivity", purged)):
                    if mask.sum() < 2: raise ValueError(f"Insufficient {endpoint}/{cohort} records")
                    rows.append(metric_row(table.loc[mask], pred[mask], endpoint, model_id, cohort))
                predictions.append(pd.DataFrame({"endpoint": endpoint, "parent_hash": table.parent_hash,
                                                 "model_id": model_id, "predicted_log10": pred,
                                                 "observed_log10": table.target, "cohort_native": True}))
        pd.DataFrame(rows).to_csv(out / "author_test_metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(out / "author_test_predictions.csv", index=False)
        lifecycle = {"test_lifecycle": "scored_once_closed", "models_changed": False, "selection_changed": False,
                     "author_test_rows_scored": True, "rerun_prohibited": True,
                     "allowed_followup": "read-only diagnostics and manuscript reporting only"}
        (out / "test_lifecycle_closure.json").write_text(__import__("json").dumps(lifecycle, indent=2), encoding="utf-8")
        (out / "README.md").write_text("# Jia author-like R3b one-time score\n\nB-grade author-like descriptive score. Native and parent-purged sensitivity are separate; no post-score selection is allowed.\n", encoding="utf-8")
        finish_stage(out, "jia2025_author_like_r3b_one_time_descriptive_score", inputs={str((args.protocol / "complete.json").resolve()): sha256(args.protocol / "complete.json"), str(args.workbook.resolve()): sha256(args.workbook)},
                     protocol_grade="B_author_like", author_test_scored=True, models_changed=False, selection_changed=False,
                     rerun_prohibited=True, partial=False)
    print(f"Jia 2025 R3b one-time descriptive score: {args.output}")


if __name__ == "__main__": run_cli(main)
