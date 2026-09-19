#!/usr/bin/env python3
"""Run the frozen N2a R1 strict nested-baseline screen after explicit confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dmpk_toolkit import featurize
from mmpk_nca_common import ENDPOINTS, approved_model_path, load_strict_train_fold
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


PRIMARY = ("AUC [ng*h/mL]", "Cmax [ng/mL]", "Tmax [h]", "t1/2 [h]")


def make_prediction(algorithm, parameters, seed, threads, device, x_fit, y_fit, x_predict):
    """Fit preprocessing only on fitting records and predict in log target space."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    a = imputer.fit_transform(x_fit)
    b = imputer.transform(x_predict)
    if algorithm == "dummy_mean":
        return np.full(len(b), float(np.mean(y_fit)))
    if algorithm == "ridge":
        scaler = StandardScaler()
        return Ridge(**parameters).fit(scaler.fit_transform(a), y_fit).predict(scaler.transform(b))
    if algorithm in {"random_forest", "extra_trees"}:
        cls = RandomForestRegressor if algorithm == "random_forest" else ExtraTreesRegressor
        return cls(random_state=seed, n_jobs=threads, **parameters).fit(a, y_fit).predict(b)
    if algorithm == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist", device=device,
                             random_state=seed, n_jobs=threads, **parameters).fit(a, y_fit)
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if device == "cuda" and not str(actual).startswith("cuda"):
            raise RuntimeError(f"XGBoost silently fell back from CUDA: {actual}")
        return model.predict(b)
    raise ValueError(f"Unknown frozen N2a algorithm: {algorithm}")


def metric_row(y, prediction) -> dict:
    err = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
    return {
        "R1_outer_records": int(len(y)),
        "log_RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "GMFE": float(10 ** np.mean(np.abs(err))),
        "AFE": float(10 ** np.mean(err)),
        "two_fold_accuracy": float(np.mean(np.abs(err) <= np.log10(2.0))),
        "log_R2": float(r2_score(y, prediction)),
    }


def load_outer_inputs(outer_fold: int, feature_view: str, n1b: Path) -> tuple[list[str], np.ndarray]:
    """Load held-out features only; target columns are intentionally not read here."""
    records = pd.read_csv(n1b / "approved_record_registry_hashed.csv")
    held_ids = records.loc[records.strict_outer_fold.eq(outer_fold), "model_record_id"].tolist()
    source = pd.read_csv(approved_model_path(), usecols=["SMILES", "Log Dose [mg/kg]"])
    source.insert(0, "model_record_id", [stable_id(f"mmpk_approved_model_row:{idx}") for idx in source.index])
    held = source[source.model_record_id.isin(held_ids)].copy()
    if len(held) != len(held_ids) or set(held.model_record_id) != set(held_ids):
        raise ValueError("Outer held input membership differs from N1b")
    log_dose = pd.to_numeric(held["Log Dose [mg/kg]"], errors="coerce").to_numpy(float)
    if not np.isfinite(log_dose).all():
        raise ValueError("Outer held input has missing author log-dose")
    feature_set = "ecfp4" if feature_view == "ecfp4_logdose" else "ecfp4_rdkit2d"
    structure, _ = featurize(held.SMILES.tolist(), feature_set=feature_set, progress=False)
    x = np.column_stack([structure, log_dose.astype(np.float32)]).astype(np.float32, copy=False)
    if np.isinf(x).any() or not np.isfinite(x[:, -1]).all():
        raise ValueError("Outer held feature matrix is invalid")
    return held.model_record_id.tolist(), x


def load_outer_r1_targets_after_selection(outer_fold: int, record_ids: list[str], n1b: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read held R1 labels only after all inner candidate scores have been chosen."""
    labels = pd.read_csv(n1b / "approved_label_tier_registry_hashed.csv")
    source = pd.read_csv(approved_model_path(), usecols=[column for _, column in ENDPOINTS])
    source.insert(0, "model_record_id", [stable_id(f"mmpk_approved_model_row:{idx}") for idx in source.index])
    source = source.set_index("model_record_id").reindex(record_ids)
    if source.index.isna().any():
        raise ValueError("Outer held source target identity mismatch")
    result = {}
    for endpoint, column in ENDPOINTS:
        if endpoint not in PRIMARY:
            continue
        frozen = labels[(labels.endpoint.eq(endpoint)) & labels.direct_observation_eligible.astype(bool)]
        allowed = set(frozen.model_record_id)
        values = pd.to_numeric(source[column], errors="coerce").to_numpy(float)
        mask = np.asarray([record in allowed for record in record_ids]) & np.isfinite(values)
        if not mask.any():
            raise ValueError(f"Outer held R1 target is empty: {endpoint}")
        result[endpoint] = mask, values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n2a_baseline_protocol_v2")
    parser.add_argument("--n1b", type=Path, default=ROOT / "data/nca_internal_development/mmpk_n1b_cohort_protocol_v3")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/mmpk_n2b_strict_r1_baseline_screen_v1")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm-formal-r1-screen", action="store_true")
    args = parser.parse_args()
    startup_self_check([args.protocol / "complete.json", args.n1b / "complete.json"], output=None if args.check_only else args.output)
    protocol = verify_stage(args.protocol, "mmpk_n2a_strict_conditional_baseline_protocol")
    n1b = verify_stage(args.n1b, "mmpk_n1b_internal_cohort_and_dual_split_protocol")
    model_path = approved_model_path()
    if n1b["inputs"].get(str(model_path.resolve())) != sha256(model_path):
        raise ValueError("N1b does not bind the current approved model source")
    candidates = pd.read_csv(args.protocol / "baseline_candidate_registry.csv")
    membership = pd.read_csv(args.protocol / "strict_nested_inner_membership_hashed.csv")
    if len(candidates) != protocol.get("candidate_cells") or candidates.candidate_id.duplicated().any() or set(PRIMARY) != set(protocol["primary_endpoints"]):
        raise ValueError("N2a candidate or endpoint registry differs from its frozen protocol")
    expected_fits = 5 * len(PRIMARY) * len(candidates) * 4 + 5 * len(PRIMARY)
    if args.check_only:
        print(f"MMPK N2b formal screen ready: candidates={len(candidates)} inner_fits={expected_fits - 20} outer_refits=20 total_fits={expected_fits} device={args.device}")
        return
    if not args.confirm_formal_r1_screen:
        raise ValueError("Formal strict CV creates performance results; pass --confirm-formal-r1-screen after GPU smoke")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing a silent CPU fallback for the frozen XGBoost candidate")
        cuda_name = torch.cuda.get_device_name(0)
    else:
        cuda_name = "not_used"
    progress = tqdm(total=expected_fits, desc="MMPK_N2B_PROGRESS", unit="fit")
    selection_rows, metric_rows, prediction_rows = [], [], []
    with stage_output(args.output) as out:
        for outer in range(5):
            nested = membership[membership.outer_test_fold.eq(outer)].copy()
            if nested.model_record_id.duplicated().any() or set(nested.inner_fold) != set(range(4)):
                raise ValueError("N2a nested membership is incomplete for an outer fold")
            views = {view: load_strict_train_fold(outer, "R1_direct", n1b=args.n1b, feature_view=view)
                     for view in sorted(candidates.feature_view.unique()) if view != "dose_only"}
            # All non-dummy candidates must share the exact N2a outer-train membership.
            expected_ids = set(nested.model_record_id)
            if any(set(value.model_record_ids) != expected_ids for value in views.values()):
                raise ValueError("N2b loader membership differs from N2a nested membership")
            local = pd.DataFrame({"model_record_id": next(iter(views.values())).model_record_ids,
                                  "row": np.arange(len(next(iter(views.values())).model_record_ids))})
            local = local.merge(nested[["model_record_id", "inner_fold", "parent_id", "scaffold_id", "strict_component_id"]], on="model_record_id", validate="one_to_one")
            winners = {}
            for endpoint_index, endpoint in enumerate(PRIMARY):
                candidate_scores = []
                for candidate in candidates.itertuples(index=False):
                    inner_scores = []
                    for inner_fold in range(4):
                        fitting = local[local.inner_fold.ne(inner_fold)].row.to_numpy()
                        evaluating = local[local.inner_fold.eq(inner_fold)].row.to_numpy()
                        if candidate.algorithm == "dummy_mean":
                            # dummy uses the same dose-only membership but no structural matrix.
                            loaded = next(iter(views.values()))
                        else:
                            loaded = views[candidate.feature_view]
                        fit_rows = fitting[loaded.target_mask[fitting, endpoint_index]]
                        eval_rows = evaluating[loaded.target_mask[evaluating, endpoint_index]]
                        if len(fit_rows) == 0 or len(eval_rows) == 0:
                            raise ValueError("Nested R1 label mask unexpectedly empty")
                        parameters = json.loads(candidate.parameters_json)
                        prediction = make_prediction(candidate.algorithm, parameters, int(candidate.random_seed), args.threads,
                                                     args.device, loaded.features[fit_rows], loaded.targets[fit_rows, endpoint_index], loaded.features[eval_rows])
                        inner_scores.append(float(np.sqrt(mean_squared_error(loaded.targets[eval_rows, endpoint_index], prediction))))
                        progress.update(1)
                    candidate_scores.append({"outer_test_fold": outer, "endpoint": endpoint, "candidate_id": candidate.candidate_id,
                                             "algorithm": candidate.algorithm, "feature_view": candidate.feature_view,
                                             "mean_inner_log_RMSE": float(np.mean(inner_scores)), "inner_log_RMSE_sd": float(np.std(inner_scores, ddof=0)),
                                             "inner_folds": 4})
                candidate_scores = pd.DataFrame(candidate_scores).sort_values(["mean_inner_log_RMSE", "candidate_id"], kind="stable")
                selected = candidate_scores.iloc[0].to_dict()
                selected["selected"] = True
                candidate_scores["selected"] = candidate_scores.candidate_id.eq(selected["candidate_id"])
                selection_rows.extend(candidate_scores.to_dict("records"))
                winners[endpoint] = selected
            # The outer feature view is loaded before labels, and R1 target values
            # are read only after all four endpoint configurations are selected.
            outer_inputs = {view: load_outer_inputs(outer, view, args.n1b) for view in views}
            target_ids = next(iter(outer_inputs.values()))[0]
            if any(ids != target_ids for ids, _ in outer_inputs.values()):
                raise ValueError("Outer feature views do not preserve a common record order")
            outer_targets = load_outer_r1_targets_after_selection(outer, target_ids, args.n1b)
            for endpoint_index, endpoint in enumerate(PRIMARY):
                winner = winners[endpoint]
                loaded = next(iter(views.values())) if winner["algorithm"] == "dummy_mean" else views[winner["feature_view"]]
                train_rows = np.flatnonzero(loaded.target_mask[:, endpoint_index])
                target_mask, target_values = outer_targets[endpoint]
                _, outer_x = outer_inputs["ecfp4_logdose" if winner["algorithm"] == "dummy_mean" else winner["feature_view"]]
                prediction = make_prediction(winner["algorithm"], json.loads(candidates.set_index("candidate_id").loc[winner["candidate_id"], "parameters_json"]),
                                             int(protocol["random_seed"]), args.threads, args.device,
                                             loaded.features[train_rows], loaded.targets[train_rows, endpoint_index], outer_x[target_mask])
                values = target_values[target_mask]
                metric_rows.append({"outer_test_fold": outer, "endpoint": endpoint, "selected_candidate_id": winner["candidate_id"],
                                    "selected_algorithm": winner["algorithm"], "selected_feature_view": winner["feature_view"], **metric_row(values, prediction)})
                for record_id, pred in zip(np.asarray(target_ids)[target_mask], prediction):
                    prediction_rows.append({"outer_test_fold": outer, "endpoint": endpoint, "model_record_id": record_id,
                                            "selected_candidate_id": winner["candidate_id"], "predicted_transformed_value": float(pred)})
                progress.update(1)
        progress.close()
        pd.DataFrame(selection_rows).to_csv(out / "inner_candidate_selection.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out / "outer_R1_metric_summary.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(out / "outer_R1_predictions_internal.csv", index=False)
        (out / "README.md").write_text(
            "# MMPK N2b strict R1 baseline screen\n\n"
            "Formal internal-only strict nested CV. Candidate selection is confined to N2a inner folds; outer R1 labels are read only after selection. This is not an external benchmark or a public data release. R1+R2 and formulation sensitivities require separately frozen follow-up runs and may not change these selections.\n",
            encoding="utf-8")
        finish_stage(out, "mmpk_n2b_strict_r1_conditional_baseline_screen", inputs={
            str((args.protocol / "complete.json").resolve()): sha256(args.protocol / "complete.json"),
            str((args.n1b / "complete.json").resolve()): sha256(args.n1b / "complete.json")},
            formal_r1_screen=True, external_labels_accessed=False, strict_outer_folds=5, strict_inner_folds=4,
            candidate_cells=len(candidates), total_fits=expected_fits, device=args.device, cuda_device_name=cuda_name,
            outer_selection_forbidden=True, R2_used_for_selection=False, formulation_used_for_selection=False, partial=False)
    print(f"MMPK N2b strict R1 baseline screen: {args.output}")


if __name__ == "__main__":
    run_cli(main)
