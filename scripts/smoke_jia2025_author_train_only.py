#!/usr/bin/env python3
"""Exercise every frozen Jia author-like pipeline using author-train rows only.

This is an engineering smoke test, not a benchmark: it saves neither targets,
predictions nor predictive metrics, and never selects a configuration.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from dmpk_toolkit import featurize
from pipeline_common import (ROOT, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check,
                             stable_id, verify_stage)


TASKS = {
    "Fu": ("Fu_final", "lgFu", "Fu_train_test"),
    "CL": ("CL_final(L/hour/kg)", "lgCL", "CL_train_test"),
    "VDss": ("VD_final(L/kg)", "lgVD", "VD_train_test"),
}
SEED = 20260918
SMOKE_ROWS_PER_ENDPOINT = 64
SMOKE_TREE_CAP = 8


def dependency_version(name: str) -> str:
    return importlib.metadata.version(name)


def rdkit_bits(molecules: list[Chem.Mol], kind: str) -> np.ndarray:
    if kind == "MACCS":
        nbits, factory = 167, lambda m: MACCSkeys.GenMACCSKeys(m)
    elif kind == "FCFP6":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=1024,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
        nbits, factory = 1024, generator.GetFingerprint
    else:
        raise ValueError(kind)
    rows = []
    for molecule in molecules:
        bits = np.zeros(nbits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(factory(molecule), bits)
        rows.append(bits)
    return np.stack(rows)


def mordred_2d(molecules: list[Chem.Mol]) -> np.ndarray:
    calculator = Calculator(descriptors, ignore_3D=True)
    raw = calculator.pandas(molecules, nproc=1, quiet=True)
    return raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def train_only_filter(x: np.ndarray, train_indices: np.ndarray, apply_indices: np.ndarray,
                      variance_filter: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit all unsupervised transforms on the fold's author-train rows only."""
    train = x[train_indices]
    finite = np.isfinite(train)
    keep = finite.all(axis=0)
    if variance_filter:
        if not keep.any():
            raise ValueError("No complete feature columns remain")
        variance = np.var(train[:, keep], axis=0)
        complete_columns = np.flatnonzero(keep)
        keep[complete_columns[variance < 0.2]] = False
        retained = np.flatnonzero(keep)
        if len(retained) > 1:
            correlation = np.abs(np.corrcoef(train[:, retained], rowvar=False))
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=1.0)
            chosen = []
            for candidate in range(len(retained)):
                if not chosen or np.all(correlation[candidate, chosen] <= 0.85):
                    chosen.append(candidate)
            keep[:] = False
            keep[retained[chosen]] = True
    if not keep.any():
        raise ValueError("No train-only feature columns remain after filtering")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fitted_train = imputer.fit_transform(train[:, keep])
    transformed = imputer.transform(x[apply_indices][:, keep])
    if not np.isfinite(fitted_train).all() or not np.isfinite(transformed).all():
        raise ValueError("Train-only imputation left nonfinite features")
    return fitted_train, transformed, int(keep.sum())


def make_model(family: str, params: dict, seed: int) -> Pipeline:
    if family == "paper_rf":
        estimator = RandomForestRegressor(n_estimators=min(int(params["rf_n_estimators"]), SMOKE_TREE_CAP),
                                          min_samples_leaf=int(params["rf_min_samples_leaf"]),
                                          max_depth=int(params["rf_max_depth"]), random_state=seed, n_jobs=1)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", estimator)])
    if family == "paper_svr":
        gamma = params["svr_gamma"]
        gamma = gamma if gamma == "auto" else float(gamma)
        estimator = SVR(kernel="rbf", C=float(params["svr_C"]), gamma=gamma)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("model", estimator)])
    if family == "paper_xgb":
        estimator = XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cpu", n_jobs=1,
                                 verbosity=0, random_state=seed, n_estimators=min(int(params["xgb_n_estimators"]), SMOKE_TREE_CAP),
                                 max_depth=int(params["xgb_max_depth"]), learning_rate=float(params["xgb_learning_rate"]))
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", estimator)])
    if family == "dummy_log_mean":
        return Pipeline([("model", DummyRegressor(strategy="mean"))])
    if family == "ridge_rdkit2d":
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()),
                         ("model", Ridge(alpha=10.0, solver="lsqr", tol=1e-6))])
    if family == "extra_trees_ecfp4_rdkit2d":
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("model", ExtraTreesRegressor(n_estimators=SMOKE_TREE_CAP, max_features=0.7, min_samples_leaf=3,
                                                        random_state=seed, n_jobs=1))])
    raise ValueError(family)


def main():
    parser = argparse.ArgumentParser(description="Jia author-train-only 10-fold feature/pipeline smoke.")
    parser.add_argument("--workbook", type=Path, default=ROOT / "基准研究参考/jm5c00340_si_001.xlsx")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/jia2025_author_like_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/jia2025_author_train_only_smoke_v2")
    args = parser.parse_args()
    startup_self_check([args.workbook, args.protocol / "complete.json"], output=args.output)
    configure_logging(ROOT, "smoke_jia2025_author_train_only")
    protocol = verify_stage(args.protocol, "jia2025_author_like_protocol_freeze")
    if protocol.get("author_test_scored") or not protocol.get("no_training"):
        raise ValueError("Unexpected protocol lifecycle state")
    RDLogger.DisableLog("rdApp.*")

    inputs = {
        str(args.workbook.resolve()): sha256(args.workbook),
        str((args.protocol / "complete.json").resolve()): sha256(args.protocol / "complete.json"),
        str((args.protocol / "paper_hyperparameter_cells.csv").resolve()): sha256(args.protocol / "paper_hyperparameter_cells.csv"),
        str((args.protocol / "predeclared_model_budget.csv").resolve()): sha256(args.protocol / "predeclared_model_budget.csv"),
    }
    with stage_output(args.output) as out:
        data = pd.read_excel(args.workbook, sheet_name="Fu_VDss_CL_modeling_set")
        cells = pd.read_csv(args.protocol / "paper_hyperparameter_cells.csv")
        budget = pd.read_csv(args.protocol / "predeclared_model_budget.csv")
        selections, validation = {}, []
        for endpoint, (raw_col, log_col, split_col) in TASKS.items():
            train = data.loc[data[split_col].astype(str).str.lower().eq("train"), ["SMILES_parent", raw_col, log_col]].copy()
            train = train.dropna().reset_index(drop=True)
            if (pd.to_numeric(train[raw_col], errors="coerce") <= 0).any():
                raise ValueError(f"{endpoint} has nonpositive author-train raw values")
            raw = pd.to_numeric(train[raw_col], errors="raise").to_numpy(dtype=float)
            target = pd.to_numeric(train[log_col], errors="raise").to_numpy(dtype=float)
            max_diff = float(np.max(np.abs(np.log10(raw) - target)))
            if max_diff > 1e-5:
                raise ValueError(f"{endpoint} raw/log target inconsistency: {max_diff:.3g}")
            train["parent_hash"] = train.SMILES_parent.map(stable_id)
            selected = train.sort_values("parent_hash", kind="stable").head(SMOKE_ROWS_PER_ENDPOINT).reset_index(drop=True)
            if len(selected) != SMOKE_ROWS_PER_ENDPOINT:
                raise ValueError(f"{endpoint} has too few train rows for 10-fold smoke")
            selections[endpoint] = selected
            validation.append({"endpoint": endpoint, "author_train_rows": int(len(train)), "smoke_rows": int(len(selected)),
                               "raw_log_max_abs_difference": max_diff, "author_test_rows_selected_or_used": False,
                               "label_values_exported": False})

        all_smiles = list(dict.fromkeys(s for frame in selections.values() for s in frame.SMILES_parent.tolist()))
        molecules = [Chem.MolFromSmiles(s) for s in all_smiles]
        if any(m is None for m in molecules):
            raise ValueError("Invalid SMILES_parent in selected author-train rows")
        rdkit2d, rdkit_names = featurize(all_smiles, progress=False, feature_set="rdkit2d")
        ecfp4_rdkit2d, _ = featurize(all_smiles, progress=False, feature_set="ecfp4_rdkit2d")
        maccs = rdkit_bits(molecules, "MACCS")
        fcfp6 = rdkit_bits(molecules, "FCFP6")
        mordred = mordred_2d(molecules)
        feature_cache = {"rdkit": rdkit2d, "MACCS": maccs, "FCFP6": fcfp6, "mordred": mordred,
                         "merged": np.concatenate([maccs, fcfp6, rdkit2d, mordred], axis=1),
                         "rdkit2d": rdkit2d, "ecfp4_rdkit2d": ecfp4_rdkit2d}
        index = {smiles: i for i, smiles in enumerate(all_smiles)}
        feature_rows = []
        smoke_rows = []
        for endpoint, selected in selections.items():
            indices = np.asarray([index[s] for s in selected.SMILES_parent], dtype=int)
            y = pd.to_numeric(selected[TASKS[endpoint][1]], errors="raise").to_numpy(dtype=float)
            kfold = KFold(n_splits=10, shuffle=True, random_state=SEED)
            fold_splits = list(kfold.split(indices))
            preprocessing_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]] = {}
            configs = []
            for _, cell in cells.loc[cells.endpoint.eq(endpoint)].iterrows():
                for family in ("paper_rf", "paper_svr", "paper_xgb"):
                    configs.append((family, str(cell.descriptor_set), cell.to_dict(), f"{family}_{cell.descriptor_set}"))
            for _, ref in budget.loc[(budget.endpoint.eq(endpoint)) & (budget.family.eq("fixed_project_reference"))].iterrows():
                configs.append((str(ref.model_id), str(ref.representation), {}, str(ref.model_id)))
            for family, representation, params, model_id in configs:
                # DummyRegressor still follows sklearn's X/y interface.  Its
                # declared representation is ``none``, so provide a fixed
                # single zero column rather than silently borrowing a chemical
                # feature view.
                x = np.zeros((len(indices), 1), dtype=np.float32) if representation == "none" else feature_cache[representation][indices]
                if x.shape[0] != len(y):
                    raise ValueError("Feature/target alignment failure")
                folds_ok = 0
                min_features = None
                for fold, (fit, held) in enumerate(fold_splits):
                    cache_key = (representation, fold)
                    if cache_key not in preprocessing_cache:
                        preprocessing_cache[cache_key] = train_only_filter(
                            x, fit, held, variance_filter=representation in {"mordred", "merged"})
                    filtered_fit, filtered_held, retained_features = preprocessing_cache[cache_key]
                    model = make_model(family, params, SEED + fold)
                    model.fit(filtered_fit, y[fit])
                    pred = np.asarray(model.predict(filtered_held), dtype=float)
                    if pred.shape != (len(held),) or not np.isfinite(pred).all():
                        raise ValueError(f"Nonfinite smoke prediction: {endpoint}/{model_id}/fold{fold}")
                    folds_ok += 1
                    features = retained_features
                    min_features = features if min_features is None else min(min_features, features)
                smoke_rows.append({"endpoint": endpoint, "model_id": model_id, "representation": representation,
                                   "folds_completed": folds_ok, "finite_predictions_only_checked": True,
                                   "metrics_computed": False, "smoke_tree_cap": SMOKE_TREE_CAP, "feature_columns_input": int(min_features)})
            for representation, matrix in feature_cache.items():
                feature_rows.append({"endpoint": endpoint, "representation": representation, "smoke_rows": int(len(indices)),
                                     "feature_columns_raw": int(matrix.shape[1]), "nonfinite_cells_raw": int((~np.isfinite(matrix[indices])).sum()),
                                     "feature_values_exported": False})
        pd.DataFrame(validation).to_csv(out / "author_train_target_contract_audit.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(out / "feature_availability_audit.csv", index=False)
        pd.DataFrame(smoke_rows).to_csv(out / "pipeline_smoke_manifest.csv", index=False)
        memberships = []
        for endpoint, selected in selections.items():
            memberships.extend({"endpoint": endpoint, "parent_hash": value, "row_role": "author_train_smoke", "target_exported": False}
                               for value in selected.parent_hash)
        pd.DataFrame(memberships).sort_values(["endpoint", "parent_hash"]).to_csv(out / "smoke_membership_hashed.csv", index=False)
        versions = {name: dependency_version(name) for name in ("rdkit", "mordredcommunity", "scikit-learn", "xgboost", "pandas", "numpy")}
        dump_json(out / "environment_manifest.json", {"versions": versions, "seed": SEED, "folds": 10,
                                                        "smoke_rows_per_endpoint": SMOKE_ROWS_PER_ENDPOINT,
                                                        "smoke_tree_cap": SMOKE_TREE_CAP,
                                                        "author_test_rows_loaded_by_workbook_reader": True,
                                                        "author_test_rows_selected_or_used": False,
                                                        "author_test_label_values_exported": False,
                                                        "no_author_test_predictions": True,
                                                        "no_predictive_metrics": True})
        report = """# Jia 2025 author-train-only smoke v2

All selected rows are author-train members. The program checked raw-to-log target consistency, constructed every frozen descriptor family, and fitted every frozen model family across ten train-only folds. Mordred and merged representations use each fold's author-train rows for missing-column removal, variance filtering and correlation pruning. It stored no targets, predictions or predictive metrics. Trees were deliberately capped at eight estimators only for engineering smoke; this cap is not part of the later frozen scientific configurations.

The full R3 run may proceed only if this output has a complete marker and its manifest shows every endpoint/model family completed ten folds. R3 must use the exact frozen parameters from the protocol, not this smoke cap, and may score the author test once descriptively.
"""
        (out / "smoke_report.md").write_text(report, encoding="utf-8")
        finish_stage(out, "jia2025_author_train_only_feature_pipeline_smoke", inputs=inputs,
                     no_training_release=True, author_test_rows_loaded_by_workbook_reader=True,
                     author_test_rows_selected_or_used=False, author_test_predictions=False,
                     predictive_metrics_computed=False, smoke_rows_per_endpoint=SMOKE_ROWS_PER_ENDPOINT,
                     folds=10, smoke_tree_cap=SMOKE_TREE_CAP, partial=False)
    print(f"Jia 2025 author-train-only feature/pipeline smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
