#!/usr/bin/env python3
"""Run frozen Jia 2025 author-train CV and final fitting; never score author test."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

from pipeline_common import (ROOT, configure_logging, dump_json, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, stable_id, verify_stage)
from smoke_jia2025_author_train_only import (SEED, TASKS, dependency_version, featurize,
                                              rdkit_bits, mordred_2d, Chem, RDLogger)


TREE_THREADS = 4


def prepare(train_x, apply_x, descriptor_filter, scale):
    """Fit all feature decisions on train_x only and return a reloadable state."""
    keep = np.isfinite(train_x).all(axis=0)
    if descriptor_filter:
        kept = np.flatnonzero(keep)
        if not len(kept): raise ValueError("No finite Mordred/merged columns")
        kept = kept[np.var(train_x[:, kept], axis=0) >= .2]
        if len(kept) > 1:
            corr = np.nan_to_num(np.abs(np.corrcoef(train_x[:, kept], rowvar=False)), nan=0., posinf=1., neginf=1.)
            selected = []
            for j in range(len(kept)):
                if not selected or np.all(corr[j, selected] <= .85): selected.append(j)
            kept = kept[selected]
        keep[:] = False; keep[kept] = True
    if not keep.any(): raise ValueError("No train-only feature columns")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fitted = imputer.fit_transform(train_x[:, keep])
    applied = imputer.transform(apply_x[:, keep])
    scaler = StandardScaler().fit(fitted) if scale else None
    if scaler is not None: fitted, applied = scaler.transform(fitted), scaler.transform(applied)
    if not np.isfinite(fitted).all() or not np.isfinite(applied).all(): raise ValueError("Nonfinite post-preprocessing feature")
    return fitted, applied, {"keep": keep, "imputer": imputer, "scaler": scaler}


def estimator(family, p, seed):
    if family == "paper_rf": return RandomForestRegressor(n_estimators=int(p["rf_n_estimators"]), min_samples_leaf=int(p["rf_min_samples_leaf"]), max_depth=int(p["rf_max_depth"]), random_state=seed, n_jobs=TREE_THREADS)
    if family == "paper_svr": return SVR(kernel="rbf", C=float(p["svr_C"]), gamma=p["svr_gamma"] if p["svr_gamma"] == "auto" else float(p["svr_gamma"]))
    if family == "paper_xgb": return XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cpu", n_jobs=TREE_THREADS, verbosity=0, random_state=seed, n_estimators=int(p["xgb_n_estimators"]), max_depth=int(p["xgb_max_depth"]), learning_rate=float(p["xgb_learning_rate"]))
    if family == "dummy_log_mean": return DummyRegressor(strategy="mean")
    if family == "ridge_rdkit2d": return Ridge(alpha=10., solver="lsqr", tol=1e-6)
    if family == "extra_trees_ecfp4_rdkit2d": return ExtraTreesRegressor(n_estimators=300, max_features=.7, min_samples_leaf=3, random_state=seed, n_jobs=TREE_THREADS)
    raise ValueError(family)


def feature_matrices(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols): raise ValueError("Invalid author-train SMILES_parent")
    rdkit, names = featurize(smiles, progress=True, feature_set="rdkit2d")
    combined, _ = featurize(smiles, progress=True, feature_set="ecfp4_rdkit2d")
    maccs, fcfp6 = rdkit_bits(mols, "MACCS"), rdkit_bits(mols, "FCFP6")
    mordred = mordred_2d(mols)
    return {"rdkit": rdkit, "MACCS": maccs, "FCFP6": fcfp6, "mordred": mordred,
            "merged": np.concatenate([maccs, fcfp6, rdkit, mordred], axis=1),
            "rdkit2d": rdkit, "ecfp4_rdkit2d": combined}, names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workbook", type=Path, default=ROOT / "基准研究参考/jm5c00340_si_001.xlsx")
    p.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/jia2025_author_like_protocol_v1")
    p.add_argument("--smoke", type=Path, default=ROOT / "results/analysis/jia2025_author_train_only_smoke_v2")
    p.add_argument("--output", type=Path, default=ROOT / "models/jia2025_author_like_r3a_v1")
    p.add_argument("--max-author-train-rows", type=int, help="Engineering smoke only; deterministic limit per endpoint and never a scientific result")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    startup_self_check([args.workbook, args.protocol / "complete.json", args.smoke / "complete.json"], output=None if args.check_only else args.output)
    protocol = verify_stage(args.protocol, "jia2025_author_like_protocol_freeze")
    smoke = verify_stage(args.smoke, "jia2025_author_train_only_feature_pipeline_smoke")
    if protocol.get("author_test_scored") or smoke.get("author_test_rows_selected_or_used") or smoke.get("predictive_metrics_computed"):
        raise ValueError("Input lifecycle does not permit R3a")
    if args.check_only:
        print("Jia R3a input valid: frozen protocol, valid v2 train-only smoke, author test remains unscored")
        return
    configure_logging(ROOT, "train_jia2025_author_like_r3a")
    RDLogger.DisableLog("rdApp.*")
    inputs = {str(x.resolve()): sha256(x) for x in [args.workbook, args.protocol / "complete.json", args.smoke / "complete.json", args.protocol / "paper_hyperparameter_cells.csv", args.protocol / "predeclared_model_budget.csv"]}
    with stage_output(args.output) as out:
        raw = pd.read_excel(args.workbook, sheet_name="Fu_VDss_CL_modeling_set")
        cells = pd.read_csv(args.protocol / "paper_hyperparameter_cells.csv")
        budget = pd.read_csv(args.protocol / "predeclared_model_budget.csv")
        task_frames, validation = {}, []
        for endpoint, (raw_col, log_col, split_col) in TASKS.items():
            x = raw.loc[raw[split_col].astype(str).str.lower().eq("train"), ["SMILES_parent", raw_col, log_col]].dropna().copy()
            values, target = pd.to_numeric(x[raw_col], errors="raise").to_numpy(float), pd.to_numeric(x[log_col], errors="raise").to_numpy(float)
            diff = float(np.max(np.abs(np.log10(values)-target)))
            if (values <= 0).any() or diff > 1e-5: raise ValueError(f"{endpoint} raw/log contract fails")
            x["parent_hash"] = x.SMILES_parent.map(stable_id); task_frames[endpoint] = x.reset_index(drop=True)
            if args.max_author_train_rows:
                task_frames[endpoint] = task_frames[endpoint].sort_values("parent_hash", kind="stable").head(args.max_author_train_rows).reset_index(drop=True)
            validation.append({"endpoint":endpoint,"train_rows":len(x),"unique_parent_hashes":x.parent_hash.nunique(),"raw_log_max_abs_difference":diff,"author_test_selected_or_used":False})
        smiles = list(dict.fromkeys(s for x in task_frames.values() for s in x.SMILES_parent.tolist()))
        print(f"FEATURE_PROGRESS 0/5 unique_author_train_structures={len(smiles)}")
        started = time.time(); features, rdkit_names = feature_matrices(smiles); print(f"FEATURE_PROGRESS 5/5 elapsed={time.time()-started:.1f}s")
        lookup = {s:i for i,s in enumerate(smiles)}
        configs = []
        for endpoint in TASKS:
            for _, row in cells.loc[cells.endpoint.eq(endpoint)].iterrows():
                for family in ("paper_rf","paper_svr","paper_xgb"): configs.append((endpoint,f"{family}_{row.descriptor_set}",family,str(row.descriptor_set),row.to_dict()))
            for _, row in budget.loc[(budget.endpoint.eq(endpoint)) & (budget.family.eq("fixed_project_reference"))].iterrows(): configs.append((endpoint,str(row.model_id),str(row.model_id),str(row.representation),{}))
        total_fits = len(configs)*11; progress = tqdm(total=total_fits, desc="R3A_TOTAL_FITS", unit="fit")
        metric_rows=[]; model_rows=[]; members=[]
        for completed, (endpoint, model_id, family, representation, params) in enumerate(configs, 1):
            frame=task_frames[endpoint]; ids=np.asarray([lookup[s] for s in frame.SMILES_parent],int); y=pd.to_numeric(frame[TASKS[endpoint][1]],errors="raise").to_numpy(float)
            X=np.zeros((len(ids),1),np.float32) if representation=="none" else features[representation][ids]
            oof=np.empty(len(y)); folds=KFold(n_splits=10,shuffle=True,random_state=SEED)
            for fold,(fit,held) in enumerate(folds.split(X)):
                a,b,_=prepare(X[fit],X[held],representation in {"mordred","merged"},family in {"paper_svr","ridge_rdkit2d"})
                model=estimator(family,params,SEED+fold); model.fit(a,y[fit]); pred=np.asarray(model.predict(b),float)
                if not np.isfinite(pred).all(): raise ValueError(f"Nonfinite CV prediction {endpoint}/{model_id}/fold{fold}")
                oof[held]=pred; progress.update(1)
            _,_,state=prepare(X,X,representation in {"mordred","merged"},family in {"paper_svr","ridge_rdkit2d"})
            final_x=state["imputer"].transform(X[:,state["keep"]]);
            if state["scaler"] is not None: final_x=state["scaler"].transform(final_x)
            final=estimator(family,params,SEED); final.fit(final_x,y); progress.update(1)
            if not np.isfinite(np.asarray(final.predict(final_x),float)).all(): raise ValueError(f"Nonfinite full fit {endpoint}/{model_id}")
            physical_error=np.abs(oof-y); metric_rows.append({"endpoint":endpoint,"model_id":model_id,"representation":representation,"cv_folds":10,"n_train_records":len(y),"n_train_parents":frame.parent_hash.nunique(),"R2_log":float(r2_score(y,oof)),"MAE_log":float(mean_absolute_error(y,oof)),"RMSE_log":float(np.sqrt(mean_squared_error(y,oof))),"GMFE":float(10**np.mean(physical_error)),"within_2fold":float(np.mean(physical_error<=np.log10(2))),"author_test_scored":False})
            artifact=f"models/{endpoint}__{model_id}.joblib"; path=out/artifact; path.parent.mkdir(exist_ok=True)
            joblib.dump({"endpoint":endpoint,"model_id":model_id,"representation":representation,"feature_state":state,"estimator":final,"parameters":params,"target":"log10"},path,compress=3)
            model_rows.append({"endpoint":endpoint,"model_id":model_id,"representation":representation,"artifact":artifact,"sha256":sha256(path),"cv_folds":10,"frozen_after_train_cv":True})
            print(f"MODEL_PROGRESS {completed}/{len(configs)} endpoint={endpoint} model={model_id}")
        progress.close()
        for endpoint,frame in task_frames.items(): members += [{"endpoint":endpoint,"parent_hash":z,"role":"author_train","label_exported":False} for z in frame.parent_hash]
        pd.DataFrame(validation).to_csv(out/"train_target_contract_audit.csv",index=False); pd.DataFrame(metric_rows).to_csv(out/"train_cv_metrics.csv",index=False); pd.DataFrame(model_rows).to_csv(out/"frozen_model_manifest.csv",index=False); pd.DataFrame(members).drop_duplicates().to_csv(out/"author_train_membership_hashed.csv",index=False)
        dump_json(out/"environment_manifest.json",{"versions":{x:dependency_version(x) for x in ["rdkit","mordredcommunity","scikit-learn","xgboost","pandas","numpy"]},"seed":SEED,"folds":10,"tree_threads":TREE_THREADS,"author_test_rows_loaded_by_workbook_reader":True,"author_test_rows_selected_or_used":False,"author_test_predictions":False,"author_test_metrics":False})
        (out/"README.md").write_text("# Jia 2025 R3a train-only frozen models\n\nAll 54 configurations were fit only on author-train rows. CV metrics are training-development diagnostics; author-test has not been predicted or scored. Models are immutable inputs to the one-time R3b descriptive scorer.\n",encoding="utf-8")
        finish_stage(out,"jia2025_author_like_r3a_traincv_and_model_freeze",inputs=inputs,protocol_grade="B_author_like",models_frozen=len(model_rows),cv_folds=10,author_test_rows_loaded_by_workbook_reader=True,author_test_rows_selected_or_used=False,author_test_predictions=False,author_test_metrics=False,partial=bool(args.max_author_train_rows),max_author_train_rows=args.max_author_train_rows)
    print(f"Jia 2025 R3a train-CV/model freeze: {args.output}")

if __name__=="__main__": run_cli(main)
