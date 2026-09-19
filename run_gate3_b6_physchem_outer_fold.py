#!/usr/bin/env python3
"""Run one independently recoverable Gate-3 B6 endpoint-by-outer-fold cell."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_gate3_b6_physchem_smoke import (
    AUX_TASKS,
    META_ALPHAS,
    bounded_scaffolds,
    bool_series,
    computed_control,
    digest_values,
    dynamic_scaffold_folds,
    feature_view,
    fit_tree,
    overlap_row,
    predict_tree,
    stable_seed,
    stagea_view,
)


TASKS = [
    "CL__human__systemic_iv",
    "CLint__human__microsome",
    "Papp__human__caco2_ab",
    "Thalf__human__terminal_iv",
    "VDss__human__steady_state_iv",
    "fu__human__plasma",
]
PRODUCER_FOLDS = 5
STAGEA_ENSEMBLE_SEEDS = [20260920, 20260921, 20260922]
TOTAL_CELL_FITS = 73


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_bar(completed: int, total: int, width: int = 20) -> str:
    filled = width if total <= 0 else min(width, int(width * completed / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class FitProgress:
    """Log-friendly nested progress bars; one durable line per completed fit."""

    def __init__(self, task_id: str, outer_fold: int, total: int = TOTAL_CELL_FITS):
        self.task_id = task_id
        self.outer_fold = outer_fold
        self.total = total
        self.completed = 0
        self.started = time.monotonic()
        self.stage_name = "initialization"
        self.stage_total = 0
        self.stage_completed = 0

    def start_stage(self, name: str, total: int) -> None:
        if self.stage_completed != self.stage_total:
            raise RuntimeError(
                f"Previous progress stage incomplete: {self.stage_name} "
                f"{self.stage_completed}/{self.stage_total}"
            )
        self.stage_name = name
        self.stage_total = total
        self.stage_completed = 0
        self.render("start")

    def advance(self, detail: str = "") -> None:
        self.stage_completed += 1
        self.completed += 1
        if self.stage_completed > self.stage_total or self.completed > self.total:
            raise RuntimeError("Progress counter exceeded the frozen fit budget")
        self.render(detail)

    def render(self, detail: str) -> None:
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.completed * (self.total - self.completed) if self.completed else 0.0
        stage = f"{progress_bar(self.stage_completed, self.stage_total)} {self.stage_completed:02d}/{self.stage_total:02d}"
        overall = f"{progress_bar(self.completed, self.total)} {self.completed:02d}/{self.total:02d}"
        suffix = f" detail={detail}" if detail else ""
        print(
            f"CELL_PROGRESS task={self.task_id} fold={self.outer_fold} "
            f"stage={self.stage_name} {stage} total={overall} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}{suffix}",
            flush=True,
        )

    def finish(self) -> None:
        if self.completed != self.total or self.stage_completed != self.stage_total:
            raise RuntimeError(
                f"Progress budget incomplete: total={self.completed}/{self.total}, "
                f"stage={self.stage_completed}/{self.stage_total}"
            )
        elapsed = time.monotonic() - self.started
        print(
            f"CELL_PROGRESS_COMPLETE task={self.task_id} fold={self.outer_fold} "
            f"fits={self.completed} elapsed={format_duration(elapsed)}",
            flush=True,
        )


def collapse_task(records: pd.DataFrame, task_id: str) -> pd.DataFrame:
    identity = records.groupby("parent_id").agg(
        features=("feature_index", "nunique"), scaffolds=("scaffold_group", "nunique"),
        structures=("canonical_parent_smiles", "nunique"), folds=("inner_fold_id", "nunique"),
    )
    if identity.max().max() != 1:
        raise ValueError("Downstream parent identity is inconsistent")
    work = records.copy()
    if task_id == "fu__human__plasma":
        work["physical_record_target"] = expit(work.target_value.to_numpy(float))
        parent = work.groupby("parent_id", as_index=False, sort=True).agg(
            scaffold_group=("scaffold_group", "first"),
            canonical_parent_smiles=("canonical_parent_smiles", "first"),
            feature_index=("feature_index", "first"), inner_fold_id=("inner_fold_id", "first"),
            fusion_target=("physical_record_target", "mean"), source_records=("row_id", "size"),
        )
        parent["stageA_model_target"] = logit(np.clip(parent.fusion_target.to_numpy(float), 1e-12, 1 - 1e-12))
    else:
        parent = work.groupby("parent_id", as_index=False, sort=True).agg(
            scaffold_group=("scaffold_group", "first"),
            canonical_parent_smiles=("canonical_parent_smiles", "first"),
            feature_index=("feature_index", "first"), inner_fold_id=("inner_fold_id", "first"),
            fusion_target=("target_value", "mean"), source_records=("row_id", "size"),
        )
        parent["stageA_model_target"] = parent.fusion_target
    return parent


def to_fusion_space(task_id: str, prediction: np.ndarray) -> np.ndarray:
    values = expit(prediction) if task_id == "fu__human__plasma" else np.asarray(prediction, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Stage-A emitted non-finite fusion-space predictions")
    return values


def constrain_route(task_id: str, prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if task_id == "fu__human__plasma":
        values = np.clip(values, 1e-12, 1 - 1e-12)
    if not np.isfinite(values).all():
        raise ValueError("A route emitted non-finite predictions")
    return values


def primary_error(task_id: str, observed: np.ndarray, predicted: np.ndarray) -> float:
    if task_id == "fu__human__plasma":
        return float(np.mean(np.abs(observed - predicted)))
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def select_meta_alpha(task_id: str, features: np.ndarray, observed: np.ndarray,
                      stage_prediction: np.ndarray, folds: np.ndarray,
                      progress: FitProgress | None = None) -> tuple[float, pd.DataFrame]:
    rows = []
    residual = observed - stage_prediction
    for alpha in META_ALPHAS:
        errors = []
        for fold in sorted(set(folds.tolist())):
            train = folds != fold; valid = folds == fold
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(features[train], residual[train])
            if progress is not None:
                progress.advance(f"alpha={alpha}|inner_fold={fold}")
            prediction = constrain_route(task_id, stage_prediction[valid] + model.predict(features[valid]))
            errors.append(primary_error(task_id, observed[valid], prediction))
        rows.append({
            "alpha": alpha, "mean_inner_primary_error": float(np.mean(errors)),
            "inner_folds": len(set(folds.tolist())),
            "primary_metric": "physical_MAE" if task_id == "fu__human__plasma" else "transformed_RMSE",
        })
    table = pd.DataFrame(rows).sort_values(["mean_inner_primary_error", "alpha"], kind="stable")
    return float(table.iloc[0].alpha), table


def add_model_ancestry(rows: list[dict], out: Path, component: str, task: str,
                       fold: int | str, train: pd.DataFrame, recipient: pd.DataFrame,
                       model_name: str) -> None:
    row = overlap_row(component, task, fold, train, recipient, model_name)
    row["model_sha256"] = sha256(out / model_name)
    rows.append(row)


def add_reload(rows: list[dict], component: str, task: str,
               expected: np.ndarray, actual: np.ndarray) -> None:
    difference = float(np.max(np.abs(np.asarray(expected) - np.asarray(actual))))
    rows.append({"component": component, "task_id": task, "maximum_absolute_difference": difference})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
    parser.add_argument("--authorization", type=Path, default=ROOT / "data/public_development/gate3_b6_formal_run_v4")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_protocol_v2")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_cohort_v1")
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--p1-freeze", type=Path, default=ROOT / "data/public_development/stl_p1_decision_freeze_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--engineering-preflight", action="store_true")
    parser.add_argument("--downstream-parents-per-fold", type=int)
    parser.add_argument("--auxiliary-parents-per-task", type=int)
    parser.add_argument("--stagea-trees", type=int)
    parser.add_argument("--producer-trees", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    slug = args.task.replace("__", "_").lower()
    if args.output is None:
        args.output = ROOT / f"results/benchmarks/gate3_b6_physchem_formal_cells_v3/{slug}_fold{args.outer_fold}"
    overrides = [args.downstream_parents_per_fold, args.auxiliary_parents_per_task,
                 args.stagea_trees, args.producer_trees]
    if any(value is not None for value in overrides) and not args.engineering_preflight:
        raise ValueError("Budget overrides require --engineering-preflight and cannot publish a formal cell")

    required = [
        args.authorization / "complete.json", args.authorization / "formal_authorization.json",
        args.authorization / "formal_cell_registry.csv",
        args.protocol / "complete.json", args.protocol / "auxiliary_producer_registry.csv",
        args.protocol / "endpoint_fusion_registry.csv",
        args.cohort / "complete.json", args.cohort / "parent_targets.csv",
        args.cohort / "parent_features_float32.npz", args.cohort / "feature_registry.json",
        args.cohort / "outer_fold_auxiliary_parent_membership.csv",
        args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
        args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
        args.stl_protocol / "feature_registry.json",
        args.p1_freeze / "complete.json", args.p1_freeze / "p1_decision_registry.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    authorization_meta = verify_stage(args.authorization, "gate3_b6_formal_run_freeze")
    protocol_meta = verify_stage(args.protocol, "gate3_b6_physchem_protocol")
    cohort_meta = verify_stage(args.cohort, "gate3_b6_physchem_cohort")
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    p1_meta = verify_stage(args.p1_freeze, "stl_p1_decision_freeze")
    metas = [authorization_meta, protocol_meta, cohort_meta, train_meta, stl_meta, p1_meta]
    if not authorization_meta.get("formal_train_cv_authorized"):
        raise ValueError("Formal train-CV authorization is absent")
    if any(meta.get("test_labels_read", False) or meta.get("validation_target_file_opened", False) for meta in metas):
        raise ValueError("Outer-fold worker accepts only fixed-validation/test-closed inputs")
    authorization = json.loads((args.authorization / "formal_authorization.json").read_text())
    if not authorization.get("formal_train_cv_authorized"):
        raise ValueError("Formal authorization JSON does not permit train-CV")
    if authorization.get("stageA_ensemble_seeds") != STAGEA_ENSEMBLE_SEEDS:
        raise ValueError("Formal authorization has different corrected Stage-A ensemble seeds")
    if authorization.get("outer_fold_runner_sha256") != sha256(Path(__file__)):
        raise ValueError("Outer-fold runner differs from the formally authorized code hash")
    registry = pd.read_csv(args.authorization / "formal_cell_registry.csv")
    registered = registry.loc[registry.task_id.eq(args.task) & registry.outer_fold.eq(args.outer_fold)]
    if len(registered) != 1:
        raise ValueError("Requested endpoint/fold is absent from the frozen 30-cell registry")

    p1 = pd.read_csv(args.p1_freeze / "p1_decision_registry.csv")
    frozen = p1.loc[p1.task_id.eq(args.task)]
    if len(frozen) != 1 or frozen.iloc[0].stageA_algorithm != "extra_trees":
        raise ValueError("Frozen ExtraTrees Stage-A specification is absent")
    stage_spec = frozen.iloc[0]
    stage_params = json.loads(stage_spec.stageA_parameters)
    formal_stage_trees = int(stage_params["n_estimators"])
    stage_trees = args.stagea_trees or formal_stage_trees
    producer_registry = pd.read_csv(args.protocol / "auxiliary_producer_registry.csv")
    formal_producer_trees = int(producer_registry.n_estimators.unique().item())
    producer_trees = args.producer_trees or formal_producer_trees
    full_configuration = bool(
        not args.engineering_preflight and not any(value is not None for value in overrides)
        and stage_trees == formal_stage_trees and producer_trees == formal_producer_trees
    )
    if args.check_only:
        print(
            "Gate 3 B6 outer-fold ready: "
            f"task={args.task} fold={args.outer_fold} full_configuration={full_configuration} "
            f"stageA_trees={stage_trees} producer_trees={producer_trees} fits=73"
        )
        return

    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv")
    records = records.loc[records.task_id.eq(args.task)].copy()
    for column in ["target_value", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records.feature_index = records.feature_index.astype(int)
    records.inner_fold_id = records.inner_fold_id.astype(int)
    downstream = collapse_task(records, args.task)
    if args.downstream_parents_per_fold is not None:
        downstream = pd.concat([
            bounded_scaffolds(frame, args.downstream_parents_per_fold, f"preflight|{args.task}|{fold}")
            for fold, frame in downstream.groupby("inner_fold_id", sort=True)
        ], ignore_index=True)
    outer_train = downstream.loc[downstream.inner_fold_id.ne(args.outer_fold)].copy().sort_values("parent_id").reset_index(drop=True)
    outer_eval = downstream.loc[downstream.inner_fold_id.eq(args.outer_fold)].copy().sort_values("parent_id").reset_index(drop=True)
    if set(outer_train.parent_id) & set(outer_eval.parent_id) or set(outer_train.scaffold_group) & set(outer_eval.scaffold_group):
        raise ValueError("Downstream outer partition leakage")
    if set(outer_train.inner_fold_id) != (set(range(5)) - {args.outer_fold}):
        raise ValueError("Outer-training partition lacks a frozen inner scaffold fold")

    stage_npz = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    stage_cache = {key: stage_npz[key] for key in stage_npz.files}
    stage_matrix = stagea_view(stage_cache, str(stage_spec.stageA_feature_set))
    stage_registry = json.loads((args.stl_protocol / "feature_registry.json").read_text())
    stage_names = stage_registry["rdkit2d"]["descriptor_names"]
    cohort_registry = json.loads((args.cohort / "feature_registry.json").read_text())
    safe_names = cohort_registry["leakage_safe_rdkit2d"]["descriptor_names"]
    safe_indices = [stage_names.index(name) for name in safe_names]
    downstream_aux_cache = {
        "ecfp4": stage_cache["ecfp4"],
        "leakage_safe_rdkit2d": stage_cache["rdkit2d"][:, safe_indices],
    }
    if downstream_aux_cache["leakage_safe_rdkit2d"].shape[1] != 195:
        raise ValueError("Leakage-safe downstream descriptor projection is not 195-dimensional")
    cohort_npz = np.load(args.cohort / "parent_features_float32.npz")
    cohort_cache = {key: cohort_npz[key] for key in cohort_npz.files}
    parent_targets = pd.read_csv(args.cohort / "parent_targets.csv")
    membership = pd.read_csv(args.cohort / "outer_fold_auxiliary_parent_membership.csv")
    membership.retained_for_outer_fold = bool_series(membership.retained_for_outer_fold)
    membership = membership.loc[
        membership.downstream_task.eq(args.task) & membership.outer_fold.eq(args.outer_fold)
        & membership.retained_for_outer_fold
    ].copy()

    inputs = {str(path.resolve()): sha256(path) for path in required}
    with stage_output(args.output) as out:
        (out / "models").mkdir()
        progress = FitProgress(args.task, args.outer_fold)
        progress.start_stage("StageA_three_seed_ensemble", 15)
        ancestry: list[dict] = []
        reloads: list[dict] = []
        imputation: list[dict] = []
        aux_cv: list[dict] = []
        aux_selected: list[dict] = []
        meta_cv: list[pd.DataFrame] = []

        stage_oof_model = np.full(len(outer_train), np.nan)
        for recipient_fold in sorted(outer_train.inner_fold_id.unique()):
            train = outer_train.loc[outer_train.inner_fold_id.ne(recipient_fold)]
            recipient_mask = outer_train.inner_fold_id.eq(recipient_fold).to_numpy()
            recipient = outer_train.loc[recipient_mask]
            seed_predictions = []
            for ensemble_seed in STAGEA_ENSEMBLE_SEEDS:
                model, audit = fit_tree(
                    stage_matrix[train.feature_index], train.stageA_model_target.to_numpy(float),
                    stable_seed("gate3-stageA-crossfit", args.task, args.outer_fold,
                                recipient_fold, stage_spec.stageA_candidate_id, ensemble_seed),
                    stage_trees, float(stage_params["max_features"]),
                    int(stage_params["min_samples_leaf"]), args.threads,
                )
                progress.advance(f"crossfit_fold={recipient_fold}|seed={ensemble_seed}")
                name = f"models/stageA_crossfit_fold{recipient_fold}_seed{ensemble_seed}.joblib"
                joblib.dump(model, out / name, compress=3)
                prediction = predict_tree(model, stage_matrix[recipient.feature_index])
                seed_predictions.append(prediction)
                add_model_ancestry(ancestry, out, "stageA_crossfit", args.task,
                                   f"{recipient_fold}|seed={ensemble_seed}", train, recipient, name)
                reloaded = predict_tree(joblib.load(out / name), stage_matrix[recipient.feature_index])
                add_reload(reloads, "stageA_crossfit",
                           f"{args.task}|{recipient_fold}|{ensemble_seed}", prediction, reloaded)
                imputation.append({"component": "stageA_crossfit", "task_id": args.task,
                                   "fold": int(recipient_fold), "ensemble_seed": ensemble_seed, **audit})
            stage_oof_model[recipient_mask] = np.mean(np.vstack(seed_predictions), axis=0)
        if not np.isfinite(stage_oof_model).all():
            raise ValueError("Stage-A crossfit did not cover every outer-training parent")
        stage_oof = to_fusion_space(args.task, stage_oof_model)
        stage_eval_seed_predictions = []
        for ensemble_seed in STAGEA_ENSEMBLE_SEEDS:
            stage_final, audit = fit_tree(
                stage_matrix[outer_train.feature_index], outer_train.stageA_model_target.to_numpy(float),
                stable_seed("p1-stagea-control", args.task, args.outer_fold,
                            stage_spec.stageA_candidate_id, ensemble_seed),
                stage_trees, float(stage_params["max_features"]),
                int(stage_params["min_samples_leaf"]), args.threads,
            )
            progress.advance(f"outer_eval|seed={ensemble_seed}")
            stage_final_name = f"models/stageA_outer_train_seed{ensemble_seed}.joblib"
            joblib.dump(stage_final, out / stage_final_name, compress=3)
            stage_eval_seed = predict_tree(stage_final, stage_matrix[outer_eval.feature_index])
            stage_eval_seed_predictions.append(stage_eval_seed)
            add_model_ancestry(
                ancestry, out, "stageA_outer_eval", args.task,
                f"{args.outer_fold}|seed={ensemble_seed}", outer_train, outer_eval, stage_final_name,
            )
            stage_reload_seed = predict_tree(
                joblib.load(out / stage_final_name), stage_matrix[outer_eval.feature_index]
            )
            add_reload(reloads, "stageA_outer_eval",
                       f"{args.task}|{ensemble_seed}", stage_eval_seed, stage_reload_seed)
            imputation.append({"component": "stageA_outer_eval", "task_id": args.task,
                               "fold": args.outer_fold, "ensemble_seed": ensemble_seed, **audit})
        stage_eval_model = np.mean(np.vstack(stage_eval_seed_predictions), axis=0)
        stage_eval = to_fusion_space(args.task, stage_eval_model)

        aux_train_predictions: dict[str, np.ndarray] = {}
        aux_eval_predictions: dict[str, np.ndarray] = {}
        for aux_task in AUX_TASKS:
            progress.start_stage(f"{aux_task}_producer", 16)
            eligible = membership.loc[membership.auxiliary_task.eq(aux_task), ["parent_id", "scaffold_group", "feature_index"]]
            targets = parent_targets.loc[parent_targets.task_id.eq(aux_task), ["parent_id", "target_value"]]
            aux = eligible.merge(targets, on="parent_id", validate="one_to_one")
            if args.auxiliary_parents_per_task is not None:
                aux = bounded_scaffolds(aux, args.auxiliary_parents_per_task,
                                        f"preflight-aux|{args.task}|{args.outer_fold}|{aux_task}")
            aux = aux.sort_values("parent_id").reset_index(drop=True)
            assignment = dynamic_scaffold_folds(
                aux, outer_train, f"formal-producer|{args.task}|{args.outer_fold}|{aux_task}"
            )
            aux["producer_fold"] = aux.scaffold_group.astype(str).map(assignment).astype(int)
            recipient_folds = outer_train.scaffold_group.astype(str).map(assignment)
            if recipient_folds.isna().any():
                raise ValueError("A downstream outer-training scaffold lacks a producer fold")
            recipient_folds = recipient_folds.astype(int).to_numpy()
            candidates = producer_registry.loc[producer_registry.auxiliary_task.eq(aux_task)]
            scores = []
            for spec in candidates.itertuples(index=False):
                matrix = feature_view(cohort_cache, str(spec.feature_set))
                errors = []
                for fold in range(PRODUCER_FOLDS):
                    train = aux.loc[aux.producer_fold.ne(fold)]
                    valid = aux.loc[aux.producer_fold.eq(fold)]
                    if train.empty or valid.empty:
                        raise ValueError(f"Empty auxiliary candidate fold: {aux_task} fold={fold}")
                    if (set(train.parent_id) & set(valid.parent_id) or
                            set(train.scaffold_group.astype(str)) & set(valid.scaffold_group.astype(str))):
                        raise ValueError(f"Auxiliary candidate-CV leakage: {aux_task} fold={fold}")
                    model, audit = fit_tree(
                        matrix[train.feature_index], train.target_value.to_numpy(float),
                        stable_seed("gate3-formal-aux-select", args.task, args.outer_fold,
                                    aux_task, spec.producer_id, fold),
                        producer_trees, float(spec.max_features), int(spec.min_samples_leaf), args.threads,
                    )
                    progress.advance(f"candidate={spec.feature_set}|fold={fold}")
                    prediction = predict_tree(model, matrix[valid.feature_index])
                    rmse = float(np.sqrt(np.mean(np.square(valid.target_value.to_numpy(float) - prediction))))
                    errors.append(rmse)
                    aux_cv.append({
                        "downstream_task": args.task, "outer_fold": args.outer_fold,
                        "auxiliary_task": aux_task, "producer_id": spec.producer_id,
                        "feature_set": spec.feature_set, "producer_fold": fold,
                        "train_parents": len(train), "validation_parents": len(valid),
                        "validation_rmse": rmse, "parent_overlap": 0, "scaffold_overlap": 0,
                    })
                    imputation.append({
                        "component": "auxiliary_candidate_cv", "task_id": aux_task,
                        "fold": fold, "producer_id": spec.producer_id,
                        "feature_set": spec.feature_set, **audit,
                    })
                scores.append((float(np.mean(errors)), spec))
            scores.sort(key=lambda item: (item[0], str(item[1].producer_id)))
            best_score = scores[0][0]
            within = [item for item in scores if item[0] <= best_score * 1.01]
            within.sort(key=lambda item: (
                item[1].feature_set != "leakage_safe_rdkit2d", item[0], item[1].producer_id
            ))
            selected_score, selected_spec = within[0]
            aux_selected.append({
                "downstream_task": args.task, "outer_fold": args.outer_fold,
                "auxiliary_task": aux_task, "selected_producer_id": selected_spec.producer_id,
                "selected_feature_set": selected_spec.feature_set,
                "mean_internal_rmse": selected_score, "best_observed_internal_rmse": best_score,
                "selection_rule": "within_1pct_choose_leakage_safe_rdkit2d_else_min_RMSE",
                "downstream_labels_used_for_selection": 0, "source_test_labels_used_for_selection": 0,
            })
            aux_matrix = feature_view(cohort_cache, str(selected_spec.feature_set))
            downstream_matrix = feature_view(downstream_aux_cache, str(selected_spec.feature_set))
            train_prediction = np.full(len(outer_train), np.nan)
            for fold in range(PRODUCER_FOLDS):
                producer_train = aux.loc[aux.producer_fold.ne(fold)]
                recipient_mask = recipient_folds == fold
                recipient = outer_train.loc[recipient_mask]
                if recipient.empty:
                    continue
                model, audit = fit_tree(
                    aux_matrix[producer_train.feature_index], producer_train.target_value.to_numpy(float),
                    stable_seed("gate3-formal-aux-crossfit", args.task, args.outer_fold, aux_task, fold),
                    producer_trees, float(selected_spec.max_features),
                    int(selected_spec.min_samples_leaf), args.threads,
                )
                progress.advance(f"selected_crossfit|fold={fold}")
                name = f"models/{aux_task}_crossfit_fold{fold}.joblib"
                joblib.dump(model, out / name, compress=3)
                prediction = predict_tree(model, downstream_matrix[recipient.feature_index])
                train_prediction[recipient_mask] = prediction
                add_model_ancestry(ancestry, out, "auxiliary_crossfit", aux_task, fold,
                                   producer_train, recipient, name)
                reloaded = predict_tree(joblib.load(out / name), downstream_matrix[recipient.feature_index])
                add_reload(reloads, "auxiliary_crossfit", f"{aux_task}|{fold}", prediction, reloaded)
                imputation.append({"component": "auxiliary_crossfit", "task_id": aux_task,
                                   "fold": fold, **audit})
            if not np.isfinite(train_prediction).all():
                raise ValueError(f"Auxiliary crossfit is incomplete: {aux_task}")
            aux_train_predictions[aux_task] = train_prediction
            final_model, audit = fit_tree(
                aux_matrix[aux.feature_index], aux.target_value.to_numpy(float),
                stable_seed("gate3-formal-aux-final", args.task, args.outer_fold, aux_task),
                producer_trees, float(selected_spec.max_features),
                int(selected_spec.min_samples_leaf), args.threads,
            )
            progress.advance("outer_eval_final")
            name = f"models/{aux_task}_outer_train.joblib"
            joblib.dump(final_model, out / name, compress=3)
            evaluation_prediction = predict_tree(final_model, downstream_matrix[outer_eval.feature_index])
            aux_eval_predictions[aux_task] = evaluation_prediction
            add_model_ancestry(ancestry, out, "auxiliary_outer_eval", aux_task, args.outer_fold,
                               aux, outer_eval, name)
            reloaded = predict_tree(joblib.load(out / name), downstream_matrix[outer_eval.feature_index])
            add_reload(reloads, "auxiliary_outer_eval", aux_task, evaluation_prediction, reloaded)
            imputation.append({"component": "auxiliary_outer_eval", "task_id": aux_task,
                               "fold": args.outer_fold, **audit})

        c1_train = computed_control(outer_train.canonical_parent_smiles.tolist())
        c1_eval = computed_control(outer_eval.canonical_parent_smiles.tolist())
        b6_train = np.column_stack([
            aux_train_predictions["experimental_logP"], aux_train_predictions["experimental_logD_pH7_4"],
            np.ones(len(outer_train)), np.ones(len(outer_train)),
        ])
        b6_eval = np.column_stack([
            aux_eval_predictions["experimental_logP"], aux_eval_predictions["experimental_logD_pH7_4"],
            np.ones(len(outer_eval)), np.ones(len(outer_eval)),
        ])
        observed_train = outer_train.fusion_target.to_numpy(float)
        residual = observed_train - stage_oof
        route_train = {"C0_corrected_StageA": constrain_route(args.task, stage_oof)}
        route_eval = {"C0_corrected_StageA": constrain_route(args.task, stage_eval)}
        route_alpha = {"C0_corrected_StageA": np.nan}
        for route, x_train, x_eval in [
            ("C1_structure_only_control", c1_train, c1_eval),
            ("B6_physchem_candidate", b6_train, b6_eval),
        ]:
            progress.start_stage(f"{route}_meta", 13)
            alpha, cv = select_meta_alpha(
                args.task, x_train, observed_train, stage_oof,
                outer_train.inner_fold_id.to_numpy(int), progress=progress,
            )
            cv.insert(0, "route", route); meta_cv.append(cv)
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(x_train, residual)
            progress.advance(f"final_alpha={alpha}")
            name = f"models/{route}_meta.joblib"
            joblib.dump(model, out / name, compress=3)
            route_train[route] = constrain_route(args.task, stage_oof + model.predict(x_train))
            route_eval[route] = constrain_route(args.task, stage_eval + model.predict(x_eval))
            route_alpha[route] = alpha
            reloaded = constrain_route(args.task, stage_eval + joblib.load(out / name).predict(x_eval))
            add_reload(reloads, "meta_model_fit_and_evaluation", route, route_eval[route], reloaded)
            add_model_ancestry(ancestry, out, "meta_model_fit_and_evaluation", route,
                               args.outer_fold, outer_train, outer_eval, name)

        # Evaluation values enter scoring only after all three route predictions exist.
        observed_eval = outer_eval.fusion_target.to_numpy(float)
        metrics = []
        metric_name = "physical_MAE" if args.task == "fu__human__plasma" else "transformed_RMSE"
        for route, prediction in route_eval.items():
            metrics.append({
                "task_id": args.task, "endpoint": stage_spec.endpoint, "outer_fold": args.outer_fold,
                "route": route, "primary_metric_name": metric_name,
                "primary_metric": primary_error(args.task, observed_eval, prediction),
                "evaluation_parents": len(outer_eval), "selected_meta_alpha": route_alpha[route],
            })
        ancestry_table = pd.DataFrame(ancestry)
        reload_table = pd.DataFrame(reloads)
        imputation_table = pd.DataFrame(imputation)
        if len(ancestry_table) != 29 or ancestry_table[["parent_overlap", "scaffold_overlap"]].to_numpy().max() != 0:
            raise ValueError("Formal cell ancestry is incomplete or leaky")
        expected_imputation_components = {
            "stageA_crossfit": 12,
            "stageA_outer_eval": 3,
            "auxiliary_candidate_cv": 20,
            "auxiliary_crossfit": 10,
            "auxiliary_outer_eval": 2,
        }
        if imputation_table.groupby("component").size().to_dict() != expected_imputation_components:
            raise ValueError("Fold-local imputation audit does not cover all 47 tree-model fits")
        if (imputation_table.all_missing_columns.ne(0).any()
                or not imputation_table.imputer_fit_scope.eq("current_model_training_parents_only").all()):
            raise ValueError("Fold-local imputation contract failed")
        if reload_table.maximum_absolute_difference.max() > 1e-12:
            raise ValueError("Formal cell reload tolerance failed")
        for row in ancestry_table.itertuples(index=False):
            if sha256(out / row.model_file) != row.model_sha256:
                raise ValueError(f"Saved model hash changed: {row.model_file}")
        progress.finish()

        train_predictions = outer_train[["parent_id", "scaffold_group", "inner_fold_id", "fusion_target"]].copy()
        train_predictions = train_predictions.rename(columns={"fusion_target": "observed_outer_train_target"})
        eval_predictions = outer_eval[["parent_id", "scaffold_group", "inner_fold_id", "fusion_target"]].copy()
        eval_predictions = eval_predictions.rename(columns={"fusion_target": "observed_outer_evaluation_target"})
        for route in route_train:
            train_predictions[route] = route_train[route]
            eval_predictions[route] = route_eval[route]
        for aux_task in AUX_TASKS:
            train_predictions[f"nested_{aux_task}"] = aux_train_predictions[aux_task]
            eval_predictions[f"nested_{aux_task}"] = aux_eval_predictions[aux_task]

        pd.DataFrame(aux_cv).to_csv(out / "auxiliary_internal_cv.csv", index=False)
        pd.DataFrame(aux_selected).to_csv(out / "selected_auxiliary_producers.csv", index=False)
        pd.concat(meta_cv, ignore_index=True).to_csv(out / "meta_alpha_internal_cv.csv", index=False)
        ancestry_table.to_csv(out / "nested_ancestry_audit.csv", index=False)
        reload_table.to_csv(out / "model_reload_audit.csv", index=False)
        imputation_table.to_csv(out / "fold_local_imputation_audit.csv", index=False)
        train_predictions.to_csv(out / "outer_train_crossfit_predictions.csv", index=False)
        eval_predictions.to_csv(out / "outer_evaluation_predictions.csv", index=False)
        pd.DataFrame(metrics).to_csv(out / "outer_fold_metrics.csv", index=False)
        lifecycle = pd.DataFrame([
            ("load_train_only_shared_container", True, False, "outer train and fold labels share one historical train-only CSV"),
            ("fit_stageA_and_auxiliary_models", True, False, "outer-evaluation targets not used"),
            ("fit_C1_and_B6_meta_models", True, False, "outer-evaluation targets not used"),
            ("all_three_routes_predicted", True, False, "predictions finite before scoring"),
            ("score_outer_fold", True, True, "outer-evaluation targets used only now"),
        ], columns=["event", "completed", "outer_evaluation_target_values_used", "note"])
        lifecycle.to_csv(out / "label_lifecycle_audit.csv", index=False)
        configuration = {
            "task_id": args.task, "outer_fold": args.outer_fold,
            "full_configuration": full_configuration, "engineering_preflight": args.engineering_preflight,
            "stageA_trees": stage_trees, "formal_stageA_trees": formal_stage_trees,
            "producer_trees": producer_trees, "formal_producer_trees": formal_producer_trees,
            "stageA_ensemble_seeds": STAGEA_ENSEMBLE_SEEDS,
            "downstream_parent_cap_per_fold": args.downstream_parents_per_fold,
            "auxiliary_parent_cap_per_task": args.auxiliary_parents_per_task,
            "outer_evaluation_target_container_shared_with_outer_train": True,
            "outer_evaluation_target_values_used_before_all_routes_predicted": 0,
            "fixed_validation_status": "closed", "test_status": "closed",
        }
        (out / "configuration.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n")
        finish_stage(
            out, "gate3_b6_physchem_outer_fold", inputs=inputs,
            partial=not full_configuration, full_configuration=full_configuration,
            engineering_preflight=args.engineering_preflight, task_id=args.task, outer_fold=args.outer_fold,
            outer_train_parents=len(outer_train), outer_evaluation_parents=len(outer_eval),
            ancestry_rows=len(ancestry_table), maximum_parent_overlap=0, maximum_scaffold_overlap=0,
            imputation_audit_rows=len(imputation_table), candidate_cv_imputation_audit_rows=20,
            maximum_all_missing_training_columns=int(imputation_table.all_missing_columns.max()),
            maximum_reload_difference=float(reload_table.maximum_absolute_difference.max()),
            route_predictions_complete=True, outer_evaluation_metrics_computed=True,
            outer_evaluation_target_values_used_before_all_routes_predicted=0,
            outer_evaluation_target_container_shared_with_outer_train=True,
            architecture_selection_authorized=False, fixed_validation_authorized=False,
            source_test_labels_read=False, validation_target_file_opened=False, test_labels_read=False,
        )
    print(f"Gate 3 B6 outer fold: {args.output}")


if __name__ == "__main__":
    run_cli(main)
