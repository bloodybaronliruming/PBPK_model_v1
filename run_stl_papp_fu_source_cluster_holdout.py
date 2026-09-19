#!/usr/bin/env python3
"""Source-cluster and scaffold-isolated Papp/fu diagnostic with fixed leaders.

The five source folds are registered by the feasibility audit.  A source fold
holds out whole parent--document connected components; every scaffold of the
held-out parents is also removed from training.  The exact Stage-A ExtraTrees
leader is fitted with target scaling learned from the remaining train parents.
This diagnostic is train-only and cannot authorize model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import metric_row
from stl_benchmark_common import feature_view, make_estimator


TASKS = ("Papp__human__caco2_ab", "fu__human__plasma")
VIEW_SCREEN = {
    "rdkit2d": "stl_stageA_first_batch_rdkit2d_v1",
    "ecfp4": "stl_stageA_ecfp4_v1",
    "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1",
}
PAPP_BINS = (0.0, 100.0, 10000.0, np.inf)
PAPP_LABELS = ("≤100", "100–10,000", ">10,000")
POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def collapse_parents(frame: pd.DataFrame) -> pd.DataFrame:
    parent = frame.groupby(["molecule_id", "feature_index", "scaffold_group"], as_index=False).agg(
        parent_target=("target_value", "mean"),
        source_records=("row_id", "size"),
    )
    if parent.molecule_id.duplicated().any():
        raise ValueError("A parent maps to more than one feature/scaffold pair")
    return parent


def max_tanimoto_to_training(ecfp4: np.ndarray, eval_indices: np.ndarray, train_indices: np.ndarray,
                             batch_size: int = 32) -> np.ndarray:
    """Exact maximum ECFP4 Tanimoto; only the fold training structures are searched."""
    if len(eval_indices) == 0 or len(train_indices) == 0:
        raise ValueError("Similarity requires non-empty evaluation and training parents")
    packed = np.packbits(ecfp4.astype(np.uint8, copy=False), axis=1)
    eval_bits, train_bits = packed[eval_indices], packed[train_indices]
    eval_count = POPCOUNT[eval_bits].sum(axis=1).astype(np.int32)
    train_count = POPCOUNT[train_bits].sum(axis=1).astype(np.int32)
    result = np.empty(len(eval_indices), dtype=float)
    for start in range(0, len(eval_indices), batch_size):
        stop = min(start + batch_size, len(eval_indices))
        intersection = POPCOUNT[np.bitwise_and(eval_bits[start:stop, None, :], train_bits[None, :, :])].sum(axis=2)
        union = eval_count[start:stop, None] + train_count[None, :] - intersection
        result[start:stop] = np.max(intersection / np.maximum(union, 1), axis=1)
    return result


def source_fold_inputs(task_records: pd.DataFrame, parent_components: pd.DataFrame,
                       assignments: pd.DataFrame, source_fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parents = collapse_parents(task_records)
    map_table = parent_components.merge(assignments[["source_component_id", "source_fold"]],
                                        on="source_component_id", validate="many_to_one")
    parents = parents.merge(map_table[["molecule_id", "source_component_id", "source_fold"]],
                            on="molecule_id", validate="one_to_one")
    evaluation_parent = parents.loc[parents.source_fold.eq(source_fold)].copy()
    eval_ids = set(evaluation_parent.molecule_id)
    evaluation_rows = task_records.loc[task_records.molecule_id.isin(eval_ids)].copy()
    evaluation_scaffolds = set(evaluation_parent.scaffold_group.astype(str))
    training_parent = parents.loc[~parents.source_fold.eq(source_fold)].copy()
    source_train_before = training_parent.copy()
    training_parent = training_parent.loc[~training_parent.scaffold_group.astype(str).isin(evaluation_scaffolds)].copy()
    if evaluation_parent.empty or training_parent.empty:
        raise ValueError("A source fold has empty evaluation or training parents")
    if set(training_parent.molecule_id) & eval_ids:
        raise ValueError("Parent overlap in source-cluster holdout")
    if set(training_parent.scaffold_group.astype(str)) & evaluation_scaffolds:
        raise ValueError("Scaffold overlap in source-cluster holdout")
    train_docs = set(task_records.loc[task_records.molecule_id.isin(set(training_parent.molecule_id)), "doc_id"].astype(str))
    eval_docs = set(evaluation_rows.doc_id.astype(str))
    if train_docs & eval_docs:
        raise ValueError("Document overlap in source-cluster holdout")
    audit = pd.DataFrame([{
        "task_id": task_records.task_id.iloc[0], "endpoint": task_records.endpoint.iloc[0], "source_fold": source_fold,
        "evaluation_parents": len(evaluation_parent), "evaluation_records": len(evaluation_rows),
        "source_train_parents_before_scaffold_exclusion": len(source_train_before),
        "scaffold_collision_parents_removed_from_train": len(source_train_before) - len(training_parent),
        "train_parents_after_scaffold_exclusion": len(training_parent),
        "train_records_after_scaffold_exclusion": int(task_records.molecule_id.isin(set(training_parent.molecule_id)).sum()),
        "parent_overlap": 0, "document_overlap": 0, "scaffold_overlap": 0,
    }])
    return training_parent, evaluation_parent, audit


def metric_subsets(oof: pd.DataFrame, task: str, view: str, seed: int) -> list[dict]:
    result = []
    overall = metric_row(oof, oof.predicted_interface_target.to_numpy(), task, "extra_trees", view, "source_cluster_holdout")
    overall.update(seed=seed, subset="all_source_clusters")
    result.append(overall)
    if task == TASKS[0]:
        papp = oof.copy()
        papp["papp_raw_1e_minus_6_cm_s"] = np.power(10.0, papp.target_value.to_numpy(float))
        papp["papp_range"] = pd.cut(papp.papp_raw_1e_minus_6_cm_s, bins=PAPP_BINS, labels=PAPP_LABELS, include_lowest=True)
        for label, frame in papp.groupby("papp_range", observed=False):
            if frame.empty:
                continue
            row = metric_row(frame, frame.predicted_interface_target.to_numpy(), task, "extra_trees", view,
                             "source_cluster_holdout_papp_range")
            row.update(seed=seed, subset=f"papp_value_{label}")
            result.append(row)
    return result


def source_figure(metrics: pd.DataFrame) -> plt.Figure:
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    subset = metrics.loc[metrics.subset.eq("all_source_clusters")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.8))
    for axis, task, title, color in zip(axes, TASKS, ("Papp", "fu"), ("#4C78A8", "#C44E52"), strict=True):
        values = subset.loc[subset.task_id.eq(task), "primary_metric"].to_numpy(float)
        axis.scatter(np.arange(1, len(values) + 1), values, color=color, s=45)
        axis.axhline(values.mean(), color="#333333", linewidth=1, linestyle="--", label=f"mean = {values.mean():.3f}")
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Fixed estimator seed")
        axis.set_xticks(np.arange(1, len(values) + 1))
        axis.set_ylabel("Parent-level transformed RMSE" if task == TASKS[0] else "Parent-level physical MAE")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")
    fig.suptitle("Source-cluster and scaffold-isolated train-CV diagnostic", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "results/analysis/stl_papp_fu_source_cluster_feasibility_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_papp_fu_source_cluster_holdout_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260920, 20260921, 20260922])
    parser.add_argument("--source-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if sorted(set(args.source_folds)) != sorted(args.source_folds) or set(args.source_folds) - set(range(5)):
        raise ValueError("--source-folds must be unique values in 0..4")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.threads < 1:
        raise ValueError("Seeds must be unique and threads positive")
    screen_paths = {view: args.benchmarks / name for view, name in VIEW_SCREEN.items()}
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
                args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
                args.stageA_audit / "complete.json", args.stageA_audit / "endpoint_decisions.csv",
                args.feasibility / "complete.json", args.feasibility / "parent_source_components.csv",
                args.feasibility / "source_component_fold_assignment.csv"]
    required.extend(path / "complete.json" for path in screen_paths.values())
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    stage_meta = verify_stage(args.stageA_audit, "stl_stageA_three_view_audit")
    feasibility_meta = verify_stage(args.feasibility, "stl_papp_fu_source_cluster_feasibility")
    screen_meta = [verify_stage(path, "stl_benchmark_screen") for path in screen_paths.values()]
    if train_meta.get("fixed_validation_targets_published") or stage_meta.get("validation_labels_read"):
        raise ValueError("Source holdout requires train-only inputs")
    if any(meta.get("test_labels_read") for meta in [train_meta, stl_meta, stage_meta, feasibility_meta, *screen_meta]):
        raise ValueError("An upstream stage reports test-label access")
    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "feature_index"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records.feature_index = records.feature_index.astype(int)
    records = records.loc[records.task_id.isin(TASKS)].copy()
    if set(records.task_id.unique()) != set(TASKS):
        raise ValueError("Required Papp/fu tasks are absent")
    components = pd.read_csv(args.feasibility / "parent_source_components.csv", dtype=str, keep_default_na=False)
    assignment = pd.read_csv(args.feasibility / "source_component_fold_assignment.csv", dtype=str, keep_default_na=False)
    assignment.source_fold = pd.to_numeric(assignment.source_fold, errors="raise").astype(int)
    cache_file = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    cache = {key: cache_file[key] for key in cache_file.files}
    decisions = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv")
    registries = {view: pd.read_csv(path / "run_registry.csv") for view, path in screen_paths.items()}
    leaders = {}
    for task in TASKS:
        decision = decisions.loc[decisions.task_id.eq(task)]
        if len(decision) != 1:
            raise ValueError(f"Missing Stage-A decision for {task}")
        decision = decision.iloc[0]
        view = str(decision.stageA_leading_feature_view)
        row = registries[view].loc[(registries[view].task_id.eq(task)) &
                                   (registries[view].candidate_id.eq(decision.stageA_leading_candidate))]
        if len(row) != 5 or not row.algorithm.eq("extra_trees").all() or row.parameters.nunique() != 1:
            raise ValueError(f"Stage-A leader registry inconsistent for {task}")
        leaders[task] = {"view": view, "parameters": json.loads(row.iloc[0].parameters),
                         "candidate": str(decision.stageA_leading_candidate)}
    expected_fits = len(TASKS) * len(args.seeds) * len(args.source_folds)
    partial = len(args.seeds) != 3 or args.source_folds != [0, 1, 2, 3, 4]
    if args.check_only:
        print(f"Papp/fu source-cluster holdout ready: tasks={len(TASKS)} seeds={len(args.seeds)} folds={len(args.source_folds)} fits={expected_fits} partial={partial}")
        return
    with stage_output(args.output) as out:
        predictions, metrics, runs, isolation, reloads, similarities = [], [], [], [], [], []
        for task, task_records in records.groupby("task_id", sort=True):
            spec = leaders[task]
            matrix = feature_view(cache, spec["view"])
            ecfp4 = cache["ecfp4"]
            task_components = components.loc[components.task_id.eq(task), ["molecule_id", "source_component_id"]].copy()
            task_assignment = assignment.loc[assignment.task_id.eq(task), ["source_component_id", "source_fold"]].copy()
            if task_components.molecule_id.duplicated().any() or task_assignment.source_component_id.duplicated().any():
                raise ValueError("Feasibility mapping is not unique")
            per_fold = {}
            for source_fold in args.source_folds:
                train_parent, eval_parent, audit = source_fold_inputs(task_records, task_components, task_assignment, source_fold)
                eval_indices = eval_parent.feature_index.to_numpy(int)
                train_indices = train_parent.feature_index.to_numpy(int)
                similarity = max_tanimoto_to_training(ecfp4, eval_indices, train_indices)
                similarity_frame = eval_parent[["molecule_id", "source_component_id"]].copy()
                similarity_frame["task_id"], similarity_frame["source_fold"] = task, source_fold
                similarity_frame["max_train_ecfp4_tanimoto"] = similarity
                similarities.append(similarity_frame)
                per_fold[source_fold] = (train_parent, eval_parent, audit, similarity_frame)
            for seed in args.seeds:
                all_eval = []
                for source_fold in args.source_folds:
                    train_parent, eval_parent, audit, similarity_frame = per_fold[source_fold]
                    mean, std = float(train_parent.parent_target.mean()), float(train_parent.parent_target.std(ddof=0))
                    if not np.isfinite(std) or std <= 0:
                        raise ValueError("Invalid fold-local target scale")
                    estimator = make_estimator("extra_trees", int(seed + 100003 * source_fold), args.threads,
                                               small_budget=True, parameters=spec["parameters"])
                    x_train = matrix[train_parent.feature_index.to_numpy(int)]
                    x_eval = matrix[eval_parent.feature_index.to_numpy(int)]
                    estimator.fit(x_train, (train_parent.parent_target.to_numpy(float) - mean) / std)
                    prediction_z = np.asarray(estimator.predict(x_eval)).reshape(-1)
                    raw_prediction = prediction_z * std + mean
                    global_mean, global_std = float(task_records.train_mean.iloc[0]), float(task_records.train_std_population.iloc[0])
                    parent_prediction = pd.DataFrame({"molecule_id": eval_parent.molecule_id,
                                                      "predicted_interface_target": (raw_prediction - global_mean) / global_std})
                    evaluation_rows = task_records.loc[task_records.molecule_id.isin(set(eval_parent.molecule_id)), [
                        "row_id", "molecule_id", "task_id", "endpoint", "doc_id", "split", "target_value", "train_mean",
                        "train_std_population", "reporting_inverse"
                    ]].copy()
                    table = evaluation_rows.merge(parent_prediction, on="molecule_id", validate="many_to_one")
                    table = table.merge(similarity_frame[["molecule_id", "source_component_id", "max_train_ecfp4_tanimoto"]],
                                        on="molecule_id", validate="many_to_one")
                    table["interface_target"] = (table.target_value - table.train_mean) / table.train_std_population
                    table["source_fold"], table["seed"] = source_fold, seed
                    table["candidate_id"], table["algorithm"], table["feature_set"] = spec["candidate"], "extra_trees", spec["view"]
                    predictions.append(table); all_eval.append(table)
                    run = audit.iloc[0].to_dict()
                    run.update(seed=seed, candidate_id=spec["candidate"], algorithm="extra_trees", feature_set=spec["view"],
                               parameters=json.dumps(spec["parameters"], sort_keys=True), fold_target_mean=mean,
                               fold_target_std_population=std)
                    runs.append(run)
                    isolation.append({**audit.iloc[0].to_dict(), "seed": seed, "evaluation_targets_used_in_preprocessing": False})
                    if source_fold == args.source_folds[0]:
                        with tempfile.TemporaryDirectory(dir=out) as temp:
                            path = Path(temp) / "model.joblib"
                            joblib.dump(estimator, path, compress=3)
                            after = np.asarray(joblib.load(path).predict(x_eval)).reshape(-1)
                        difference = float(np.max(np.abs(prediction_z - after)))
                        if difference > 1e-8:
                            raise ValueError("Model reload mismatch")
                        reloads.append({"task_id": task, "seed": seed, "source_fold": source_fold,
                                        "max_abs_difference": difference})
                oof = pd.concat(all_eval, ignore_index=True)
                metrics.extend(metric_subsets(oof, task, spec["view"], seed))
        prediction_frame = pd.concat(predictions, ignore_index=True)
        keys = ["task_id", "seed", "row_id"]
        if prediction_frame.duplicated(keys).any() or prediction_frame.predicted_interface_target.isna().any():
            raise ValueError("Duplicate or incomplete source-holdout predictions")
        if not partial:
            expected = records.groupby("task_id").row_id.nunique()
            coverage = prediction_frame.groupby(["task_id", "seed"]).row_id.nunique()
            if any(count != expected[task] for (task, _), count in coverage.items()):
                raise ValueError("Full source-holdout does not cover train-only records")
        metric_frame = pd.DataFrame(metrics)
        similarity_frame = pd.concat(similarities, ignore_index=True).drop_duplicates(["task_id", "source_fold", "molecule_id"])
        prediction_frame.to_csv(out / "predictions.csv", index=False)
        metric_frame.to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(runs).to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(isolation).to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(reloads).to_csv(out / "reload_checks.csv", index=False)
        similarity_frame.to_csv(out / "evaluation_parent_max_train_ecfp4_tanimoto.csv", index=False)
        figure = source_figure(metric_frame)
        figure.savefig(out / "Figure_source_cluster_holdout_metrics.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        import matplotlib.pyplot as plt
        plt.close(figure)
        summary = {
            "tasks": len(TASKS), "seeds": args.seeds, "source_folds": args.source_folds, "fits": len(runs),
            "source_cluster_definition": "parent_document_bipartite_connected_component",
            "training_exclusion": "evaluation_source_components_plus_evaluation_scaffolds",
            "similarity_definition": "exact_maximum_ecfp4_tanimoto_to_retained_fold_training_parent",
            "validation_target_file_opened": False, "validation_rows_evaluated": False, "test_labels_read": False,
            "model_selection_authorized": False, "selection_scope": "source_generalization_diagnostic_only",
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Source-cluster holdout diagnostic\n\n"
            "This uses fixed Stage-A ExtraTrees leaders and parent--document connected source clusters. Evaluation source clusters and their scaffolds are excluded from each training fold. The output measures a deliberately stricter train-CV diagnostic and is not eligible for model selection or comparison with fixed validation/test.\n",
            encoding="utf-8",
        )
        finish_stage(out, "stl_papp_fu_source_cluster_holdout", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json"),
            "feasibility_complete_sha256": sha256(args.feasibility / "complete.json"),
        }, partial=partial, **summary)
    print(f"Papp/fu source-cluster holdout: {args.output}")


if __name__ == "__main__":
    run_cli(main)
