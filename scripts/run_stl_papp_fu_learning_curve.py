#!/usr/bin/env python3
"""Train-only scaffold-group learning curves for the Stage-A Papp and fu leaders.

For every fixed outer fold, whole scaffold groups are sampled only from that
fold's outer-training parents at pre-registered fractions and seeds.  The
exact Stage-A ExtraTrees leader is then fit with target scaling learned only
from the selected parents.  Outer-fold evaluation, validation and test labels
remain unavailable.  Source seen/unseen is descriptive sensitivity only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_stl_benchmark_screen import limit_records, metric_row
from stl_benchmark_common import feature_view, make_estimator


TASKS = ("Papp__human__caco2_ab", "fu__human__plasma")
VIEW_SCREEN = {
    "rdkit2d": "stl_stageA_first_batch_rdkit2d_v1",
    "ecfp4": "stl_stageA_ecfp4_v1",
    "ecfp4_rdkit2d": "stl_stageA_ecfp4_rdkit2d_v1",
}


def collapsed_parents(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse records before sampling so repeats cannot inflate train size."""
    parent = frame.groupby(["molecule_id", "feature_index", "scaffold_group"], as_index=False).agg(
        parent_target=("target_value", "mean"),
        parent_document_ids=("doc_id", lambda x: "|".join(sorted(set(x.astype(str))))),
        source_records=("row_id", "size"),
    )
    if parent.molecule_id.duplicated().any():
        raise ValueError("A parent has inconsistent feature/scaffold assignment")
    return parent


def choose_scaffold_groups(parents: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Sample complete scaffold groups; select nearest attainable parent count."""
    if not 0 < fraction <= 1:
        raise ValueError("Learning-curve fractions must be in (0, 1]")
    groups = parents.groupby("scaffold_group", sort=True).molecule_id.size()
    if fraction == 1.0:
        return parents.copy(), groups.index.astype(str).tolist()
    target = max(1, int(round(len(parents) * fraction)))
    labels = groups.index.astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    order = labels[rng.permutation(len(labels))]
    selected, count = [], 0
    for label in order:
        proposed = count + int(groups.loc[label])
        # Keep the first group even if it exceeds target.  Subsequently choose
        # the closer of including or stopping, so fractions remain transparent.
        if selected and abs(proposed - target) > abs(count - target):
            break
        selected.append(label)
        count = proposed
        if count >= target:
            break
    if not selected:
        raise ValueError("Scaffold sampler selected no groups")
    result = parents.loc[parents.scaffold_group.astype(str).isin(selected)].copy()
    if result.empty or result.molecule_id.nunique() != len(result):
        raise ValueError("Invalid scaffold-group sample")
    return result, selected


def source_membership(evaluation: pd.DataFrame, train_documents: set[str]) -> pd.DataFrame:
    rows = []
    for molecule, group in evaluation.groupby("molecule_id", sort=True):
        docs = set(group.doc_id.astype(str))
        rows.append({
            "molecule_id": molecule,
            "source_document_count": len(docs),
            "source_all_seen_in_selected_train": docs <= train_documents,
            "source_any_unseen_in_selected_train": bool(docs - train_documents),
        })
    return pd.DataFrame(rows)


def learning_curve_figure(metrics: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    selected = metrics.loc[metrics.source_subset.eq("all_sources")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.8), sharex=True)
    for axis, task, label, color in zip(
        axes, TASKS, ("Papp", "fu"), ("#4C78A8", "#C44E52"), strict=True
    ):
        table = selected.loc[selected.task_id.eq(task)].groupby("fraction", as_index=False).agg(
            mean_metric=("primary_metric", "mean"), std_metric=("primary_metric", "std"),
            mean_train_parents=("selected_train_parents", "mean"),
        ).sort_values("fraction")
        yerr = table.std_metric.fillna(0.0)
        axis.errorbar(table.fraction * 100, table.mean_metric, yerr=yerr, marker="o", markersize=5,
                      linewidth=1.6, capsize=3, color=color)
        for row in table.itertuples(index=False):
            axis.annotate(f"n≈{row.mean_train_parents:.0f}", (row.fraction * 100, row.mean_metric),
                          xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5)
        metric = "Parent-level transformed RMSE" if task.startswith("Papp") else "Parent-level physical MAE"
        axis.set_title(label, fontweight="bold")
        axis.set_ylabel(metric)
        axis.set_xlabel("Outer-training scaffold-group sample (%)")
        axis.set_xticks([25, 50, 75, 100])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Train-only scaffold-group learning curves of Stage-A ExtraTrees leaders", fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/stl_papp_fu_learning_curve_v1")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260917, 20260918, 20260919])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-parents-per-task-fold", type=int, help="Engineering smoke only")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.fractions or any(not 0 < value <= 1 for value in args.fractions):
        raise ValueError("--fractions must be non-empty values in (0, 1]")
    if sorted(set(args.fractions)) != sorted(args.fractions) or len(set(args.fractions)) != len(args.fractions):
        raise ValueError("--fractions must be unique and ascending")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.threads < 1:
        raise ValueError("Seeds must be unique and threads positive")
    screen_paths = {view: args.benchmarks / name for view, name in VIEW_SCREEN.items()}
    required = [args.train_protocol / "complete.json", args.train_protocol / "benchmark_train_records.csv",
                args.stl_protocol / "complete.json", args.stl_protocol / "canonical_parent_features_float32.npz",
                args.stageA_audit / "complete.json", args.stageA_audit / "endpoint_decisions.csv"]
    required.extend(path / "complete.json" for path in screen_paths.values())
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.train_protocol, "stl_train_only_protocol")
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    audit_meta = verify_stage(args.stageA_audit, "stl_stageA_three_view_audit")
    screens = {view: verify_stage(path, "stl_benchmark_screen") for view, path in screen_paths.items()}
    if train_meta.get("fixed_validation_targets_published") or audit_meta.get("validation_labels_read"):
        raise ValueError("Learning curves require train-only inputs with fixed validation closed")
    if any(meta.get("test_labels_read") for meta in [train_meta, stl_meta, audit_meta, *screens.values()]):
        raise ValueError("An upstream stage reports test-label access")
    records = pd.read_csv(args.train_protocol / "benchmark_train_records.csv", dtype=str, keep_default_na=False)
    for column in ["target_value", "train_mean", "train_std_population", "feature_index", "inner_fold_id"]:
        records[column] = pd.to_numeric(records[column], errors="raise")
    records.feature_index = records.feature_index.astype(int)
    records.inner_fold_id = records.inner_fold_id.astype(int)
    records = records.loc[records.task_id.isin(TASKS)].copy()
    if set(records.task_id.unique()) != set(TASKS):
        raise ValueError("Train-only protocol does not contain both Papp and fu")
    records = limit_records(records, args.max_parents_per_task_fold)
    cache_file = np.load(args.stl_protocol / "canonical_parent_features_float32.npz")
    cache = {key: cache_file[key] for key in cache_file.files}
    decisions = pd.read_csv(args.stageA_audit / "endpoint_decisions.csv")
    registries = {view: pd.read_csv(path / "run_registry.csv") for view, path in screen_paths.items()}
    leaders = {}
    for task in TASKS:
        decision = decisions.loc[decisions.task_id.eq(task)]
        if len(decision) != 1:
            raise ValueError(f"Missing unique Stage-A decision for {task}")
        decision = decision.iloc[0]
        view = str(decision.stageA_leading_feature_view)
        row = registries[view].loc[(registries[view].task_id.eq(task)) &
                                   (registries[view].candidate_id.eq(decision.stageA_leading_candidate))]
        if len(row) != 5 or not row.algorithm.eq("extra_trees").all() or row.parameters.nunique() != 1:
            raise ValueError(f"Expected five parameter-consistent ExtraTrees Stage-A leader rows for {task}")
        leaders[task] = {"view": view, "parameters": json.loads(row.iloc[0].parameters),
                         "candidate_id": str(decision.stageA_leading_candidate), "endpoint": str(decision.endpoint)}
    expected_fits = len(TASKS) * len(args.fractions) * len(args.seeds) * 5
    if args.check_only:
        print(f"Papp/fu learning curve ready: tasks={len(TASKS)} fractions={args.fractions} seeds={len(args.seeds)} fits={expected_fits} limited_smoke={bool(args.max_parents_per_task_fold)}")
        return
    limited = bool(args.max_parents_per_task_fold)
    with stage_output(args.output) as out:
        predictions, runs, isolation, reloads, samples, metrics = [], [], [], [], [], []
        for task, task_records in records.groupby("task_id", sort=True):
            spec = leaders[task]
            matrix = feature_view(cache, spec["view"])
            for fraction in args.fractions:
                for seed in args.seeds:
                    all_eval = []
                    candidate = f"stageA_leader__fraction_{fraction:g}__seed_{seed}"
                    for fold in range(5):
                        train_rows = task_records.loc[task_records.inner_fold_id.ne(fold)].copy()
                        eval_rows = task_records.loc[task_records.inner_fold_id.eq(fold)].copy()
                        train_parents, eval_parents = collapsed_parents(train_rows), collapsed_parents(eval_rows)
                        parent_overlap = len(set(train_parents.molecule_id) & set(eval_parents.molecule_id))
                        scaffold_overlap = len(set(train_rows.scaffold_group) & set(eval_rows.scaffold_group))
                        if parent_overlap or scaffold_overlap:
                            raise ValueError(f"Outer-fold leakage: {task} fold={fold}")
                        selected, groups = choose_scaffold_groups(train_parents, fraction, seed + 1009 * fold)
                        selected_ids = set(selected.molecule_id)
                        if selected_ids & set(eval_parents.molecule_id):
                            raise ValueError("Selected training parents overlap evaluation")
                        selected_docs = set(train_rows.loc[train_rows.molecule_id.isin(selected_ids), "doc_id"].astype(str))
                        mean, std = float(selected.parent_target.mean()), float(selected.parent_target.std(ddof=0))
                        if not np.isfinite(std) or std <= 0:
                            raise ValueError(f"Invalid selected target scale: {task} fraction={fraction} seed={seed} fold={fold}")
                        estimator = make_estimator("extra_trees", int(seed + 100003 * fold), args.threads, small_budget=True,
                                                   parameters=spec["parameters"])
                        x_train = matrix[selected.feature_index.to_numpy(int)]
                        x_eval = matrix[eval_parents.feature_index.to_numpy(int)]
                        estimator.fit(x_train, (selected.parent_target.to_numpy(float) - mean) / std)
                        z = np.asarray(estimator.predict(x_eval)).reshape(-1)
                        raw_prediction = z * std + mean
                        global_mean = float(task_records.train_mean.iloc[0])
                        global_std = float(task_records.train_std_population.iloc[0])
                        parent_prediction = pd.DataFrame({"molecule_id": eval_parents.molecule_id,
                                                          "predicted_interface_target": (raw_prediction - global_mean) / global_std})
                        table = eval_rows[["row_id", "molecule_id", "task_id", "endpoint", "doc_id", "split", "inner_fold_id",
                                           "target_value", "train_mean", "train_std_population", "reporting_inverse"]].merge(
                            parent_prediction, on="molecule_id", how="left", validate="many_to_one")
                        table["interface_target"] = (table.target_value - table.train_mean) / table.train_std_population
                        table["candidate_id"], table["fraction"], table["seed"] = candidate, fraction, seed
                        table["feature_set"], table["algorithm"], table["evaluation_partition"] = spec["view"], "extra_trees", f"fold_{fold}"
                        membership = source_membership(table, selected_docs)
                        table = table.merge(membership, on="molecule_id", how="left", validate="many_to_one")
                        predictions.append(table); all_eval.append(table)
                        samples.append({"task_id": task, "endpoint": spec["endpoint"], "fraction": fraction, "seed": seed,
                                        "outer_fold": fold, "available_outer_train_parents": len(train_parents),
                                        "selected_train_parents": len(selected), "selected_scaffold_groups": len(groups),
                                        "achieved_fraction": len(selected) / len(train_parents),
                                        "evaluation_parents": len(eval_parents), "selected_document_count": len(selected_docs)})
                        runs.append({"task_id": task, "candidate_id": candidate, "fraction": fraction, "seed": seed, "outer_fold": fold,
                                     "parameters": json.dumps(spec["parameters"], sort_keys=True), "feature_set": spec["view"],
                                     "selected_train_parents": len(selected), "evaluation_parents": len(eval_parents),
                                     "fold_target_mean": mean, "fold_target_std_population": std})
                        isolation.append({"task_id": task, "fraction": fraction, "seed": seed, "outer_fold": fold,
                                          "parent_overlap": parent_overlap, "scaffold_overlap": scaffold_overlap,
                                          "evaluation_targets_used_in_preprocessing": False})
                        if fold == 0:
                            with tempfile.TemporaryDirectory(dir=out) as temp:
                                model_path = Path(temp) / "model.joblib"
                                joblib.dump(estimator, model_path, compress=3)
                                repeat = np.asarray(joblib.load(model_path).predict(x_eval)).reshape(-1)
                            difference = float(np.max(np.abs(z - repeat)))
                            if difference > 1e-8:
                                raise ValueError("Model reload mismatch")
                            reloads.append({"task_id": task, "fraction": fraction, "seed": seed,
                                            "max_abs_difference": difference})
                    oof = pd.concat(all_eval, ignore_index=True)
                    for subset, subset_frame in [
                        ("all_sources", oof),
                        ("all_seen_documents", oof.loc[oof.source_all_seen_in_selected_train]),
                        ("any_unseen_document", oof.loc[oof.source_any_unseen_in_selected_train]),
                    ]:
                        if subset_frame.empty:
                            continue
                        result = metric_row(subset_frame, subset_frame.predicted_interface_target.to_numpy(), task,
                                            "extra_trees", spec["view"], "train_cv_learning_curve")
                        result.update(fraction=fraction, seed=seed, source_subset=subset,
                                      selected_train_parents=int(np.mean([r["selected_train_parents"] for r in samples
                                                                           if r["task_id"] == task and r["fraction"] == fraction and r["seed"] == seed])))
                        metrics.append(result)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        keys = ["task_id", "fraction", "seed", "row_id"]
        if prediction_frame.duplicated(keys).any() or prediction_frame.predicted_interface_target.isna().any():
            raise ValueError("Incomplete or duplicate learning-curve OOF predictions")
        metric_frame = pd.DataFrame(metrics)
        sample_frame = pd.DataFrame(samples)
        prediction_frame.to_csv(out / "predictions.csv", index=False)
        metric_frame.to_csv(out / "metrics.csv", index=False)
        sample_frame.to_csv(out / "sample_manifest.csv", index=False)
        pd.DataFrame(runs).to_csv(out / "run_registry.csv", index=False)
        pd.DataFrame(isolation).to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(reloads).to_csv(out / "reload_checks.csv", index=False)
        figure = learning_curve_figure(metric_frame)
        figure.savefig(out / "Figure_Papp_fu_learning_curves.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        summary = {"tasks": len(TASKS), "fractions": args.fractions, "seeds": args.seeds, "fits": len(runs),
                   "limited_smoke": limited, "sampling_unit": "complete_scaffold_group_from_outer_training_parents",
                   "source_sensitivity_definition": "evaluation-parent documents seen in selected outer-training sample",
                   "validation_target_file_opened": False, "validation_rows_evaluated": False, "test_labels_read": False,
                   "model_selection_authorized": False, "selection_scope": "learning_curve_diagnostic_only"}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Papp/fu learning curves\n\n"
            "Each point uses the fixed Stage-A ExtraTrees leader, fixed outer scaffold folds, and a whole-scaffold-group sample "
            "from the corresponding outer training fold. Values are train-CV diagnostics only; no validation/test labels are opened and "
            "the runner cannot authorize model selection. Source groups are descriptive because source, protocol and chemical space are confounded.\n",
            encoding="utf-8")
        finish_stage(out, "stl_papp_fu_learning_curve", inputs={
            "train_protocol_complete_sha256": sha256(args.train_protocol / "complete.json"),
            "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
            "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json"),
        }, partial=limited, **summary)
    print(f"Papp/fu learning curves: {args.output}")


if __name__ == "__main__":
    run_cli(main)
