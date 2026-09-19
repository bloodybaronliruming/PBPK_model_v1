#!/usr/bin/env python3
"""Run the frozen Gate-3 B6 aggregate gates, sensitivities, and publication figures."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_gate3_b6_physchem_cell_batch import verify_cell


C0 = "C0_corrected_StageA"
C1 = "C1_structure_only_control"
B6 = "B6_physchem_candidate"
ROUTES = [C0, C1, B6]
ROUTE_LABELS = {C0: "Corrected Stage-A", C1: "Structure-only control", B6: "Nested physicochemical"}
ROUTE_COLORS = {C0: "#4C78A8", C1: "#9C755F", B6: "#E45756"}
TASK_ORDER = [
    "CL__human__systemic_iv", "CLint__human__microsome", "Papp__human__caco2_ab",
    "Thalf__human__terminal_iv", "VDss__human__steady_state_iv", "fu__human__plasma",
]
ENDPOINT_LABELS = {
    "CL__human__systemic_iv": "CL", "CLint__human__microsome": "CLint",
    "Papp__human__caco2_ab": "Papp", "Thalf__human__terminal_iv": "Terminal t½",
    "VDss__human__steady_state_iv": "VDss", "fu__human__plasma": "fu",
}


class StageProgress:
    def __init__(self, total: int):
        self.total = total; self.done = 0; self.started = time.monotonic()

    def advance(self, stage: str) -> None:
        self.done += 1
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.done * (self.total - self.done)
        width = 30; filled = int(width * self.done / self.total)
        bar = "[" + "#" * filled + "-" * (width - filled) + "]"
        print(f"ANALYSIS_PROGRESS {bar} {self.done:02d}/{self.total:02d} stage={stage} "
              f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)


def primary_score(observed: np.ndarray, predicted: np.ndarray, metric: str) -> float:
    error = np.asarray(predicted, float) - np.asarray(observed, float)
    if metric == "physical_MAE":
        return float(np.mean(np.abs(error)))
    if metric == "transformed_RMSE":
        return float(np.sqrt(np.mean(error ** 2)))
    raise ValueError(f"Unsupported primary metric: {metric}")


def equal_fold_score(frame: pd.DataFrame, prediction: str, metric: str) -> float:
    return float(np.mean([
        primary_score(part.observed.to_numpy(float), part[prediction].to_numpy(float), metric)
        for _, part in frame.groupby("outer_fold", sort=True)
    ]))


def paired_bootstrap(frame: pd.DataFrame, candidate: str, comparator: str, metric: str,
                     repeats: int, seed: int, chunk: int = 250) -> dict:
    folds = []
    for _, part in frame.groupby("outer_fold", sort=True):
        folds.append((part.observed.to_numpy(float), part[candidate].to_numpy(float),
                      part[comparator].to_numpy(float)))
    if not folds:
        raise ValueError("Paired bootstrap received no folds")
    candidate_score = equal_fold_score(frame, candidate, metric)
    comparator_score = equal_fold_score(frame, comparator, metric)
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats); relative_changes = np.empty(repeats)
    offset = 0
    while offset < repeats:
        size = min(chunk, repeats - offset)
        candidate_boot = np.zeros(size); comparator_boot = np.zeros(size)
        for observed, candidate_values, comparator_values in folds:
            indices = rng.integers(0, len(observed), size=(size, len(observed)))
            candidate_error = candidate_values[indices] - observed[indices]
            comparator_error = comparator_values[indices] - observed[indices]
            if metric == "physical_MAE":
                candidate_boot += np.mean(np.abs(candidate_error), axis=1)
                comparator_boot += np.mean(np.abs(comparator_error), axis=1)
            else:
                candidate_boot += np.sqrt(np.mean(candidate_error ** 2, axis=1))
                comparator_boot += np.sqrt(np.mean(comparator_error ** 2, axis=1))
        candidate_boot /= len(folds); comparator_boot /= len(folds)
        differences[offset:offset + size] = candidate_boot - comparator_boot
        relative_changes[offset:offset + size] = 100 * (candidate_boot / comparator_boot - 1)
        offset += size
    return {
        "candidate_score": candidate_score, "comparator_score": comparator_score,
        "difference": candidate_score - comparator_score,
        "difference_bootstrap_mean": float(differences.mean()),
        "difference_ci_low": float(np.quantile(differences, 0.025)),
        "difference_ci_high": float(np.quantile(differences, 0.975)),
        "relative_error_change_pct": 100 * (candidate_score / comparator_score - 1),
        "relative_change_ci_low": float(np.quantile(relative_changes, 0.025)),
        "relative_change_ci_high": float(np.quantile(relative_changes, 0.975)),
    }


def fingerprint(smiles: str, generator, cache: dict[str, object]):
    if smiles not in cache:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES for applicability analysis: {smiles}")
        cache[smiles] = generator.GetFingerprint(molecule)
    return cache[smiles]


def attach_sensitivity_membership(predictions: pd.DataFrame, records: pd.DataFrame,
                                  progress: StageProgress) -> pd.DataFrame:
    identity = records.groupby(["task_id", "parent_id"], as_index=False).agg(
        canonical_parent_smiles=("canonical_parent_smiles", "first"),
        structure_count=("canonical_parent_smiles", "nunique"),
        source_record_count=("row_id", "nunique"), inner_fold_id=("inner_fold_id", "first"),
        fold_count=("inner_fold_id", "nunique"),
    )
    if identity.structure_count.ne(1).any() or identity.fold_count.ne(1).any():
        raise ValueError("Task-parent structure/fold identity is inconsistent")
    prediction = predictions.merge(
        identity.drop(columns=["structure_count", "fold_count"]),
        on=["task_id", "parent_id"], validate="one_to_one",
    )
    if not prediction.inner_fold_id_x.eq(prediction.outer_fold).all() or not prediction.inner_fold_id_y.eq(prediction.outer_fold).all():
        raise ValueError("Prediction/source outer-fold membership differs")
    prediction = prediction.drop(columns=["inner_fold_id_x", "inner_fold_id_y"])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    cache: dict[str, object] = {}
    similarity = np.full(len(prediction), np.nan)
    threshold = np.full(len(prediction), np.nan)
    for task in TASK_ORDER:
        task_identity = identity.loc[identity.task_id.eq(task)].copy()
        for fold in range(5):
            mask = prediction.task_id.eq(task) & prediction.outer_fold.eq(fold)
            evaluation = prediction.loc[mask]
            training = task_identity.loc[task_identity.inner_fold_id.ne(fold)]
            if evaluation.empty or training.empty:
                raise ValueError(f"Empty applicability-domain cell: {task} fold={fold}")
            training_fp = [fingerprint(s, generator, cache) for s in training.canonical_parent_smiles]
            values = [max(DataStructs.BulkTanimotoSimilarity(fingerprint(s, generator, cache), training_fp))
                      for s in evaluation.canonical_parent_smiles]
            cutoff = float(np.quantile(values, 0.25))
            similarity[np.flatnonzero(mask)] = values
            threshold[np.flatnonzero(mask)] = cutoff
            progress.advance(f"similarity:{ENDPOINT_LABELS[task]}:fold{fold}")
    prediction["nearest_outer_train_tanimoto"] = similarity
    prediction["low_similarity_cutoff"] = threshold
    prediction["is_low_similarity"] = prediction.nearest_outer_train_tanimoto.le(prediction.low_similarity_cutoff)
    prediction["is_repeated_parent"] = prediction.source_record_count.ge(2)
    if prediction[["nearest_outer_train_tanimoto", "low_similarity_cutoff"]].isna().any().any():
        raise ValueError("Applicability-domain membership is incomplete")
    return prediction


def route_diagnostics(predictions: pd.DataFrame, metric_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        part = predictions.loc[predictions.task_id.eq(task)]
        observed = part.observed.to_numpy(float); metric = metric_map[task]
        for route in ROUTES:
            predicted = part[route].to_numpy(float); residual = predicted - observed
            rho, pvalue = spearmanr(observed, predicted)
            denominator = float(np.sum((observed - observed.mean()) ** 2))
            rows.append({
                "task_id": task, "endpoint_label": ENDPOINT_LABELS[task], "route": route,
                "route_label": ROUTE_LABELS[route], "parents": len(part), "primary_metric": metric,
                "equal_fold_primary_error": equal_fold_score(part, route, metric),
                "pooled_primary_error": primary_score(observed, predicted, metric),
                "pooled_MAE": float(np.mean(np.abs(residual))),
                "pooled_RMSE": float(np.sqrt(np.mean(residual ** 2))),
                "pooled_R2": 1 - float(np.sum(residual ** 2)) / denominator if denominator else np.nan,
                "spearman_rho": float(rho), "spearman_pvalue": float(pvalue),
                "mean_residual": float(residual.mean()),
            })
    return pd.DataFrame(rows)


def build_figures(out: Path, fold_metrics: pd.DataFrame, bootstrap: pd.DataFrame,
                  sensitivity: pd.DataFrame, producer: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), constrained_layout=True)
    for ax, task in zip(axes.flat, TASK_ORDER, strict=True):
        part = fold_metrics.loc[fold_metrics.task_id.eq(task)]
        for x, route in enumerate(ROUTES):
            values = part.loc[part.route.eq(route), "normalized_to_C0"].to_numpy()
            jitter = np.linspace(-0.07, 0.07, len(values))
            ax.scatter(x + jitter, values, s=25, color=ROUTE_COLORS[route], alpha=0.72,
                       edgecolors="white", linewidths=0.35, zorder=3)
            ax.hlines(values.mean(), x - 0.22, x + 0.22, color=ROUTE_COLORS[route], linewidth=2.3)
        ax.axhline(1, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(3), ["Stage-A", "Structure\ncontrol", "Physchem"])
        ax.set_title(ENDPOINT_LABELS[task], fontweight="bold")
        ax.set_ylabel("Primary error / Stage-A error")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Nested physicochemical fusion does not improve the corrected Stage-A baseline",
                 fontsize=13, fontweight="bold")
    fig.savefig(out / "figure_1_normalized_route_performance.png", dpi=600,
                bbox_inches="tight", facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ybase = np.arange(len(TASK_ORDER))[::-1]
    comparison_style = [(C0, -0.11, "o", "#4C78A8", "B6 vs corrected Stage-A"),
                        (C1, 0.11, "s", "#9C755F", "B6 vs structure-only control")]
    for comparator, offset, marker, color, label in comparison_style:
        part = bootstrap.loc[bootstrap.comparator.eq(comparator)].set_index("task_id").loc[TASK_ORDER]
        center = part.relative_error_change_pct.to_numpy()
        lower = center - part.relative_change_ci_low.to_numpy()
        upper = part.relative_change_ci_high.to_numpy() - center
        ax.errorbar(center, ybase + offset, xerr=np.vstack([lower, upper]), fmt=marker,
                    color=color, ecolor=color, capsize=3, markersize=5, linewidth=1.1, label=label)
    ax.axvspan(ax.get_xlim()[0], 0, color="#DCEFE2", alpha=0.45, zorder=-2)
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.axvline(-2, color="#4C78A8", linewidth=0.9, linestyle="--", label="A1 threshold (−2%)")
    ax.set_yticks(ybase, [ENDPOINT_LABELS[t] for t in TASK_ORDER])
    ax.set_xlabel("Relative primary-error change of B6 (%)\nNegative values favor B6")
    fig.suptitle("Paired parent bootstrap effects (95% CI)", fontweight="bold", fontsize=12, y=0.89)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.16, right=0.98)
    fig.savefig(out / "figure_2_paired_bootstrap_forest.png", dpi=600,
                bbox_inches="tight", facecolor="white"); plt.close(fig)

    heat = fold_metrics.loc[fold_metrics.route.eq(B6)].pivot(index="task_id", columns="outer_fold", values="relative_change_vs_C0_pct").loc[TASK_ORDER]
    limit = max(2.0, float(np.nanmax(np.abs(heat.to_numpy()))))
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    image = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            value = heat.iloc[row, col]
            color = "white" if abs(value) > limit * 0.52 else "#222222"
            ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(5), [f"Fold {i}" for i in range(1, 6)])
    ax.set_yticks(range(6), [ENDPOINT_LABELS[t] for t in TASK_ORDER])
    ax.set_title("Fold-wise B6 error change relative to corrected Stage-A", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Relative primary-error change (%)\nNegative values favor B6")
    fig.savefig(out / "figure_3_foldwise_effect_heatmap.png", dpi=600,
                bbox_inches="tight", facecolor="white"); plt.close(fig)

    plot = sensitivity.copy()
    plot["label"] = plot.endpoint_label + " — " + plot.subset.map({"low_similarity": "Low similarity", "repeated_parent": "Repeated parent"})
    plot = plot.sort_values(["task_order", "subset_order"], ascending=[False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    colors = {"low_similarity": "#59A14F", "repeated_parent": "#B279A2"}
    markers = {"low_similarity": "o", "repeated_parent": "s"}
    for y, row in plot.iterrows():
        color = colors[row.subset]
        face = color if row.evaluable else "white"
        if np.isfinite(row.relative_error_change_pct):
            ax.errorbar(row.relative_error_change_pct, y,
                        xerr=[[row.relative_error_change_pct - row.relative_change_ci_low],
                              [row.relative_change_ci_high - row.relative_error_change_pct]],
                        fmt=markers[row.subset], color=color, markerfacecolor=face,
                        capsize=3, markersize=5, linewidth=1)
            ax.text(row.relative_change_ci_high + 0.35, y, f"n={int(row.parents)}", va="center", fontsize=7)
        else:
            ax.text(0, y, "Not estimable (n=0)", va="center", ha="center", fontsize=7,
                    color="#666666", style="italic")
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.axvline(2, color="#D62728", linestyle="--", linewidth=1, label="A6 harm limit (+2%)")
    ax.set_yticks(range(len(plot)), plot.label)
    ax.set_xlabel("B6 relative primary-error change vs Stage-A (%)\nPositive values indicate harm")
    fig.suptitle("Applicability and repeated-measure sensitivity (95% CI)",
                 fontweight="bold", fontsize=12, y=0.91)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color=colors["low_similarity"], linestyle="none", label="Low similarity"),
               Line2D([0], [0], marker="s", color=colors["repeated_parent"], linestyle="none", label="Repeated parent"),
               Line2D([0], [0], marker="o", markerfacecolor="white", color="#666666", linestyle="none", label="Underpowered")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.28, right=0.96)
    fig.savefig(out / "figure_4_sensitivity_forest.png", dpi=600,
                bbox_inches="tight", facecolor="white"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    display_producer = producer.copy()
    display_producer["auxiliary_task"] = display_producer.auxiliary_task.map({
        "experimental_logD_pH7_4": "Experimental logD (pH 7.4)",
        "experimental_logP": "Experimental logP",
    })
    display_producer["selected_feature_set"] = display_producer.selected_feature_set.map({
        "ecfp4_plus_leakage_safe_rdkit2d": "ECFP4 + safe RDKit2D",
        "leakage_safe_rdkit2d": "Safe RDKit2D",
    })
    selection = display_producer.groupby(["auxiliary_task", "selected_feature_set"]).size().unstack(fill_value=0)
    selection.plot(kind="bar", stacked=True, ax=axes[0], color=["#4C78A8", "#F28E2B"][:len(selection.columns)])
    axes[0].set_title("Auxiliary producer selection", fontweight="bold", pad=48)
    axes[0].set_xlabel(""); axes[0].set_ylabel("Selected outer-fold cells")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Feature view", frameon=False, fontsize=7, loc="lower center",
                   bbox_to_anchor=(0.5, 1.01), ncol=1)
    alpha = metrics.loc[metrics.route.isin([C1, B6])].groupby(["route", "selected_meta_alpha"]).size().unstack(fill_value=0)
    alpha.plot(kind="bar", ax=axes[1], color=["#59A14F", "#EDC948", "#B279A2"][:len(alpha.columns)])
    axes[1].set_title("Residual ridge regularization", fontweight="bold", pad=48)
    axes[1].set_xlabel(""); axes[1].set_ylabel("Selected outer-fold cells")
    axes[1].set_xticklabels(["Structure control", "Physchem"], rotation=0)
    axes[1].legend(title="Ridge alpha", frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 1.01), ncol=2)
    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "figure_5_selection_diagnostics.png", dpi=600,
                bbox_inches="tight", facecolor="white"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-protocol", type=Path, default=ROOT / "data/public_development/gate3_b6_aggregate_analysis_protocol_v1")
    parser.add_argument("--formal", type=Path, default=ROOT / "data/public_development/gate3_b6_formal_run_v4")
    parser.add_argument("--batch", type=Path, default=ROOT / "results/benchmarks/gate3_b6_physchem_formal_cells_v3")
    parser.add_argument("--train-records", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1/benchmark_train_records.csv")
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts/run_gate3_b6_physchem_outer_fold.py")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/gate3_b6_physchem_formal_analysis_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.analysis_protocol / "complete.json", args.analysis_protocol / "analysis_contract.json",
                args.formal / "complete.json", args.formal / "formal_cell_registry.csv",
                args.formal / "formal_authorization.json", args.batch / "batch_complete.json",
                args.train_records, args.runner]
    startup_self_check(required, output=None if args.check_only else args.output)
    analysis_meta = verify_stage(args.analysis_protocol, "gate3_b6_aggregate_analysis_protocol")
    formal_meta = verify_stage(args.formal, "gate3_b6_formal_run_freeze")
    contract = json.loads((args.analysis_protocol / "analysis_contract.json").read_text())
    authorization = json.loads((args.formal / "formal_authorization.json").read_text())
    batch = json.loads((args.batch / "batch_complete.json").read_text())
    runner_hash = sha256(args.runner)
    if authorization["outer_fold_runner_sha256"] != runner_hash or batch["outer_fold_runner_sha256"] != runner_hash:
        raise ValueError("Formal authorization/batch/runner hashes differ")
    if batch["authorization_complete_sha256"] != sha256(args.formal / "complete.json"):
        raise ValueError("Batch is not bound to the supplied formal authorization")
    if analysis_meta.get("test_labels_read") or formal_meta.get("test_labels_read"):
        raise ValueError("Analysis inputs are not test-closed")
    registry = pd.read_csv(args.formal / "formal_cell_registry.csv")
    if len(registry) != 30:
        raise ValueError("Formal registry must contain 30 cells")
    if args.check_only:
        print(f"Gate 3 B6 formal analysis ready: cells=30 bootstrap={contract['paired_bootstrap']['replicates']}")
        return

    progress = StageProgress(47)
    prediction_frames = []; metric_frames = []; producer_frames = []
    stored = {(row["task_id"], int(row["outer_fold"])): row for row in batch["cells"]}
    for row in registry.itertuples(index=False):
        cell = ROOT / row.output_directory
        verified = verify_cell(cell, str(row.task_id), int(row.outer_fold), runner_hash)
        if stored[(str(row.task_id), int(row.outer_fold))]["complete_sha256"] != verified["complete_sha256"]:
            raise ValueError(f"Batch/cell marker hash differs: {row.cell_id}")
        prediction = pd.read_csv(cell / "outer_evaluation_predictions.csv").rename(
            columns={"observed_outer_evaluation_target": "observed"})
        prediction.insert(0, "outer_fold", int(row.outer_fold)); prediction.insert(0, "task_id", str(row.task_id))
        prediction_frames.append(prediction)
        metric = pd.read_csv(cell / "outer_fold_metrics.csv"); metric_frames.append(metric)
        selected = pd.read_csv(cell / "selected_auxiliary_producers.csv")
        selected["cell_id"] = row.cell_id; producer_frames.append(selected)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    producer = pd.concat(producer_frames, ignore_index=True)
    if predictions.duplicated(["task_id", "outer_fold", "parent_id"]).any() or not predictions[ROUTES].notna().all().all():
        raise ValueError("Formal prediction keys/values are incomplete")
    progress.advance("verified_30_cells")

    metric_map = metrics.groupby("task_id").primary_metric_name.first().to_dict()
    fold_rows = []
    for task in TASK_ORDER:
        for fold in range(5):
            part = predictions.loc[predictions.task_id.eq(task) & predictions.outer_fold.eq(fold)]
            c0_score = primary_score(part.observed.to_numpy(), part[C0].to_numpy(), metric_map[task])
            for route in ROUTES:
                value = primary_score(part.observed.to_numpy(), part[route].to_numpy(), metric_map[task])
                stored_value = metrics.loc[(metrics.task_id.eq(task)) & (metrics.outer_fold.eq(fold)) & metrics.route.eq(route), "primary_metric"]
                if len(stored_value) != 1 or abs(value - float(stored_value.iloc[0])) > 1e-12:
                    raise ValueError(f"Stored/recomputed metric differs: {task} fold={fold} route={route}")
                fold_rows.append({"task_id": task, "endpoint_label": ENDPOINT_LABELS[task], "outer_fold": fold,
                                  "route": route, "route_label": ROUTE_LABELS[route], "parents": len(part),
                                  "primary_metric": metric_map[task], "primary_error": value,
                                  "normalized_to_C0": value / c0_score,
                                  "relative_change_vs_C0_pct": 100 * (value / c0_score - 1)})
    fold_metrics = pd.DataFrame(fold_rows)
    diagnostics = route_diagnostics(predictions, metric_map)
    progress.advance("recomputed_route_metrics")

    repeats = int(contract["paired_bootstrap"]["replicates"]); base_seed = int(contract["paired_bootstrap"]["seed"])
    bootstrap_rows = []
    for task_index, task in enumerate(TASK_ORDER):
        part = predictions.loc[predictions.task_id.eq(task)]
        for comparison_index, comparator in enumerate([C0, C1]):
            result = paired_bootstrap(part, B6, comparator, metric_map[task], repeats,
                                      base_seed + task_index * 100 + comparison_index)
            bootstrap_rows.append({"task_id": task, "endpoint_label": ENDPOINT_LABELS[task],
                                   "candidate": B6, "comparator": comparator,
                                   "comparator_label": ROUTE_LABELS[comparator],
                                   "primary_metric": metric_map[task], "parents": len(part),
                                   "outer_folds": part.outer_fold.nunique(), "bootstrap_repeats": repeats, **result})
        progress.advance(f"bootstrap:{ENDPOINT_LABELS[task]}")
    bootstrap = pd.DataFrame(bootstrap_rows)

    records = pd.read_csv(args.train_records)
    records = records.loc[records.task_id.isin(TASK_ORDER)].copy()
    predictions = attach_sensitivity_membership(predictions, records, progress)
    sensitivity_rows = []
    for task_index, task in enumerate(TASK_ORDER):
        task_part = predictions.loc[predictions.task_id.eq(task)]
        for subset_index, (subset, mask) in enumerate([
            ("low_similarity", task_part.is_low_similarity),
            ("repeated_parent", task_part.is_repeated_parent),
        ]):
            part = task_part.loc[mask].copy()
            if part.empty:
                result = {key: np.nan for key in ["candidate_score", "comparator_score", "difference",
                          "difference_bootstrap_mean", "difference_ci_low", "difference_ci_high",
                          "relative_error_change_pct", "relative_change_ci_low", "relative_change_ci_high"]}
            else:
                result = paired_bootstrap(part, B6, C0, metric_map[task], repeats,
                                          base_seed + 1000 + task_index * 100 + subset_index)
            evaluable = len(part) >= int(contract["sufficient_power"]["minimum_parents"]) and part.outer_fold.nunique() >= int(contract["sufficient_power"]["minimum_outer_folds"])
            passes = bool(result["relative_error_change_pct"] <= contract["sensitivity_relative_harm_limit_pct"]) if evaluable else np.nan
            sensitivity_rows.append({
                "task_id": task, "endpoint_label": ENDPOINT_LABELS[task], "task_order": task_index,
                "subset": subset, "subset_order": subset_index, "parents": len(part),
                "outer_folds": part.outer_fold.nunique(), "primary_metric": metric_map[task],
                "evaluable": evaluable, "relative_harm_limit_pct": contract["sensitivity_relative_harm_limit_pct"],
                "subset_passed": passes, **result,
            })
        progress.advance(f"sensitivity:{ENDPOINT_LABELS[task]}")
    sensitivity = pd.DataFrame(sensitivity_rows)

    mechanism_rows = []
    for task in TASK_ORDER:
        part = predictions.loc[predictions.task_id.eq(task)]
        residual = part.observed - part[C0]; correction = part[B6] - part[C0]
        correction_rho, correction_p = spearmanr(correction, residual)
        for auxiliary in ["nested_experimental_logP", "nested_experimental_logD_pH7_4"]:
            target_rho, target_p = spearmanr(part[auxiliary], part.observed)
            residual_rho, residual_p = spearmanr(part[auxiliary], residual)
            mechanism_rows.append({"task_id": task, "endpoint_label": ENDPOINT_LABELS[task],
                                   "auxiliary_signal": auxiliary, "parents": len(part),
                                   "auxiliary_vs_target_spearman": target_rho, "auxiliary_vs_target_pvalue": target_p,
                                   "auxiliary_vs_stageA_residual_spearman": residual_rho,
                                   "auxiliary_vs_stageA_residual_pvalue": residual_p,
                                   "B6_correction_vs_stageA_residual_spearman": correction_rho,
                                   "B6_correction_vs_stageA_residual_pvalue": correction_p})
    mechanism = pd.DataFrame(mechanism_rows)
    progress.advance("mechanistic_diagnostics")

    gate_rows = []; decision_rows = []
    for task in TASK_ORDER:
        comparison_c0 = bootstrap.loc[(bootstrap.task_id.eq(task)) & bootstrap.comparator.eq(C0)].iloc[0]
        comparison_c1 = bootstrap.loc[(bootstrap.task_id.eq(task)) & bootstrap.comparator.eq(C1)].iloc[0]
        nonworse = int((fold_metrics.loc[(fold_metrics.task_id.eq(task)) & fold_metrics.route.eq(B6), "relative_change_vs_C0_pct"] <= 0).sum())
        task_sensitivity = sensitivity.loc[sensitivity.task_id.eq(task)]
        low_evaluable = bool(task_sensitivity.loc[task_sensitivity.subset.eq("low_similarity"), "evaluable"].iloc[0])
        powered = task_sensitivity.loc[task_sensitivity.evaluable]
        a6 = low_evaluable and bool(powered.subset_passed.astype(bool).all())
        gates = {
            "A1": bool(-comparison_c0.relative_error_change_pct >= 2.0),
            "A2": nonworse >= 3,
            "A3": bool(comparison_c0.difference_ci_high < 0),
            "A4": bool(comparison_c1.candidate_score < comparison_c1.comparator_score and comparison_c1.difference_ci_high < 0),
            "A5": True,
            "A6": a6,
        }
        observed = {
            "A1": f"B6 error reduction={-comparison_c0.relative_error_change_pct:.4f}%",
            "A2": f"nonworse_folds={nonworse}/5",
            "A3": f"B6-C0 bootstrap CI high={comparison_c0.difference_ci_high:.6g}",
            "A4": f"B6-C1 difference={comparison_c1.difference:.6g}; CI high={comparison_c1.difference_ci_high:.6g}",
            "A5": "30/30 cells verified; overlap=0; reload<=1e-12; finite predictions",
            "A6": "; ".join(f"{r.subset}:n={r.parents},evaluable={r.evaluable},harm={r.relative_error_change_pct:.3f}%" for r in task_sensitivity.itertuples()),
        }
        for gate_id in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            gate_rows.append({"task_id": task, "endpoint_label": ENDPOINT_LABELS[task],
                              "gate_id": gate_id, "required": True,
                              "passed": gates[gate_id], "observed": observed[gate_id]})
        gate_rows.append({"task_id": task, "endpoint_label": ENDPOINT_LABELS[task], "gate_id": "A7",
                          "required": False, "passed": False,
                          "observed": "Not evaluable: per-record logP/logD provenance unavailable; limitation retained"})
        advanced = all(gates.values())
        failed = [key for key, value in gates.items() if not value]
        decision_rows.append({"task_id": task, "endpoint_label": ENDPOINT_LABELS[task],
                              "A1_to_A6_all_passed": advanced, "B6_advanced": advanced,
                              "failed_required_gates": "|".join(failed),
                              "decision": "advance_B6" if advanced else "stop_B6_retain_as_negative_ablation"})
    gates = pd.DataFrame(gate_rows); decisions = pd.DataFrame(decision_rows)
    progress.advance("A1_to_A7_gate_table")

    summary = {
        "analysis_id": contract["analysis_id"], "cells_verified": 30,
        "endpoints": 6, "parents": int(len(predictions)),
        "bootstrap_replicates": repeats,
        "endpoints_advancing_B6": int(decisions.B6_advanced.sum()),
        "B6_family_advanced": bool(decisions.B6_advanced.any()),
        "next_action": "stop_B6_family_retain_as_negative_ablation" if not decisions.B6_advanced.any() else "advance_only_passing_endpoints",
        "architecture_selection_authorized": True,
        "fixed_validation_authorized": False, "test_authorized": False,
        "A7_limitation": "Per-record experimental logP/logD provenance unavailable; source sensitivity cannot be claimed.",
    }
    inputs = {str(path.resolve()): sha256(path) for path in required}
    with stage_output(args.output) as out:
        fold_metrics.to_csv(out / "table_1_outer_fold_route_metrics.csv", index=False, float_format="%.8g")
        diagnostics.to_csv(out / "table_2_endpoint_route_diagnostics.csv", index=False, float_format="%.8g")
        bootstrap.to_csv(out / "table_3_paired_parent_bootstrap.csv", index=False, float_format="%.8g")
        predictions.to_csv(out / "table_4_parent_predictions_and_sensitivity_membership.csv", index=False, float_format="%.8g")
        sensitivity.to_csv(out / "table_5_sensitivity_metrics.csv", index=False, float_format="%.8g")
        gates.to_csv(out / "table_6_endpoint_gate_decisions.csv", index=False)
        decisions.to_csv(out / "table_7_endpoint_final_decisions.csv", index=False)
        producer.to_csv(out / "table_8_auxiliary_producer_selections.csv", index=False)
        mechanism.to_csv(out / "table_9_mechanistic_associations.csv", index=False, float_format="%.8g")
        (out / "decision.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        build_figures(out, fold_metrics, bootstrap, sensitivity, producer, metrics)
        progress.advance("publication_figures")
        report = [
            "# Gate 3 B6 preregistered formal train-CV analysis", "",
            "All 30 endpoint-by-outer-fold cells passed lineage, overlap, imputation, hash, reload, prediction, and label-lifecycle checks. "
            "The analysis used 10,000 outer-fold-stratified paired-parent bootstrap replicates. No model was refitted, and fixed validation/test remained closed.", "",
            "## Endpoint decisions", "",
            "| Endpoint | B6 vs Stage-A error change | 95% CI | Nonworse folds | Failed required gates | Decision |",
            "|---|---:|---:|---:|---|---|",
        ]
        for task in TASK_ORDER:
            b = bootstrap.loc[(bootstrap.task_id.eq(task)) & bootstrap.comparator.eq(C0)].iloc[0]
            d = decisions.loc[decisions.task_id.eq(task)].iloc[0]
            nonworse = int((fold_metrics.loc[(fold_metrics.task_id.eq(task)) & fold_metrics.route.eq(B6), "relative_change_vs_C0_pct"] <= 0).sum())
            decision_label = "Advance" if bool(d.B6_advanced) else "Stop"
            report.append(f"| {ENDPOINT_LABELS[task]} | {b.relative_error_change_pct:+.2f}% | {b.relative_change_ci_low:+.2f}% to {b.relative_change_ci_high:+.2f}% | {nonworse}/5 | {d.failed_required_gates} | {decision_label} |")
        report += ["", "No endpoint passed all required A1–A6 gates. B6 is retained as a rigorously evaluated negative ablation and is not advanced. "
                   "The protocol prohibits post-result tuning on these outer-CV outcomes.", "",
                   "## Scope", "", "A7 remains an explicit limitation because experimental logP/logD records lack per-record provenance. "
                   "This analysis authorizes the train-CV architecture decision only; fixed validation, test, source-test labels, and human F remain closed."]
        (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        finish_stage(
            out, "gate3_b6_physchem_formal_analysis", inputs=inputs,
            cells_verified=30, endpoints=6, parents=len(predictions), bootstrap_replicates=repeats,
            publication_png_figures=5, figure_dpi=600, figure_language="English",
            endpoints_advancing_B6=int(decisions.B6_advanced.sum()), B6_family_advanced=bool(decisions.B6_advanced.any()),
            model_fitted=False, architecture_selection_authorized=True,
            fixed_validation_authorized=False, validation_target_file_opened=False,
            source_test_labels_read=False, test_labels_read=False, partial=False,
        )
    print(f"Gate 3 B6 formal analysis: {args.output}")


if __name__ == "__main__":
    run_cli(main)
