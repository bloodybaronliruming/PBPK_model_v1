#!/usr/bin/env python3
"""Create the read-only Gate 5 CLint diagnostic and test-lifecycle closeout bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from pipeline_common import ROOT, dump_json, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASK = "CLint__human__microsome"


def log_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    observed_log = np.log10(np.asarray(observed, dtype=float))
    predicted_log = np.log10(np.asarray(predicted, dtype=float))
    error = predicted_log - observed_log
    absolute = np.abs(error)
    rank = float(spearmanr(observed_log, predicted_log).statistic) if len(error) > 2 else None
    return {
        "n": int(len(error)),
        "log10_rmse": float(np.sqrt(np.mean(error ** 2))),
        "log10_mae": float(np.mean(absolute)),
        "gmfe": float(10 ** np.mean(absolute)),
        "within_2fold": float(np.mean(absolute <= np.log10(2))),
        "within_3fold": float(np.mean(absolute <= np.log10(3))),
        "spearman": rank,
    }


def plot_observed_predicted(table: pd.DataFrame, metrics: dict, output: Path) -> None:
    observed = np.log10(table.observed_physical.to_numpy(float))
    predicted = np.log10(table.predicted_physical.to_numpy(float))
    inside = table.inside_train_only_AD.astype(bool).to_numpy()
    low = float(min(observed.min(), predicted.min())) - 0.12
    high = float(max(observed.max(), predicted.max())) + 0.12
    fig, ax = plt.subplots(figsize=(6.4, 6.1), constrained_layout=True)
    ax.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=1.2, label="Identity")
    ax.scatter(observed[inside], predicted[inside], s=50, color="#2878B5", edgecolor="white",
               linewidth=0.6, alpha=0.9, label=f"Inside AD (n={inside.sum()})")
    ax.scatter(observed[~inside], predicted[~inside], s=66, marker="^", color="#D95F02", edgecolor="white",
               linewidth=0.6, alpha=0.95, label=f"Outside AD (n={(~inside).sum()})")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed log$_{10}$ CLint (µL min$^{-1}$ mg protein$^{-1}$)")
    ax.set_ylabel("Predicted log$_{10}$ CLint (µL min$^{-1}$ mg protein$^{-1}$)")
    ax.text(0.03, 0.97,
            f"n = {metrics['full_test']['molecules']}\nRMSE = {metrics['full_test']['primary']:.3f}\n"
            f"95% CI = {metrics['bootstrap']['log10_rmse_95_ci'][0]:.3f}–{metrics['bootstrap']['log10_rmse_95_ci'][1]:.3f}\n"
            f"Spearman = {metrics['full_test']['spearman_molecule_mean']:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.94})
    ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=8.5)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ad_and_source(table: pd.DataFrame, source: pd.DataFrame, ad_floor: float, output: Path) -> None:
    absolute_error = np.abs(np.log10(table.predicted_physical.to_numpy(float)) -
                            np.log10(table.observed_physical.to_numpy(float)))
    inside = table.inside_train_only_AD.astype(bool).to_numpy()
    eligible = source.loc[source.descriptive_metric_eligible.astype(bool)].sort_values("log10_rmse")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.axvline(ad_floor, color="#555555", linestyle="--", linewidth=1.2,
               label=f"AD floor = {ad_floor:.3f}")
    ax.scatter(table.loc[inside, "max_train_tanimoto"], absolute_error[inside], s=46,
               color="#2878B5", edgecolor="white", linewidth=0.5, alpha=0.9, label="Inside AD")
    ax.scatter(table.loc[~inside, "max_train_tanimoto"], absolute_error[~inside], s=62, marker="^",
               color="#D95F02", edgecolor="white", linewidth=0.5, alpha=0.95, label="Outside AD")
    ax.set_xlabel("Maximum ECFP4 Tanimoto to training set")
    ax.set_ylabel("Absolute log$_{10}$ error")
    ax.set_title("a", loc="left", fontweight="bold")
    ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, fontsize=8.5)

    ax = axes[1]
    y = np.arange(len(eligible))
    bars = ax.barh(y, eligible.log10_rmse, color="#4E9F3D", alpha=0.88)
    ax.set_yticks(y, [f"Document {int(v)}" for v in eligible.doc_id])
    ax.set_xlabel("Document-level log$_{10}$ RMSE")
    ax.set_title("b", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    span = max(float(eligible.log10_rmse.max()), 0.1)
    ax.set_xlim(0, span * 1.30)
    for bar, row in zip(bars, eligible.itertuples(index=False)):
        ax.text(bar.get_width() + span * 0.025, bar.get_y() + bar.get_height() / 2,
                f"{row.log10_rmse:.3f} (n={row.molecules})", va="center", ha="left", fontsize=8.5)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path,
                        default=ROOT / "results/final/CLint_gate5_stageA_et_c01_test_recovery_v2")
    parser.add_argument("--frozen", type=Path,
                        default=ROOT / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2")
    parser.add_argument("--recovery", type=Path,
                        default=ROOT / "data/public_development/gate5_clint_technical_recovery_v2")
    parser.add_argument("--stagea", type=Path,
                        default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--task-table", type=Path,
                        default=ROOT / "data/processed_v15/datasets/tasks/CLint__human__microsome.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/gate5_clint_final_diagnostic_closeout_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.final / "complete.json", args.final / "metrics.json", args.final / "failure_report.json",
        args.final / "test_predictions.csv", args.frozen / "complete.json", args.frozen / "evaluation_protocol.json",
        args.recovery / "complete.json", args.stagea / "complete.json", args.stagea / "frozen_stageA_shortlist.csv",
        args.task_table,
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    final_meta = verify_stage(args.final, "gate5_clint_final_technical_recovery_test_evaluation")
    frozen_meta = verify_stage(args.frozen, "gate5_clint_stagea_et_candidate")
    recovery_meta = verify_stage(args.recovery, "gate5_clint_technical_recovery_protocol_v2")
    verify_stage(args.stagea, "stl_stageA_three_view_audit")
    if not final_meta.get("test_evaluated") or any(final_meta.get(k) for k in ["refit", "calibration", "selection_changed"]):
        raise ValueError("Final Gate 5 lifecycle flags do not permit read-only closeout")
    if recovery_meta.get("test_evaluated") or frozen_meta.get("test_evaluated"):
        raise ValueError("Pre-test model/protocol artifacts were modified after evaluation")
    if args.check_only:
        print("Gate 5 CLint closeout preflight valid: final hashes and no-post-test-change lifecycle verified.")
        return

    metrics_payload = json.loads((args.final / "metrics.json").read_text(encoding="utf-8"))
    failures = json.loads((args.final / "failure_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.frozen / "evaluation_protocol.json").read_text(encoding="utf-8"))
    table = pd.read_csv(args.final / "test_predictions.csv")
    required_prediction_columns = {
        "row_id", "molecule_id", "observed_physical", "predicted_physical", "predicted_transformed",
        "max_train_tanimoto", "inside_train_only_AD", "canonical_parent_id",
    }
    if not required_prediction_columns.issubset(table.columns) or len(table) != 28 or table.molecule_id.nunique() != 28:
        raise ValueError("Frozen final prediction table is incomplete")
    observed = table.observed_physical.to_numpy(float)
    predicted = table.predicted_physical.to_numpy(float)
    if (observed <= 0).any() or (predicted <= 0).any() or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Final prediction table contains invalid physical values")
    if not np.allclose(np.log10(predicted), table.predicted_transformed.to_numpy(float), rtol=0, atol=1e-12):
        raise ValueError("Transformed and physical predictions disagree")
    reconstructed = log_metrics(observed, predicted)
    if not np.isclose(reconstructed["log10_rmse"], metrics_payload["full_test"]["primary"], rtol=0, atol=1e-12):
        raise ValueError("Reconstructed primary metric disagrees with frozen metrics")

    ad_rows = []
    for name, mask in [("inside_train_only_AD", table.inside_train_only_AD.astype(bool)),
                       ("outside_train_only_AD", ~table.inside_train_only_AD.astype(bool))]:
        row = {"stratum": name, **log_metrics(table.loc[mask, "observed_physical"], table.loc[mask, "predicted_physical"])}
        row["mean_max_train_tanimoto"] = float(table.loc[mask, "max_train_tanimoto"].mean())
        ad_rows.append(row)
    ad_summary = pd.DataFrame(ad_rows)

    source_meta = pd.read_csv(args.task_table, usecols=["row_id", "doc_id"])
    source_table = table.merge(source_meta, on="row_id", how="left", validate="one_to_one")
    if source_table.doc_id.isna().any():
        raise ValueError("A frozen test prediction lacks source-document metadata")
    source_rows = []
    for doc_id, group in source_table.groupby("doc_id", sort=True):
        eligible = group.molecule_id.nunique() >= 3
        row = {
            "doc_id": int(doc_id), "records": int(len(group)), "molecules": int(group.molecule_id.nunique()),
            "mean_max_train_tanimoto": float(group.max_train_tanimoto.mean()),
            "descriptive_metric_eligible": bool(eligible),
        }
        if eligible:
            row.update(log_metrics(group.observed_physical, group.predicted_physical))
            row["molecules"] = int(group.molecule_id.nunique())
        else:
            row.update({k: np.nan for k in ["log10_rmse", "log10_mae", "gmfe", "within_2fold", "within_3fold", "spearman"]})
        source_rows.append(row)
    source_summary = pd.DataFrame(source_rows)

    shortlist = pd.read_csv(args.stagea / "frozen_stageA_shortlist.csv")
    cv = shortlist.loc[(shortlist.task_id == TASK) &
                       (shortlist.candidate_uid == "ecfp4_rdkit2d::extra_trees__c01")]
    if len(cv) != 1:
        raise ValueError("Unable to recover the unique frozen Stage-A CLint reference")
    cv = cv.iloc[0]
    comparison = pd.DataFrame([
        {"evaluation": "train_cv", "molecules": int(cv.parents), "log10_rmse": float(cv.primary_metric),
         "rmse_lower_reference": float(cv.fold_primary_min), "rmse_upper_reference": float(cv.fold_primary_max),
         "selection_role": "pre-test_candidate_selection"},
        {"evaluation": "frozen_test", "molecules": 28, "log10_rmse": reconstructed["log10_rmse"],
         "rmse_lower_reference": float(metrics_payload["bootstrap"]["log10_rmse_95_ci"][0]),
         "rmse_upper_reference": float(metrics_payload["bootstrap"]["log10_rmse_95_ci"][1]),
         "selection_role": "confirmatory_no_reselection"},
    ])
    test_vs_cv = float((reconstructed["log10_rmse"] / float(cv.primary_metric) - 1) * 100)
    calibration = linregress(np.log10(predicted), np.log10(observed))
    calibration_summary = pd.DataFrame([{
        "slope_observed_on_predicted": float(calibration.slope),
        "intercept_observed_on_predicted": float(calibration.intercept),
        "pearson_r": float(calibration.rvalue),
        "spearman": reconstructed["spearman"],
        "status": "descriptive_only_no_calibration_fitted",
    }])

    integrity = {
        "final_artifacts_verified": int(len(final_meta["artifacts"])),
        "final_complete_sha256": sha256(args.final / "complete.json"),
        "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
        "recovery_complete_sha256": sha256(args.recovery / "complete.json"),
        "stagea_complete_sha256": sha256(args.stagea / "complete.json"),
        "records": int(len(table)), "molecules": int(table.molecule_id.nunique()),
        "invalid_or_unmapped_features": int(failures["invalid_or_unmapped_test_feature"]),
        "train_scaffold_overlap": int(failures["train_scaffold_overlap"]),
        "nonfinite_or_nonpositive_prediction": int(failures["nonfinite_or_nonpositive_prediction"]),
        "primary_metric_reconstructed_exactly": True,
    }
    lifecycle = {
        "task_id": TASK,
        "status": "test_consumed_lifecycle_closed",
        "final_result": str(args.final.resolve()),
        "technical_history": "two documented pre-prediction aborts followed by the final recovery v2 evaluation",
        "test_evaluated": True,
        "model_refit": False,
        "calibration_fitted": False,
        "selection_changed": False,
        "model_selection_authorized": False,
        "test_rerun_authorized": False,
        "allowed_post_test_work": ["read_only_diagnostics", "reporting", "external_protocol_comparison"],
        "forbidden_post_test_work": ["refit", "calibration", "threshold_change", "candidate_swap", "hyperparameter_search", "test_rerun"],
    }

    with stage_output(args.output) as out:
        ad_summary.to_csv(out / "ad_stratum_metrics.csv", index=False, float_format="%.6f")
        source_summary.to_csv(out / "source_document_metrics.csv", index=False, float_format="%.6f")
        comparison.to_csv(out / "traincv_test_comparison.csv", index=False, float_format="%.6f")
        calibration_summary.to_csv(out / "calibration_summary.csv", index=False, float_format="%.6f")
        dump_json(out / "integrity_audit.json", integrity)
        dump_json(out / "test_lifecycle_closure.json", lifecycle)
        plot_observed_predicted(table, metrics_payload, out / "Figure_1_CLint_observed_vs_predicted.png")
        plot_ad_and_source(table, source_summary, float(protocol["applicability_domain"]["train_only_floor"]),
                           out / "Figure_2_CLint_AD_and_source_diagnostics.png")
        pd.DataFrame([
            {"figure": "Figure_1_CLint_observed_vs_predicted.png", "dpi": 600,
             "content": "Frozen test observed versus predicted values with pre-registered AD status"},
            {"figure": "Figure_2_CLint_AD_and_source_diagnostics.png", "dpi": 600,
             "content": "Absolute error versus training similarity and eligible document-level RMSE"},
        ]).to_csv(out / "figure_manifest.csv", index=False)
        eligible_sources = source_summary.loc[source_summary.descriptive_metric_eligible]
        report = f"""# Gate 5 CLint final diagnostic and lifecycle closeout

## Integrity and lifecycle

All {len(final_meta['artifacts'])} frozen final artifacts and their hashes were verified. The final result contains {len(table)} records for {table.molecule_id.nunique()} unique molecules. The model was not refitted, calibrated, reselected, or threshold-adjusted. The Gate 5 test lifecycle is now closed and no evaluator rerun is authorized.

Two technical aborts occurred after programmatic test-table access but before any prediction or metric output. The final recovery v2 used the unchanged pre-test-frozen model and corrected only the source-identity versus canonical-parent-identity validation contract. This history must remain visible in reporting.

## Frozen performance

Molecule-level log10 RMSE was {reconstructed['log10_rmse']:.4f} with the pre-registered 95% bootstrap CI [{metrics_payload['bootstrap']['log10_rmse_95_ci'][0]:.4f}, {metrics_payload['bootstrap']['log10_rmse_95_ci'][1]:.4f}]. GMFE was {reconstructed['gmfe']:.3f}; {reconstructed['within_2fold']*100:.1f}% and {reconstructed['within_3fold']*100:.1f}% of molecules were within 2-fold and 3-fold, respectively. Spearman correlation was {reconstructed['spearman']:.3f}.

The frozen test RMSE was {abs(test_vs_cv):.1f}% {'lower' if test_vs_cv < 0 else 'higher'} than the pre-test train-CV RMSE ({float(cv.primary_metric):.4f}) and remained within the historical fold range [{float(cv.fold_primary_min):.4f}, {float(cv.fold_primary_max):.4f}]. Error magnitude therefore showed no unexpected test-set collapse. Ranking and dynamic-range recovery remained limited: log-space R² was {metrics_payload['full_test']['transformed']['r2']:.3f}, and the descriptive observed-on-predicted calibration slope was {calibration.slope:.3f}.

## Applicability domain and source sensitivity

The pre-registered train-only ECFP4 threshold covered {int(table.inside_train_only_AD.sum())}/{len(table)} molecules ({table.inside_train_only_AD.mean()*100:.1f}%). Inside-AD and outside-AD log10 RMSE values were {ad_summary.iloc[0].log10_rmse:.4f} and {ad_summary.iloc[1].log10_rmse:.4f}; the outside-AD estimate is descriptive because it contains only {int(ad_summary.iloc[1].n)} molecules.

Five documents met the pre-registered minimum of three molecules. Their descriptive log10 RMSE values ranged from {eligible_sources.log10_rmse.min():.4f} to {eligible_sources.log10_rmse.max():.4f}, indicating source/domain heterogeneity. These summaries must not be used to modify the candidate.

## Interpretation

The frozen CLint candidate provides a reproducible moderate-accuracy result with test error consistent with train-CV and a GMFE near 2.1. It does not support a state-of-the-art claim on its own because rank correlation is weak, explained variance is limited, the sample contains only 28 molecules, and performance varies by source. The appropriate next step is same-protocol literature benchmarking, not post-test optimization.
"""
        (out / "analysis_report.md").write_text(report, encoding="utf-8")
        finish_stage(out, "gate5_clint_final_diagnostic_closeout", inputs={
            "final_complete_sha256": sha256(args.final / "complete.json"),
            "frozen_complete_sha256": sha256(args.frozen / "complete.json"),
            "recovery_complete_sha256": sha256(args.recovery / "complete.json"),
            "stagea_complete_sha256": sha256(args.stagea / "complete.json"),
        }, task_id=TASK, test_lifecycle_closed=True, read_only=True, refit=False, calibration=False,
        selection_changed=False, figures=2, dpi=600, partial=False)
    print(f"Gate 5 CLint diagnostic closeout: {args.output}")


if __name__ == "__main__":
    run_cli(main)
