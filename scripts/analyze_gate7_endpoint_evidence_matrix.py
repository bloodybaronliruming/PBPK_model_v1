"""Create the read-only Gate 7 endpoint diagnostic evidence matrix.

This stage aggregates already published metric/diagnostic summaries only.  It
does not open any raw test labels, prediction-level tables, models, or R4
technical predictions; it fits no model and calculates no new performance
metric.  Its role is to make claim boundaries and manuscript figure choices
auditable before drafting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate7_endpoint_diagnostic_evidence_matrix"
ORDER = [
    "fu__human__plasma",
    "CLint__human__microsome",
    "Papp__human__caco2_ab",
    "CL__human__systemic_iv",
    "VDss__human__steady_state_iv",
    "Thalf__human__terminal_iv",
    "F__human__absolute_oral",
]
LABELS = {
    "fu__human__plasma": "Plasma fu",
    "CLint__human__microsome": "Microsomal CLint",
    "Papp__human__caco2_ab": "Caco-2 Papp A→B",
    "CL__human__systemic_iv": "Systemic CL (IV)",
    "VDss__human__steady_state_iv": "VDss (IV)",
    "Thalf__human__terminal_iv": "Terminal t½ (IV)",
    "F__human__absolute_oral": "Absolute oral F",
}


def csv(path: Path, columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in published diagnostic table {path}: {sorted(missing)}")
    return frame


def source_paths(root: Path) -> dict[str, tuple[Path, str]]:
    return {
        "gate4": (root / "data/public_development/cross_gate_candidate_freeze_v1", "cross_gate_endpoint_candidate_freeze"),
        "r0": (root / "results/analysis/gate7_release_readiness_audit_v3", "gate7_release_readiness_audit"),
        "r3": (root / "results/analysis/gate7_model_data_card_audit_v1", "gate7_model_data_card_availability_audit"),
        "r4": (root / "results/analysis/gate7_release_cli_smoke_v1", "gate7_dual_lane_batch_inference"),
        "clint": (root / "results/analysis/gate5_clint_final_diagnostic_closeout_v1", "gate5_clint_final_diagnostic_closeout"),
        "legacy": (root / "results/final/frozen_test_diagnostics_v1", "frozen_test_diagnostics"),
        "thalf": (root / "results/final/Thalf_v15_rdkit2d_et3_diagnostics_v2", "thalf_v15_frozen_test_diagnostics"),
        "thalf_source": (root / "results/analysis/thalf_v15_source_sensitivity_v1", "thalf_source_sensitivity_analysis"),
    }


def verify_sources(root: Path) -> dict[str, Path]:
    verified = {}
    for name, (folder, expected_stage) in source_paths(root).items():
        verify_stage(folder, expected_stage)
        verified[name] = folder
    return verified


def scalar(frame: pd.DataFrame, condition: pd.Series, column: str, name: str):
    values = frame.loc[condition, column].tolist()
    if len(values) != 1:
        raise ValueError(f"Expected one published {name} value, found {len(values)}")
    return values[0]


def diagnostic_matrix(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    folders = verify_sources(root)
    gate4 = csv(folders["gate4"] / "endpoint_candidate_freeze_registry.csv", {"task_id", "train_cv_score", "primary_metric"})
    r0 = csv(folders["r0"] / "release_readiness_by_endpoint.csv", {"task_id", "release_lane", "test_lifecycle", "inferential_status", "model_kind"})
    r3 = csv(folders["r3"] / "endpoint_ad_uncertainty_audit.csv", {
        "task_id", "ad_method", "ad_threshold", "ad_release_status", "uncertainty_technical_source", "uncertainty_release_status",
    })
    if set(gate4.task_id) != set(ORDER) or set(r0.task_id) != set(ORDER) or set(r3.task_id) != set(ORDER):
        raise ValueError("Gate 4/R0/R3 endpoint membership does not match the seven-endpoint contract")
    if r0.loc[r0.task_id.eq("CLint__human__microsome"), "test_lifecycle"].item() != "test_consumed_no_reselection":
        raise ValueError("R5 requires the corrected R0 v3 CLint lifecycle")
    r4_meta = json.loads((folders["r4"] / "complete.json").read_text(encoding="utf-8"))
    if not (r4_meta.get("no_labels_accessed") and r4_meta.get("technical_smoke") and r4_meta.get("output_rows") == 14):
        raise ValueError("R4 technical smoke does not establish the required label-free interface evidence")

    legacy_primary = csv(folders["legacy"] / "table_1_bootstrap_primary_ci.csv", {"Task", "Molecules", "Primary metric", "Primary estimate", "Status"})
    legacy_ad = csv(folders["legacy"] / "table_3_applicability_domain.csv", {"Task", "Test molecules", "Median nearest-train Tanimoto", "Molecules below 0.30"})
    legacy_calibration = csv(folders["legacy"] / "table_4_residual_diagnostics.csv", {"Task", "Calibration slope"})
    legacy_tasks = {"fu__human__plasma", "Papp__human__caco2_ab", "CL__human__systemic_iv", "VDss__human__steady_state_iv"}
    if set(legacy_primary.Task) - legacy_tasks - {"Thalf__human__terminal_iv"}:
        raise ValueError("Unexpected legacy frozen-test task in R5 source")

    clint_test = csv(folders["clint"] / "traincv_test_comparison.csv", {"evaluation", "molecules", "log10_rmse"})
    clint_ad = csv(folders["clint"] / "ad_stratum_metrics.csv", {"stratum", "n", "log10_rmse"})
    clint_cal = csv(folders["clint"] / "calibration_summary.csv", {"slope_observed_on_predicted", "status"})
    thalf = csv(folders["thalf"] / "table_1_diagnostic_summary.csv", {
        "molecules", "log10_rmse", "calibration_slope", "median_nearest_train_tanimoto", "molecules_below_tanimoto_0_30",
    })
    thalf_overlap = csv(folders["thalf"] / "table_2_tdc_overlap_diagnostics.csv", {"tdc_overlap", "molecules"})
    thalf_source = csv(folders["thalf_source"] / "table_1b_source_composition.csv", {"split", "source_group", "molecules"})

    frozen = legacy_primary.loc[legacy_primary.Task.isin(legacy_tasks)].set_index("Task")
    frozen_ad = legacy_ad.loc[legacy_ad.Task.isin(legacy_tasks)].set_index("Task")
    frozen_cal = legacy_calibration.loc[legacy_calibration.Task.isin(legacy_tasks)].set_index("Task")
    if set(frozen.index) != legacy_tasks or set(frozen_ad.index) != legacy_tasks or set(frozen_cal.index) != legacy_tasks:
        raise ValueError("The historical frozen-test summary is incomplete")
    clint_row = clint_test.loc[clint_test.evaluation.eq("frozen_test")]
    if len(clint_row) != 1:
        raise ValueError("Expected exactly one already-closed CLint frozen-test summary")
    clint_inside = clint_ad.loc[clint_ad.stratum.eq("inside_train_only_AD")]
    clint_outside = clint_ad.loc[clint_ad.stratum.eq("outside_train_only_AD")]
    if len(clint_inside) != 1 or len(clint_outside) != 1 or len(clint_cal) != 1:
        raise ValueError("Published CLint diagnostic summaries are incomplete")
    if len(thalf) != 1 or int(thalf.molecules.item()) != 53 or int(thalf_overlap.molecules.sum()) != 53:
        raise ValueError("Published terminal t½ v15 diagnostic membership is inconsistent")
    thalf_train = thalf_source.loc[thalf_source.split.eq("train")]
    obach_train = scalar(thalf_train, thalf_train.source_group.eq("obach"), "molecules", "Obach train composition")
    all_train = int(thalf_train.molecules.sum())

    base = gate4.loc[:, ["task_id", "canonical_unit", "train_records", "train_parents", "primary_metric", "train_cv_score"]]
    base = base.merge(r0.loc[:, ["task_id", "release_lane", "test_lifecycle", "inferential_status", "model_kind"]], on="task_id", validate="one_to_one")
    base = base.merge(r3.loc[:, ["task_id", "ad_method", "ad_threshold", "ad_release_status", "uncertainty_technical_source", "uncertainty_release_status"]], on="task_id", validate="one_to_one")
    rows = []
    for task_id in ORDER:
        row = base.loc[base.task_id.eq(task_id)].iloc[0].to_dict()
        row["endpoint_label"] = LABELS[task_id]
        row["diagnostic_source_scope"] = "not_applicable"
        row["published_test_molecules"] = np.nan
        row["published_test_primary_metric"] = "not_available"
        row["published_test_primary_estimate"] = np.nan
        row["published_calibration_slope"] = np.nan
        row["published_median_nearest_train_tanimoto"] = np.nan
        row["published_low_similarity_or_AD_outside_count"] = np.nan
        row["published_low_similarity_or_AD_reference"] = "not_applicable"
        row["source_or_definition_limitation"] = ""
        row["manuscript_interpretation"] = ""
        row["figure_panel"] = ""
        if task_id in legacy_tasks:
            row.update({
                "diagnostic_source_scope": "already-published historical frozen-test summary; not the later Gate-4 train-CV reference",
                "published_test_molecules": int(frozen.loc[task_id, "Molecules"]),
                "published_test_primary_metric": frozen.loc[task_id, "Primary metric"],
                "published_test_primary_estimate": float(frozen.loc[task_id, "Primary estimate"]),
                "published_calibration_slope": float(frozen_cal.loc[task_id, "Calibration slope"]),
                "published_median_nearest_train_tanimoto": float(frozen_ad.loc[task_id, "Median nearest-train Tanimoto"]),
                "published_low_similarity_or_AD_outside_count": int(frozen_ad.loc[task_id, "Molecules below 0.30"]),
                "published_low_similarity_or_AD_reference": "nearest-train Tanimoto <0.30",
                "manuscript_interpretation": "Report as a historical frozen-test result; do not use it to replace the Gate-4 research reference.",
                "figure_panel": "Frozen performance and similarity diagnostics",
            })
            if task_id == "Papp__human__caco2_ab":
                row["source_or_definition_limitation"] = "Caco-2 A→B unit/source sensitivity remains a required limitation."
            elif task_id in {"CL__human__systemic_iv", "VDss__human__steady_state_iv"}:
                row["source_or_definition_limitation"] = "Low-similarity coverage and mean shrinkage require explicit reporting."
            else:
                row["source_or_definition_limitation"] = "Historical test lifecycle is closed; output uncertainty is not calibrated."
        elif task_id == "CLint__human__microsome":
            test = clint_row.iloc[0]
            row.update({
                "diagnostic_source_scope": "Gate-5 one-time frozen-test closeout; pre-registered descriptive diagnostics only",
                "published_test_molecules": int(test.molecules),
                "published_test_primary_metric": "RMSE (log10)",
                "published_test_primary_estimate": float(test.log10_rmse),
                "published_calibration_slope": float(clint_cal.slope_observed_on_predicted.item()),
                "published_median_nearest_train_tanimoto": np.nan,
                "published_low_similarity_or_AD_outside_count": int(clint_outside.n.item()),
                "published_low_similarity_or_AD_reference": "outside pre-registered train-only AD threshold 0.4740325",
                "source_or_definition_limitation": f"AD-outside n={int(clint_outside.n.item())}; document-level heterogeneity is already descriptively reported; no post-test calibration.",
                "manuscript_interpretation": "Use the frozen result with its finite-sample CI and disclose source heterogeneity and dynamic-range compression.",
                "figure_panel": "CLint frozen-test, AD and source diagnostics",
            })
        elif task_id == "Thalf__human__terminal_iv":
            row.update({
                "diagnostic_source_scope": "v15 frozen-test diagnostic; source-overlap limitation retained",
                "published_test_molecules": int(thalf.molecules.item()),
                "published_test_primary_metric": "RMSE (log10)",
                "published_test_primary_estimate": float(thalf.log10_rmse.item()),
                "published_calibration_slope": float(thalf.calibration_slope.item()),
                "published_median_nearest_train_tanimoto": float(thalf.median_nearest_train_tanimoto.item()),
                "published_low_similarity_or_AD_outside_count": int(thalf.molecules_below_tanimoto_0_30.item()),
                "published_low_similarity_or_AD_reference": "nearest-train Tanimoto <0.30",
                "source_or_definition_limitation": (
                    f"Published source-composition table contains {int(obach_train)}/{all_train} Obach molecules before model structure-availability filtering; "
                    "50/53 frozen-test molecules overlap the TDC ecosystem."
                ),
                "manuscript_interpretation": "Report the public benchmark and frozen test as non-independent of TDC/Obach; do not claim independent external validation.",
                "figure_panel": "Terminal t½ source composition and TDC-overlap diagnostic",
            })
        else:
            row.update({
                "diagnostic_source_scope": "exploratory interface only; no numeric model or valid test evaluation",
                "source_or_definition_limitation": "Insufficient independent validation; numeric output and test-performance claim are prohibited.",
                "manuscript_interpretation": "Keep as a status-only exploratory endpoint and report the data gap rather than a predictive result.",
                "figure_panel": "Endpoint scope/status matrix only",
            })
        rows.append(row)
    matrix = pd.DataFrame(rows)
    if len(matrix) != 7 or matrix.task_id.duplicated().any():
        raise ValueError("R5 matrix must have exactly one row per endpoint")

    figure_plan = pd.DataFrame([
        {"figure_id": "Figure 1", "purpose": "Dataset governance and split/lineage workflow", "evidence_scope": "protocol and source/derivation registries", "status": "existing/process archive", "claim_boundary": "Methods only; no efficacy ranking."},
        {"figure_id": "Figure 2", "purpose": "Endpoint evidence and release-readiness matrix", "evidence_scope": "Gate 4, R0–R4 and published diagnostics", "status": "R5 new read-only matrix", "claim_boundary": "Status/limitations only; no cross-protocol performance ranking."},
        {"figure_id": "Figure 3", "purpose": "Matched strong-STL versus architecture ablations", "evidence_scope": "Gate 4 ledger and Gate 3 B6", "status": "existing analysis", "claim_boundary": "Within-project matched comparisons only."},
        {"figure_id": "Figure 4", "purpose": "Frozen endpoint performance, similarity and calibration diagnostics", "evidence_scope": "existing frozen-test and Gate-5 closeout summaries", "status": "existing figures plus table assembly", "claim_boundary": "Closed test results; no post-test selection or calibration."},
        {"figure_id": "Figure 5", "purpose": "Cross-species and multitask positive/negative findings", "evidence_scope": "completed matched train-CV analyses", "status": "existing analysis", "claim_boundary": "Do not claim replacement of strong STL without the registered gate."},
        {"figure_id": "Figure 6", "purpose": "Terminal t½ public benchmark and source-overlap limitation", "evidence_scope": "TDC benchmark, v15 source and frozen diagnostics", "status": "existing analysis plus limitation callout", "claim_boundary": "Do not call the v15 frozen test independent external validation."},
    ])
    source_table = pd.DataFrame([
        {"source_key": key, "path": str(folder), "stage": source_paths(root)[key][1], "use": use}
        for key, folder, use in [
            ("gate4", folders["gate4"], "train-CV reference and endpoint definition only"),
            ("r0", folders["r0"], "current asset lane and lifecycle truth"),
            ("r3", folders["r3"], "AD/uncertainty availability only"),
            ("r4", folders["r4"], "label-free CLI smoke status only; no predictions opened"),
            ("clint", folders["clint"], "already-closed CLint diagnostic summary tables only"),
            ("legacy", folders["legacy"], "already-published frozen-test summary tables only; terminal t½ legacy row excluded"),
            ("thalf", folders["thalf"], "already-published v15 diagnostic summary tables only"),
            ("thalf_source", folders["thalf_source"], "published source-composition table only"),
        ]
    ])
    inputs = {str((folder / "complete.json").resolve()): sha256(folder / "complete.json") for folder in folders.values()}
    return matrix, figure_plan, source_table, {"inputs": inputs, "r4": r4_meta}


def render_matrix(matrix: pd.DataFrame, path: Path) -> None:
    columns = ["Bound asset", "Closed/protected\nlifecycle", "Technical\ninterface", "AD status", "Uncertainty", "Independent\nexternal claim"]
    states = []
    text = []
    for row in matrix.itertuples(index=False):
        is_f = row.task_id == "F__human__absolute_oral"
        states.append([
            2 if not is_f else 0,
            2 if row.test_lifecycle == "test_consumed_no_reselection" else 0,
            1 if not is_f else 0,
            2 if row.ad_release_status == "pre_registered_train_only_AD_available" else (1 if not is_f else 0),
            1 if "available_not_calibrated" in row.uncertainty_release_status else 0,
            0 if is_f else 1,
        ])
        text.append([
            "bound" if not is_f else "none",
            "closed" if not is_f else "protected",
            "technical" if not is_f else "status only",
            "pre-reg" if row.ad_release_status == "pre_registered_train_only_AD_available" else ("technical" if not is_f else "N/A"),
            "uncal." if "available_not_calibrated" in row.uncertainty_release_status else "none",
            "not established" if not is_f else "prohibited",
        ])
    colors = {0: "#D9D9D9", 1: "#F4B183", 2: "#70AD47"}
    fig, ax = plt.subplots(figsize=(12.0, 5.6), constrained_layout=True)
    for i, (state_row, label_row) in enumerate(zip(states, text)):
        for j, (state, label) in enumerate(zip(state_row, label_row)):
            ax.add_patch(plt.Rectangle((j, len(states) - 1 - i), 1, 1, facecolor=colors[state], edgecolor="white", linewidth=1.5))
            ax.text(j + 0.5, len(states) - 0.5 - i, label, ha="center", va="center", fontsize=8.5)
    ax.set_xlim(0, len(columns))
    ax.set_ylim(0, len(states))
    ax.set_xticks(np.arange(len(columns)) + 0.5, columns, fontsize=10)
    ax.set_yticks(np.arange(len(states)) + 0.5, list(reversed(matrix.endpoint_label.tolist())), fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Gate 7 endpoint evidence and release-readiness matrix", fontsize=14, weight="bold", pad=12)
    fig.text(0.5, 0.005, "Green: verified within its stated scope; orange: technical/limited; grey: unavailable, protected or not established.", ha="center", fontsize=9)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-read-only-evidence-synthesis", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "results/analysis/gate7_endpoint_evidence_matrix_v3"
    matrix, figure_plan, sources, payload = diagnostic_matrix(root)
    if args.check_only:
        print(f"Gate 7 R5 preflight: endpoints={len(matrix)}; published summaries only; labels/predictions not read.")
        return
    if not args.confirm_read_only_evidence_synthesis:
        raise ValueError("Gate 7 R5 requires --confirm-read-only-evidence-synthesis")
    with stage_output(output) as folder:
        matrix.to_csv(folder / "endpoint_diagnostic_evidence_matrix.csv", index=False)
        figure_plan.to_csv(folder / "manuscript_figure_plan.csv", index=False)
        sources.to_csv(folder / "read_only_source_registry.csv", index=False)
        render_matrix(matrix, folder / "Figure_1_gate7_evidence_matrix.png")
        folder.joinpath("README.md").write_text(
            "# Gate 7 R5 endpoint diagnostic evidence matrix\n\n"
            "This is a read-only synthesis of existing, completed summary tables. It opens no raw label, prediction, "
            "or model artifact and computes no new performance metric. The matrix deliberately separates Gate-4 train-CV "
            "research references from historical frozen-test assets, retains closed test lifecycle boundaries, and makes "
            "F status-only. The PNG is an English 600-dpi manuscript-planning/status figure, not a performance ranking.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=payload["inputs"], partial=False, no_model_fitted=True,
            no_labels_accessed=True, no_prediction_rows_accessed=True, no_performance_metrics_computed=True,
            no_calibration=True, no_candidate_selection_changed=True, read_only_evidence_synthesis=True,
            endpoint_count=int(len(matrix)), figure_dpi=600, unified_numeric_release_authorized=False,
        )
    print(f"Gate 7 R5 endpoint evidence matrix: {output}")


if __name__ == "__main__":
    run_cli(main)
