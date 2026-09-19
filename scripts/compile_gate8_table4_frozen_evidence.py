"""Freeze and compile Gate 8 Table 4 from R0-approved diagnostic summaries.

This stage assembles frozen-test/AD/calibration and clean-reload engineering
evidence only. It does not open raw labels, prediction rows, models, fold
members, or test data, and it does not recompute metrics or select/calibrate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_table4_frozen_evidence"
R0_STAGE = "gate8_manuscript_evidence_registry"
R0_OUTPUT = "data/public_development/gate8_manuscript_evidence_registry_v2"
OUTPUT = "data/public_development/gate8_table4_frozen_evidence_v1"
ENDPOINT_ORDER = [
    "fu__human__plasma", "CLint__human__microsome", "Papp__human__caco2_ab",
    "CL__human__systemic_iv", "VDss__human__steady_state_iv", "Thalf__human__terminal_iv",
    "F__human__absolute_oral",
]
LEGACY_TASKS = {"fu__human__plasma", "Papp__human__caco2_ab", "CL__human__systemic_iv", "VDss__human__steady_state_iv"}


def required_asset(manifest: pd.DataFrame, source_id: str, artifact: str) -> pd.Series:
    found = manifest.loc[manifest.source_id.eq(source_id) & manifest.artifact.eq(artifact)]
    if len(found) != 1:
        raise ValueError(f"R0 must contain exactly one Table 4 asset: {source_id}/{artifact}")
    return found.iloc[0]


def approved_assets(root: Path) -> tuple[dict[str, Path], pd.DataFrame, dict[str, str]]:
    r0 = root / R0_OUTPUT
    r0_meta = verify_stage(r0, R0_STAGE)
    flags = (
        "no_model_loaded", "no_labels_accessed", "no_prediction_rows_accessed",
        "no_performance_metrics_computed", "no_model_fitted", "no_calibration",
        "no_candidate_selection_changed",
    )
    if not all(r0_meta.get(flag) is True for flag in flags):
        raise ValueError("R0 does not satisfy the required no-label/no-reselection boundary")
    manifest_path = r0 / "source_artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    selected = {
        "matrix": required_asset(manifest, "gate7_evidence_matrix", "endpoint_diagnostic_evidence_matrix.csv"),
        "clint_test": required_asset(manifest, "clint_gate5_closeout", "traincv_test_comparison.csv"),
        "clint_calibration": required_asset(manifest, "clint_gate5_closeout", "calibration_summary.csv"),
        "legacy_primary": required_asset(manifest, "frozen_test_diagnostics", "table_1_bootstrap_primary_ci.csv"),
        "legacy_ad": required_asset(manifest, "frozen_test_diagnostics", "table_3_applicability_domain.csv"),
        "delivery": required_asset(manifest, "gate7_reproducibility", "delivery_manifest.json"),
        "reload": required_asset(manifest, "gate7_clean_reload", "verification_checks.json"),
    }
    paths: dict[str, Path] = {}
    inputs = {
        str((r0 / "complete.json").resolve()): sha256(r0 / "complete.json"),
        str(manifest_path.resolve()): sha256(manifest_path),
    }
    folders: set[Path] = set()
    for key, row in selected.items():
        folder = Path(row.source_path)
        if folder not in folders:
            verify_stage(folder, row.source_stage)
            inputs[str((folder / "complete.json").resolve())] = sha256(folder / "complete.json")
            folders.add(folder)
        path = folder / row.artifact
        if not path.is_file() or sha256(path) != row.artifact_sha256:
            raise ValueError(f"R0-approved Table 4 asset changed after registration: {path}")
        paths[key] = path
        inputs[str(path.resolve())] = sha256(path)
    return paths, manifest, inputs


def endpoint_schema() -> pd.DataFrame:
    rows = [
        ("task_id", "Task ID", "string", "yes", "Gate 7 matrix", "copied", "Internal key; not a comparison class."),
        ("endpoint", "Endpoint", "string", "yes", "Gate 7 matrix", "copied", "Definition/unit are governed by Table 1."),
        ("evidence_class", "Evidence class", "string", "yes", "compiler", "literal not_comparison", "Frozen diagnostics are not a model-selection comparison."),
        ("evidence_lane", "Evidence lane", "string", "yes", "Gate 7 matrix", "copied", "Historical test-locked and current frozen-final lanes remain distinct."),
        ("test_lifecycle", "Test lifecycle", "string", "yes", "Gate 7 matrix", "copied", "No consumed/prohibited test may be reused."),
        ("diagnostic_status", "Diagnostic status", "string", "yes", "Gate 7 matrix", "copied", "Do not call documented-overlap evidence independent external validation."),
        ("test_molecules", "Frozen-test molecules (n)", "string", "yes", "approved summaries", "copied or not_applicable", "No test membership is read or reconstructed."),
        ("primary_metric", "Primary metric", "string", "yes", "approved summaries", "copied or not_available", "No cross-endpoint ranking."),
        ("primary_estimate", "Primary estimate", "string", "yes", "approved summaries", "copied or not_available", "Existing descriptive result only; no post-test selection."),
        ("frozen_95ci", "Frozen 95% CI", "string", "yes", "approved summaries", "copied or unavailable", "Unavailable is reported rather than recalculated."),
        ("ad_or_similarity", "AD/similarity evidence", "string", "yes", "approved summaries", "copied or not_applicable", "AD availability does not establish calibrated uncertainty."),
        ("calibration_status", "Calibration status", "string", "yes", "approved summaries", "copied or unavailable", "No post-test calibration is fitted."),
        ("uncertainty_status", "Uncertainty status", "string", "yes", "Gate 7 matrix", "copied", "Technical spread is not a calibrated predictive interval."),
        ("source_definition_limitation", "Source/definition limitation", "string", "yes", "Gate 7 matrix", "copied", "Must remain visible in manuscript text."),
        ("source_artifacts", "Source artifacts", "string", "yes", "R0 manifest", "exact asset list", "Only R0-approved artifacts are admitted."),
        ("claim_guardrail", "Claim guardrail", "string", "yes", "compiler", "fixed", "No post-test selection, calibration, or external-independence overclaim."),
    ]
    return pd.DataFrame(rows, columns=["column_id", "display_header", "dtype", "required", "allowed_source", "assembly_rule", "claim_guardrail"])


def engineering_schema() -> pd.DataFrame:
    rows = [
        ("evidence_id", "Evidence ID", "string", "yes", "compiler", "fixed", "Technical evidence only."),
        ("evidence_class", "Evidence class", "string", "yes", "compiler", "literal not_comparison", "Not performance evidence."),
        ("scope", "Scope", "string", "yes", "Gate 7 delivery manifest", "copied", "Not a unified numeric release."),
        ("environment", "Environment", "string", "yes", "Gate 7 delivery manifest", "copied", "Exact dependency restoration details remain in manifest."),
        ("technical_output_contract", "Technical output contract", "string", "yes", "delivery/reload summaries", "copied", "No label-bearing input or performance score is involved."),
        ("replay_result", "Clean-reload result", "string", "yes", "clean-reload verification", "copied", "Technical replay does not imply predictive validation."),
        ("numeric_tolerance", "Numeric tolerance", "string", "yes", "clean-reload verification", "copied", "A zero difference is a diagnostic, not a boolean check."),
        ("source_artifacts", "Source artifacts", "string", "yes", "R0 manifest", "exact asset list", "Only R0-approved artifacts are admitted."),
        ("claim_guardrail", "Claim guardrail", "string", "yes", "compiler", "fixed", "Do not infer endpoint performance or calibrated uncertainty."),
    ]
    return pd.DataFrame(rows, columns=["column_id", "display_header", "dtype", "required", "allowed_source", "assembly_rule", "claim_guardrail"])


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def compile_endpoint_table(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(paths["matrix"])
    legacy_primary = pd.read_csv(paths["legacy_primary"])
    legacy_ad = pd.read_csv(paths["legacy_ad"])
    clint_test = pd.read_csv(paths["clint_test"])
    clint_calibration = pd.read_csv(paths["clint_calibration"])
    matrix_columns = {
        "task_id", "endpoint_label", "release_lane", "test_lifecycle", "inferential_status",
        "published_test_molecules", "published_test_primary_metric", "published_test_primary_estimate",
        "published_calibration_slope", "ad_method", "ad_threshold", "ad_release_status",
        "uncertainty_release_status", "published_median_nearest_train_tanimoto",
        "published_low_similarity_or_AD_outside_count", "published_low_similarity_or_AD_reference",
        "source_or_definition_limitation",
    }
    if not matrix_columns <= set(matrix.columns):
        raise ValueError("Gate 7 matrix does not satisfy Table 4 schema")
    if set(matrix.task_id) != set(ENDPOINT_ORDER) or matrix.task_id.duplicated().any():
        raise ValueError("Table 4 requires exactly the frozen seven-endpoint matrix")
    if not {"Task", "Molecules", "Primary metric", "Primary estimate", "95% CI low", "95% CI high", "Status"} <= set(legacy_primary.columns):
        raise ValueError("Legacy frozen primary summary lacks Table 4 fields")
    if not {"Task", "Test molecules", "Median nearest-train Tanimoto", "Molecules below 0.30"} <= set(legacy_ad.columns):
        raise ValueError("Legacy AD summary lacks Table 4 fields")
    if not {"evaluation", "molecules", "log10_rmse", "rmse_lower_reference", "rmse_upper_reference", "selection_role"} <= set(clint_test.columns):
        raise ValueError("CLint frozen summary lacks Table 4 fields")
    if len(clint_calibration) != 1 or "status" not in clint_calibration or "slope_observed_on_predicted" not in clint_calibration:
        raise ValueError("CLint calibration summary lacks Table 4 fields")

    legacy_primary = legacy_primary.set_index("Task")
    legacy_ad = legacy_ad.set_index("Task")
    if not LEGACY_TASKS <= set(legacy_primary.index) or not LEGACY_TASKS <= set(legacy_ad.index):
        raise ValueError("Legacy approved summaries do not contain the four intended historical endpoints")
    if "Thalf__human__terminal_iv" not in legacy_primary.index or "Thalf__human__terminal_iv" not in legacy_ad.index:
        raise ValueError("Expected legacy t½ n=2 row is absent; exclusion lineage cannot be documented")
    legacy_thalf = legacy_primary.loc["Thalf__human__terminal_iv"]
    if int(legacy_thalf["Molecules"]) != 2 or legacy_thalf["Status"] != "exploratory_n2":
        raise ValueError("Legacy terminal t½ row no longer matches its registered exploratory n=2 limitation")
    clint_frozen = clint_test.loc[clint_test.evaluation.eq("frozen_test")]
    if len(clint_frozen) != 1 or clint_frozen.selection_role.item() != "confirmatory_no_reselection":
        raise ValueError("CLint Table 4 requires exactly one closed frozen-test summary")

    rows = []
    for task_id in ENDPOINT_ORDER:
        row = matrix.loc[matrix.task_id.eq(task_id)].iloc[0]
        base = {
            "task_id": task_id, "endpoint": row.endpoint_label, "evidence_class": "not_comparison",
            "evidence_lane": row.release_lane, "test_lifecycle": row.test_lifecycle,
            "diagnostic_status": row.inferential_status, "uncertainty_status": row.uncertainty_release_status,
            "source_definition_limitation": row.source_or_definition_limitation,
        }
        if task_id in LEGACY_TASKS:
            primary = legacy_primary.loc[task_id]
            ad = legacy_ad.loc[task_id]
            if int(primary["Molecules"]) != int(row.published_test_molecules) or int(ad["Test molecules"]) != int(row.published_test_molecules):
                raise ValueError(f"Historical frozen membership disagreement for {task_id}")
            if abs(float(primary["Primary estimate"]) - float(row.published_test_primary_estimate)) > 1e-12:
                raise ValueError(f"Historical frozen metric disagreement for {task_id}")
            base.update({
                "test_molecules": str(int(primary["Molecules"])), "primary_metric": primary["Primary metric"],
                "primary_estimate": fmt(primary["Primary estimate"]),
                "frozen_95ci": f"[{fmt(primary['95% CI low'])}, {fmt(primary['95% CI high'])}]",
                "ad_or_similarity": (
                    f"median nearest-train Tanimoto={fmt(ad['Median nearest-train Tanimoto'])}; "
                    f"below 0.30={int(ad['Molecules below 0.30'])}"
                ),
                "calibration_status": f"published slope={fmt(row.published_calibration_slope)}; no post-test calibration",
                "source_artifacts": "frozen_test_diagnostics/table_1_bootstrap_primary_ci.csv; table_3_applicability_domain.csv; gate7_evidence_matrix/endpoint_diagnostic_evidence_matrix.csv",
                "claim_guardrail": "Historical frozen diagnostic only; do not use for later candidate selection or calibrated-interval claims.",
            })
        elif task_id == "CLint__human__microsome":
            frozen = clint_frozen.iloc[0]
            if int(frozen.molecules) != int(row.published_test_molecules) or abs(float(frozen.log10_rmse) - float(row.published_test_primary_estimate)) > 1e-12:
                raise ValueError("CLint closeout and Gate 7 matrix disagree")
            cal = clint_calibration.iloc[0]
            base.update({
                "test_molecules": str(int(frozen.molecules)), "primary_metric": "RMSE (log10)",
                "primary_estimate": fmt(frozen.log10_rmse),
                "frozen_95ci": f"[{fmt(frozen.rmse_lower_reference)}, {fmt(frozen.rmse_upper_reference)}]",
                "ad_or_similarity": f"{row.ad_method}; threshold={fmt(row.ad_threshold)}; outside={int(row.published_low_similarity_or_AD_outside_count)}",
                "calibration_status": f"slope={fmt(cal.slope_observed_on_predicted)}; {cal.status}",
                "source_artifacts": "clint_gate5_closeout/traincv_test_comparison.csv; calibration_summary.csv; gate7_evidence_matrix/endpoint_diagnostic_evidence_matrix.csv",
                "claim_guardrail": "One-time closed frozen test; diagnostic-only calibration summary and no post-test model change.",
            })
        elif task_id == "Thalf__human__terminal_iv":
            base.update({
                "test_molecules": str(int(row.published_test_molecules)), "primary_metric": row.published_test_primary_metric,
                "primary_estimate": fmt(row.published_test_primary_estimate),
                "frozen_95ci": "not_available in R0-approved summary assets",
                "ad_or_similarity": (
                    f"{row.published_low_similarity_or_AD_reference}; "
                    f"count={int(row.published_low_similarity_or_AD_outside_count)}; "
                    f"median={fmt(row.published_median_nearest_train_tanimoto)}"
                ),
                "calibration_status": f"published slope={fmt(row.published_calibration_slope)}; no post-test calibration",
                "source_artifacts": "gate7_evidence_matrix/endpoint_diagnostic_evidence_matrix.csv",
                "claim_guardrail": "Use the v15 n=53 overlap-limited summary only; legacy n=2 exploratory t½ row is excluded and no independent external-validation claim is allowed.",
            })
        else:
            base.update({
                "test_molecules": "not_available", "primary_metric": "not_available", "primary_estimate": "not_available",
                "frozen_95ci": "not_available", "ad_or_similarity": "not_applicable",
                "calibration_status": "not_applicable", "source_artifacts": "gate7_evidence_matrix/endpoint_diagnostic_evidence_matrix.csv",
                "claim_guardrail": "Exploratory status-only endpoint; numeric prediction and test-performance claims are prohibited.",
            })
        rows.append(base)
    output = pd.DataFrame(rows)
    if len(output) != 7 or output.isna().any().any() or set(output.evidence_class) != {"not_comparison"}:
        raise ValueError("Endpoint panel must contain seven complete non-comparison rows")
    exclusions = pd.DataFrame([{
        "excluded_source_row": "frozen_test_diagnostics/table_1_bootstrap_primary_ci.csv::Thalf__human__terminal_iv",
        "reason": "Legacy exploratory n=2 terminal-half-life summary is incompatible with the R0-approved Gate 7 v15 n=53 overlap-limited diagnostic scope.",
        "replacement_in_endpoint_panel": "gate7_evidence_matrix/endpoint_diagnostic_evidence_matrix.csv::Thalf__human__terminal_iv",
        "claim_guardrail": "Do not merge, pool, or numerically rank the legacy n=2 and v15 n=53 terminal-half-life summaries.",
    }])
    return output, exclusions


def compile_engineering_table(paths: dict[str, Path]) -> pd.DataFrame:
    delivery = json.loads(paths["delivery"].read_text(encoding="utf-8"))
    reload_checks = json.loads(paths["reload"].read_text(encoding="utf-8"))
    if delivery.get("model_or_label_accessed") is not False or delivery.get("scope") != "Gate 7 technical dual-lane inference only; not a unified numeric release":
        raise ValueError("Delivery manifest does not meet the frozen technical-only scope")
    checks = reload_checks.get("checks", {})
    bool_names = reload_checks.get("boolean_check_names", [])
    if not bool_names or not all(checks.get(name) is True for name in bool_names):
        raise ValueError("Clean reload verification has a failed boolean check")
    if checks.get("numeric_output_count") != 12 or checks.get("max_abs_cross_process_prediction_difference") != 0.0:
        raise ValueError("Clean reload numerical diagnostics differ from the frozen R6 verification")
    audit = reload_checks.get("audit", {})
    if audit.get("cross_process_prediction_tolerance") != 1e-12 or audit.get("in_process_repeat_tolerance") != 1e-12:
        raise ValueError("Clean reload tolerances differ from the frozen contract")
    return pd.DataFrame([{
        "evidence_id": "gate7_r6_clean_reload", "evidence_class": "not_comparison",
        "scope": delivery["scope"], "environment": delivery["active_environment"],
        "technical_output_contract": "14 technical output rows; 12 numeric outputs; F status-only; label-free input contract.",
        "replay_result": "All 15 boolean checks passed; schema/non-prediction fields matched; cross-process numeric maximum difference=0.0.",
        "numeric_tolerance": "cross-process=1e-12; in-process repeat=1e-12",
        "source_artifacts": "gate7_reproducibility/delivery_manifest.json; gate7_clean_reload/verification_checks.json",
        "claim_guardrail": "Technical reproducibility only; it neither estimates endpoint performance nor authorizes a unified numeric release.",
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT
    paths, manifest, inputs = approved_assets(root)
    endpoint = compile_endpoint_table(paths)
    engineering = compile_engineering_table(paths)
    endpoint_columns, exclusions = endpoint
    source_contract = manifest.loc[
        ((manifest.source_id.eq("gate7_evidence_matrix")) & manifest.artifact.eq("endpoint_diagnostic_evidence_matrix.csv"))
        | ((manifest.source_id.eq("clint_gate5_closeout")) & manifest.artifact.isin(["traincv_test_comparison.csv", "calibration_summary.csv"]))
        | ((manifest.source_id.eq("frozen_test_diagnostics")) & manifest.artifact.isin(["table_1_bootstrap_primary_ci.csv", "table_3_applicability_domain.csv"]))
        | ((manifest.source_id.eq("gate7_reproducibility")) & manifest.artifact.eq("delivery_manifest.json"))
        | ((manifest.source_id.eq("gate7_clean_reload")) & manifest.artifact.eq("verification_checks.json"))
    ].copy()
    if len(source_contract) != 7:
        raise ValueError("Table 4 must bind exactly seven R0-approved diagnostic/engineering assets")
    if args.check_only:
        print("Gate 8 R3 preflight: endpoint_rows=7 engineering_rows=1 exclusions=1 approved_assets=7; no labels, prediction rows, models, folds, tests, or metrics read.")
        return
    with stage_output(output) as folder:
        endpoint_schema().to_csv(folder / "table_4_endpoint_schema.csv", index=False)
        engineering_schema().to_csv(folder / "table_4_engineering_schema.csv", index=False)
        endpoint_columns.to_csv(folder / "Table_4_frozen_diagnostic_evidence.csv", index=False)
        engineering.to_csv(folder / "Table_4_engineering_reproducibility_evidence.csv", index=False)
        exclusions.to_csv(folder / "table_4_exclusion_lineage.csv", index=False)
        source_contract.to_csv(folder / "table_4_source_contract.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R3 Table 4 frozen diagnostic and engineering evidence\n\n"
            "This directory compiles only R0-approved summaries: frozen-test/AD/calibration status and Gate 7 clean-reload engineering evidence. "
            "It does not open raw labels, predictions, models, fold members, or tests and does not recompute metrics, fit, calibrate or select. "
            "The legacy terminal t½ n=2 exploratory row is documented as excluded; the endpoint panel instead reports the R0-approved v15 n=53 "
            "source-overlap-limited summary from the Gate 7 matrix. These records must never be pooled. All endpoint rows are non-comparison evidence; "
            "technical replay must not be interpreted as performance validation or a unified numeric release.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_fold_members_accessed=True, no_test_labels_accessed=True,
            no_performance_metrics_computed=True, no_bootstrap_recomputed=True, no_model_fitted=True,
            no_calibration=True, no_candidate_selection_changed=True, endpoint_row_count=len(endpoint_columns),
            engineering_row_count=len(engineering), exclusion_count=len(exclusions), approved_asset_count=len(source_contract),
            evidence_class="not_comparison_only", source_registry_version="gate8_manuscript_evidence_registry_v2",
        )
    print(f"Gate 8 R3 Table 4 frozen diagnostic and engineering evidence: {output}")


if __name__ == "__main__":
    run_cli(main)
