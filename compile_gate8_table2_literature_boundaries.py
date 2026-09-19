"""Freeze Gate 8 Table 2 with separated Jia B/C evidence and MMPK boundaries.

Only R0-approved Jia summary tables are transformed.  Each formerly paired
published/author-like value is emitted as separate B or C rows; differences are
never exported. MMPK assets are hash-verified but never read for values or
written as public comparison rows, because they remain internal-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_table2_literature_boundaries"
R0_STAGE = "gate8_manuscript_evidence_registry"
R0_OUTPUT = "data/public_development/gate8_manuscript_evidence_registry_v2"
OUTPUT = "data/public_development/gate8_table2_literature_boundaries_v1"


def required_asset(manifest: pd.DataFrame, source_id: str, artifact: str) -> pd.Series:
    result = manifest.loc[manifest.source_id.eq(source_id) & manifest.artifact.eq(artifact)]
    if len(result) != 1:
        raise ValueError(f"R0 must register exactly one Table 2 asset: {source_id}/{artifact}")
    return result.iloc[0]


def approved_assets(root: Path) -> tuple[dict[str, Path], pd.DataFrame, dict[str, str]]:
    r0 = root / R0_OUTPUT
    r0_meta = verify_stage(r0, R0_STAGE)
    flags = (
        "no_model_loaded", "no_labels_accessed", "no_prediction_rows_accessed",
        "no_performance_metrics_computed", "no_model_fitted", "no_calibration",
        "no_candidate_selection_changed",
    )
    if not all(r0_meta.get(flag) is True for flag in flags):
        raise ValueError("R0 does not meet the required no-label/no-reselection boundary")
    manifest_path = r0 / "source_artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    selected = {
        "jia_paired": required_asset(manifest, "jia_author_like_closeout", "published_vs_author_like_primary.csv"),
        "jia_strict": required_asset(manifest, "jia_author_like_closeout", "project_strict_protocol_context.csv"),
        "mmpk_decision": required_asset(manifest, "mmpk_n2c_internal", "sensitivity_signal_decision.csv"),
        "mmpk_metrics": required_asset(manifest, "mmpk_n2c_internal", "pooled_paired_metrics.csv"),
    }
    paths: dict[str, Path] = {}
    inputs = {
        str((r0 / "complete.json").resolve()): sha256(r0 / "complete.json"),
        str(manifest_path.resolve()): sha256(manifest_path),
    }
    checked_folders: set[Path] = set()
    for key, row in selected.items():
        folder = Path(row.source_path)
        if folder not in checked_folders:
            verify_stage(folder, row.source_stage)
            inputs[str((folder / "complete.json").resolve())] = sha256(folder / "complete.json")
            checked_folders.add(folder)
        path = folder / row.artifact
        if not path.is_file() or sha256(path) != row.artifact_sha256:
            raise ValueError(f"R0-approved Table 2 asset changed after registration: {path}")
        paths[key] = path
        inputs[str(path.resolve())] = sha256(path)
    return paths, manifest, inputs


def public_schema() -> pd.DataFrame:
    rows = [
        ("evidence_row_id", "Evidence row ID", "string", "yes", "compiler", "deterministic source row/level key", "Traceability only."),
        ("evidence_section", "Evidence section", "string", "yes", "compiler", "published C / author-like B / strict-context B", "Rows from sections cannot be numerically ranked across protocols."),
        ("endpoint", "Endpoint", "string", "yes", "Jia summaries", "copied", "No claim of endpoint-definition equivalence outside its stated protocol."),
        ("model_id", "Model ID", "string", "yes", "Jia paired summary/compiler", "copied or project strict context label", "Identifier is not an endorsement or model-selection result."),
        ("metric", "Metric", "string", "yes", "Jia summaries", "copied", "Metric values are not transformed or recomputed."),
        ("reported_value", "Reported value", "string", "yes", "Jia summaries", "copied separately by level", "No difference or pooled comparison is exported."),
        ("evidence_level", "Evidence level", "string", "yes", "compiler", "literal B or C", "A-level is prohibited because an exact matched protocol is not established."),
        ("comparison_id", "Comparison ID", "string", "yes", "compiler", "fixed source/protocol key", "Numerical comparison is allowed only within the same frozen comparison_id, if separately pre-specified."),
        ("protocol_scope", "Protocol scope", "string", "yes", "Jia summaries/compiler", "fixed disclosure", "Author-like and project-strict rows are B-level, not direct reproduction claims."),
        ("source_artifact", "Source artifact", "string", "yes", "R0 manifest", "exact approved asset", "No non-R0 artifact is permitted."),
        ("claim_guardrail", "Claim guardrail", "string", "yes", "compiler", "fixed", "Never rank B and C rows together or call B author-like results A-level reproduction."),
    ]
    return pd.DataFrame(rows, columns=["column_id", "display_header", "dtype", "required", "allowed_source", "assembly_rule", "claim_guardrail"])


def internal_schema() -> pd.DataFrame:
    rows = [
        ("source_family", "Source family", "string", "yes", "R0 manifest", "literal MMPK internal source", "No numerical endpoint content allowed."),
        ("release_status", "Release status", "string", "yes", "compiler", "literal internal_only", "Not A/B/C comparison evidence."),
        ("permitted_use", "Permitted use", "string", "yes", "R0 allowed usage", "copied", "Internal methodology/sensitivity discussion only."),
        ("prohibited_outputs", "Prohibited outputs", "string", "yes", "compiler", "fixed", "No public comparison row, row-level data, predictions, or redistributable model."),
        ("bound_artifacts", "Bound artifacts", "string", "yes", "R0 manifest", "artifact names only; no values", "Hash-bound lineage only."),
    ]
    return pd.DataFrame(rows, columns=["column_id", "display_header", "dtype", "required", "allowed_source", "assembly_rule", "claim_guardrail"])


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def compile_jia_public_table(paths: dict[str, Path]) -> pd.DataFrame:
    paired = pd.read_csv(paths["jia_paired"])
    strict = pd.read_csv(paths["jia_strict"])
    paired_fields = {
        "endpoint", "model_id", "metric", "published_jia2025", "author_like_r3b",
        "difference_author_like_minus_published", "comparison_level",
    }
    strict_fields = {"endpoint", "train_records", "train_parents", "train_cv_score", "primary_metric", "comparison_level_current_project", "test_lifecycle"}
    if not paired_fields <= set(paired.columns) or not strict_fields <= set(strict.columns):
        raise ValueError("Jia approved summary schema no longer satisfies Table 2 contract")
    if paired.empty or paired.duplicated(["endpoint", "model_id", "metric"]).any():
        raise ValueError("Jia paired summary must contain unique endpoint/model/metric rows")
    if set(paired.comparison_level) != {"B_author_like_same_workbook_author_membership"}:
        raise ValueError("Jia author-like source does not retain its B-level membership boundary")
    if set(strict.comparison_level_current_project) != {"B"} or strict.endpoint.duplicated().any():
        raise ValueError("Project strict context must remain unique B-level context")
    rows = []
    for index, source in paired.reset_index(drop=True).iterrows():
        base = {
            "endpoint": source.endpoint, "model_id": source.model_id, "metric": source.metric,
            "source_artifact": "jia_author_like_closeout/published_vs_author_like_primary.csv",
        }
        rows.append({
            "evidence_row_id": f"C{index + 1}", "evidence_section": "A. Jia 2025 published literature background",
            **base, "reported_value": fmt(source.published_jia2025), "evidence_level": "C",
            "comparison_id": "jia2025_published_literature_background",
            "protocol_scope": "Published Jia 2025 value; background context only, with no claim of identical data/split/input protocol.",
            "claim_guardrail": "C-level literature context only; do not rank against project B-level rows or claim direct replication.",
        })
        rows.append({
            "evidence_row_id": f"B{index + 1}", "evidence_section": "B. Project author-like rerun",
            **base, "reported_value": fmt(source.author_like_r3b), "evidence_level": "B",
            "comparison_id": "jia2025_author_like_same_workbook_author_membership",
            "protocol_scope": "Project author-like rerun using the Jia workbook author membership; not an exact independent or strict-project reproduction.",
            "claim_guardrail": "B-level author-like result; do not merge with the published C value or promote to A-level reproduction.",
        })
    for index, source in strict.reset_index(drop=True).iterrows():
        rows.append({
            "evidence_row_id": f"S{index + 1}", "evidence_section": "C. Project strict-protocol context",
            "endpoint": source.endpoint, "model_id": "project_strict_protocol_reference", "metric": source.primary_metric,
            "reported_value": fmt(source.train_cv_score), "evidence_level": "B",
            "comparison_id": "project_strict_endpoint_context",
            "protocol_scope": (
                f"Project strict endpoint context (train records={int(source.train_records)}; canonical parents={int(source.train_parents)}; "
                f"test lifecycle={source.test_lifecycle}); not numerically comparable to Jia author-like membership."
            ),
            "source_artifact": "jia_author_like_closeout/project_strict_protocol_context.csv",
            "claim_guardrail": "B-level project context only; do not rank against Jia C or author-like B rows across different comparison_id values.",
        })
    table = pd.DataFrame(rows)
    if len(table) != len(paired) * 2 + len(strict) or table.isna().any().any():
        raise ValueError("Table 2 public table has incomplete separated evidence rows")
    if not set(table.evidence_level) <= {"B", "C"} or "A" in set(table.evidence_level):
        raise ValueError("Table 2 public rows must have unique B/C levels and no A-level claim")
    if table.reported_value.str.contains("difference", case=False).any():
        raise ValueError("Table 2 must not export paired difference values")
    return table


def compile_mmpk_boundary(manifest: pd.DataFrame) -> pd.DataFrame:
    mmpk = manifest.loc[manifest.source_id.eq("mmpk_n2c_internal")].copy()
    expected = {"sensitivity_signal_decision.csv", "pooled_paired_metrics.csv", "Figure_1_MMPK_N2c_paired_bootstrap.png"}
    if set(mmpk.artifact) != expected or set(mmpk.evidence_class) != {"internal_only_not_public_comparison"}:
        raise ValueError("MMPK R0 registration does not preserve its internal-only restriction")
    use = mmpk.allowed_usage.iloc[0]
    return pd.DataFrame([{
        "source_family": "MMPK N2c paired sensitivity", "release_status": "internal_only",
        "permitted_use": use,
        "prohibited_outputs": "No public comparison row, row-level data, predictions, or redistributable model pending explicit data rights.",
        "bound_artifacts": "; ".join(sorted(mmpk.artifact.tolist())),
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
    public_table = compile_jia_public_table(paths)
    mmpk_boundary = compile_mmpk_boundary(manifest)
    source_contract = manifest.loc[
        ((manifest.source_id.eq("jia_author_like_closeout")) & manifest.artifact.isin(["published_vs_author_like_primary.csv", "project_strict_protocol_context.csv"]))
        | ((manifest.source_id.eq("mmpk_n2c_internal")) & manifest.artifact.isin(["sensitivity_signal_decision.csv", "pooled_paired_metrics.csv"]))
    ].copy()
    if len(source_contract) != 4:
        raise ValueError("Table 2 must bind exactly four R0-approved assets")
    if args.check_only:
        print(f"Gate 8 R4 preflight: public_rows={len(public_table)} B_rows={(public_table.evidence_level == 'B').sum()} C_rows={(public_table.evidence_level == 'C').sum()} internal_only_rows={len(mmpk_boundary)} approved_assets=4; no labels, prediction rows, models, or MMPK numeric output read.")
        return
    with stage_output(output) as folder:
        public_schema().to_csv(folder / "table_2_public_schema.csv", index=False)
        internal_schema().to_csv(folder / "table_2_internal_only_schema.csv", index=False)
        public_table.to_csv(folder / "Table_2_Jia_literature_and_author_like_context.csv", index=False)
        mmpk_boundary.to_csv(folder / "table_2_internal_only_boundary.csv", index=False)
        source_contract.to_csv(folder / "table_2_source_contract.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R4 Table 2 literature and author-like boundaries\n\n"
            "This directory separates Jia 2025 published C-level background rows from project author-like B-level rows and project strict-context B-level rows. "
            "The paired source's difference column is deliberately not exported. MMPK files are hash-verified but never read for values or emitted as public "
            "comparison rows; only a non-numeric internal-only boundary record is produced. No raw labels, prediction rows, models, fold members or tests are read, "
            "and no metric is recomputed, fitted, calibrated, ranked, or selected.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_fold_members_accessed=True, no_test_labels_accessed=True,
            no_performance_metrics_computed=True, no_bootstrap_recomputed=True, no_model_fitted=True,
            no_calibration=True, no_candidate_selection_changed=True, no_mmpk_numeric_values_read=True,
            no_mmpk_numeric_values_exported=True, public_row_count=len(public_table),
            b_level_row_count=int((public_table.evidence_level == "B").sum()),
            c_level_row_count=int((public_table.evidence_level == "C").sum()),
            internal_only_boundary_row_count=len(mmpk_boundary), approved_asset_count=len(source_contract),
            source_registry_version="gate8_manuscript_evidence_registry_v2",
        )
    print(f"Gate 8 R4 Table 2 literature and author-like boundaries: {output}")


if __name__ == "__main__":
    run_cli(main)
