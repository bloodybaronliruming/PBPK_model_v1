"""Freeze actual R0-approved PNG assets and caption boundaries for Figures 1–6.

No image is regenerated, relabelled, or numerically recomputed. The stage only
checks R0 provenance hashes, PNG dimensions, panel placement, and caption claim
boundaries. It does not open labels, predictions, models, or metric tables.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_figure_contract"
R0_STAGE = "gate8_manuscript_evidence_registry"
R0_OUTPUT = "data/public_development/gate8_manuscript_evidence_registry_v2"
OUTPUT = "data/public_development/gate8_figure_contract_v1"

PANELS = [
    ("Figure 1", "A", "gate4_candidate_freeze", "Figure_gate4_endpoint_decisions.png", "Endpoint-specific candidate and decision lineage"),
    ("Figure 1", "B", "gate7_evidence_matrix", "Figure_1_gate7_evidence_matrix.png", "Endpoint lifecycle, AD, uncertainty and evidence lanes"),
    ("Figure 2", "A", "p1_matched_stl", "Figure_P1_vs_matched_control.png", "P1 versus matched strong-STL control"),
    ("Figure 2", "B", "gate4_candidate_freeze", "Figure_gate4_endpoint_decisions.png", "Endpoint-specific decision context; reused without modification"),
    ("Figure 3", "A", "multitask_gradient_report", "Figure_multitask_gradient_affinity_report.png", "Multitask gradient-affinity diagnostic"),
    ("Figure 4", "A", "cross_species_cl", "Figure_CL_transfer_trainCV.png", "CL matched human-only versus transfer ablation"),
    ("Figure 4", "B", "cross_species_fu", "Figure_fu_transfer_trainCV.png", "fu matched human-only versus transfer ablation"),
    ("Figure 4", "C", "cross_species_vdss", "Figure_VDss_transfer_trainCV.png", "VDss matched human-only versus transfer ablation"),
    ("Figure 4", "D", "cross_species_thalf", "Figure_Thalf_transfer_trainCV.png", "Terminal t½ matched human-only versus transfer ablation"),
    ("Figure 5", "A", "gate3_b6_formal", "figure_2_paired_bootstrap_forest.png", "B6 paired-bootstrap forest with formal 0/6 stop decision"),
    ("Figure 6", "A", "clint_gate5_closeout", "Figure_2_CLint_AD_and_source_diagnostics.png", "CLint closed-test AD and source diagnostic"),
    ("Figure 6", "B", "frozen_test_diagnostics", "figure_3_applicability_domain.png", "Historical frozen-test applicability-domain diagnostic"),
    ("Figure 6", "C", "gate7_evidence_matrix", "Figure_1_gate7_evidence_matrix.png", "Cross-endpoint release/lifecycle/evidence matrix; reused without modification"),
]

CAPTIONS = [
    ("Figure 1", "not_comparison", "Study governance and endpoint evidence lanes.", "Separate strict train-CV research references, historical test-locked assets, status-only F, and closed test lifecycles; this is a workflow/status figure, not a performance ranking."),
    ("Figure 2", "B", "Matched strong-STL baseline and endpoint-specific decisions.", "Show only within-project matched evidence and endpoint-specific decisions; do not claim a universal algorithm champion or compare with historical tests."),
    ("Figure 3", "B", "Multitask gradient affinity and negative-transfer diagnostic.", "Gradient affinity is an associative optimization diagnostic; it does not demonstrate a causal pharmacokinetic mechanism or reopen stopped multitask architectures."),
    ("Figure 4", "B", "Controlled cross-species transfer ablations across four endpoints.", "Each panel retains its matched human-only and strong-STL controls; do not infer that animal transfer universally replaces endpoint-specific human models."),
    ("Figure 5", "B", "Nested experimental-physchem (B6) negative ablation.", "Report the preregistered 0/6 advance outcome and retain the B6 stop decision; do not selectively reopen the architecture family."),
    ("Figure 6", "not_comparison", "Frozen performance, applicability-domain limitations and release evidence status.", "Report closed-test and technical-reproducibility evidence with lifecycle limits; do not make an independent external-validation claim where terminal t½ source overlap is documented."),
]
NONVISUAL = [
    ("Figure 6", "gate7_reproducibility", "delivery_manifest.json", "Reproducibility provenance supporting the Figure 6 release-status caption; not a plotted performance panel."),
    ("Figure 6", "gate7_clean_reload", "verification_checks.json", "Clean-environment technical replay supporting the Figure 6 engineering-status caption; not predictive validation."),
]


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Expected a valid PNG with IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def r0_asset(manifest: pd.DataFrame, source_id: str, artifact: str) -> pd.Series:
    found = manifest.loc[manifest.source_id.eq(source_id) & manifest.artifact.eq(artifact)]
    if len(found) != 1:
        raise ValueError(f"R0 must register exactly one figure asset: {source_id}/{artifact}")
    return found.iloc[0]


def verify_r0_asset(manifest: pd.DataFrame, source_id: str, artifact: str, inputs: dict[str, str]) -> tuple[pd.Series, Path]:
    row = r0_asset(manifest, source_id, artifact)
    folder = Path(row.source_path)
    verify_stage(folder, row.source_stage)
    complete = folder / "complete.json"
    inputs[str(complete.resolve())] = sha256(complete)
    path = folder / artifact
    if not path.is_file() or sha256(path) != row.artifact_sha256:
        raise ValueError(f"R0-approved figure asset changed after registration: {path}")
    inputs[str(path.resolve())] = sha256(path)
    return row, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT
    r0 = root / R0_OUTPUT
    r0_meta = verify_stage(r0, R0_STAGE)
    flags = (
        "no_model_loaded", "no_labels_accessed", "no_prediction_rows_accessed",
        "no_performance_metrics_computed", "no_model_fitted", "no_calibration",
        "no_candidate_selection_changed",
    )
    if not all(r0_meta.get(flag) is True for flag in flags):
        raise ValueError("R0 registry does not meet required no-label/no-reselection boundary")
    manifest_path = r0 / "source_artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    inputs = {
        str((r0 / "complete.json").resolve()): sha256(r0 / "complete.json"),
        str(manifest_path.resolve()): sha256(manifest_path),
    }
    panel_rows = []
    used_assets: set[tuple[str, str]] = set()
    for figure_id, panel, source_id, artifact, purpose in PANELS:
        source, path = verify_r0_asset(manifest, source_id, artifact, inputs)
        width, height = png_dimensions(path)
        if width < 1000 or height < 700:
            raise ValueError(f"Figure panel has insufficient frozen pixel dimensions: {path} ({width}x{height})")
        used_assets.add((source_id, artifact))
        panel_rows.append({
            "figure_id": figure_id, "panel": panel, "source_id": source_id,
            "asset_path": str(path.resolve()), "asset_sha256": sha256(path),
            "pixel_width": width, "pixel_height": height, "source_evidence_class": source.evidence_class,
            "panel_purpose": purpose, "assembly_instruction": "Reuse the frozen PNG without crop, rescale-induced label loss, recoloring, or data-layer modification.",
        })
    caption = pd.DataFrame(CAPTIONS, columns=["figure_id", "figure_evidence_class", "english_caption_title", "claim_guardrail"])
    if set(caption.figure_id) != {f"Figure {n}" for n in range(1, 7)} or set(caption.figure_evidence_class) != {"B", "not_comparison"}:
        raise ValueError("Figure caption contract must contain exactly six registered B/non-comparison figures")
    nonvisual_rows = []
    for figure_id, source_id, artifact, purpose in NONVISUAL:
        source, path = verify_r0_asset(manifest, source_id, artifact, inputs)
        used_assets.add((source_id, artifact))
        nonvisual_rows.append({
            "figure_id": figure_id, "source_id": source_id, "artifact": artifact,
            "asset_path": str(path.resolve()), "asset_sha256": sha256(path),
            "source_evidence_class": source.evidence_class, "caption_support_only": True,
            "purpose": purpose,
        })
    exclusions = pd.DataFrame([
        ("jia_author_like_closeout/Figure_2_Jia2025_evidence_boundaries.png", "Not part of the R0 Figure 1–6 plan; Jia B/C boundaries are reported in Table 2 and may not be silently substituted into a core figure."),
        ("mmpk_n2c_internal/Figure_1_MMPK_N2c_paired_bootstrap.png", "Internal-only MMPK asset; data rights do not permit its public main-figure use."),
    ], columns=["excluded_asset", "reason"])
    unique_sources = sorted(used_assets)
    source_contract = manifest.loc[manifest.apply(lambda x: (x.source_id, x.artifact) in set(unique_sources), axis=1)].copy()
    if len(panel_rows) != 13 or len(nonvisual_rows) != 2 or len(source_contract) != 13:
        raise ValueError("Figure contract asset-count invariant failed")
    if args.check_only:
        print("Gate 8 R6 preflight: figures=6 PNG_panels=13 nonvisual_caption_assets=2 unique_R0_assets=13 exclusions=2; no labels, predictions, models, or metrics read/recomputed.")
        return
    with stage_output(output) as folder:
        pd.DataFrame(panel_rows).to_csv(folder / "figure_panel_asset_manifest.csv", index=False)
        caption.to_csv(folder / "figure_caption_contract.csv", index=False)
        pd.DataFrame(nonvisual_rows).to_csv(folder / "figure_nonvisual_caption_evidence.csv", index=False)
        exclusions.to_csv(folder / "figure_exclusion_register.csv", index=False)
        source_contract.to_csv(folder / "figure_source_contract.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R6 figure asset and caption contract\n\n"
            "This contract freezes the existing R0-approved PNG panels and the English caption claim boundaries for Figures 1–6. "
            "It does not generate a new plot, read a label/prediction/model/metric table, or make a new model decision. Figure 4 is a fixed four-panel "
            "cross-species ablation layout. Figure 6 has three PNG panels plus nonvisual reproducibility evidence for its caption. Jia is Table 2 context; "
            "MMPK is internal-only and excluded.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_metric_tables_accessed=True, no_performance_metrics_computed=True,
            no_plot_regenerated=True, no_model_fitted=True, no_calibration=True, no_candidate_selection_changed=True,
            figure_count=6, png_panel_count=len(panel_rows), nonvisual_caption_asset_count=len(nonvisual_rows),
            excluded_asset_count=len(exclusions), source_registry_version="gate8_manuscript_evidence_registry_v2",
        )
    print(f"Gate 8 R6 figure asset and caption contract: {output}")


if __name__ == "__main__":
    run_cli(main)
