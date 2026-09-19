#!/usr/bin/env python3
"""Package Gate-3 B6 reports, tables, figures, and contracts into one flat directory."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/analysis/gate3_b6_physchem_formal_analysis_v2")
    parser.add_argument("--analysis-protocol", type=Path, default=ROOT / "data/public_development/gate3_b6_aggregate_analysis_protocol_v1")
    parser.add_argument("--gate-protocol", type=Path, default=ROOT / "data/public_development/gate3_b6_physchem_protocol_v2")
    parser.add_argument("--batch", type=Path, default=ROOT / "results/benchmarks/gate3_b6_physchem_formal_cells_v3")
    parser.add_argument("--chinese-report", type=Path, default=ROOT / "进度/2026-09-17_Gate3B6正式汇总分析.md")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/gate3_b6_publication_bundle_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    sources = {
        "REPORT_zh.md": args.chinese_report,
        "REPORT_en.md": args.analysis / "REPORT.md",
        "decision.json": args.analysis / "decision.json",
        "analysis_contract.json": args.analysis_protocol / "analysis_contract.json",
        "endpoint_advance_gates.csv": args.gate_protocol / "endpoint_advance_gates.csv",
        "formal_batch_complete.json": args.batch / "batch_complete.json",
    }
    for index in range(1, 10):
        matches = sorted(args.analysis.glob(f"table_{index}_*.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one table_{index} CSV, found {matches}")
        sources[matches[0].name] = matches[0]
    for index in range(1, 6):
        matches = sorted(args.analysis.glob(f"figure_{index}_*.png"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one figure_{index} PNG, found {matches}")
        sources[matches[0].name] = matches[0]

    startup_self_check(list(sources.values()) + [args.analysis / "complete.json",
                                                args.analysis_protocol / "complete.json"],
                       output=None if args.check_only else args.output)
    analysis_meta = verify_stage(args.analysis, "gate3_b6_physchem_formal_analysis")
    protocol_meta = verify_stage(args.analysis_protocol, "gate3_b6_aggregate_analysis_protocol")
    if analysis_meta.get("publication_png_figures") != 5 or analysis_meta.get("figure_dpi") != 600:
        raise ValueError("Source analysis does not contain the frozen five 600-dpi figures")
    if analysis_meta.get("test_labels_read") or protocol_meta.get("test_labels_read"):
        raise ValueError("Only validation/test-closed analysis may be packaged")
    if args.check_only:
        print(f"Gate 3 B6 publication bundle ready: files={len(sources)} figures=5 tables=9")
        return

    with stage_output(args.output) as out:
        rows = []
        for packaged_name, source in sorted(sources.items()):
            destination = out / packaged_name
            shutil.copy2(source, destination)
            source_hash = sha256(source)
            packaged_hash = sha256(destination)
            if source_hash != packaged_hash:
                raise ValueError(f"Packaged copy differs from source: {source}")
            rows.append({"packaged_file": packaged_name,
                         "source_path": str(source.resolve().relative_to(ROOT)),
                         "sha256": source_hash})
        with (out / "source_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["packaged_file", "source_path", "sha256"])
            writer.writeheader(); writer.writerows(rows)
        (out / "README.md").write_text(
            "# Gate 3 B6 publication analysis bundle\n\n"
            "This flat directory contains the complete recommended v2 formal analysis: Chinese and English "
            "reports, the frozen statistical contract, A1–A7 gates, batch provenance, nine machine-readable "
            "tables, and five English 600-dpi PNG figures. Files are byte-identical copies of their versioned "
            "sources; `source_manifest.csv` records every source path and SHA-256.\n\n"
            "Decision: 0/6 endpoints advanced B6. B6 is retained as a negative ablation. Fixed validation, "
            "test, source-test labels, and human F remained closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "gate3_b6_publication_bundle",
            inputs={"analysis_complete_sha256": sha256(args.analysis / "complete.json"),
                    "analysis_protocol_complete_sha256": sha256(args.analysis_protocol / "complete.json")},
            packaged_source_files=len(sources), tables=9, png_figures=5,
            figure_dpi=600, endpoints_advancing_B6=0,
            model_fitted=False, fixed_validation_authorized=False,
            source_test_labels_read=False, test_labels_read=False, partial=False,
        )
    print(f"Gate 3 B6 publication bundle: {args.output}")


if __name__ == "__main__":
    run_cli(main)
