#!/usr/bin/env python3
"""将已发布研究图复制到进度/研究过程，并生成可追溯清单。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "进度" / "研究过程" / "figures"

FIGURES = [
    ("fig01_ml_readiness.png", "results/data_curation_figures/Fig5_ML_Readiness_Overview.png", "Data readiness and endpoint coverage"),
    ("fig02_public_f_isolation.png", "results/figures/public_f_cohort_isolation_figure_v2/Figure_S_public_F_triple_isolation.png", "Public-F parent/scaffold/source isolation"),
    ("fig03_stagea_feature_views.png", "results/figures/stl_stageA_feature_view_comparison_v2/Figure_STL_StageA_feature_views.png", "Stage-A feature-view comparison"),
    ("fig04_cl_transfer.png", "results/analysis/cl_cross_species_transfer_traincv_analysis_v2/Figure_CL_transfer_trainCV.png", "CL cross-species transfer"),
    ("fig05_thalf_transfer.png", "results/analysis/thalf_cross_species_transfer_traincv_analysis_v2/Figure_Thalf_transfer_trainCV.png", "Terminal half-life cross-species transfer"),
    ("fig06_vdss_transfer.png", "results/analysis/vdss_cross_species_transfer_traincv_analysis_v1/Figure_VDss_transfer_trainCV.png", "VDss cross-species transfer"),
    ("fig07_fu_transfer.png", "results/analysis/fu_cross_species_transfer_traincv_analysis_v2/Figure_fu_transfer_trainCV.png", "fu cross-species transfer"),
    ("fig08_shared_private.png", "results/analysis/multitask_shared_private_traincv_analysis_v2/Figure_shared_vs_private_trainCV.png", "Shared versus private multitask encoders"),
    ("fig09_gradient_affinity.png", "results/analysis/multitask_gradient_affinity_report_v2/Figure_multitask_gradient_affinity_report.png", "Multitask gradient affinity"),
    ("fig10_two_group.png", "results/analysis/multitask_two_group_traincv_analysis_v1/Figure_two_group_vs_full_shared_trainCV.png", "Two-group multitask ablation"),
    ("fig11_gate2b_stoploss.png", "data/public_development/gate2b_decision_freeze_gate3_audit_v1/Figure_gate2b_stop_loss_gate3_entry.png", "Architecture stop-loss and Gate-3 entry"),
    ("fig12_physchem_feasibility.png", "data/public_development/physchem_auxiliary_feasibility_audit_v3/Figure_physchem_auxiliary_feasibility.png", "Experimental physicochemical auxiliary feasibility"),
    ("fig13_gate3_bootstrap.png", "results/analysis/gate3_b6_physchem_formal_analysis_v2/figure_2_paired_bootstrap_forest.png", "Gate-3 B6 paired bootstrap"),
    ("fig14_gate3_sensitivity.png", "results/analysis/gate3_b6_physchem_formal_analysis_v2/figure_4_sensitivity_forest.png", "Gate-3 B6 sensitivity analysis"),
    ("fig15_thalf_frozen_test.png", "results/final/Thalf_v15_rdkit2d_et3_diagnostics_v2/figure_1_observed_vs_predicted.png", "Frozen strict-IV half-life test diagnostics"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只核验源图，不复制")
    args = parser.parse_args()

    missing = [source for _, source, _ in FIGURES if not (ROOT / source).is_file()]
    if missing:
        raise FileNotFoundError("缺少已登记源图:\n" + "\n".join(missing))
    if args.check_only:
        print(f"Research archive figures ready: {len(FIGURES)}")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, source_rel, caption in FIGURES:
        source = ROOT / source_rel
        destination = DEST / name
        with tempfile.NamedTemporaryFile(dir=DEST, prefix=f".{name}.", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        rows.append({
            "archive_name": name,
            "caption": caption,
            "source_path": source_rel,
            "source_sha256": sha256(source),
            "archive_sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        })

    manifest = DEST / "source_manifest.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=DEST,
                                     prefix=".source_manifest.", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary_manifest = Path(handle.name)
    temporary_manifest.replace(manifest)
    print(f"Research process archive: {DEST} ({len(rows)} figures)")


if __name__ == "__main__":
    main()
