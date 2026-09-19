#!/usr/bin/env python3
"""Consolidate formal cross-gate evidence and freeze endpoint decisions.

Metadata-only Gate 4 stage. It fits no model, opens no labels, and explicitly
separates unified train-CV references from historical test-locked assets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASKS = [
    "CL__human__systemic_iv",
    "CLint__human__microsome",
    "Papp__human__caco2_ab",
    "Thalf__human__terminal_iv",
    "VDss__human__steady_state_iv",
    "fu__human__plasma",
]
ENDPOINT_ORDER = ["fu", "CLint", "Papp", "CL", "VDss", "Thalf", "F"]


def require_closed(label: str, meta: dict) -> None:
    if meta.get("partial", False):
        raise ValueError(f"{label} is partial")
    for key in ["test_labels_read", "test_evaluated", "validation_labels_read", "validation_target_file_opened"]:
        if meta.get(key, False):
            raise ValueError(f"{label} unexpectedly opened labels: {key}")


def evidence_row(**kwargs) -> dict:
    base = {
        "task_id": "", "endpoint": "", "evidence_family": "", "candidate_id": "",
        "comparator_id": "", "comparison_role": "", "primary_metric": "",
        "candidate_score": np.nan, "comparator_score": np.nan, "improvement_pct": np.nan,
        "folds_nonworse": np.nan, "outer_folds": 5, "paired_ci_low": np.nan,
        "paired_ci_high": np.nan, "advance_gate_passed": False, "decision": "",
        "comparison_level": "B", "source_path": "",
    }
    base.update(kwargs)
    return base


def build_figure(decisions: pd.DataFrame) -> plt.Figure:
    endpoints = ["fu", "CLint", "Papp", "CL", "VDss", "Thalf"]
    families = ["D-MPNN", "Frozen SMILES", "Cross-species", "Full shared", "Two-group", "Physchem B6"]
    matrix = np.full((len(families), len(endpoints)), np.nan)
    for i, family in enumerate(families):
        for j, endpoint in enumerate(endpoints):
            row = decisions.loc[(decisions.endpoint.eq(endpoint)) & decisions.family.eq(family)]
            if not row.empty:
                matrix[i, j] = 1.0 if bool(row.iloc[0].advanced) else -1.0
    cmap = matplotlib.colors.ListedColormap(["#C44E52", "#D9D9D9", "#2E8B57"])
    norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.imshow(np.nan_to_num(matrix, nan=0.0), cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(range(len(endpoints)), labels=endpoints)
    axis.set_yticks(range(len(families)), labels=families)
    for i in range(len(families)):
        for j in range(len(endpoints)):
            value = matrix[i, j]
            label = "N/A" if np.isnan(value) else "ADVANCE" if value > 0 else "STOP"
            axis.text(j, i, label, ha="center", va="center", fontsize=7.5,
                      color="#333333" if np.isnan(value) else "white", fontweight="bold")
    axis.set_title("Gate 4 endpoint architecture decisions", fontweight="bold", pad=12)
    axis.set_xlabel("Endpoint")
    axis.set_ylabel("Candidate family")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.015,
             "All decisions use matched train-CV evidence; fixed test labels were not accessed in Gate 4.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stagea", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--p1", type=Path, default=ROOT / "results/analysis/stl_p1_matched_comparison_v1")
    parser.add_argument("--dmpnn", type=Path, default=ROOT / "results/analysis/gate1b_dmpnn_stage1_audit_v1")
    parser.add_argument("--smiles", type=Path, default=ROOT / "results/analysis/gate1b_frozen_smiles_probe_audit_v1")
    parser.add_argument("--cl-transfer", type=Path, default=ROOT / "results/analysis/cl_cross_species_transfer_traincv_analysis_v2")
    parser.add_argument("--vdss-transfer", type=Path, default=ROOT / "results/analysis/vdss_cross_species_transfer_traincv_analysis_v1")
    parser.add_argument("--thalf-transfer", type=Path, default=ROOT / "results/analysis/thalf_cross_species_transfer_traincv_analysis_v2")
    parser.add_argument("--fu-transfer", type=Path, default=ROOT / "results/analysis/fu_cross_species_transfer_traincv_analysis_v2")
    parser.add_argument("--shared", type=Path, default=ROOT / "results/analysis/multitask_shared_private_traincv_analysis_v2")
    parser.add_argument("--two-group", type=Path, default=ROOT / "results/analysis/multitask_two_group_traincv_analysis_v1")
    parser.add_argument("--b6", type=Path, default=ROOT / "results/analysis/gate3_b6_physchem_formal_analysis_v2")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--historical-freeze", type=Path, default=ROOT / "models/frozen/final_candidates_v1")
    parser.add_argument("--clint-freeze", type=Path, default=ROOT / "models/frozen/CLint__human__microsome/v6_rf3_v1")
    parser.add_argument("--thalf-freeze", type=Path, default=ROOT / "models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1")
    parser.add_argument("--historical-test", type=Path, default=ROOT / "results/final/frozen_test_v1")
    parser.add_argument("--thalf-test", type=Path, default=ROOT / "results/final/Thalf_v15_rdkit2d_et3_test_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/cross_gate_candidate_freeze_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    stages = {
        "stagea": (args.stagea, "stl_stageA_three_view_audit"),
        "p1": (args.p1, "stl_p1_matched_comparison"),
        "dmpnn": (args.dmpnn, "gate1b_dmpnn_stage1_audit"),
        "smiles": (args.smiles, "gate1b_frozen_smiles_probe_audit"),
        "cl_transfer": (args.cl_transfer, "cross_species_transfer_traincv_analysis"),
        "vdss_transfer": (args.vdss_transfer, "vdss_cross_species_transfer_traincv_analysis"),
        "thalf_transfer": (args.thalf_transfer, "cross_species_transfer_traincv_analysis"),
        "fu_transfer": (args.fu_transfer, "cross_species_transfer_traincv_analysis"),
        "shared": (args.shared, "multitask_shared_private_traincv_analysis"),
        "two_group": (args.two_group, "multitask_two_group_traincv_analysis"),
        "b6": (args.b6, "gate3_b6_physchem_formal_analysis"),
        "protocol": (args.protocol, "stl_benchmark_protocol"),
        "historical_freeze": (args.historical_freeze, "frozen_candidate_registry"),
        "clint_freeze": (args.clint_freeze, "clint_v6_frozen_candidate"),
        "thalf_freeze": (args.thalf_freeze, "thalf_v15_frozen_candidate"),
    }
    required = [path / "complete.json" for path, _ in stages.values()] + [
        args.stagea / "frozen_stageA_shortlist.csv", args.p1 / "overall_matched_comparison.csv",
        args.dmpnn / "endpoint_decisions.csv", args.smiles / "endpoint_decisions.csv",
        args.shared / "paired_bootstrap_comparison.csv", args.two_group / "model_candidacy_gate.csv",
        args.b6 / "table_3_paired_parent_bootstrap.csv", args.b6 / "table_7_endpoint_final_decisions.csv",
        args.protocol / "task_manifest.csv", args.historical_freeze / "candidate_registry.json",
        args.clint_freeze / "candidate_registry.json", args.thalf_freeze / "candidate_registry.json",
        args.historical_test / "complete.json", args.historical_test / "metrics.json",
        args.thalf_test / "complete.json", args.thalf_test / "metrics.json",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    metas = {}
    for label, (path, stage) in stages.items():
        metas[label] = verify_stage(path, stage)
        if label not in {"historical_freeze", "clint_freeze", "thalf_freeze"}:
            require_closed(label, metas[label])
    historical_test_meta = verify_stage(args.historical_test, "frozen_test_evaluation")
    thalf_test_meta = verify_stage(args.thalf_test, "thalf_v15_frozen_test_evaluation")
    if not historical_test_meta.get("test_evaluated") or not thalf_test_meta.get("test_evaluated"):
        raise ValueError("Historical test lifecycle records are incomplete")

    task_manifest = pd.read_csv(args.protocol / "task_manifest.csv")
    if set(task_manifest.task_id) != set(TASKS):
        raise ValueError("Protocol task manifest differs from the six Gate 4 tasks")
    units = task_manifest.set_index("task_id").canonical_unit.to_dict()
    stagea = pd.read_csv(args.stagea / "frozen_stageA_shortlist.csv")
    stagea = stagea.loc[stagea.shortlist_position.eq(1)].copy()
    if set(stagea.task_id) != set(TASKS) or len(stagea) != 6:
        raise ValueError("Stage-A leaders do not cover exactly six endpoints")

    evidence = []
    for row in stagea.itertuples(index=False):
        evidence.append(evidence_row(
            task_id=row.task_id, endpoint=row.endpoint, evidence_family="corrected_stageA",
            candidate_id=f"{row.feature_set}::{row.candidate_id}", comparison_role="strong_train_cv_reference",
            primary_metric=row.primary_metric_name, candidate_score=float(row.primary_metric),
            advance_gate_passed=True, decision="retain_unified_research_reference", source_path=str(args.stagea)))

    p1 = pd.read_csv(args.p1 / "overall_matched_comparison.csv")
    for row in p1.itertuples(index=False):
        evidence.append(evidence_row(
            task_id=row.task_id, endpoint=row.endpoint, evidence_family="P1_nested_STL",
            candidate_id="P1_nested_candidate", comparator_id="corrected_stageA_matched_control",
            comparison_role="stageA_replacement", primary_metric=row.primary_metric_name,
            candidate_score=float(row.p1_primary_metric), comparator_score=float(row.matched_control_primary_metric),
            improvement_pct=-float(row.relative_change_pct), paired_ci_low=float(row.paired_bootstrap_difference_95ci_low),
            paired_ci_high=float(row.paired_bootstrap_difference_95ci_high), advance_gate_passed=False,
            decision="stop_no_preregistered_advance", source_path=str(args.p1)))

    dmpnn = pd.read_csv(args.dmpnn / "endpoint_decisions.csv")
    for row in dmpnn.itertuples(index=False):
        evidence.append(evidence_row(
            task_id=row.task_id, endpoint=row.endpoint, evidence_family="D-MPNN",
            candidate_id=row.best_dmpnn_candidate_id, comparator_id="corrected_stageA",
            comparison_role="stageA_replacement", primary_metric="physical_MAE" if row.endpoint == "fu" else "transformed_RMSE",
            candidate_score=float(row.best_dmpnn_primary_metric), comparator_score=float(row.stageA_primary_metric),
            improvement_pct=-float(row.relative_change_percent), folds_nonworse=int(row.folds_nonworse_out_of_5),
            advance_gate_passed=False, decision="stop_retain_negative_ablation", source_path=str(args.dmpnn)))

    smiles = pd.read_csv(args.smiles / "endpoint_decisions.csv")
    for row in smiles.itertuples(index=False):
        evidence.append(evidence_row(
            task_id=row.task_id, endpoint=row.endpoint, evidence_family="Frozen_SMILES",
            candidate_id=row.best_sequence_candidate_id, comparator_id="corrected_stageA",
            comparison_role="stageA_replacement", primary_metric="physical_MAE" if row.endpoint == "fu" else "transformed_RMSE",
            candidate_score=float(row.best_sequence_primary_metric), comparator_score=float(row.stageA_primary_metric),
            improvement_pct=-float(row.relative_change_percent), folds_nonworse=int(row.folds_nonworse_out_of_5),
            advance_gate_passed=False, decision="stop_retain_negative_ablation", source_path=str(args.smiles)))

    shared = pd.read_csv(args.shared / "paired_bootstrap_comparison.csv")
    for row in shared.itertuples(index=False):
        evidence.append(evidence_row(
            task_id=row.task_id, endpoint=row.endpoint, evidence_family="Full_shared_MTL",
            candidate_id="shared_encoder_private_heads", comparator_id="corrected_stageA",
            comparison_role="stageA_replacement", primary_metric=row.primary_metric,
            improvement_pct=float(row.shared_improvement_pct), folds_nonworse=int(row.noninferior_outer_folds),
            paired_ci_low=float(row.bootstrap_95ci_low), paired_ci_high=float(row.bootstrap_95ci_high),
            advance_gate_passed=bool(row.endpoint_gate_passed), decision="stop_retain_negative_transfer_evidence",
            source_path=str(args.shared)))

    grouped = pd.read_csv(args.two_group / "model_candidacy_gate.csv")
    for row in grouped.itertuples(index=False):
        evidence.append(evidence_row(
            task_id=row.task_id, endpoint=row.endpoint, evidence_family="Two_group_MTL",
            candidate_id="fu_family_vs_non_fu_capacity_matched", comparator_id="corrected_stageA",
            comparison_role="stageA_replacement", primary_metric="physical_MAE" if row.endpoint == "fu" else "transformed_RMSE",
            improvement_pct=float(row.improvement_vs_corrected_stageA_pct),
            folds_nonworse=int(row.noninferior_outer_folds_vs_stageA),
            paired_ci_high=float(row.bootstrap_95ci_high_grouped_minus_stageA),
            advance_gate_passed=bool(row.final_model_candidacy_confirmed),
            decision="stop_retain_mechanistic_ablation", source_path=str(args.two_group)))

    transfer_specs = [
        ("CL", args.cl_transfer), ("VDss", args.vdss_transfer),
        ("Thalf", args.thalf_transfer), ("fu", args.fu_transfer),
    ]
    for endpoint, path in transfer_specs:
        transfer = pd.read_csv(path / "matched_comparison.csv")
        transfer = transfer.loc[transfer.comparator.eq("corrected_stageA_STL")]
        if len(transfer) != 2:
            raise ValueError(f"Expected two Stage-A transfer comparisons for {endpoint}")
        for row in transfer.itertuples(index=False):
            if endpoint == "fu":
                candidate_score = row.candidate_physical_MAE
                comparator_score = row.comparator_physical_MAE
            else:
                candidate_score = row.candidate_transformed_RMSE
                comparator_score = row.comparator_transformed_RMSE
            task_id = stagea.loc[stagea.endpoint.eq(endpoint), "task_id"].iloc[0]
            evidence.append(evidence_row(
                task_id=task_id, endpoint=endpoint, evidence_family="Cross_species", candidate_id=row.route_id,
                comparator_id="corrected_stageA", comparison_role="stageA_replacement",
                primary_metric="physical_MAE" if endpoint == "fu" else "transformed_RMSE",
                candidate_score=float(candidate_score), comparator_score=float(comparator_score),
                improvement_pct=float(row.improvement_pct), folds_nonworse=int(row.noninferior_outer_folds),
                paired_ci_low=float(row.paired_bootstrap_difference_95ci_low),
                paired_ci_high=float(row.paired_bootstrap_difference_95ci_high), advance_gate_passed=False,
                decision="retain_transfer_signal_ablation_not_stageA_replacement", source_path=str(path)))

    b6 = pd.read_csv(args.b6 / "table_3_paired_parent_bootstrap.csv")
    b6 = b6.loc[b6.comparator.eq("C0_corrected_StageA")]
    b6_decisions = pd.read_csv(args.b6 / "table_7_endpoint_final_decisions.csv")
    for row in b6.itertuples(index=False):
        endpoint = "Thalf" if row.task_id == "Thalf__human__terminal_iv" else stagea.loc[
            stagea.task_id.eq(row.task_id), "endpoint"].iloc[0]
        advanced = bool(b6_decisions.loc[b6_decisions.task_id.eq(row.task_id), "B6_advanced"].iloc[0])
        evidence.append(evidence_row(
            task_id=row.task_id, endpoint=endpoint, evidence_family="Physchem_B6",
            candidate_id="nested_predicted_logP_logD_late_fusion", comparator_id="corrected_stageA",
            comparison_role="stageA_replacement", primary_metric=row.primary_metric,
            candidate_score=float(row.candidate_score), comparator_score=float(row.comparator_score),
            improvement_pct=-float(row.relative_error_change_pct), paired_ci_low=float(row.difference_ci_low),
            paired_ci_high=float(row.difference_ci_high), advance_gate_passed=advanced,
            decision="stop_retain_negative_ablation", source_path=str(args.b6)))
    ledger = pd.DataFrame(evidence)
    nonreference = ledger.loc[ledger.comparison_role.eq("stageA_replacement")]
    if nonreference.advance_gate_passed.astype(bool).any():
        raise ValueError("A non-Stage-A family unexpectedly passed the replacement gate")

    old_test = json.loads((args.historical_test / "metrics.json").read_text(encoding="utf-8"))
    thalf_test = json.loads((args.thalf_test / "metrics.json").read_text(encoding="utf-8"))
    candidate_rows = []
    for row in stagea.sort_values("task_id").itertuples(index=False):
        if row.task_id == "CLint__human__microsome":
            publication_asset, asset_policy = "pending_Gate5_packaging", "supersede_pretest_RF3_with_later_trainCV_only_StageA_leader"
            lifecycle, next_action = "test_unread", "package_exact_seed20260916_configuration_then_one_time_test"
            inferential_status = "confirmatory_after_one_time_test_with_finite_sample_CI"
        elif row.task_id == "Thalf__human__terminal_iv":
            publication_asset, asset_policy = str(args.thalf_freeze.resolve()), "retain_test_locked_v15_rdkit2d_ET3_asset"
            lifecycle, next_action = "test_consumed_no_reselection", "report_existing_frozen_test_and_source_overlap_limit"
            inferential_status = "confirmatory_scaffold_split_not_independent_of_TDC_Obach"
        else:
            publication_asset, asset_policy = str(args.historical_freeze.resolve()), "retain_historical_pretest_locked_asset"
            lifecycle, next_action = "test_consumed_no_reselection", "report_existing_frozen_test_no_post_test_model_change"
            inferential_status = "confirmatory_historical_frozen_test"
        candidate_rows.append({
            "task_id": row.task_id, "endpoint": row.endpoint, "canonical_unit": units[row.task_id],
            "unified_research_reference": f"{row.feature_set}::{row.candidate_id}",
            "reference_algorithm": row.algorithm, "reference_feature_set": row.feature_set,
            "reference_parameters": row.parameters, "primary_metric": row.primary_metric_name,
            "train_records": int(row.records), "train_parents": int(row.parents),
            "train_cv_score": float(row.primary_metric), "publication_asset": publication_asset,
            "publication_asset_policy": asset_policy, "test_lifecycle": lifecycle,
            "inferential_status": inferential_status, "comparison_level_current_project": "B",
            "next_action": next_action,
        })
    candidate_rows.append({
        "task_id": "F__human__absolute_oral", "endpoint": "F", "canonical_unit": "fraction",
        "unified_research_reference": "none", "reference_algorithm": "none", "reference_feature_set": "none",
        "reference_parameters": "{}", "primary_metric": "physical_MAE", "train_records": 20,
        "train_parents": 20, "train_cv_score": np.nan, "publication_asset": "none",
        "publication_asset_policy": "no_model_selection_without_validation",
        "test_lifecycle": "test_prohibited_insufficient_validation",
        "inferential_status": "exploratory_insufficient_validation", "comparison_level_current_project": "C",
        "next_action": "retain_interface_and_expand_data_without_blocking_current_manuscript",
    })
    candidates = pd.DataFrame(candidate_rows)
    candidates["endpoint_order"] = candidates.endpoint.map({value: i for i, value in enumerate(ENDPOINT_ORDER)})
    candidates = candidates.sort_values("endpoint_order").drop(columns="endpoint_order").reset_index(drop=True)

    lifecycle_rows = []
    for task in TASKS:
        endpoint = stagea.loc[stagea.task_id.eq(task), "endpoint"].iloc[0]
        if task == "CLint__human__microsome":
            lifecycle_rows.append({"task_id": task, "endpoint": endpoint, "test_status": "unread",
                "test_records": 28, "test_molecules": 28, "test_primary_metric": np.nan, "test_gmfe": np.nan,
                "test_within_2fold": np.nan, "further_selection_allowed": False,
                "allowed_next_action": "one_time_evaluation_after_Gate5_packaging"})
        elif task == "Thalf__human__terminal_iv":
            result = thalf_test["full_test"]
            lifecycle_rows.append({"task_id": task, "endpoint": endpoint, "test_status": "consumed",
                "test_records": result["records"], "test_molecules": result["molecules"],
                "test_primary_metric": result["primary"], "test_gmfe": result["gmfe_positive_subset"],
                "test_within_2fold": result["within_2fold_positive_subset"], "further_selection_allowed": False,
                "allowed_next_action": "report_only_no_reselection"})
        else:
            result = old_test[task]["test"]
            lifecycle_rows.append({"task_id": task, "endpoint": endpoint, "test_status": "consumed",
                "test_records": result["records"], "test_molecules": result["molecules"],
                "test_primary_metric": result["primary"], "test_gmfe": result.get("gmfe_positive_subset", np.nan),
                "test_within_2fold": result.get("within_2fold_positive_subset", np.nan),
                "further_selection_allowed": False, "allowed_next_action": "report_only_no_reselection"})
    lifecycle_rows.append({"task_id": "F__human__absolute_oral", "endpoint": "F", "test_status": "prohibited",
        "test_records": 2, "test_molecules": 2, "test_primary_metric": np.nan, "test_gmfe": np.nan,
        "test_within_2fold": np.nan, "further_selection_allowed": False,
        "allowed_next_action": "do_not_open_until_independent_validation_exists"})
    lifecycle = pd.DataFrame(lifecycle_rows)

    family_rows = []
    for endpoint in stagea.endpoint.tolist():
        for family, role in [("D-MPNN", "negative_ablation"), ("Frozen SMILES", "negative_ablation"),
                             ("Full shared", "negative_transfer_ablation"), ("Two-group", "mechanistic_ablation"),
                             ("Physchem B6", "negative_multimodal_ablation")]:
            family_rows.append({"endpoint": endpoint, "family": family, "advanced": False, "publication_role": role})
        if endpoint in {"CL", "VDss", "Thalf", "fu"}:
            role = "local_positive_transfer_mechanism_not_final_replacement" if endpoint != "fu" else "negative_transfer_ablation"
            family_rows.append({"endpoint": endpoint, "family": "Cross-species", "advanced": False, "publication_role": role})
    family_decisions = pd.DataFrame(family_rows)

    benchmark_map = pd.DataFrame([
        {"endpoint": "fu", "published_anchor": "Jia_2025", "current_level": "C", "next_level": "A_after_author_split_reproduction"},
        {"endpoint": "CL", "published_anchor": "Jia_2025", "current_level": "C", "next_level": "A_after_author_split_reproduction"},
        {"endpoint": "VDss", "published_anchor": "Jia_2025", "current_level": "C", "next_level": "A_after_author_split_reproduction"},
        {"endpoint": "Thalf", "published_anchor": "TDC_Half_Life_Obach|MMPK_2025", "current_level": "A_for_TDC|C_for_MMPK", "next_level": "retain_TDC_A; MMPK_A_requires_oral_dose_data"},
        {"endpoint": "Papp", "published_anchor": "no_direct_anchor_in_two_core_papers", "current_level": "B_internal", "next_level": "add_endpoint_specific_literature_context_without_delaying_freeze"},
        {"endpoint": "CLint", "published_anchor": "no_direct_anchor_in_two_core_papers", "current_level": "B_internal", "next_level": "one_time_test_then_endpoint_specific_context"},
        {"endpoint": "F", "published_anchor": "MMPK_2025", "current_level": "C", "next_level": "A_only_in_separate_dose_conditioned_NCA_track"},
    ])
    roadmap_proposal = (
        "# Gate 4 route-change proposal (not yet applied)\n\n"
        "1. Separate historical test-locked publication assets from the unified Stage-A train-CV research references.\n"
        "2. For CLint only, supersede the pretest RF3 asset with the later train-CV-only ECFP4+RDKit2D ExtraTrees configuration; "
        "package and hash it in Gate 5 before a single test read.\n"
        "3. Keep Jia direct reproduction on the critical path; keep MMPK/NCA data acquisition parallel and non-blocking.\n"
        "4. Do not reopen D-MPNN, frozen-SMILES, cross-species replacement, shared/two-group MTL or B6 searches without new independent evidence.\n\n"
        "This proposal is intentionally not written into 进度/研究路线.md until the user approves it.\n")

    if args.check_only:
        print(f"Gate 4 ready: endpoints={len(candidates)} evidence_rows={len(ledger)} "
              f"non_stageA_advanced={int(nonreference.advance_gate_passed.astype(bool).sum())} "
              "CLint_test=unread model_fitted=False")
        return

    inputs = {f"{label}_complete_sha256": sha256(path / "complete.json") for label, (path, _) in stages.items()}
    inputs.update({"historical_test_complete_sha256": sha256(args.historical_test / "complete.json"),
                   "thalf_test_complete_sha256": sha256(args.thalf_test / "complete.json")})
    with stage_output(args.output) as out:
        candidates.to_csv(out / "endpoint_candidate_freeze_registry.csv", index=False)
        ledger.sort_values(["task_id", "evidence_family", "candidate_id"]).to_csv(out / "cross_gate_evidence_ledger.csv", index=False)
        family_decisions.sort_values(["endpoint", "family"]).to_csv(out / "architecture_family_decisions.csv", index=False)
        lifecycle.to_csv(out / "test_lifecycle_registry.csv", index=False)
        benchmark_map.to_csv(out / "literature_benchmark_map.csv", index=False)
        pd.DataFrame([{"evidence_id": label, "path": str(path.resolve()), "stage": stage,
                       "complete_sha256": sha256(path / "complete.json")}
                      for label, (path, stage) in stages.items()]).to_csv(out / "input_evidence_registry.csv", index=False)
        (out / "roadmap_change_proposal.md").write_text(roadmap_proposal, encoding="utf-8")
        figure = build_figure(family_decisions)
        figure.savefig(out / "Figure_gate4_endpoint_decisions.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# Gate 4 cross-gate endpoint candidate freeze\n\n"
            "The corrected Stage-A endpoint leaders remain the unified train-CV research references. No graph, frozen-SMILES, "
            "cross-species, shared/two-group multitask or nested physicochemical route passed the Stage-A replacement gate. "
            "Historical test-locked assets remain immutable. CLint test remains unread; its later train-CV-only Stage-A ExtraTrees "
            "configuration is frozen for Gate 5 packaging before one-time evaluation. Human absolute F remains exploratory.\n",
            encoding="utf-8")
        finish_stage(out, "cross_gate_endpoint_candidate_freeze", inputs=inputs,
                     endpoints_registered=7, evaluable_endpoints=6, exploratory_endpoints=1,
                     evidence_rows=len(ledger), architecture_families_advanced=0,
                     unified_reference="corrected_StageA_endpoint_leaders",
                     clint_candidate="ecfp4_rdkit2d::extra_trees__c01_seed20260916",
                     clint_test_status="unread", historical_test_locked_assets_preserved=True,
                     roadmap_modified=False, validation_labels_read=False, test_labels_read=False,
                     model_fitted=False, data_modified=False, partial=False)
    print(f"Gate 4 cross-gate candidate freeze: {args.output}")


if __name__ == "__main__":
    run_cli(main)
