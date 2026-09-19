#!/usr/bin/env python3
"""Freeze Gate 2B decisions and select one bounded Gate 3 protocol candidate.

This is a metadata-only stop-loss stage. It verifies formal train-CV evidence,
fits no model, and does not open fixed-validation or test targets.
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASKS = [
    "CL__human__systemic_iv", "CLint__human__microsome", "Papp__human__caco2_ab",
    "Thalf__human__terminal_iv", "VDss__human__steady_state_iv", "fu__human__plasma",
]


def require_closed(label: str, meta: dict) -> None:
    if meta.get("partial", False):
        raise ValueError(f"{label} is partial")
    if meta.get("test_labels_read", False) or meta.get("test_evaluated", False):
        raise ValueError(f"{label} is not test-closed")
    if meta.get("validation_target_file_opened", False) or meta.get("validation_labels_read", False):
        raise ValueError(f"{label} opened fixed-validation targets")


def build_stop_loss_registry() -> pd.DataFrame:
    rows = [
        ("B0", "Corrected Stage-A endpoint leaders", "structure_2d_classical", "retain_reference",
         "Strong matched STL reference; no later route passed its replacement gate.", False, False),
        ("G2A", "Single full-shared encoder + private heads", "rdkit2d_multitask", "stop",
         "Zero of six human endpoints advanced; human fu worsened in all five folds.", False, False),
        ("G2B", "Capacity-matched fu/non-fu two-group encoders", "rdkit2d_multitask", "stop",
         "Mechanistic gate passed 4/5 only; zero endpoints passed the Stage-A candidacy gate.", False, False),
        ("G2C", "More RDKit2D grouping, PCGrad, GradNorm or MMoE", "rdkit2d_multitask", "defer",
         "No independent evidence justifies adaptive expansion on the same outer-CV results.", False, False),
        ("B2", "D-MPNN graph + RDKit2D", "structure_graph", "stop_current_family",
         "All six endpoint candidates were 2.80% to 9.85% worse than Stage-A.", False, False),
        ("B3", "Frozen MoLFormer representation", "structure_sequence", "stop_current_family",
         "Matched structure controls ranked first for all six endpoints; nested MLP did not beat Stage-A.", False, False),
        ("B4", "2D + graph + SMILES gated fusion", "structure_multimodal", "blocked_stop_loss",
         "Both required incremental modalities B2 and B3 failed their Stage-A gates.", False, False),
        ("B5", "Structure fusion + semantic context", "context_multimodal", "blocked_stop_loss",
         "B4 prerequisite failed and controlled species/context routes did not replace Stage-A.", False, False),
        ("B7", "Mechanistic PK OOF augmentation", "predicted_mechanistic", "defer",
         "Current Thalf protected upstream implementation was negative; other endpoints lack a locked ancestry protocol.", False, False),
        ("B8", "Matching animal PK OOF augmentation", "predicted_cross_species", "stop_current_family",
         "Formal CL, VDss, Thalf and fu transfer routes produced zero Stage-A replacements.", False, False),
        ("S1", "Post-hoc Stage-A/neural stacking", "prediction_fusion", "defer",
         "Not preregistered for Gate 2 and would require fully nested base/meta cross-fitting to avoid leakage.", False, False),
        ("B6", "Structure + experimental physchem OOF", "predicted_physchem", "select_protocol_only",
         "Only independently preregistered, untested modality family with sizable logP/logD auxiliary data.", True, False),
    ]
    return pd.DataFrame(rows, columns=[
        "candidate_id", "architecture_or_input", "family", "gate2b_stop_loss_status", "evidence_basis",
        "gate3_protocol_authorized", "gate3_training_authorized",
    ])


def build_figure(registry: pd.DataFrame) -> plt.Figure:
    colors = {
        "retain_reference": "#4C78A8", "stop": "#C44E52", "stop_current_family": "#C44E52",
        "blocked_stop_loss": "#8172B2", "defer": "#999999", "select_protocol_only": "#2E8B57",
    }
    labels = {
        "retain_reference": "REFERENCE", "stop": "STOP", "stop_current_family": "STOP",
        "blocked_stop_loss": "BLOCKED", "defer": "DEFER", "select_protocol_only": "PROTOCOL ONLY",
    }
    fig, axis = plt.subplots(figsize=(9.2, 6.2))
    plot = registry.iloc[::-1].reset_index(drop=True)
    for index, row in plot.iterrows():
        status = row.gate2b_stop_loss_status
        axis.barh(index, 1, color=colors[status], height=0.72)
        axis.text(0.02, index, row.candidate_id, va="center", ha="left", color="white", fontweight="bold", fontsize=8)
        axis.text(0.98, index, labels[status], va="center", ha="right", color="white", fontweight="bold", fontsize=8)
    axis.set_yticks(range(len(plot)))
    axis.set_yticklabels(["\n".join(textwrap.wrap(value, 36)) for value in plot.architecture_or_input], fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.set_title("Gate 2B stop-loss audit and bounded Gate 3 entry", fontweight="bold", pad=12)
    axis.spines[:].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=8)
    fig.text(
        0.5, 0.01,
        "Green authorizes protocol construction only; fixed validation, test and model training remain closed.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-freeze", type=Path, default=ROOT / "data/public_development/stl_p1_decision_freeze_v1")
    parser.add_argument("--shared", type=Path, default=ROOT / "results/analysis/multitask_shared_private_traincv_analysis_v2")
    parser.add_argument("--affinity", type=Path, default=ROOT / "results/analysis/multitask_gradient_affinity_report_v2")
    parser.add_argument("--two-group", type=Path, default=ROOT / "results/analysis/multitask_two_group_traincv_analysis_v1")
    parser.add_argument("--vdss", type=Path, default=ROOT / "results/analysis/vdss_cross_species_transfer_traincv_analysis_v1")
    parser.add_argument("--cl", type=Path, default=ROOT / "results/analysis/cl_cross_species_transfer_traincv_analysis_v2")
    parser.add_argument("--thalf", type=Path, default=ROOT / "results/analysis/thalf_cross_species_transfer_traincv_analysis_v2")
    parser.add_argument("--fu", type=Path, default=ROOT / "results/analysis/fu_cross_species_transfer_traincv_analysis_v2")
    parser.add_argument("--multimodal", type=Path, default=ROOT / "data/public_development/gate1b_multimodal_protocol_v2")
    parser.add_argument("--dmpnn", type=Path, default=ROOT / "results/analysis/gate1b_dmpnn_stage1_audit_v1")
    parser.add_argument("--smiles", type=Path, default=ROOT / "results/benchmarks/gate1b_frozen_smiles_adaptation_v1")
    parser.add_argument("--nested-mlp", type=Path, default=ROOT / "results/benchmarks/gate1b_nested_scaffold_mlp_v1")
    parser.add_argument("--endpoint-expansion", type=Path, default=ROOT / "results/analysis/endpoint_expansion_feasibility_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/gate2b_decision_freeze_gate3_audit_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    required = [
        args.p1_freeze / "complete.json", args.p1_freeze / "p1_decision_registry.csv",
        args.shared / "complete.json", args.shared / "endpoint_overall_metrics.csv", args.shared / "decision.json",
        args.affinity / "complete.json", args.affinity / "next_architecture_candidate.csv",
        args.two_group / "complete.json", args.two_group / "endpoint_overall_metrics.csv",
        args.two_group / "mechanistic_gate.csv", args.two_group / "model_candidacy_gate.csv",
        args.vdss / "complete.json", args.vdss / "decision_registry.csv",
        args.cl / "complete.json", args.cl / "decision_registry.csv",
        args.thalf / "complete.json", args.thalf / "decision_registry.csv",
        args.fu / "complete.json", args.fu / "decision_registry.csv",
        args.multimodal / "complete.json", args.multimodal / "incremental_ablation_ladder.csv",
        args.multimodal / "oof_ancestry_contract.csv", args.multimodal / "leakage_prohibition_registry.csv",
        args.dmpnn / "complete.json", args.dmpnn / "endpoint_decisions.csv",
        args.smiles / "complete.json", args.smiles / "candidate_ranking.csv",
        args.nested_mlp / "complete.json", args.nested_mlp / "metrics.csv",
        args.endpoint_expansion / "complete.json", args.endpoint_expansion / "active_endpoint_registry.csv",
        args.endpoint_expansion / "physchem_feature_leakage_exclusions.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    stages = {
        "p1_freeze": (args.p1_freeze, "stl_p1_decision_freeze"),
        "shared": (args.shared, "multitask_shared_private_traincv_analysis"),
        "affinity": (args.affinity, "multitask_gradient_affinity_report"),
        "two_group": (args.two_group, "multitask_two_group_traincv_analysis"),
        "vdss": (args.vdss, "vdss_cross_species_transfer_traincv_analysis"),
        "cl": (args.cl, "cross_species_transfer_traincv_analysis"),
        "thalf": (args.thalf, "cross_species_transfer_traincv_analysis"),
        "fu": (args.fu, "cross_species_transfer_traincv_analysis"),
        "multimodal": (args.multimodal, "gate1b_multimodal_input_protocol"),
        "dmpnn": (args.dmpnn, "gate1b_dmpnn_stage1_audit"),
        "smiles": (args.smiles, "gate1b_frozen_smiles_adaptation"),
        "nested_mlp": (args.nested_mlp, "gate1b_nested_scaffold_mlp"),
        "endpoint_expansion": (args.endpoint_expansion, "endpoint_expansion_feasibility"),
    }
    metas = {}
    for label, (path, stage) in stages.items():
        metas[label] = verify_stage(path, stage)
        require_closed(label, metas[label])

    p1 = pd.read_csv(args.p1_freeze / "p1_decision_registry.csv")
    shared = pd.read_csv(args.shared / "endpoint_overall_metrics.csv")
    grouped = pd.read_csv(args.two_group / "endpoint_overall_metrics.csv")
    grouped_gate = pd.read_csv(args.two_group / "model_candidacy_gate.csv")
    if set(p1.task_id) != set(TASKS) or set(shared.task_id) != set(TASKS) or set(grouped.task_id) != set(TASKS):
        raise ValueError("Gate 2B endpoint evidence must cover exactly the six authorized human tasks")
    if grouped_gate.statistical_candidacy_gate_passed.astype(bool).any():
        raise ValueError("A two-group endpoint unexpectedly passed the Stage-A gate")
    two_group_decision = json.loads((args.two_group / "decision.json").read_text(encoding="utf-8"))
    shared_decision = json.loads((args.shared / "decision.json").read_text(encoding="utf-8"))
    if two_group_decision["mechanistic_gate_passed"] or two_group_decision["statistical_model_candidacy_endpoints"]:
        raise ValueError("Two-group decision does not support a stop-loss freeze")
    if shared_decision["shared_architecture_advanced"]:
        raise ValueError("Full sharing unexpectedly advanced")

    endpoint = grouped[[
        "task_id", "endpoint", "parents", "primary_metric", "stageA_reference_score", "private_score",
        "shared_score", "grouped_score", "grouped_relative_change_vs_shared_pct",
        "grouped_relative_change_vs_private_pct", "grouped_relative_change_vs_stageA_pct",
        "noninferior_folds_vs_shared", "noninferior_folds_vs_private", "noninferior_folds_vs_stageA",
    ]].copy()
    endpoint = endpoint.merge(
        shared[["task_id", "shared_relative_change_pct", "shared_noninferior_outer_folds"]],
        on="task_id", validate="one_to_one",
    ).merge(
        grouped_gate[["task_id", "statistical_candidacy_gate_passed", "status"]],
        on="task_id", validate="one_to_one",
    )
    endpoint["full_shared_decision"] = "reject"
    endpoint["two_group_decision"] = "retain_mechanistic_ablation_not_candidate"
    endpoint["stageA_reference_retained"] = True
    endpoint["source_sensitivity_triggered"] = False
    endpoint["fixed_validation_authorized"] = False
    endpoint["test_authorized"] = False

    transfer_rows = []
    for endpoint_name, path in [("VDss", args.vdss), ("CL", args.cl), ("Thalf", args.thalf), ("fu", args.fu)]:
        frame = pd.read_csv(path / "decision_registry.csv")
        stagea = frame.loc[frame.comparator.eq("corrected_stageA_STL")].copy()
        if len(stagea) != 2 or stagea.core_traincv_gate.astype(bool).any() or stagea.fixed_validation_ready.astype(bool).any():
            raise ValueError(f"{endpoint_name} transfer evidence unexpectedly advances versus Stage-A")
        for row in stagea.itertuples(index=False):
            transfer_rows.append({
                "endpoint": endpoint_name, "route_id": row.route_id, "comparator": row.comparator,
                "core_traincv_gate": bool(row.core_traincv_gate),
                "fixed_validation_ready": bool(row.fixed_validation_ready), "next_action": row.next_action,
                "gate2b_freeze_decision": "retain_transfer_ablation_not_stageA_replacement",
            })
    transfer = pd.DataFrame(transfer_rows)
    if len(transfer) != 8:
        raise ValueError("Expected four endpoints x two cross-species routes")

    dmpnn = pd.read_csv(args.dmpnn / "endpoint_decisions.csv")
    smiles = pd.read_csv(args.smiles / "candidate_ranking.csv")
    nested = pd.read_csv(args.nested_mlp / "metrics.csv")
    if len(dmpnn) != 6 or dmpnn.relative_change_percent.le(0).any():
        raise ValueError("D-MPNN stop evidence is incomplete")
    structure_winners = smiles.loc[smiles.rank_within_task.eq(1)]
    if len(structure_winners) != 6 or structure_winners.algorithm.ne("structure_control").any():
        raise ValueError("Frozen-SMILES matched adaptation does not support the registered stop decision")
    if nested.task_id.nunique() != 5 or set(nested.feature_set) != {"stageA_structure", "frozen_molformer"}:
        raise ValueError("Nested MLP representation audit is incomplete")

    active = pd.read_csv(args.endpoint_expansion / "active_endpoint_registry.csv")
    physchem = active.loc[active.endpoint_family.isin(["logP", "logD", "pKa"])].copy()
    logp = physchem.loc[physchem.endpoint_family.eq("logP")].iloc[0]
    logd = physchem.loc[physchem.endpoint_family.eq("logD")].iloc[0]
    pka = physchem.loc[physchem.endpoint_family.eq("pKa")]
    if int(logp.train_records) < 1000 or int(logd.train_records) < 1000 or pka.train_records.astype(int).sum() != 0:
        raise ValueError("Physchem Gate 3 feasibility differs from the frozen endpoint audit")

    stop_loss = build_stop_loss_registry()
    if stop_loss.gate3_protocol_authorized.sum() != 1 or stop_loss.gate3_training_authorized.any():
        raise ValueError("Exactly one protocol-only Gate 3 candidate must be selected")
    gate3 = pd.DataFrame([{
        "candidate_id": "G3_B6_nested_physchem_oof_late_fusion",
        "source_ablation_id": "B6",
        "candidate_family": "endpoint_specific_structure_plus_predicted_physchem",
        "structure_anchor": "frozen_corrected_StageA_endpoint_leader",
        "auxiliary_targets": "experimental_logP|experimental_logD_pH7.4",
        "pka_status": "deferred_until_licensed_or_public_experimental_cohort_is_audited",
        "pk_row_input_policy": "predictions_only_never_measured_physchem_values",
        "ancestry": "strict_nested_crossfit_excluding_downstream_outer_parent_and_scaffold",
        "fusion": "late_projection_with_explicit_missingness_masks_and_direct_structure_path",
        "comparator": "capacity_matched_structure_only_route_and_corrected_StageA",
        "primary_gate": "at_least_2pct_error_reduction_and_at_least_3of5_nonworse_folds",
        "bootstrap_gate": "paired_parent_bootstrap_95ci_high_below_zero",
        "source_sensitivity": "required_only_after_primary_gate",
        "human_F_selection": "prohibited",
        "fixed_validation_status": "closed",
        "test_status": "closed",
        "protocol_construction_authorized": True,
        "cohort_construction_authorized": False,
        "model_training_authorized": False,
        "reason": "independently_preregistered_untested_modality_with_7267_logP_and_4374_logD_training_records",
        "next_required_stage": "physchem_auxiliary_cohort_provenance_and_leakage_feasibility_audit",
    }])

    lifecycle = {
        "schema_version": 1,
        "gate2b_status": "closed_no_architecture_advanced",
        "strong_reference": "corrected_StageA_endpoint_leaders",
        "full_shared_status": "rejected",
        "two_group_status": "mechanistic_ablation_only",
        "cross_species_status": "mechanistic_transfer_ablations_only_no_StageA_replacement",
        "gate3_selected_candidate": gate3.iloc[0].candidate_id,
        "gate3_authorization": "protocol_and_feasibility_audit_only",
        "fixed_validation_status": "closed",
        "test_status": "closed",
        "forbidden_actions": [
            "continue_adaptive_rdkit2d_task_group_search_on_same_outer_cv",
            "start_PCGrad_GradNorm_or_MMoE_without_new_independent_evidence",
            "use_measured_physchem_values_as_PK_row_features",
            "use_non_nested_auxiliary_predictions",
            "pool_logP_and_logD_or_treat_predicted_pKa_as_experimental_truth",
            "select_human_F_or_open_fixed_validation_or_test",
        ],
        "next_stage": gate3.iloc[0].next_required_stage,
    }
    if args.check_only:
        print(
            "Gate 2B freeze ready: endpoints=6 transfer_routes=8 stopped_or_deferred=10 "
            "gate3_candidate=G3_B6_nested_physchem_oof_late_fusion training_authorized=False"
        )
        return

    input_hashes = {f"{label}_complete_sha256": sha256(path / "complete.json") for label, (path, _) in stages.items()}
    with stage_output(args.output) as out:
        endpoint.to_csv(out / "gate2b_endpoint_decision_registry.csv", index=False)
        transfer.to_csv(out / "cross_species_stop_loss_registry.csv", index=False)
        stop_loss.to_csv(out / "architecture_stop_loss_registry.csv", index=False)
        gate3.to_csv(out / "gate3_candidate_registry.csv", index=False)
        physchem.to_csv(out / "physchem_feasibility_snapshot.csv", index=False)
        pd.DataFrame([
            {"evidence_id": label, "path": str(path), "stage": stage,
             "complete_sha256": sha256(path / "complete.json")}
            for label, (path, stage) in stages.items()
        ]).to_csv(out / "input_evidence_registry.csv", index=False)
        (out / "lifecycle_contract.json").write_text(
            json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        figure = build_figure(stop_loss)
        figure.savefig(out / "Figure_gate2b_stop_loss_gate3_entry.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# Gate 2B decision freeze and Gate 3 stop-loss audit\n\n"
            "No Gate 2B neural architecture replaces the corrected Stage-A references. Full sharing is rejected; "
            "the capacity-matched fu/non-fu model is retained only as a mechanistic ablation. Current graph, frozen "
            "SMILES, cross-species and additional RDKit2D-sharing families are stopped or deferred. The only Gate 3 "
            "entry is protocol construction for nested-crossfit predicted experimental logP/logD late fusion. This "
            "stage authorizes neither cohort publication nor model training and opens no fixed-validation or test labels.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "gate2b_decision_freeze_gate3_audit", inputs=input_hashes,
            human_endpoints=6, cross_species_stageA_comparisons=8,
            architecture_families_audited=len(stop_loss), gate2b_architectures_advanced=0,
            gate3_candidates_selected=1, gate3_candidate_id=gate3.iloc[0].candidate_id,
            gate3_protocol_construction_authorized=True, gate3_cohort_construction_authorized=False,
            gate3_model_training_authorized=False, validation_target_file_opened=False,
            test_labels_read=False, model_fitted=False, data_modified=False, partial=False,
        )
    print(f"Gate 2B decision freeze and Gate 3 audit: {args.output}")


if __name__ == "__main__":
    run_cli(main)
