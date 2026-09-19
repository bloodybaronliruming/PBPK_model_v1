#!/usr/bin/env python3
"""Publish the leakage-guarded Gate 1B multimodal input protocol.

The protocol is metadata-only.  It binds each modality to an availability
gate, fold-local preprocessing rule, task permission, fusion rule, and OOF
ancestry requirement.  It does not compute representations, fit models, or
read fixed-validation/test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def modality_registry(feature_registry: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "modality_id": "ecfp4",
            "modality_class": "structure_2d",
            "provider": "RDKit Morgan radius=2",
            "current_status": "ready_cached",
            "dimensions": feature_registry["ecfp4"]["dimensions"],
            "fit_scope": "deterministic; no label fit",
            "missingness_rule": "not_missing_after_valid_parent_structure",
            "allowed_fusion": "single_view|early_concat|late_projection",
            "hard_gate": "canonical-parent hash must match frozen manifest",
        },
        {
            "modality_id": "rdkit2d",
            "modality_class": "structure_2d",
            "provider": "versioned RDKit descriptor registry",
            "current_status": "ready_cached",
            "dimensions": feature_registry["rdkit2d"]["dimensions"],
            "fit_scope": "imputation/scaling fit on training fold only",
            "missingness_rule": "descriptor NaN -> training-fold median plus descriptor-missing mask when used by neural fusion",
            "allowed_fusion": "single_view|early_concat|late_projection",
            "hard_gate": "remove target-derived descriptors for experimental logP/logD heads",
        },
        {
            "modality_id": "molecular_graph",
            "modality_class": "structure_graph",
            "provider": "project D-MPNN; 23 atom and 6 directed-bond features",
            "current_status": "adapter_ready_cache_not_materialized",
            "dimensions": "variable_graph_to_fixed_embedding",
            "fit_scope": "graph deterministic; encoder fit on training fold only",
            "missingness_rule": "not_missing_after_valid_parent_structure",
            "allowed_fusion": "graph_only_after_adapter|graph_plus_rdkit2d|late_projection",
            "hard_gate": "capacity-matched structure baseline and saved reload check required",
        },
        {
            "modality_id": "frozen_smiles_embedding",
            "modality_class": "structure_sequence",
            "provider": "provider_not_yet_selected",
            "current_status": "blocked_until_encoder_license_version_and_hash_are_frozen",
            "dimensions": "provider_specific",
            "fit_scope": "encoder weights frozen; downstream projection fit on training fold only",
            "missingness_rule": "tokenization failure -> modality missing plus explicit mask; never silently zero without mask",
            "allowed_fusion": "single_view|late_projection|gated_late_fusion",
            "hard_gate": "record model name, revision, license, tokenizer, pooling, max length and weight hash",
        },
        {
            "modality_id": "experimental_physchem_oof",
            "modality_class": "predicted_physchem",
            "provider": "leakage-safe logP/logD auxiliary heads; future acidic/basic pKa",
            "current_status": "blocked_until_auxiliary_cohort_and_crossfit_predictions_exist",
            "dimensions": "2 now; up to 4 after pKa audit",
            "fit_scope": "strict cross-fit by downstream frozen scaffold fold",
            "missingness_rule": "missing prediction -> explicit per-endpoint mask; no population or target-conditional fill",
            "allowed_fusion": "late_projection|gated_late_fusion",
            "hard_gate": "no measured physchem value and no target-derived RDKit descriptors may enter a PK row",
        },
        {
            "modality_id": "semantic_context",
            "modality_class": "experimental_context",
            "provider": "species|system|route|matrix/formulation when explicitly available",
            "current_status": "schema_ready",
            "dimensions": "training-vocabulary dependent",
            "fit_scope": "vocabulary and normalization fit on training fold only; UNK and MISSING reserved",
            "missingness_rule": "one explicit MISSING token per field",
            "allowed_fusion": "embedding_projection|gated_late_fusion",
            "hard_gate": "only cross-species/multitask use; constant context is prohibited in endpoint-specific human STL",
        },
        {
            "modality_id": "mechanistic_pk_oof",
            "modality_class": "predicted_mechanistic",
            "provider": "permitted upstream PK relations from physics contract",
            "current_status": "blocked_until_nested_crossfit_ancestry_is_materialized",
            "dimensions": "target_specific",
            "fit_scope": "strict cross-fit by downstream fold; validation producer fit on train only",
            "missingness_rule": "per-upstream prediction mask; retain direct structure path",
            "allowed_fusion": "late_projection|residual_candidate|gated_late_fusion",
            "hard_gate": "predictions only; measured upstream values and hard physical closure are prohibited",
        },
        {
            "modality_id": "animal_pk_oof",
            "modality_class": "predicted_cross_species",
            "provider": "species-specific auxiliary heads with explicit species token",
            "current_status": "data_ready_predictions_not_materialized",
            "dimensions": "available-species-by-endpoint dependent",
            "fit_scope": "strict cross-fit excluding downstream held-out parent and scaffold",
            "missingness_rule": "per-species prediction mask; never substitute animal observation for human truth",
            "allowed_fusion": "late_projection|gated_late_fusion",
            "hard_gate": "human-only STL remains the comparator; report negative transfer",
        },
    ])


def task_permissions(tasks: pd.DataFrame, pk_tasks: pd.DataFrame) -> pd.DataFrame:
    core = tasks[["task_id", "endpoint"]].copy()
    core = pd.concat([core, pd.DataFrame([{"task_id": "F__human__absolute_oral", "endpoint": "F"}])], ignore_index=True)
    animal = (pk_tasks.loc[pk_tasks.species.ne("human")]
              .groupby("endpoint", as_index=False)
              .agg(animal_tasks=("task_id", "nunique"), animal_records=("development_records", "sum")))
    core = core.merge(animal, on="endpoint", how="left").fillna({"animal_tasks": 0, "animal_records": 0})
    relationship = {
        "CL": "fu__human__plasma|CLint__human__microsome",
        "F": "Papp__human__caco2_ab",
        "Thalf": "CL__human__systemic_iv|VDss__human__steady_state_iv",
    }
    rows = []
    for row in core.itertuples(index=False):
        rows.append({
            "task_id": row.task_id,
            "endpoint": row.endpoint,
            "ecfp4_rdkit2d": "allowed_structure_baseline",
            "molecular_graph": "allowed_incremental_ablation",
            "frozen_smiles_embedding": "allowed_after_provider_gate",
            "semantic_context": "prohibited_in_endpoint_specific_human_STL; allowed_in_cross_species_or_multitask",
            "experimental_physchem_oof": "allowed_after_nested_crossfit_gate",
            "mechanistic_pk_oof_inputs": relationship.get(row.endpoint, "none_per_physics_contract"),
            "animal_pk_oof": "allowed_matching_endpoint_after_nested_crossfit" if row.animal_tasks else "unavailable_no_matching_animal_task",
            "matching_animal_tasks": int(row.animal_tasks),
            "matching_animal_records": int(row.animal_records),
            "head_selection_authorized": row.endpoint != "F",
            "selection_boundary": "train_CV_then_fixed_validation" if row.endpoint != "F" else "representation_and_sensitivity_only_no_validation",
        })
    return pd.DataFrame(rows).sort_values("task_id", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--stageA-audit", type=Path, default=ROOT / "results/analysis/stl_stageA_three_view_audit_v2")
    parser.add_argument("--endpoint-expansion", type=Path, default=ROOT / "results/analysis/endpoint_expansion_feasibility_v1")
    parser.add_argument("--physics-contract", type=Path, default=ROOT / "results/analysis/multitask_physics_explainability_contract_v1")
    parser.add_argument("--training-interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--pk-registry", type=Path, default=ROOT / "data/public_development/multitask_pk_24h_v6/task_registry.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/gate1b_multimodal_protocol_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.stl_protocol / "complete.json", args.stl_protocol / "task_manifest.csv", args.stl_protocol / "feature_registry.json",
        args.stageA_audit / "complete.json", args.stageA_audit / "frozen_stageA_shortlist.csv",
        args.endpoint_expansion / "complete.json", args.endpoint_expansion / "physchem_feature_leakage_exclusions.csv",
        args.physics_contract / "complete.json", args.physics_contract / "relationship_contract.csv",
        args.training_interface / "complete.json", args.pk_registry,
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    audit_meta = verify_stage(args.stageA_audit, "stl_stageA_three_view_audit")
    expansion_meta = verify_stage(args.endpoint_expansion, "endpoint_expansion_feasibility")
    physics_meta = verify_stage(args.physics_contract, "multitask_physics_explainability_contract")
    interface_meta = verify_stage(args.training_interface, "multitask_pk_training_interface")
    if stl_meta.get("test_labels_read") or audit_meta.get("validation_labels_read") or audit_meta.get("test_labels_read"):
        raise ValueError("Gate 1B protocol requires the Stage A shortlist to remain validation/test blind")
    if expansion_meta.get("validation_or_test_labels_read") or interface_meta.get("test_labels_read"):
        raise ValueError("An upstream endpoint/interface stage reports protected-label access")
    if not physics_meta.get("label_safe") or audit_meta.get("next_gate") != "Gate1B_multimodal_incremental_ablation":
        raise ValueError("Gate 1B prerequisites are not satisfied")

    input_hashes = {
        "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
        "stageA_audit_complete_sha256": sha256(args.stageA_audit / "complete.json"),
        "endpoint_expansion_complete_sha256": sha256(args.endpoint_expansion / "complete.json"),
        "physics_contract_complete_sha256": sha256(args.physics_contract / "complete.json"),
        "training_interface_complete_sha256": sha256(args.training_interface / "complete.json"),
        "pk_registry_sha256": sha256(args.pk_registry),
    }
    if args.check_only and (args.output / "complete.json").exists():
        published = verify_stage(args.output, "gate1b_multimodal_input_protocol")
        if published.get("inputs") != input_hashes:
            raise ValueError("Published Gate 1B protocol no longer matches its frozen inputs")
        print(
            f"Gate 1B protocol valid: modalities={published['registered_modalities']} "
            f"tasks={published['core_tasks']} ablations={published['ablation_steps']}"
        )
        return

    tasks = pd.read_csv(args.stl_protocol / "task_manifest.csv")
    pk_tasks = pd.read_csv(args.pk_registry)
    feature_registry = json.loads((args.stl_protocol / "feature_registry.json").read_text(encoding="utf-8"))
    modalities = modality_registry(feature_registry)
    permissions = task_permissions(tasks, pk_tasks)
    if len(tasks) != 6 or len(permissions) != 7 or permissions.head_selection_authorized.sum() != 6:
        raise ValueError("Gate 1B requires six selectable human heads plus non-selectable human F")

    ablations = pd.DataFrame([
        ("B0", "RDKit2D Stage-A endpoint leader", "existing", "common classical structure reference", "none"),
        ("B1", "ECFP4 + RDKit2D", "existing", "same-source two-view structure reference", "B0"),
        ("B2", "D-MPNN graph + RDKit2D", "implement_then_long_GPU", "graph incremental value with numeric anchor", "B0"),
        ("B3", "frozen SMILES embedding", "provider_gate_then_long_compute", "independent frozen sequence representation", "B0"),
        ("B4", "best 2D + graph + SMILES gated late fusion", "blocked", "structure-multimodal interaction", "B1|B2|B3"),
        ("B5", "B4 + semantic context", "cross_species_or_multitask_only", "conditional translation; never endpoint-specific constant tokens", "B4"),
        ("B6", "best structure + experimental physchem OOF", "blocked", "incremental physicochemical representation", "leakage_safe_logP_logD_cohort"),
        ("B7", "best structure + permitted mechanistic PK OOF", "blocked", "target-specific conditional mechanism", "nested_upstream_ancestry"),
        ("B8", "best human structure + matching animal PK OOF", "blocked", "cross-species transfer and negative-transfer test", "nested_animal_ancestry"),
    ], columns=["ablation_id", "input_configuration", "current_status", "scientific_question", "prerequisite"])

    ancestry = pd.DataFrame([
        ("OOF_train", "For downstream fold k, every upstream producer excludes all parents and scaffolds assigned to fold k.", "producer_train_parent_hash|producer_train_scaffold_hash|heldout_fold|model_hash"),
        ("fixed_validation", "Producer is fit only on frozen train after train-OOF hyperparameter selection; validation labels are never producer inputs.", "producer_train_parent_hash|selected_config_hash|model_hash"),
        ("external_or_test", "Only a fully frozen producer may infer; labels, calibration and selection remain inaccessible.", "frozen_model_hash|input_membership_hash"),
        ("same_molecule_cross_task", "Measured upstream labels are prohibited; prediction must come from a producer that excluded the downstream held-out parent/scaffold.", "downstream_row_id|upstream_task|producer_ancestry_hash"),
    ], columns=["prediction_scope", "required_exclusion", "required_saved_ancestry"])

    fusion = pd.DataFrame([
        ("early_concat", "fixed-size deterministic 2D only", "fold-fit imputation/scaling", "same head and bounded budget as comparator"),
        ("late_projection", "graph/SMILES/predicted modalities", "one projection per modality; concatenate projections and masks", "capacity-matched direct-structure path retained"),
        ("gated_late_fusion", "two or more optional modalities", "soft gates conditioned on availability masks", "report gate distribution; no source/reliability inputs"),
        ("missing_modality", "all optional modalities", "zero after projection plus explicit binary mask", "no target-conditioned or global-label imputation"),
    ], columns=["fusion_mode", "allowed_modalities", "implementation_rule", "comparison_rule"])

    advancement = pd.DataFrame([
        (1, "train_CV_completeness", "100% authorized train rows predicted exactly once; finite values; reload check passes", "reject"),
        (2, "numerical_stability", "no predeclared extreme-output or unattributed convergence failure", "reject"),
        (3, "material_OOF_gain", "primary error improves >=2% and at least 3/5 folds are non-worse", "strong_advance"),
        (4, "modality_diversity", "within 1% of endpoint best; at most one distinct-modality candidate per endpoint", "diversity_advance"),
        (5, "shortlist_cap", "maximum 3 Gate1B candidates per endpoint before fixed validation", "rank_train_CV_only"),
        (6, "fixed_validation_confirmation", "opened only after shortlist freeze; require <=2% degradation vs structure reference and report source-disjoint sensitivity", "confirm_or_reject"),
        (7, "multi_seed", "three seeds required before replacing a structure baseline; report mean, SD and paired OOF bootstrap", "final_candidate_gate"),
    ], columns=["order", "gate", "criterion", "decision"])

    leakage = pd.read_csv(args.endpoint_expansion / "physchem_feature_leakage_exclusions.csv")
    generic = pd.DataFrame([
        ("all", "source_id|doc_id|assay_id", "prohibited_predictor", "prevents source shortcut"),
        ("all", "reliability_tier|evidence_quality_tier", "prohibited_predictor", "weighting/stratification only"),
        ("all", "split|inner_fold_id|test_membership", "prohibited_predictor", "administrative metadata"),
        ("PK", "measured_upstream_endpoint", "prohibited_predictor", "use strict OOF prediction only"),
        ("F", "relative_F_or_oral_only_exposure", "prohibited_label", "not absolute oral bioavailability"),
        ("pKa", "predicted_database_pKa", "prohibited_experimental_label", "prediction is not an experimental target"),
    ], columns=["target_family", "excluded_feature", "rule", "reason"])
    leakage = pd.concat([leakage[["target_family", "excluded_feature", "rule", "reason"]], generic], ignore_index=True)

    compute = pd.DataFrame([
        (1, "graph_manifest_and_adapter_smoke", "short", "assistant", "no user command yet", "B2 implementation prerequisite"),
        (2, "D-MPNN five-fold Stage1B screen", "long_GPU", "user_background", "runner to be created and smoke-tested first", "execute sequentially on one GPU"),
        (3, "frozen_SMILES_provider_license_and_revision_audit", "short_research", "assistant", "no download until provider frozen", "B3 prerequisite"),
        (4, "frozen_SMILES_embedding_materialization", "medium_or_long_GPU", "user_background", "runner to be created after provider gate", "one immutable parent-keyed cache"),
        (5, "logP_logD_isolated_cohort_and_crossfit", "medium_CPU", "assistant_then_user_if_runtime_exceeds_15min", "not started", "B6 prerequisite"),
        (6, "mechanistic_and_animal_nested_OOF", "long_CPU_or_GPU", "user_background", "only after structural shortlist", "B7/B8 prerequisite"),
    ], columns=["priority", "job", "runtime_class", "owner", "command_status", "dependency"])

    summary = {
        "core_tasks": 7,
        "head_selection_authorized_tasks": 6,
        "registered_modalities": len(modalities),
        "ablation_steps": len(ablations),
        "fixed_validation_authorized_now": False,
        "test_labels_read": False,
        "model_fitted": False,
        "next_implementation": "graph_manifest_adapter_and_smoke",
        "first_delegated_long_task": "DMPNN_five_fold_Stage1B_after_runner_smoke",
    }
    if args.check_only:
        print(f"Gate 1B protocol valid: modalities={len(modalities)} tasks={len(permissions)} ablations={len(ablations)}")
        return
    with stage_output(args.output) as out:
        modalities.to_csv(out / "modality_registry.csv", index=False)
        permissions.to_csv(out / "task_modality_permissions.csv", index=False)
        ablations.to_csv(out / "incremental_ablation_ladder.csv", index=False)
        ancestry.to_csv(out / "oof_ancestry_contract.csv", index=False)
        fusion.to_csv(out / "fusion_and_missingness_contract.csv", index=False)
        advancement.to_csv(out / "candidate_advancement_rules.csv", index=False)
        leakage.to_csv(out / "leakage_prohibition_registry.csv", index=False)
        compute.to_csv(out / "compute_and_delegation_plan.csv", index=False)
        (out / "protocol.json").write_text(json.dumps({
            "comparison_anchor": "same endpoint, same frozen folds, same primary metric, Stage-A structure leader",
            "selection_sequence": "train CV -> frozen Gate1B shortlist -> fixed validation -> three-seed confirmation",
            "fusion_policy": "incremental and capacity-audited; no unconstrained all-modality search",
            "context_policy": "semantic context only in cross-species/multitask settings where it varies",
            "missingness_policy": "explicit per-modality masks; no target-informed imputation",
            "test_policy": "test/external labels unavailable for representation, selection, calibration and attribution",
            "human_F_policy": "representation/sensitivity only until a real validation cohort exists",
            "AUC_Cmax_policy": "deferred and absent from Gate1B",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / "README.md").write_text(
            "# Gate 1B multimodal input protocol v1\n\n"
            "Metadata-only protocol for incremental structure, sequence, context, predicted physicochemical, mechanistic, and cross-species modalities. "
            "Every optional modality has an availability mask and a prerequisite gate. OOF-derived inputs carry producer ancestry that excludes the downstream held-out parent and scaffold. "
            "Fixed validation remains closed until a bounded Gate 1B shortlist is frozen; test labels remain unread.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate1b_multimodal_input_protocol", inputs=input_hashes, **summary, partial=False)
    print(f"Gate 1B multimodal protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
