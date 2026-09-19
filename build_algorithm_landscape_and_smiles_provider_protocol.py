#!/usr/bin/env python3
"""Freeze the broad algorithm landscape and the Gate 1B SMILES-provider contract.

This is a metadata-only stage.  It keeps every scientifically plausible model
family visible, but separates evidence retention from permission to spend
compute or open protected validation labels.  Remote provider metadata is
recorded at immutable revisions; this script performs no download and fits no
model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def algorithm_landscape() -> pd.DataFrame:
    columns = [
        "family", "algorithm", "representation", "current_status", "project_role",
        "principal_strength", "principal_limit", "compute_class", "dependency_status",
        "priority", "activation_gate", "retention_policy",
    ]
    rows = [
        # References and linear models
        ("reference", "Dummy median/mean", "target only", "completed_reference", "non-informative floor", "sanity and leakage detection", "no chemistry", "tiny_cpu", "ready", "mandatory", "always report", "retain_all_results"),
        ("linear", "Ridge", "RDKit2D|ECFP4|frozen embedding", "completed_and_scheduled", "stable linear reference", "strong high-dimensional regularization", "linear response", "small_cpu", "ready", "mandatory", "bounded train-CV", "retain_all_results"),
        ("linear", "ElasticNet/Lasso", "scaled RDKit2D|ECFP4", "completed_reference", "sparse linear ablation", "feature selection and interpretability", "correlated features destabilize selection", "small_cpu", "ready", "reference", "reactivate for sparse interpretation", "retain_artifacts_reactivate_if_gate_met"),
        ("linear", "Partial least squares", "scaled RDKit2D", "scheduled", "chemometric baseline", "handles collinearity", "component selection can overfit", "small_cpu", "ready_sklearn", "secondary", "bounded components in inner CV", "retain_artifacts_reactivate_if_gate_met"),
        ("robust_linear", "Huber/RANSAC/Theil-Sen", "scaled RDKit2D", "conditional_future", "outlier sensitivity ablation", "robust loss", "poor scaling or weak nonlinear capacity", "small_to_medium_cpu", "ready_sklearn", "conditional", "activate if influence audit finds outliers", "retain_registry"),
        ("robust_linear", "Quantile regression", "scaled RDKit2D", "conditional_future", "conditional interval/median model", "distributional target view", "not a primary mean-error optimizer", "medium_cpu", "ready_sklearn", "conditional", "activate after point-model shortlist", "retain_registry"),
        ("additive", "GAM/splines", "selected RDKit2D", "conditional_future", "interpretable nonlinear baseline", "shape plots and smooth effects", "feature-count and interaction limits", "medium_cpu", "additional_package_needed", "conditional", "activate for interpretation subset", "retain_registry"),
        # Similarity, kernel and probabilistic models
        ("instance", "k-nearest neighbours", "scaled RDKit2D|Tanimoto ECFP4", "completed_reference", "local-similarity baseline", "transparent applicability domain", "weak extrapolation and expensive inference", "medium_cpu", "ready", "reference", "retain even if noncompetitive", "retain_all_results"),
        ("kernel", "SVR", "scaled RDKit2D", "completed_reference", "nonlinear classical baseline", "sample-efficient nonlinear fit", "quadratic scaling", "medium_cpu", "ready", "secondary", "bounded train-CV only", "retain_artifacts_reactivate_if_gate_met"),
        ("kernel", "Tanimoto SVM/SVR", "ECFP4", "scheduled", "fingerprint-native kernel", "chemically meaningful similarity", "kernel matrix scaling", "medium_to_large_cpu", "custom_adapter_needed", "secondary", "Nyström or endpoint-size gate", "retain_registry"),
        ("kernel", "Kernel ridge", "scaled RDKit2D|Tanimoto ECFP4", "scheduled", "regularized kernel reference", "stable closed-form objective", "quadratic/cubic scaling", "medium_to_large_cpu", "ready_partial", "secondary", "endpoint-size or Nyström gate", "retain_registry"),
        ("probabilistic", "Exact Gaussian process", "compact descriptors", "conditional_future", "uncertainty reference", "principled posterior", "cubic scaling", "large_cpu", "gpytorch_not_installed", "conditional", "small-task size gate", "retain_registry"),
        ("probabilistic", "Sparse GP/deep-kernel GP", "descriptors|learned embedding", "conditional_future", "calibrated uncertainty", "scalable posterior approximation", "inducing-point and kernel sensitivity", "large_gpu", "gpytorch_not_installed", "conditional", "after frozen representation shortlist", "retain_registry"),
        # Tree and boosting family
        ("tree", "Random forest", "RDKit2D|ECFP4|combined", "completed_reference", "bagged nonlinear baseline", "robust and interpretable", "limited smooth extrapolation", "medium_cpu", "ready", "mandatory", "bounded train-CV", "retain_all_results"),
        ("tree", "ExtraTrees", "RDKit2D|ECFP4|combined|frozen embedding", "completed_and_scheduled", "strong randomized-tree baseline", "strong on mixed nonlinear descriptors", "piecewise-constant extrapolation", "medium_cpu", "ready", "mandatory", "bounded train-CV", "retain_all_results"),
        ("boosting", "Histogram gradient boosting", "RDKit2D|combined", "completed_reference", "native sklearn boosting", "fast nonlinear tabular fit", "hyperparameter sensitivity", "medium_cpu", "ready", "mandatory", "bounded train-CV", "retain_all_results"),
        ("boosting", "XGBoost", "RDKit2D|ECFP4|combined|frozen embedding", "completed_and_scheduled", "strong boosting baseline", "regularized scalable trees", "search and calibration cost", "medium_cpu_or_gpu", "ready", "mandatory", "bounded train-CV", "retain_all_results"),
        ("boosting", "LightGBM", "RDKit2D|combined|frozen embedding", "completed_and_scheduled", "strong fast tabular baseline", "efficient leaf-wise boosting", "small-data overfit risk", "medium_cpu_or_gpu", "ready", "mandatory", "bounded train-CV", "retain_all_results"),
        ("boosting", "CatBoost", "RDKit2D|context|combined", "completed_reference", "ordered-boosting/context candidate", "handles mixed context and missingness", "extra compute and tuning", "medium_cpu_or_gpu", "ready", "secondary", "context only when nonconstant and authorized", "retain_all_results"),
        ("boosting", "NGBoost", "RDKit2D|frozen embedding", "conditional_future", "probabilistic boosting", "predictive distribution", "additional dependency and runtime", "medium_cpu", "ngboost_not_installed", "conditional", "after point-model shortlist", "retain_registry"),
        # Tabular neural models
        ("tabular_deep", "MLP/ResMLP", "scaled RDKit2D|combined|frozen embedding", "completed_and_scheduled", "capacity-matched neural baseline", "flexible fusion and multitask reuse", "seed sensitivity", "medium_gpu", "ready_torch", "mandatory", "multi-seed after screening", "retain_all_results"),
        ("tabular_deep", "TabNet", "RDKit2D", "conditional_future", "attention-based tabular comparator", "built-in feature masks", "often unstable on small data", "medium_gpu", "pytorch_tabnet_not_installed", "conditional", "only after strong trees", "retain_registry"),
        ("tabular_deep", "FT-Transformer/SAINT", "RDKit2D|context", "conditional_future", "tabular transformer comparator", "feature interactions", "compute and tuning hungry", "medium_gpu", "adapter_needed", "conditional", "enough rows and strong tree anchor", "retain_registry"),
        ("tabular_foundation", "TabPFN", "compact descriptor subset", "conditional_future", "small-tabular foundation baseline", "fast low-data prior", "feature/row limits and license/version audit", "medium_gpu", "tabpfn_not_installed", "conditional", "provider and dimensionality gate", "retain_registry"),
        # Graph neural networks
        ("graph", "Project D-MPNN", "2D molecular graph", "retained_negative_ablation", "Gate1B graph evidence", "direct learned graph message passing", "0/6 endpoints passed current advancement gate", "large_gpu", "ready", "reference", "reactivate only as pretrained/shared or materially changed encoder", "retain_all_results_no_validation"),
        ("graph", "Chemprop D-MPNN", "2D molecular graph", "available_reference", "external implementation cross-check", "mature molecular property stack", "implementation/search duplication", "large_gpu", "chemprop_ready", "secondary", "only if project D-MPNN implementation doubt arises", "retain_registry"),
        ("graph", "GCN", "2D molecular graph", "conditional_future", "minimal graph architecture", "simple capacity control", "oversmoothing and weaker edge handling", "medium_gpu", "pyg_ready", "reference", "one bounded config if graph family revisited", "retain_registry"),
        ("graph", "GAT/GATv2", "2D molecular graph", "conditional_future", "attention graph ablation", "learned neighbor weighting", "attention does not guarantee chemistry gain", "medium_gpu", "pyg_ready", "conditional", "graph reactivation gate", "retain_registry"),
        ("graph", "GIN/GINE", "2D molecular graph", "conditional_future", "expressive graph comparator", "strong graph-isomorphism inductive bias", "needs regularization and edge design", "medium_gpu", "pyg_ready", "secondary", "graph reactivation gate", "retain_registry"),
        ("graph", "AttentiveFP", "2D molecular graph", "conditional_future", "molecule-focused graph comparator", "attention/readout designed for molecules", "additional search cost", "medium_gpu", "adapter_needed", "secondary", "graph reactivation gate", "retain_registry"),
        ("graph", "Graph Transformer/GPS", "2D graph plus positional encoding", "conditional_future", "long-range graph comparator", "global interactions", "small-data and compute risk", "large_gpu", "adapter_needed", "conditional", "only after pretrained/frozen sequence evidence", "retain_registry"),
        ("graph_pretrained", "GROVER/Mole-BERT/GraphMVP", "pretrained 2D graph", "conditional_future", "pretraining benefit test", "may overcome current supervised D-MPNN limit", "provider/license/revision heterogeneity", "large_gpu", "provider_audit_needed", "secondary", "only one audited provider after sequence gate", "retain_registry"),
        # SMILES and molecular foundation models
        ("smiles_frozen", "MoLFormer XL 10%", "canonical non-isomeric SMILES embedding", "selected_primary_provider", "Gate1B frozen sequence representation", "large-scale bidirectional molecular pretraining", "202-token limit and remote custom code", "medium_gpu_cache_then_small_models", "transformers_ready_weights_pending", "next", "pinned-code smoke then immutable cache", "retain_all_results"),
        ("smiles_frozen", "ChemBERTa 100M MLM", "canonical SMILES embedding", "selected_backup_provider", "provider-failure backup", "permissive license and standard Transformers", "not primary to avoid provider zoo", "medium_gpu_cache_then_small_models", "transformers_ready_weights_pending", "backup", "activate only if primary compatibility/coverage fails", "retain_registry"),
        ("smiles_frozen", "ChemBERTa-77M-MTR", "SMILES embedding", "literature_reference_blocked", "multitask-pretraining literature reference", "property-aware pretraining rationale", "model card lacks license metadata", "medium_gpu", "weights_exist_license_gate_failed", "reference", "do not download/use until license resolved", "retain_registry"),
        ("smiles_frozen", "SMI-TED/SELFIES encoders", "SMILES|SELFIES embedding", "conditional_future", "alternative sequence representation", "robust or compact pretrained encoding", "new provider audit and model-zoo risk", "medium_to_large_gpu", "provider_audit_needed", "conditional", "activate only after primary frozen-embedding result", "retain_registry"),
        ("smiles_finetune", "Full pretrained-SMILES fine-tuning", "token sequence", "conditional_future", "upper-capacity sequence model", "task adaptation", "overfit and high seed/search cost", "large_gpu", "provider_dependent", "conditional", "frozen encoder must first show signal", "retain_registry"),
        ("smiles_finetune", "LoRA/adapters/prompt tuning", "token sequence", "conditional_future", "parameter-efficient adaptation", "lower trainable capacity", "method-specific tuning", "large_gpu", "adapter_dependency_needed", "conditional", "frozen encoder signal plus multi-seed gate", "retain_registry"),
        # 3D methods
        ("three_dimensional", "SchNet/DimeNet/PaiNN", "generated conformer geometry", "conditional_future", "3D inductive-bias ablation", "geometric equivariance/invariance", "conformer uncertainty and compute", "large_gpu", "conformer_pipeline_missing", "conditional", "frozen conformer/protonation protocol required", "retain_registry"),
        ("three_dimensional", "Uni-Mol or audited 3D foundation model", "conformer ensemble", "conditional_future", "pretrained 3D comparator", "large-scale geometric pretraining", "provider/license and conformer-domain shift", "large_gpu", "provider_audit_needed", "conditional", "2D/sequence plateau plus conformer gate", "retain_registry"),
        ("three_dimensional", "Conformer ensemble pooling", "multiple low-energy conformers", "conditional_future", "uncertainty-aware 3D aggregation", "reduces arbitrary single-conformer choice", "substantial preprocessing", "very_large_cpu_gpu", "pipeline_missing", "conditional", "single-conformer smoke and reproducibility gate", "retain_registry"),
        # Transfer, multitask, mechanisms and ensembles
        ("cross_species", "Species-conditioned pretrain then human fine-tune", "structure plus species token", "planned", "animal-to-human transfer", "uses more labels without mixing truth", "negative transfer", "large_gpu", "interface_ready", "high_after_STL", "human-only STL anchor and nested isolation", "retain_all_results"),
        ("cross_species", "Animal-head OOF stacking", "animal OOF predictions plus structure", "planned", "explicit translational feature", "auditable species contribution", "nested crossfit cost", "large_cpu_gpu", "ancestry_pipeline_needed", "secondary", "producer excludes held-out parent/scaffold", "retain_registry"),
        ("domain_adaptation", "CORAL/MMD/adversarial domain adaptation", "learned representation", "conditional_future", "species/source shift correction", "targets distribution shift", "can erase useful biology or leak domains", "large_gpu", "adapter_needed", "conditional", "measured domain-shift and negative-transfer gate", "retain_registry"),
        ("multitask", "Shared encoder with private towers", "multi-endpoint/multi-species", "implemented_smoke", "core future MTL comparator", "statistical sharing with endpoint specificity", "negative transfer", "large_gpu", "ready", "after_STL_transfer", "only after strong STL and transfer baselines", "retain_all_results"),
        ("multitask", "Cross-stitch/Sluice", "task-specific latent features", "conditional_future", "soft representation sharing", "learned sharing strength", "parameter/search growth", "large_gpu", "adapter_needed", "conditional", "shared-private benchmark gate", "retain_registry"),
        ("multitask", "MMoE/PLE", "multi-gate experts", "planned", "heterogeneous-task sharing", "reduces task interference", "expert collapse and tuning cost", "large_gpu", "adapter_needed", "secondary", "shared-private must first be competitive", "retain_registry"),
        ("multitask", "Task-conditioned hypernetwork/FiLM", "task and species tokens", "conditional_future", "conditional parameter sharing", "continuous context conditioning", "complex attribution", "large_gpu", "adapter_needed", "conditional", "MMoE/PLE evidence gate", "retain_registry"),
        ("physics_informed", "Mechanistic OOF cascade/residual model", "structure plus upstream OOF predictions", "planned", "soft PK relationship test", "adds interpretable physical pathway", "nested ancestry and error propagation", "large_cpu_gpu", "contract_ready_predictions_missing", "high_after_structure", "strict nested crossfit", "retain_all_results"),
        ("physics_informed", "Soft consistency regularization", "joint PK outputs", "conditional_future", "physical plausibility", "encourages coherent predictions", "incorrect constraints can bias", "large_gpu", "contract_ready", "conditional", "unconstrained MTL comparator first", "retain_registry"),
        ("ensemble", "Simple mean/median/rank ensemble", "OOF predictions", "planned", "low-variance ensemble", "robust and transparent", "needs complementary models", "small_cpu", "ready", "after_shortlist", "OOF-only member selection", "retain_all_results"),
        ("ensemble", "Nonnegative linear/NNLS stacking", "OOF predictions", "planned", "bounded learned ensemble", "interpretable weights", "nested fitting required", "small_cpu", "ready", "after_shortlist", "outer-fold-safe meta-fit", "retain_all_results"),
        ("ensemble", "Bayesian model averaging", "predictive distributions", "conditional_future", "uncertainty-weighted ensemble", "principled model averaging", "prior/likelihood assumptions", "medium_cpu", "adapter_needed", "conditional", "calibrated member distributions required", "retain_registry"),
        ("uncertainty", "Deep ensemble/MC dropout", "neural predictions", "conditional_future", "epistemic uncertainty", "simple and useful uncertainty signal", "multiplicative compute or approximation", "large_gpu", "ready_partial", "conditional", "final neural candidate only", "retain_registry"),
        ("uncertainty", "Conformal prediction", "held-out or cross-conformal residuals", "planned", "coverage intervals", "distribution-free marginal coverage", "small calibration sets and shift", "small_cpu", "ready", "publication", "model selection frozen before calibration", "retain_all_results"),
    ]
    return pd.DataFrame(rows, columns=columns)


def provider_registry() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "provider_role": "primary", "model_id": "ibm-research/MoLFormer-XL-both-10pct",
            "immutable_revision": "7b12d946c181a37f6012b9dc3b002275de070314",
            "compatibility_ref": "compat-v4", "license": "apache-2.0", "license_gate": "pass",
            "parameters": 46_800_000, "weight_file": "model.safetensors", "weight_bytes": 187_248_784,
            "hidden_size": 768, "max_model_tokens": 202, "pooling": "pooler_output",
            "remote_custom_code": True, "trust_policy": "exact_revision_only_then_local_code_audit",
            "pretraining_corpus": "10% ZINC plus 10% PubChem; non-isomeric canonical SMILES",
            "input_contract": "project graph_smiles; non-isomeric H-suppressed parent; no silent truncation",
            "selection_status": "selected_for_smoke", "activation_rule": "primary unless smoke safety/coverage/repeatability fails",
            "official_url": "https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct",
        },
        {
            "provider_role": "backup", "model_id": "DeepChem/ChemBERTa-100M-MLM",
            "immutable_revision": "f5c45f44d3061f0346888f5c09db17ec1146d29d",
            "compatibility_ref": "immutable_sha", "license": "mit", "license_gate": "pass",
            "parameters": 92_100_000, "weight_file": "model.safetensors", "weight_bytes": 368_569_864,
            "hidden_size": 768, "max_model_tokens": 512, "pooling": "attention_masked_mean_last_hidden_state",
            "remote_custom_code": False, "trust_policy": "standard_transformers_exact_revision",
            "pretraining_corpus": "subset of 100M ZINC20 SMILES; masked-language modelling",
            "input_contract": "same project graph_smiles; no silent truncation",
            "selection_status": "not_downloaded_backup", "activation_rule": "only if primary fails compatibility, coverage or determinism gate",
            "official_url": "https://huggingface.co/DeepChem/ChemBERTa-100M-MLM",
        },
        {
            "provider_role": "literature_reference_only", "model_id": "DeepChem/ChemBERTa-77M-MTR",
            "immutable_revision": "66b895cab8adebea0cb59a8effa66b2020f204ca",
            "compatibility_ref": "immutable_sha", "license": "missing_from_model_card", "license_gate": "fail_until_resolved",
            "parameters": pd.NA, "weight_file": "pytorch_model.bin", "weight_bytes": pd.NA,
            "hidden_size": 384, "max_model_tokens": 512, "pooling": "not_authorized",
            "remote_custom_code": False, "trust_policy": "do_not_use_without_license_resolution",
            "pretraining_corpus": "ChemBERTa-2 MTR literature family; 200 computed RDKit properties",
            "input_contract": "none; reference only",
            "selection_status": "blocked_not_discarded", "activation_rule": "license metadata must become explicit before any download/use",
            "official_url": "https://huggingface.co/DeepChem/ChemBERTa-77M-MTR",
        },
    ])


def cache_contract() -> pd.DataFrame:
    rows = [
        ("identity", "cache_primary_key", "molecule_id", "one immutable parent-keyed row"),
        ("identity", "input_smiles", "graph_smiles", "same canonical non-isomeric H-suppressed parent as graph manifest"),
        ("identity", "input_sha256", "required", "hash UTF-8 SMILES and bind to graph-manifest hash"),
        ("provider", "model_id_and_revision", "required", "exact model ID and immutable commit SHA"),
        ("provider", "weights_and_code_hashes", "required", "hash cached weight, tokenizer, config and every remote-code file"),
        ("tokenization", "truncation", "false", "overflow is an explicit failure; never silently truncate"),
        ("tokenization", "token_count", "required", "count including model special tokens"),
        ("tokenization", "unknown_token_count", "required", "coverage diagnostic"),
        ("tokenization", "maximum_length", "202_primary_512_backup", "provider config, not tokenizer sentinel"),
        ("embedding", "dtype_and_dimension", "float32_x_768", "save a dense parent-keyed matrix"),
        ("embedding", "pooling", "provider_registry", "primary pooler_output; backup masked mean"),
        ("embedding", "normalization", "none_in_cache", "fold-local scaling/projection only downstream"),
        ("failure", "valid_mask", "required", "invalid rows never become unmarked zero vectors"),
        ("failure", "failure_reason", "required_if_invalid", "token_overflow, tokenization, inference, or nonfinite"),
        ("runtime", "evaluation_mode", "model.eval_and_no_grad", "set deterministic_eval for primary"),
        ("runtime", "batch_order", "sorted_molecule_id", "batching cannot change row alignment"),
        ("runtime", "repeatability", "max_abs_difference_le_1e-6", "same device, same batch, two passes"),
        ("runtime", "cpu_gpu_consistency", "max_abs_difference_le_1e-4", "diagnostic on a fixed small subset; not bitwise"),
        ("runtime", "reload_integrity", "embedding_file_sha256", "verify metadata and shape after reload"),
        ("leakage", "labels_read", "false", "cache materialization is label-free"),
        ("leakage", "split_fit", "none", "same frozen parent embedding can be indexed by authorized folds"),
    ]
    return pd.DataFrame(rows, columns=["contract_group", "field", "required_value", "scientific_reason"])


def ablation_plan() -> pd.DataFrame:
    return pd.DataFrame([
        ("S0", "primary frozen embedding + Ridge", "mandatory", "linear information probe", "same frozen five folds", "train-CV only"),
        ("S1", "primary frozen embedding + ExtraTrees", "mandatory", "nonlinear low-tuning probe", "same frozen five folds", "train-CV only"),
        ("S2", "primary frozen embedding + LightGBM", "mandatory", "strong boosting probe", "same frozen five folds", "train-CV only"),
        ("S3", "primary frozen embedding + small MLP", "conditional_after_S0_S2_smoke", "capacity-matched neural probe", "same frozen five folds and saved reload", "train-CV only"),
        ("S4", "backup encoder with S0-S2", "provider_failure_only", "continuity, not model-zoo search", "activate only by written primary failure", "train-CV only"),
        ("S5", "best frozen sequence + Stage-A structure leader late fusion", "after_sequence_incremental_gate", "test complementary modalities", "advance if >=2% and >=3/5 non-worse or diversity rule", "bounded shortlist"),
    ], columns=["step", "configuration", "status", "question", "comparison_rule", "selection_boundary"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1b-protocol", type=Path, default=ROOT / "data/public_development/gate1b_multimodal_protocol_v2")
    parser.add_argument("--graph-manifest", type=Path, default=ROOT / "data/public_development/gate1b_graph_manifest_v1")
    parser.add_argument("--dmpnn-audit", type=Path, default=ROOT / "results/analysis/gate1b_dmpnn_stage1_audit_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/algorithm_landscape_smiles_provider_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    required = [
        args.gate1b_protocol / "complete.json", args.gate1b_protocol / "candidate_advancement_rules.csv",
        args.graph_manifest / "complete.json", args.graph_manifest / "graph_manifest.csv",
        args.dmpnn_audit / "complete.json", args.dmpnn_audit / "endpoint_decisions.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    gate = verify_stage(args.gate1b_protocol, "gate1b_multimodal_input_protocol")
    graph = verify_stage(args.graph_manifest, "gate1b_graph_manifest")
    dmpnn = verify_stage(args.dmpnn_audit, "gate1b_dmpnn_stage1_audit")
    if gate.get("test_labels_read") or graph.get("test_labels_read") or dmpnn.get("test_labels_read"):
        raise ValueError("A prerequisite reports protected test-label access")
    if dmpnn.get("validation_labels_read") or dmpnn.get("validation_authorized"):
        raise ValueError("D-MPNN audit must remain fixed-validation blind")
    if int(graph.get("parents", 0)) != 7939 or int(dmpnn.get("tasks_passing_strong_advance_gate", -1)) != 0:
        raise ValueError("Frozen parent count or D-MPNN decision has changed")

    inputs = {
        "gate1b_protocol_complete_sha256": sha256(args.gate1b_protocol / "complete.json"),
        "graph_manifest_complete_sha256": sha256(args.graph_manifest / "complete.json"),
        "dmpnn_audit_complete_sha256": sha256(args.dmpnn_audit / "complete.json"),
    }
    if args.check_only:
        published = verify_stage(args.output, "algorithm_landscape_smiles_provider_protocol")
        if published.get("inputs") != inputs:
            raise ValueError("Published provider protocol no longer matches frozen inputs")
        print(f"Algorithm/provider protocol valid: algorithms={published['registered_algorithms']} providers={published['registered_providers']}")
        return

    algorithms = algorithm_landscape()
    providers = provider_registry()
    contract = cache_contract()
    plan = ablation_plan()
    if algorithms.algorithm.duplicated().any() or len(algorithms) < 45:
        raise ValueError("Algorithm landscape is unexpectedly narrow or contains duplicate names")
    if set(providers.provider_role) != {"primary", "backup", "literature_reference_only"}:
        raise ValueError("Provider roles must be exactly one primary, one backup and one blocked reference")
    if providers.loc[providers.provider_role.eq("primary"), "license_gate"].iloc[0] != "pass":
        raise ValueError("Primary provider license gate is not satisfied")

    dependencies = pd.DataFrame([
        ("torch", "2.5.1+cu121", "ready", "encoder inference and downstream MLP"),
        ("transformers", "4.50.3", "ready", "primary requires compat-v4 remote code"),
        ("tokenizers", "0.21.4", "ready", "pinned tokenizer artifacts still required"),
        ("huggingface_hub", "0.36.2", "ready", "immutable-revision snapshot"),
        ("torch_geometric", "2.8.0.post1", "ready", "future graph-family variants"),
        ("chemprop", "2.0.4", "ready", "D-MPNN reference implementation"),
        ("xgboost", "3.2.0", "ready", "frozen embedding probe"),
        ("lightgbm", "4.7.0", "ready", "frozen embedding probe"),
        ("catboost", "1.2.10", "ready", "retained tabular family"),
        ("scikit-learn", "1.9.0", "ready", "linear/tree/kernel/stacking"),
        ("rdkit", "2023.9.6", "ready", "frozen parent standardization"),
        ("deepchem", "not_installed", "not_required_now", "install only for an activated DeepChem-native algorithm"),
        ("dgl", "not_installed", "not_required_now", "PyG is the current graph backend"),
        ("gpytorch|ngboost|tabpfn|pytorch-tabnet", "not_installed", "conditional", "install only after corresponding activation gate"),
    ], columns=["dependency", "observed_version", "status", "role"])

    summary = {
        "registered_algorithms": len(algorithms),
        "algorithm_families": int(algorithms.family.nunique()),
        "registered_providers": len(providers),
        "primary_provider": "ibm-research/MoLFormer-XL-both-10pct",
        "primary_revision": "7b12d946c181a37f6012b9dc3b002275de070314",
        "parents_to_cache": 7939,
        "dmpnn_retained_as_negative_ablation": True,
        "algorithms_deleted_from_registry": 0,
        "fixed_validation_authorized": False,
        "validation_labels_read": False,
        "test_labels_read": False,
        "model_fitted": False,
        "next_action": "audit_pinned_remote_code_then_run_32_parent_molformer_smoke",
    }
    with stage_output(args.output) as out:
        algorithms.to_csv(out / "algorithm_landscape.csv", index=False)
        providers.to_csv(out / "smiles_provider_registry.csv", index=False)
        contract.to_csv(out / "embedding_cache_contract.csv", index=False)
        plan.to_csv(out / "smiles_ablation_plan.csv", index=False)
        dependencies.to_csv(out / "dependency_and_compute_plan.csv", index=False)
        (out / "protocol.json").write_text(json.dumps({
            "retention_principle": "no scientifically plausible algorithm is deleted; negative results remain reportable evidence",
            "compute_principle": "retention does not imply unrestricted tuning; costly methods require predeclared activation gates",
            "provider_principle": "one primary and at most one backup; backup activation requires a documented primary failure",
            "primary_input": "graph_manifest.graph_smiles",
            "token_overflow_policy": "explicit invalid mask and reason; no silent truncation",
            "cache_policy": "label-free immutable parent-keyed float32 cache; fold-local downstream preprocessing",
            "selection_policy": "same five folds and metrics; train-CV gate before any fixed validation",
            "remote_code_policy": "download and statically inspect exact pinned revision before trust_remote_code execution",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Algorithm landscape and frozen-SMILES provider protocol v1\n\n"
            "This registry separates algorithm retention from advancement. Current D-MPNN results remain a negative ablation and can be reactivated only under a materially different pretrained/shared-encoder hypothesis. "
            "IBM MoLFormer is the single primary frozen encoder; ChemBERTa-100M-MLM is a failure-only backup. All weights, code, tokenizer and cache rows are revision/hash bound, label-free and validation/test blind.\n",
            encoding="utf-8",
        )
        finish_stage(out, "algorithm_landscape_smiles_provider_protocol", inputs=inputs, partial=False, **summary)
    print(f"Algorithm landscape and SMILES provider protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
