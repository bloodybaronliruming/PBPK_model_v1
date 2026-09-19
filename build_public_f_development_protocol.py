#!/usr/bin/env python3
"""Publish the pre-registered public-F development cohort protocol, without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def protocol() -> dict:
    return {
        "protocol_version": "v1",
        "endpoint": {"task_id": "F__human__oral_bioavailability_public", "canonical_unit": "fraction", "scope": "human oral bioavailability public-development only"},
        "evidence_tracks": {
            "strict_A": {
                "definition": "Primary human oral+IV evidence with parent analyte and directly reported absolute F.",
                "uses": ["strict evidence", "future source-isolated external validation"],
                "prohibited": ["automatic merge into public development", "post hoc split reassignment"],
            },
            "public_B": {
                "definition": "Continuous human oral-F records with traceable source and endpoint definition; a source may be primary but is assigned this role only for public development.",
                "uses": ["future development training", "source/scaffold-aware validation after readiness gates"],
                "prohibited": ["strict external validation", "unreviewed record-level merge", "current model training before readiness"],
            },
            "auxiliary_C": {
                "definition": "Broad public oral-bioavailability tasks, including binary or definition-mismatched endpoints.",
                "uses": ["representation learning", "auxiliary classification", "ablation"],
                "prohibited": ["continuous human absolute-F labels", "strict-F metrics", "confirmation claims"],
            },
        },
        "admission_pipeline": {
            "B0_discovery": "Source-level candidate; endpoint value remains hidden.",
            "B1_definition_audited": "Human/oral/parent/endpoint definition and provenance reviewed; no numeric admission yet.",
            "B2_numeric_locked": "Source-located reported statistic, unit transformation, identity mapping, and strict/Boulton isolation are locked.",
            "B3_cohort_ready": "Pre-registered source/scaffold-aware development role assigned only after readiness gates are met.",
        },
        "source_cluster_rule": "Normalised DOI; if unavailable, PMID; otherwise an immutable manual source identifier. A source cluster is never split across train, validation, or test.",
        "structure_rule": "Use manually confirmed parent mapping and versioned parent/Murcko grouping. Salts, prodrugs, metabolites, markers, mixtures, and unresolved identities are excluded pending a separate mapping decision.",
        "numeric_rule": "Accept a directly reported study-level absolute-F statistic only. Convert percent to fraction without clipping; retain original statistic and interval type. Do not infer F from AUC, pool incompatible statistics, or expand participant values into duplicate rows.",
        "strict_separation_rule": "Every B2/B3 record must be audited for parent, scaffold, and DOI/source-cluster overlap against frozen strict-F and Boulton reserve. Any hit is excluded from the public development cohort until separately adjudicated.",
        "readiness_gates": {
            "exploratory_development_fit": {"minimum_unique_parents": 60, "minimum_source_clusters": 30, "purpose": "engineering-only direct-F baseline; no confirmatory claim"},
            "model_selection_validation": {"minimum_unique_parents": 15, "minimum_source_clusters": 10, "additional_rule": "source and scaffold disjoint from development training"},
            "public_holdout_evaluation": {"minimum_unique_parents": 20, "minimum_source_clusters": 12, "additional_rule": "source and scaffold disjoint from all development roles; separate from strict-F"},
        },
        "current_authorization": {
            "model_training": False,
            "validation_or_test_assignment": False,
            "allowed_today": ["protocol publication", "source-level candidate aggregation", "definition audit", "source discovery"],
        },
    }


def validate(preflight: dict, candidate_summary: dict, cohort: pd.DataFrame, clusters: pd.DataFrame) -> None:
    if preflight["public_f_training_authorized"] or preflight["public_f_validation_or_test_authorized"]:
        raise ValueError("Preflight unexpectedly authorizes public-F training or held-out assignment")
    if candidate_summary.get("chembl_public_candidates", 0) <= 0 or candidate_summary.get("pkdb_public_candidates", 0) <= 0:
        raise ValueError("Public-F source-level discovery pool is empty")
    gate_columns = [column for column in cohort.columns if column.startswith("overlap_strict_f_") or column.startswith("overlap_boulton_")]
    if len(gate_columns) != 6 or cohort[gate_columns].astype(bool).any(axis=None):
        raise ValueError("Public-F B2 candidates fail the required triple-isolation audit")
    if len(cohort) != 3 or len(clusters) != 3 or not clusters.records.eq(1).all():
        raise ValueError("Current B2 source-cluster boundary differs from preflight")
    if cohort.eligible_for_strict_f_external.astype(bool).any() or cohort.eligible_for_validation_or_test_assignment.astype(bool).any():
        raise ValueError("Current B2 candidates cannot have strict/held-out roles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=ROOT / "results/analysis/public_f_development_preflight_2026-09-16_v1")
    parser.add_argument("--candidate-registry", type=Path, default=ROOT / "results/analysis/public_f_development_candidate_registry_v1")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/literature/public_f_cohort_isolation_view_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_development_cohort_protocol_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    preflight_file = args.preflight / "preflight_summary.json"
    candidate_file = args.candidate_registry / "summary.json"
    cohort_file, cluster_file = args.cohort / "cohort_records.csv", args.cohort / "source_cluster_manifest.csv"
    required = [preflight_file, args.preflight / "complete.json", candidate_file, args.candidate_registry / "complete.json",
                cohort_file, cluster_file, args.cohort / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.preflight, "public_f_development_preflight")
    verify_stage(args.candidate_registry, "public_f_development_candidate_registration")
    verify_stage(args.cohort, "public_f_cohort_isolation_view")
    preflight = json.loads(preflight_file.read_text(encoding="utf-8"))
    candidate_summary = json.loads(candidate_file.read_text(encoding="utf-8"))
    cohort = pd.read_csv(cohort_file)
    clusters = pd.read_csv(cluster_file)
    validate(preflight, candidate_summary, cohort, clusters)
    contract = protocol()
    source_registry = clusters.assign(admission_status="B2_numeric_locked_pending_B3_readiness", split_assignment="unassigned")
    routing = pd.DataFrame([
        {"track": "strict_A", "label_semantics": "human absolute oral F", "allowed_use": "strict evidence/future external validation", "prohibited_use": "automatic public merge"},
        {"track": "public_B", "label_semantics": "traceable continuous human oral F", "allowed_use": "future source-aware development", "prohibited_use": "strict external or current training"},
        {"track": "auxiliary_C", "label_semantics": "broad or binary oral bioavailability", "allowed_use": "representation/auxiliary task", "prohibited_use": "continuous human absolute-F target"},
    ])
    checklist = pd.DataFrame([
        {"admission_stage": "B0_discovery", "required_gate": "Source-level record and provenance pointer", "current_action": "Aggregate candidates by document/assay; values hidden"},
        {"admission_stage": "B1_definition_audited", "required_gate": "Human, oral, parent analyte, compatible endpoint definition", "current_action": "Reject endpoint-mismatched descriptions"},
        {"admission_stage": "B2_numeric_locked", "required_gate": "Source locator, numeric statistic, identity and triple-isolation lock", "current_action": "Current 3 candidates are B2"},
        {"admission_stage": "B3_cohort_ready", "required_gate": "Readiness thresholds and pre-registered source/scaffold-aware roles", "current_action": "Not authorized at current scale"},
    ])
    if args.check_only:
        print("Public-F development protocol valid: B2 candidates=3, training_authorized=false")
        return
    with stage_output(args.output) as out:
        (out / "protocol.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        source_registry.to_csv(out / "current_B2_source_cluster_registry.csv", index=False)
        routing.to_csv(out / "evidence_track_routing.csv", index=False)
        checklist.to_csv(out / "admission_checklist.csv", index=False)
        (out / "README.md").write_text(
            "# Public-F development cohort protocol v1\\n\\n"
            "This is a governance and admission contract, not a training dataset. It locks the separation of strict A, public B, and "
            "auxiliary C evidence; source-cluster and structure rules; numeric semantics; and readiness gates before any public-F fit, "
            "validation, or holdout evaluation. The current three B2 records remain unassigned and training is explicitly unauthorized.\\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_development_cohort_protocol", inputs={
            "preflight_complete_sha256": sha256(args.preflight / "complete.json"),
            "candidate_registry_complete_sha256": sha256(args.candidate_registry / "complete.json"),
            "cohort_complete_sha256": sha256(args.cohort / "complete.json"),
        }, current_B2_records=len(cohort), current_B2_source_clusters=len(clusters),
           training_authorized=False, validation_or_test_authorized=False, partial=False)
    print(f"Public-F development cohort protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
