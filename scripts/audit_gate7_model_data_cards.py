"""Build read-only minimum data/model cards and AD/uncertainty availability audit for Gate 7.

The program summarizes only completed registries, model metadata, and existing
metrics already recorded by Gate 4/5.  It neither loads a predictive model nor
reads labels, predictions, or performance values outside those registries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


CONTRACT_STAGE = "gate7_release_contract_freeze"
SMOKE_STAGE = "gate7_no_label_batch_inference_smoke"
GATE4_STAGE = "cross_gate_endpoint_candidate_freeze"


DEFINITIONS = {
    "fu__human__plasma": "Human plasma unbound fraction (fu).",
    "CLint__human__microsome": "Human microsomal intrinsic clearance (CLint).",
    "Papp__human__caco2_ab": "Human Caco-2 apical-to-basolateral apparent permeability (Papp A→B).",
    "CL__human__systemic_iv": "Human systemic intravenous total clearance (CL).",
    "VDss__human__steady_state_iv": "Human steady-state intravenous volume of distribution (VDss).",
    "Thalf__human__terminal_iv": "Human intravenous terminal elimination half-life (terminal t½).",
    "F__human__absolute_oral": "Human absolute oral bioavailability (F).",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_versions(root: Path) -> dict[str, str]:
    folder = root / "models/frozen/final_candidates_v1"
    verify_stage(folder, "frozen_candidate_registry")
    registry = read_json(folder / "candidate_registry.json")
    return {item["task_id"]: item["data_version"] for item in registry["candidates"]}


def card_payload(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    contract_dir = root / "data/public_development/gate7_release_contract_v2"
    smoke_dir = root / "results/analysis/gate7_release_no_label_smoke_v2"
    gate4_dir = root / "data/public_development/cross_gate_candidate_freeze_v1"
    contract_meta = verify_stage(contract_dir, CONTRACT_STAGE)
    smoke_meta = verify_stage(smoke_dir, SMOKE_STAGE)
    gate4_meta = verify_stage(gate4_dir, GATE4_STAGE)
    if (not contract_meta.get("dual_lane_contract_frozen") or not smoke_meta.get("technical_predictions_generated")
            or smoke_meta.get("unified_numeric_release_authorized")):
        raise ValueError("Gate 7 upstream state does not allow card audit")
    contract = read_json(contract_dir / "release_contract.json")
    contract_rows = pd.read_csv(contract_dir / "endpoint_release_contract.csv", keep_default_na=False)
    gate4 = pd.read_csv(gate4_dir / "endpoint_candidate_freeze_registry.csv", keep_default_na=False)
    required = {"task_id", "endpoint", "canonical_unit", "train_records", "train_parents", "primary_metric", "train_cv_score"}
    if not required <= set(gate4.columns) or len(contract_rows) != 7 or len(gate4) != 7:
        raise ValueError("Gate 7 card inputs must each contain seven endpoint rows")
    tasks = {item["task_id"]: item for item in contract["tasks"]}
    if set(tasks) != set(contract_rows.task_id) or set(tasks) != set(gate4.task_id):
        raise ValueError("Contract and Gate-4 endpoint membership differ")
    versions = legacy_versions(root)
    thalf_registry = read_json(root / "models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1/candidate_registry.json")
    clint_contract = read_json(root / "models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2/model_contract.json")
    if thalf_registry.get("data_version") != "processed_v15" or clint_contract.get("task_id") != "CLint__human__microsome":
        raise ValueError("Frozen model-card metadata differs from the contract")

    data_cards, model_cards, audit_rows, cards = [], [], [], []
    gate4_by_task = gate4.set_index("task_id")
    contract_by_task = contract_rows.set_index("task_id")
    for task_id in sorted(tasks):
        task = tasks[task_id]
        gate = gate4_by_task.loc[task_id].to_dict()
        release = contract_by_task.loc[task_id].to_dict()
        if task["model_lane"] != release["release_lane"] or task["test_lifecycle"] != release["test_lifecycle"]:
            raise ValueError(f"Contract CSV/JSON mismatch: {task_id}")
        if task_id == "CLint__human__microsome":
            data_version = "processed_v15"
            model_detail = "ExtraTrees c01; ECFP4+RDKit2D; 120 trees; seed 20260916"
            ad_method, ad_threshold, ad_status = "max train ECFP4 Tanimoto", 0.4740325077399381, "pre_registered_train_only_AD_available"
            uncertainty_source, uncertainty_status = "single ExtraTrees asset", "no_calibrated_predictive_interval_packaged"
        elif task_id == "Thalf__human__terminal_iv":
            data_version = thalf_registry["data_version"]
            model_detail = "RDKit2D ExtraTrees, three-seed geometric physical-scale ensemble"
            ad_method, ad_threshold, ad_status = "nearest train Tanimoto", 0.30, "technical_similarity_only_not_release_calibrated"
            uncertainty_source, uncertainty_status = "three frozen seed predictions", "ensemble_spread_available_not_calibrated"
        elif task_id == "F__human__absolute_oral":
            data_version = "no_releasable_model"
            model_detail = "No numeric model"
            ad_method, ad_threshold, ad_status = "not applicable", None, "not_applicable_status_only"
            uncertainty_source, uncertainty_status = "none", "not_applicable_status_only"
        else:
            data_version = versions[task_id]
            model_detail = release["model_kind"]
            ad_method, ad_threshold, ad_status = "nearest train Tanimoto", 0.30, "technical_similarity_only_not_release_calibrated"
            if task_id == "fu__human__plasma":
                uncertainty_source, uncertainty_status = "three frozen Shared-MLP seed predictions", "ensemble_spread_available_not_calibrated"
            else:
                uncertainty_source, uncertainty_status = "single historical tree asset", "no_calibrated_predictive_interval_packaged"

        data_card = {
            "task_id": task_id, "endpoint": task["endpoint"], "definition": DEFINITIONS[task_id],
            "canonical_unit": task["unit"], "gate4_reference_train_records": int(gate["train_records"]),
            "gate4_reference_train_parents": int(gate["train_parents"]), "data_version_for_bound_asset": data_version,
            "gate4_research_reference": gate["unified_research_reference"], "data_evidence_scope": "Gate4 registry; source-count re-estimation is outside R3",
            "test_lifecycle": task["test_lifecycle"], "release_limitation": task["limitation"],
        }
        model_card = {
            "task_id": task_id, "endpoint": task["endpoint"], "model_lane": task["model_lane"],
            "model_kind": task["model_kind"], "model_detail": model_detail, "asset_source": task["asset_source"],
            "contract_numeric_prediction_status": task["numeric_prediction_status"],
            "technical_smoke_status": ("passed_no_label_technical_only" if task["no_label_smoke_eligible"]
                                        else "not_applicable_status_only"),
            "numeric_release_status": "not_authorized", "test_lifecycle": task["test_lifecycle"],
            "evidence_status": task["inferential_status"], "gate4_primary_metric": gate["primary_metric"],
            "gate4_reference_train_cv_score": gate["train_cv_score"],
            "performance_scope": "Gate4 research-reference train-CV; not a new score for the bound historical asset",
            "release_limitation": task["limitation"],
        }
        ad_audit = {
            "task_id": task_id, "endpoint": task["endpoint"], "ad_method": ad_method,
            "ad_threshold": ad_threshold, "ad_release_status": ad_status,
            "uncertainty_technical_source": uncertainty_source, "uncertainty_release_status": uncertainty_status,
            "smoke_verified": bool(task["no_label_smoke_eligible"]) or task_id == "F__human__absolute_oral",
            "publication_action": "report status/limitation; do not infer calibrated coverage or interval validity from technical smoke",
        }
        data_cards.append(data_card)
        model_cards.append(model_card)
        audit_rows.append(ad_audit)
        cards.append({"data_card": data_card, "model_card": model_card, "ad_uncertainty": ad_audit})
    inputs = {
        str((contract_dir / "complete.json").resolve()): sha256(contract_dir / "complete.json"),
        str((smoke_dir / "complete.json").resolve()): sha256(smoke_dir / "complete.json"),
        str((gate4_dir / "complete.json").resolve()): sha256(gate4_dir / "complete.json"),
        str((gate4_dir / "endpoint_candidate_freeze_registry.csv").resolve()): sha256(gate4_dir / "endpoint_candidate_freeze_registry.csv"),
    }
    return pd.DataFrame(data_cards), pd.DataFrame(model_cards), pd.DataFrame(audit_rows), {"cards": cards, "inputs": inputs}


def write_readme(folder: Path, audit: pd.DataFrame) -> None:
    ad_ready = int(audit.ad_release_status.eq("pre_registered_train_only_AD_available").sum())
    uncertain = int(audit.uncertainty_release_status.str.contains("not_calibrated", regex=False).sum())
    folder.joinpath("README.md").write_text(
        "# Gate 7 R3 model/data cards and AD/uncertainty availability audit\n\n"
        "This read-only package records minimum publication fields from already completed evidence. "
        "It does not load models, predict, read labels, or create a new performance result.\n\n"
        f"Only {ad_ready}/7 endpoint(s) currently have a pre-registered train-only AD threshold in the bound release asset. "
        f"{uncertain}/7 endpoint(s) have some technical uncertainty source but no calibrated predictive interval packaged. "
        "These are availability statements, not claims of calibration or coverage.\n\n"
        "The Gate-4 train-CV score is retained as research-reference evidence. It is deliberately not presented as a new "
        "score of a different historical test-locked inference asset.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "results/analysis/gate7_model_data_card_audit_v1"
    data_cards, model_cards, audit, payload = card_payload(root)
    if args.check_only:
        print(f"Gate 7 R3 card audit check: endpoints={len(data_cards)}; no labels or models loaded.")
        return
    with stage_output(output) as folder:
        data_cards.to_csv(folder / "endpoint_data_cards.csv", index=False)
        model_cards.to_csv(folder / "endpoint_model_cards.csv", index=False)
        audit.to_csv(folder / "endpoint_ad_uncertainty_audit.csv", index=False)
        (folder / "endpoint_cards.json").write_text(json.dumps(payload["cards"], ensure_ascii=False, indent=2), encoding="utf-8")
        write_readme(folder, audit)
        finish_stage(
            folder, "gate7_model_data_card_availability_audit", inputs=payload["inputs"], partial=False,
            no_model_fitted=True, no_predictions_generated=True, no_labels_accessed=True,
            no_performance_metrics_computed=True, no_candidate_selection_changed=True,
            endpoint_count=int(len(data_cards)), unified_numeric_release_authorized=False,
            pre_registered_ad_available_endpoints=int(audit.ad_release_status.eq("pre_registered_train_only_AD_available").sum()),
            calibrated_predictive_interval_available_endpoints=0,
        )
    print(f"Gate 7 R3 model/data card audit: {output}")


if __name__ == "__main__":
    run_cli(main)
