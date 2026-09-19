#!/usr/bin/env python3
"""Freeze and verify the input boundary before public-F development work."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


STAGES = {
    "processed_v15_datasets": (ROOT / "data/processed_v15/datasets", "datasets"),
    "processed_v15_splits": (ROOT / "data/processed_v15/splits", "splits"),
    "boulton_eligibility_lock": (ROOT / "results/analysis/pkdb_absolute_f_eligibility_lock_v1", "pkdb_absolute_f_eligibility_lock"),
    "boulton_extraction_lock": (ROOT / "results/analysis/pkdb_absolute_f_extraction_lock_v1", "pkdb_absolute_f_extraction_lock"),
    "boulton_reserve_view": (ROOT / "data/literature/absolute_f_increment_view_v1", "absolute_f_increment_view"),
    "public_f_candidate_registry": (ROOT / "results/analysis/public_f_development_candidate_registry_v1", "public_f_development_candidate_registration"),
    "public_f_eligibility_lock": (ROOT / "results/analysis/public_f_pkdb_eligibility_lock_v1", "public_f_pkdb_eligibility_lock"),
    "public_f_extraction_lock": (ROOT / "results/analysis/public_f_pkdb_extraction_lock_v1", "public_f_pkdb_extraction_lock"),
    "public_f_cohort_view": (ROOT / "data/literature/public_f_cohort_isolation_view_v1", "public_f_cohort_isolation_view"),
    "public_f_isolation_figure": (ROOT / "results/figures/public_f_cohort_isolation_figure_v2", "public_f_cohort_isolation_figure"),
}


def git_state() -> dict[str, int | bool]:
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True)
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {"diff_check_passed": diff.returncode == 0, "tracked_or_untracked_entries": len(lines)}


def validate_boundary() -> tuple[pd.DataFrame, dict]:
    metas = {name: verify_stage(folder, expected) for name, (folder, expected) in STAGES.items()}
    datasets = STAGES["processed_v15_datasets"][0]
    splits = STAGES["processed_v15_splits"][0]
    task_records = pd.read_csv(datasets / "task_records.csv", usecols=["task_id", "molecule_id", "split"])
    strict_f = task_records.loc[task_records.task_id.eq("F__human__absolute_oral")].copy()
    strict_counts = strict_f.groupby("split").molecule_id.nunique().to_dict()
    if strict_counts != {"test": 2, "train": 16}:
        raise ValueError(f"Unexpected frozen strict-F membership: {strict_counts}")

    reserve = pd.read_csv(STAGES["boulton_reserve_view"][0] / "increment_records.csv")
    public = pd.read_csv(STAGES["public_f_cohort_view"][0] / "cohort_records.csv")
    clusters = pd.read_csv(STAGES["public_f_cohort_view"][0] / "source_cluster_manifest.csv")
    gate_columns = [
        "overlap_strict_f_parent", "overlap_strict_f_scaffold", "overlap_strict_f_source",
        "overlap_boulton_parent", "overlap_boulton_scaffold", "overlap_boulton_source",
    ]
    if len(reserve) != 2 or reserve.parent_id.nunique() != 2:
        raise ValueError("Boulton reserve boundary changed")
    if len(public) != 3 or public.parent_id.nunique() != 3 or len(clusters) != 3:
        raise ValueError("Public-F cohort cardinality changed")
    if public[gate_columns].astype(bool).any(axis=None):
        raise ValueError("Public-F cohort has an unresolved strict/Boulton isolation overlap")
    if public.eligible_for_strict_f_external.astype(bool).any() or public.eligible_for_validation_or_test_assignment.astype(bool).any():
        raise ValueError("Public-F cohort is not permitted to receive strict/validation/test roles")
    expected_role = "public_development_candidate_pending_pre_registered_training_cohort"
    if not public.cohort_role.eq(expected_role).all():
        raise ValueError("Public-F cohort role changed without a new protocol")

    rows = []
    for name, (folder, _) in STAGES.items():
        complete = folder / "complete.json"
        rows.append({"logical_input": name, "path": str(folder.relative_to(ROOT)), "file": "complete.json", "sha256": sha256(complete),
                     "stage": metas[name]["stage"], "verified": True})
    for logical, path in {
        "frozen_strict_f_records": datasets / "tasks/F__human__absolute_oral.csv",
        "frozen_split_manifest": splits / "split_manifest.csv",
        "public_f_cohort_records": STAGES["public_f_cohort_view"][0] / "cohort_records.csv",
        "public_f_source_clusters": STAGES["public_f_cohort_view"][0] / "source_cluster_manifest.csv",
    }.items():
        rows.append({"logical_input": logical, "path": str(path.relative_to(ROOT)), "file": path.name, "sha256": sha256(path),
                     "stage": "direct_artifact", "verified": True})
    summary = {
        "strict_f_molecules_by_split": strict_counts,
        "strict_f_validation_molecules": 0,
        "boulton_reserved_molecules": 2,
        "public_f_candidate_molecules": 3,
        "public_f_source_clusters": 3,
        "public_f_triple_isolation_gates": len(gate_columns),
        "public_f_training_authorized": False,
        "public_f_validation_or_test_authorized": False,
        "strict_f_external_members_modifiable": False,
        "clint_frozen_test_status": "not_scored_by_this_preflight",
        "git_state": git_state(),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_development_preflight_2026-09-16_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [folder / "complete.json" for folder, _ in STAGES.values()]
    startup_self_check(required, output=None if args.check_only else args.output)
    manifest, summary = validate_boundary()
    if args.check_only:
        print("Public-F development preflight valid: "
              f"strict-F={summary['strict_f_molecules_by_split']}, public={summary['public_f_candidate_molecules']} candidates")
        return
    with stage_output(args.output) as out:
        manifest.to_csv(out / "input_manifest.csv", index=False)
        (out / "preflight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F development preflight: 2026-09-16\\n\\n"
            "This preflight freezes the verified input boundary before public-F protocol and candidate-discovery work. It does not "
            "merge labels, train a model, assign any public-F member to train/validation/test, or score a frozen test. The strict-F "
            "dataset remains the frozen processed_v15 membership. Boulton reserve and public-F cohort candidates remain separate; the "
            "public cohort is checked against strict-F and Boulton parent, scaffold and source-cluster overlap.\\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_development_preflight", inputs={row.logical_input: row.sha256 for row in manifest.itertuples(index=False)},
                     **summary, partial=False)
    print(f"Public-F development preflight: {args.output}")


if __name__ == "__main__":
    run_cli(main)
