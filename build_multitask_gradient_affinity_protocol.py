#!/usr/bin/env python3
"""Freeze a train-label-only gradient-affinity diagnostic for multitask PK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traincv", type=Path, default=ROOT / "results/benchmarks/multitask_shared_private_traincv_v2")
    parser.add_argument("--shared-protocol", type=Path, default=ROOT / "data/public_development/multitask_shared_private_protocol_v2")
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/multitask_gradient_affinity_protocol_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.traincv / "complete.json",
        args.shared_protocol / "complete.json", args.shared_protocol / "protocol.json",
        args.shared_protocol / "task_target_registry.csv",
        args.interface / "complete.json", args.interface / "training_records.csv",
        args.stl / "complete.json", args.stl / "benchmark_train_records.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    train_meta = verify_stage(args.traincv, "multitask_shared_private_traincv")
    protocol_meta = verify_stage(args.shared_protocol, "multitask_shared_private_protocol")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if train_meta.get("partial") or not train_meta.get("full_configuration"):
        raise ValueError("Gradient diagnostic requires the full shared/private train-CV")
    if any(m.get("test_labels_read", False) for m in [train_meta, protocol_meta, interface_meta, stl_meta]):
        raise ValueError("Inputs must remain test-closed")
    protocol = json.loads((args.shared_protocol / "protocol.json").read_text())
    tasks = list(map(str, protocol["tasks"]))
    human_tasks = list(map(str, protocol["human_evaluation_tasks"]))
    registry = pd.read_csv(args.shared_protocol / "task_target_registry.csv")
    if len(tasks) != 45 or len(human_tasks) != 6 or set(registry.task_id) != set(tasks):
        raise ValueError("Frozen shared/private task registry is incomplete")
    models = sorted((args.traincv / "models").glob("*__shared_encoder_private_heads.pt"))
    if len(models) != 15:
        raise ValueError(f"Expected 15 shared checkpoints, found {len(models)}")
    contract = {
        "schema_version": 1,
        "diagnostic": "shared_encoder_per_task_gradient_cosine",
        "tasks": tasks,
        "human_tasks": human_tasks,
        "formal_outer_folds": [0, 1, 2, 3, 4],
        "formal_seeds": [20260917, 20260918, 20260919],
        "formal_checkpoints": 15,
        "formal_batches_per_task": 3,
        "formal_batch_size": 128,
        "batch_selection": "deterministic permutation of retained parent rows keyed by fold, seed, task and batch replicate",
        "gradient_scope": "shared encoder parameters only",
        "loss_scope": "the task-specific frozen physical-MAE or standardized-Huber training loss",
        "target_lifecycle": "retained outer-train labels only; outer-evaluation, fixed-validation and test labels prohibited",
        "fold_isolation": "reconstruct the same six-head union parent/scaffold purge as formal train-CV",
        "fold_seed_reduction": "median cosine across three deterministic task batches",
        "supportive_rule": "median cosine > 0 and positive in at least 10 of 15 fold-seed contexts",
        "conflicting_rule": "median cosine < 0 and negative in at least 10 of 15 fold-seed contexts",
        "otherwise_rule": "unstable",
        "gradient_norm_dominance_flag": "task median norm divided by within-context median task norm > 10",
        "use": "pre-register one finite next architecture ablation; never revise completed shared/private result",
        "architecture_selection_authorized": False,
        "fixed_validation_status": "closed",
        "test_status": "closed",
    }
    json.dumps(contract, allow_nan=False)
    if args.check_only:
        print(f"Gradient-affinity protocol ready: tasks={len(tasks)} human_tasks={len(human_tasks)} checkpoints={len(models)}")
        return
    with stage_output(args.output) as out:
        registry.to_csv(out / "task_registry.csv", index=False)
        pd.DataFrame({"checkpoint": [str(path.relative_to(args.traincv)) for path in models]}).to_csv(
            out / "shared_checkpoint_registry.csv", index=False
        )
        (out / "protocol.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Multitask gradient-affinity diagnostic protocol\n\n"
            "This diagnostic uses only retained outer-training labels and frozen shared-encoder checkpoints. "
            "It measures task-gradient direction and norm without refitting or reading evaluation, fixed-validation or test targets.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "multitask_gradient_affinity_protocol",
            inputs={
                "traincv_complete_sha256": sha256(args.traincv / "complete.json"),
                "shared_protocol_complete_sha256": sha256(args.shared_protocol / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_complete_sha256": sha256(args.stl / "complete.json"),
            },
            tasks=45, human_tasks=6, checkpoints=15, formal_batches_per_task=3,
            evaluation_labels_read=False, validation_target_file_opened=False, test_labels_read=False,
            model_fitted=False, architecture_selection_authorized=False, partial=False,
        )
    print(f"Gradient-affinity protocol: {args.output}")


if __name__ == "__main__":
    run_cli(main)
