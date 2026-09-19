#!/usr/bin/env python3
"""Measure train-only task-gradient affinity in frozen shared encoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dmpk_toolkit import featurize
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_multitask_shared_private_traincv import (
    META_COLUMNS, SharedPrivateNet, aggregate_parents, apply_eval_scaler,
)
from run_vdss_cross_species_transfer_traincv import load_labels_only, parse_int_list, seed_everything, stable_limit


def deterministic_batch(ids: np.ndarray, size: int, key: str) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.permutation(ids)[:min(size, len(ids))]


def encoder_gradient(
    model: SharedPrivateNet, x: torch.Tensor, task_index: torch.Tensor, target: torch.Tensor,
    ids: np.ndarray, loss_name: str,
) -> tuple[torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    prediction = model(x[ids], task_index[ids])
    observed = target[ids]
    loss = (
        torch.mean(torch.abs(prediction - observed)) if loss_name == "physical_MAE"
        else torch.nn.functional.huber_loss(prediction, observed, delta=1.0)
    )
    loss.backward()
    pieces = []
    for parameter in model.encoder.parameters():
        if parameter.grad is None:
            raise ValueError("Shared encoder parameter lacks a task gradient")
        pieces.append(parameter.grad.detach().reshape(-1))
    vector = torch.cat(pieces)
    if not torch.isfinite(vector).all():
        raise ValueError("Non-finite shared encoder gradient")
    return vector, float(loss.detach().cpu())


def build_heatmap(human: pd.DataFrame, tasks: list[str], human_tasks: list[str]) -> plt.Figure:
    matrix = human.pivot(index="human_task", columns="partner_task", values="median_cosine").reindex(index=human_tasks, columns=tasks)
    labels = [value.replace("__human__", " | human | ").replace("__", " | ") for value in tasks]
    row_labels = [value.split("__", 1)[0] for value in human_tasks]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6})
    figure, axis = plt.subplots(figsize=(13.2, 4.8))
    image = axis.imshow(matrix.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axis.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
    axis.set_xlabel("Partner task")
    axis.set_ylabel("Human primary task")
    axis.set_title("Shared-encoder task-gradient affinity (median across fold–seed contexts)", fontweight="bold", fontsize=10)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015)
    colorbar.set_label("Gradient cosine")
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/multitask_gradient_affinity_protocol_v1")
    parser.add_argument("--traincv", type=Path, default=ROOT / "results/benchmarks/multitask_shared_private_traincv_v2")
    parser.add_argument("--shared-protocol", type=Path, default=ROOT / "data/public_development/multitask_shared_private_protocol_v2")
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/multitask_gradient_affinity_v1")
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="20260917,20260918,20260919")
    parser.add_argument("--batches-per-task", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-parents-per-task", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    folds, seeds = parse_int_list(args.outer_folds), parse_int_list(args.seeds)
    if not set(folds) <= set(range(5)) or min(args.batches_per_task, args.batch_size, args.threads) < 1:
        raise ValueError("Invalid gradient diagnostic configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records_path = args.interface / "training_records.csv"
    required = [args.protocol / "complete.json", args.protocol / "protocol.json",
                args.traincv / "complete.json", args.shared_protocol / "complete.json",
                args.shared_protocol / "task_target_registry.csv", args.interface / "complete.json", records_path,
                args.stl / "complete.json", args.stl / "benchmark_train_records.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    diagnostic_meta = verify_stage(args.protocol, "multitask_gradient_affinity_protocol")
    train_meta = verify_stage(args.traincv, "multitask_shared_private_traincv")
    shared_meta = verify_stage(args.shared_protocol, "multitask_shared_private_protocol")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(m.get("test_labels_read", False) for m in [diagnostic_meta, train_meta, shared_meta, interface_meta, stl_meta]):
        raise ValueError("Inputs must remain test-closed")
    contract = json.loads((args.protocol / "protocol.json").read_text())
    tasks, human_tasks = list(map(str, contract["tasks"])), list(map(str, contract["human_tasks"]))
    registry = pd.read_csv(args.shared_protocol / "task_target_registry.csv")
    registry_index = registry.set_index("task_id")
    task_index_map = {task: index for index, task in enumerate(tasks)}
    full_configuration = bool(
        folds == contract["formal_outer_folds"] and seeds == contract["formal_seeds"]
        and args.batches_per_task == contract["formal_batches_per_task"]
        and args.batch_size == contract["formal_batch_size"] and args.max_parents_per_task == 0
    )
    expected_checkpoints = len(folds) * len(seeds)
    checkpoints = [
        args.traincv / "models" / f"fold{fold}__seed{seed}__shared_encoder_private_heads.pt"
        for fold in folds for seed in seeds
    ]
    startup_self_check(checkpoints)
    if args.check_only:
        print(f"Gradient-affinity diagnostic ready: folds={folds} seeds={seeds} tasks={len(tasks)} batches={args.batches_per_task} device={args.device} full_configuration={full_configuration}")
        return

    metadata = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=META_COLUMNS)
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(tasks)].copy()
    stl = pd.read_csv(args.stl / "benchmark_train_records.csv", dtype=str, keep_default_na=False,
                      usecols=["row_id", "task_id", "parent_id", "scaffold_group", "inner_fold_id"])
    stl = stl.loc[stl.task_id.isin(human_tasks)].copy()
    stl["inner_fold_id"] = pd.to_numeric(stl.inner_fold_id, errors="raise").astype(int)
    pair_rows, norm_rows, isolation_rows = [], [], []
    for fold in folds:
        evaluation = stl.loc[stl.inner_fold_id.eq(fold)]
        eval_parents, eval_scaffolds = set(evaluation.parent_id), set(evaluation.scaffold_group)
        fold_meta = metadata.copy()
        retained = fold_meta.loc[
            ~fold_meta.parent_id.isin(eval_parents) & ~fold_meta.scaffold_group.isin(eval_scaffolds)
        ].copy()
        isolation_rows.append({
            "outer_fold": fold, "retained_tasks": retained.task_id.nunique(), "retained_records": len(retained),
            "post_filter_parent_overlap": int(retained.parent_id.isin(eval_parents).sum()),
            "post_filter_scaffold_overlap": int(retained.scaffold_group.isin(eval_scaffolds).sum()),
        })
        labels = load_labels_only(records_path, set(retained.row_id))
        retained = retained.merge(labels, on="row_id", validate="one_to_one")
        retained["target_value"] = pd.to_numeric(retained.target_value, errors="raise")
        parents = aggregate_parents(retained, registry)
        fitting = pd.concat([
            stable_limit(part, args.max_parents_per_task, 20260917 + fold)
            for _, part in parents.groupby("task_id", sort=True)
        ], ignore_index=True)
        stats = fitting.groupby("task_id").target_value.agg(["mean", lambda x: x.std(ddof=0)]).reset_index()
        stats.columns = ["task_id", "target_mean", "target_std_population"]
        stats = stats.merge(registry[["task_id", "target_standardization"]], on="task_id", validate="one_to_one")
        fitting = fitting.merge(stats, on=["task_id", "target_standardization"], validate="many_to_one")
        fitting["model_target"] = np.where(
            fitting.target_standardization.eq("none_physical_fraction"), fitting.target_value,
            (fitting.target_value - fitting.target_mean) / fitting.target_std_population,
        )
        fitting["task_index"] = fitting.task_id.map(task_index_map).astype(int)
        unique_smiles = list(dict.fromkeys(fitting.smiles.astype(str)))
        features, _ = featurize(unique_smiles, feature_set="rdkit2d", progress=False)
        lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
        raw_x = features[np.asarray([lookup[value] for value in fitting.smiles])]
        ids_by_task = {task: fitting.index[fitting.task_id.eq(task)].to_numpy(int) for task in tasks}
        for seed in seeds:
            checkpoint_path = args.traincv / "models" / f"fold{fold}__seed{seed}__shared_encoder_private_heads.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["task_order"] != tasks or checkpoint["route_id"] != "shared_encoder_private_heads":
                raise ValueError("Checkpoint task order/route differs from diagnostic protocol")
            x_scaled = apply_eval_scaler(raw_x, fitting.task_id.tolist(), checkpoint["scaler"])
            seed_everything(seed + fold * 100000, args.threads)
            model = SharedPrivateNet(checkpoint["n_features"], len(tasks), checkpoint["width"], checkpoint["latent"], checkpoint["bounded"]).to(args.device)
            model.load_state_dict(checkpoint["state_dict"]); model.eval()
            x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=args.device)
            task_tensor = torch.as_tensor(fitting.task_index.to_numpy(int), dtype=torch.long, device=args.device)
            target_tensor = torch.as_tensor(fitting.model_target.to_numpy(float), dtype=torch.float32, device=args.device)
            for replicate in range(args.batches_per_task):
                gradients, losses = [], []
                for task in tasks:
                    ids = deterministic_batch(ids_by_task[task], args.batch_size, f"{fold}|{seed}|{task}|{replicate}")
                    vector, loss = encoder_gradient(
                        model, x_tensor, task_tensor, target_tensor, ids,
                        str(registry_index.loc[task, "training_loss"]),
                    )
                    gradients.append(vector); losses.append(loss)
                gradient_matrix = torch.stack(gradients)
                norms = torch.linalg.vector_norm(gradient_matrix, dim=1)
                if (norms <= 0).any():
                    raise ValueError("Zero shared-encoder gradient norm")
                cosine = (gradient_matrix / norms[:, None]) @ (gradient_matrix / norms[:, None]).T
                cosine = np.clip(cosine.detach().cpu().numpy(), -1.0, 1.0)
                norm_values = norms.detach().cpu().numpy()
                median_norm = float(np.median(norm_values))
                for index, task in enumerate(tasks):
                    norm_rows.append({
                        "outer_fold": fold, "seed": seed, "batch_replicate": replicate,
                        "task_id": task, "endpoint": registry_index.loc[task, "endpoint"],
                        "gradient_norm": float(norm_values[index]), "context_median_norm": median_norm,
                        "norm_ratio_to_context_median": float(norm_values[index] / median_norm),
                        "batch_loss": losses[index], "batch_parents": min(args.batch_size, len(ids_by_task[task])),
                    })
                for left in range(len(tasks)):
                    for right in range(left, len(tasks)):
                        pair_rows.append({
                            "outer_fold": fold, "seed": seed, "batch_replicate": replicate,
                            "task_a": tasks[left], "task_b": tasks[right], "gradient_cosine": float(cosine[left, right]),
                        })
            del model, x_tensor, task_tensor, target_tensor
            if args.device == "cuda":
                torch.cuda.empty_cache()
    pairs = pd.DataFrame(pair_rows); norms = pd.DataFrame(norm_rows); isolation = pd.DataFrame(isolation_rows)
    context = pairs.groupby(["outer_fold", "seed", "task_a", "task_b"], as_index=False).gradient_cosine.median()
    summaries = context.groupby(["task_a", "task_b"], as_index=False).agg(
        median_cosine=("gradient_cosine", "median"), positive_contexts=("gradient_cosine", lambda x: int((x > 0).sum())),
        negative_contexts=("gradient_cosine", lambda x: int((x < 0).sum())), contexts=("gradient_cosine", "size"),
    )
    summaries["classification"] = np.where(
        (summaries.median_cosine > 0) & (summaries.positive_contexts >= 10), "supportive",
        np.where((summaries.median_cosine < 0) & (summaries.negative_contexts >= 10), "conflicting", "unstable"),
    )
    human_rows = []
    for row in summaries.itertuples(index=False):
        for human, partner in [(row.task_a, row.task_b), (row.task_b, row.task_a)]:
            if human in human_tasks and (human != partner or row.task_a == row.task_b):
                human_rows.append({"human_task": human, "partner_task": partner, "median_cosine": row.median_cosine,
                                   "positive_contexts": row.positive_contexts, "negative_contexts": row.negative_contexts,
                                   "contexts": row.contexts, "classification": row.classification})
    human = pd.DataFrame(human_rows).drop_duplicates(["human_task", "partner_task"])
    norm_summary = norms.groupby(["task_id", "endpoint"], as_index=False).agg(
        median_gradient_norm=("gradient_norm", "median"),
        median_norm_ratio=("norm_ratio_to_context_median", "median"),
        dominance_contexts=("norm_ratio_to_context_median", lambda x: int((x > 10).sum())),
        measurements=("norm_ratio_to_context_median", "size"),
    )
    with stage_output(args.output) as out:
        pairs.to_csv(out / "batch_pairwise_gradient_cosines.csv", index=False)
        context.to_csv(out / "fold_seed_pairwise_gradient_cosines.csv", index=False)
        summaries.to_csv(out / "pairwise_affinity_summary.csv", index=False)
        human.to_csv(out / "human_task_affinity_summary.csv", index=False)
        norms.to_csv(out / "batch_task_gradient_norms.csv", index=False)
        norm_summary.to_csv(out / "task_gradient_norm_summary.csv", index=False)
        isolation.to_csv(out / "fold_isolation_audit.csv", index=False)
        figure = build_heatmap(human, tasks, human_tasks)
        figure.savefig(out / "Figure_human_task_gradient_affinity.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        (out / "README.md").write_text(
            "# Train-only multitask gradient affinity\n\nCosines use frozen shared encoders and retained outer-training labels only. "
            "Each fold-seed value is the median across deterministic task batches. This diagnostic can pre-register one finite architecture ablation but cannot revise completed performance results.\n",
            encoding="utf-8",
        )
        finish_stage(
            out, "multitask_gradient_affinity",
            inputs={"diagnostic_protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                    "traincv_complete_sha256": sha256(args.traincv / "complete.json"),
                    "interface_complete_sha256": sha256(args.interface / "complete.json"),
                    "stl_complete_sha256": sha256(args.stl / "complete.json")},
            folds=folds, seeds=seeds, checkpoints=expected_checkpoints, tasks=len(tasks), human_tasks=len(human_tasks),
            batches_per_task=args.batches_per_task, batch_size=args.batch_size,
            pair_measurements=len(pairs), norm_measurements=len(norms),
            max_post_filter_parent_overlap=int(isolation.post_filter_parent_overlap.max()),
            max_post_filter_scaffold_overlap=int(isolation.post_filter_scaffold_overlap.max()),
            gradients_finite=bool(np.isfinite(pairs.gradient_cosine).all() and np.isfinite(norms.gradient_norm).all()),
            evaluation_labels_read=False, validation_target_file_opened=False, test_labels_read=False,
            model_fitted=False, architecture_selection_authorized=False,
            partial=not full_configuration, full_configuration=full_configuration,
            device=args.device, cuda_device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "not_used",
        )
    print(f"Multitask gradient affinity: {args.output}")


if __name__ == "__main__":
    run_cli(main)
