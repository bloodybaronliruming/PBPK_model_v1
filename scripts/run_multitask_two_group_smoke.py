#!/usr/bin/env python3
"""Run a label-closed engineering smoke for the fu/non-fu two-group encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmpk_toolkit import featurize
from multitask_two_group_common import FU_GROUP, NON_FU_GROUP, ROUTE_ID, TwoGroupSharedPrivateNet, cpu_predict
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_multitask_shared_private_traincv import (
    META_COLUMNS, aggregate_parents, apply_eval_scaler, scaled_features, train_phases,
)
from run_vdss_cross_species_transfer_traincv import load_labels_only, seed_everything, stable_limit


def module_vector(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([value.detach().cpu().reshape(-1) for value in module.parameters()])


def gradient_norm(module: torch.nn.Module) -> float:
    values = [value.grad.detach().reshape(-1) for value in module.parameters() if value.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(values)).cpu()) if values else 0.0


def routing_gradient_audit(
    model: TwoGroupSharedPrivateNet, x: np.ndarray, frame: pd.DataFrame,
    task_order: list[str], task_groups: list[str], device: str,
) -> pd.DataFrame:
    rows = []
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    target = torch.as_tensor(frame.model_target.to_numpy(float), dtype=torch.float32, device=device)
    task_index = torch.as_tensor(frame.task_index.to_numpy(int), dtype=torch.long, device=device)
    for active_group in [FU_GROUP, NON_FU_GROUP]:
        task = next(value for value, group in zip(task_order, task_groups, strict=True) if group == active_group)
        ids = frame.index[frame.task_id.eq(task)].to_numpy(int)[:min(16, frame.task_id.eq(task).sum())]
        model.zero_grad(set_to_none=True)
        prediction = model(x_tensor[ids], task_index[ids])
        loss_name = str(frame.loc[ids[0], "training_loss"])
        loss = (torch.mean(torch.abs(prediction - target[ids])) if loss_name == "physical_MAE"
                else torch.nn.functional.huber_loss(prediction, target[ids], delta=1.0))
        loss.backward()
        inactive_group = NON_FU_GROUP if active_group == FU_GROUP else FU_GROUP
        active_task_index = task_order.index(task)
        active_encoder_norm = gradient_norm(model.encoders[active_group])
        inactive_encoder_norm = gradient_norm(model.encoders[inactive_group])
        active_head_norm = gradient_norm(model.heads[active_task_index])
        inactive_head_norm = sum(
            gradient_norm(head) for index, head in enumerate(model.heads) if index != active_task_index
        )
        rows.append({
            "active_group": active_group, "active_task": task,
            "active_encoder_gradient_norm": active_encoder_norm,
            "inactive_encoder_gradient_norm": inactive_encoder_norm,
            "active_head_gradient_norm": active_head_norm,
            "inactive_heads_gradient_norm_sum": inactive_head_norm,
            "active_path_nonzero": active_encoder_norm > 0 and active_head_norm > 0,
            "inactive_path_zero": inactive_encoder_norm == 0 and inactive_head_norm == 0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/multitask_two_group_protocol_v2")
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/multitask_two_group_smoke_v2")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--phase1-epochs", type=int, default=1)
    parser.add_argument("--phase2-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-parents-per-task", type=int, default=32)
    parser.add_argument("--max-eval-parents-per-task", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.outer_fold not in range(5) or min(
        args.phase1_epochs, args.phase2_epochs, args.batch_size, args.max_parents_per_task,
        args.max_eval_parents_per_task, args.threads,
    ) < 1:
        raise ValueError("Invalid smoke configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records_path = args.interface / "training_records.csv"
    required = [
        args.protocol / "complete.json", args.protocol / "protocol.json",
        args.protocol / "route_registry.csv", args.protocol / "task_group_registry.csv",
        args.protocol / "task_target_registry.csv", args.protocol / "task_sampling_registry.csv",
        args.interface / "complete.json", records_path,
        args.stl / "complete.json", args.stl / "benchmark_train_records.csv",
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_meta = verify_stage(args.protocol, "multitask_two_group_protocol")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(meta.get("test_labels_read", False) for meta in [protocol_meta, interface_meta, stl_meta]):
        raise ValueError("Smoke inputs must remain test-closed")
    protocol = json.loads((args.protocol / "protocol.json").read_text(encoding="utf-8"))
    task_order = list(map(str, protocol["tasks"]))
    human_tasks = list(map(str, protocol["human_evaluation_tasks"]))
    registry = pd.read_csv(args.protocol / "task_target_registry.csv")
    groups = pd.read_csv(args.protocol / "task_group_registry.csv").set_index("task_id").loc[task_order]
    task_groups = groups.encoder_group.astype(str).tolist()
    if task_groups.count(FU_GROUP) != 9 or task_groups.count(NON_FU_GROUP) != 36:
        raise ValueError("Smoke task routing differs from protocol")
    task_index_map = {task: index for index, task in enumerate(task_order)}
    bounded = registry.set_index("task_id").loc[task_order].model_output.eq("bounded_sigmoid_fraction").tolist()
    sampling = pd.read_csv(args.protocol / "task_sampling_registry.csv")
    phase1_probabilities = (
        sampling.set_index("task_id").tier_loss_weight / sampling.set_index("task_id").tasks_in_tier
    ).to_dict()
    route = pd.read_csv(args.protocol / "route_registry.csv").iloc[0]
    width, latent = 64, 64
    if (route.route_id != ROUTE_ID or int(route.encoder_width) != width or int(route.latent) != latent
            or route.architecture != "two_MLP_64_64_GELU_layernorm_encoders"):
        raise ValueError("Smoke architecture differs from protocol")
    metadata = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=META_COLUMNS)
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(task_order)].copy()
    stl = pd.read_csv(args.stl / "benchmark_train_records.csv", dtype=str, keep_default_na=False,
                      usecols=["row_id", "task_id", "parent_id", "scaffold_group", "inner_fold_id"])
    stl = stl.loc[stl.task_id.isin(human_tasks)].copy()
    stl["inner_fold_id"] = pd.to_numeric(stl.inner_fold_id, errors="raise").astype(int)
    if set(stl.task_id) != set(human_tasks):
        raise ValueError("Human evaluation membership is incomplete")
    if args.check_only:
        print("Two-group smoke ready: fold=0 seed=20260917 tasks=45 groups=2 evaluation_labels=closed")
        return

    fold = args.outer_fold
    eval_membership = stl.loc[stl.inner_fold_id.eq(fold)].copy()
    eval_parents, eval_scaffolds = set(eval_membership.parent_id), set(eval_membership.scaffold_group)
    fold_meta = metadata.copy()
    retained = fold_meta.loc[
        ~fold_meta.parent_id.isin(eval_parents) & ~fold_meta.scaffold_group.isin(eval_scaffolds)
    ].copy()
    if set(retained.task_id) != set(task_order):
        raise ValueError("Smoke isolation removed a registered task")
    isolation_rows = []
    for task, before in fold_meta.groupby("task_id", sort=True):
        after = retained.loc[retained.task_id.eq(task)]
        isolation_rows.append({
            "outer_fold": fold, "task_id": task, "records_before": len(before), "records_after": len(after),
            "parents_after": after.parent_id.nunique(), "scaffolds_after": after.scaffold_group.nunique(),
            "post_filter_parent_overlap": int(after.parent_id.isin(eval_parents).sum()),
            "post_filter_scaffold_overlap": int(after.scaffold_group.isin(eval_scaffolds).sum()),
        })
    isolation = pd.DataFrame(isolation_rows)
    labels = load_labels_only(records_path, set(retained.row_id))
    retained = retained.merge(labels, on="row_id", validate="one_to_one")
    retained["target_value"] = pd.to_numeric(retained.target_value, errors="raise")
    parents = aggregate_parents(retained, registry)
    fitting = pd.concat([
        stable_limit(part, args.max_parents_per_task, args.seed + fold)
        for _, part in parents.groupby("task_id", sort=True)
    ], ignore_index=True)
    stats = fitting.groupby("task_id").target_value.agg(["mean", lambda values: values.std(ddof=0)]).reset_index()
    stats.columns = ["task_id", "target_mean", "target_std_population"]
    stats = stats.merge(registry[["task_id", "target_standardization"]], on="task_id", validate="one_to_one")
    if (stats.loc[stats.target_standardization.ne("none_physical_fraction"), "target_std_population"] <= 0).any():
        raise ValueError("Smoke target standardization is invalid")
    fitting = fitting.merge(stats, on=["task_id", "target_standardization"], validate="many_to_one")
    fitting["model_target"] = np.where(
        fitting.target_standardization.eq("none_physical_fraction"), fitting.target_value,
        (fitting.target_value - fitting.target_mean) / fitting.target_std_population,
    )
    fitting["task_index"] = fitting.task_id.map(task_index_map).astype(int)

    eval_meta = metadata.loc[metadata.row_id.isin(set(eval_membership.row_id)) & metadata.task_id.isin(human_tasks)].copy()
    eval_parent = eval_meta.groupby(["task_id", "parent_id"], as_index=False).agg(
        endpoint=("endpoint", "first"), smiles=("smiles", "first"), scaffold_group=("scaffold_group", "first"),
        source_records=("row_id", "size"),
    )
    eval_parent = pd.concat([
        stable_limit(part, args.max_eval_parents_per_task, args.seed + 1000 + fold)
        for _, part in eval_parent.groupby("task_id", sort=True)
    ], ignore_index=True)
    eval_parent["task_index"] = eval_parent.task_id.map(task_index_map).astype(int)
    unique_smiles = list(dict.fromkeys([*fitting.smiles.astype(str), *eval_parent.smiles.astype(str)]))
    features, _ = featurize(unique_smiles, feature_set="rdkit2d", progress=False)
    lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
    x_fit_raw = features[np.asarray([lookup[value] for value in fitting.smiles])]
    x_eval_raw = features[np.asarray([lookup[value] for value in eval_parent.smiles])]
    x_fit, scaler = scaled_features(x_fit_raw, fitting, ROUTE_ID, task_order)
    x_eval = apply_eval_scaler(x_eval_raw, eval_parent.task_id.tolist(), scaler)

    seed_everything(args.seed + fold * 100000, args.threads)
    model = TwoGroupSharedPrivateNet(
        x_fit.shape[1], len(task_order), width, latent, bounded, task_groups,
    ).to(args.device)
    routing = routing_gradient_audit(model, x_fit, fitting, task_order, task_groups, args.device)
    if not routing.active_path_nonzero.all() or not routing.inactive_path_zero.all():
        raise ValueError("Cross-branch gradient routing audit failed")
    encoder_before = {group: module_vector(model.encoders[group]) for group in [FU_GROUP, NON_FU_GROUP]}
    heads_before = [module_vector(head) for head in model.heads]
    coverage, losses = train_phases(
        model, x_fit, fitting, task_order, human_tasks, phase1_probabilities,
        phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size, learning_rate=3e-4, weight_decay=1e-3,
        seed=args.seed + 101, device=args.device,
    )
    update_rows = []
    for group in [FU_GROUP, NON_FU_GROUP]:
        change = float(torch.max(torch.abs(module_vector(model.encoders[group]) - encoder_before[group])))
        update_rows.append({"module_type": "encoder", "module_id": group, "max_parameter_change": change})
    for index, task in enumerate(task_order):
        change = float(torch.max(torch.abs(module_vector(model.heads[index]) - heads_before[index])))
        update_rows.append({"module_type": "private_head", "module_id": task, "max_parameter_change": change})
    updates = pd.DataFrame(update_rows)
    if not updates.max_parameter_change.gt(0).all():
        raise ValueError("A smoke encoder or private head did not update")

    checkpoint = {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "route_id": ROUTE_ID, "task_order": task_order, "task_groups": task_groups, "bounded": bounded,
        "n_features": x_fit.shape[1], "width": width, "latent": latent, "scaler": scaler,
        "seed": args.seed, "outer_fold": fold,
    }
    first = cpu_predict(checkpoint, x_eval, eval_parent.task_index.to_numpy(int))
    if not np.isfinite(first).all():
        raise ValueError("Smoke predictions are not finite")
    predictions = eval_parent.copy()
    predictions["predicted_model_target"] = first
    predictions = predictions.merge(stats, on="task_id", validate="many_to_one")
    predictions["predicted_primary_target"] = np.where(
        predictions.target_standardization.eq("none_physical_fraction"), predictions.predicted_model_target,
        predictions.predicted_model_target * predictions.target_std_population + predictions.target_mean,
    )
    bounded_tasks = set(registry.loc[registry.model_output.eq("bounded_sigmoid_fraction"), "task_id"])
    bounded_predictions = predictions.loc[predictions.task_id.isin(bounded_tasks), "predicted_primary_target"]
    if not bounded_predictions.between(0, 1).all():
        raise ValueError("Bounded smoke prediction lies outside [0,1]")

    with stage_output(args.output) as out:
        model_path = out / f"fold{fold}__seed{args.seed}__two_group_smoke.pt"
        torch.save(checkpoint, model_path)
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        second = cpu_predict(loaded, x_eval, eval_parent.task_index.to_numpy(int))
        reload_difference = float(np.max(np.abs(first - second)))
        if reload_difference > 1e-7:
            raise ValueError("Smoke checkpoint reload differs")
        isolation.to_csv(out / "fold_isolation_audit.csv", index=False)
        stats.to_csv(out / "fold_local_target_statistics.csv", index=False)
        pd.DataFrame(coverage).to_csv(out / "training_coverage.csv", index=False)
        pd.DataFrame(losses).to_csv(out / "training_losses.csv", index=False)
        routing.to_csv(out / "routing_gradient_audit.csv", index=False)
        updates.to_csv(out / "module_update_audit.csv", index=False)
        predictions[["task_id", "endpoint", "parent_id", "scaffold_group", "source_records",
                     "predicted_model_target", "predicted_primary_target"]].to_csv(out / "structure_only_smoke_predictions.csv", index=False)
        (out / "README.md").write_text(
            "# Two-group multitask engineering smoke\n\n"
            "This partial run checks fu/non-fu routing, branch isolation, task coverage, bounded outputs and checkpoint reload. "
            "It reads retained outer-training labels only and never reads evaluation, fixed-validation or test targets. "
            "Its predictions and losses cannot be used for architecture selection.\n",
            encoding="utf-8",
        )
        coverage_frame = pd.DataFrame(coverage)
        finish_stage(
            out, "multitask_two_group_smoke",
            inputs={"protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                    "interface_complete_sha256": sha256(args.interface / "complete.json"),
                    "stl_complete_sha256": sha256(args.stl / "complete.json")},
            outer_fold=fold, seed=args.seed, tasks=45, fu_tasks=9, non_fu_tasks=36,
            training_parents=len(fitting), structure_only_evaluation_parents=len(eval_parent),
            max_post_filter_parent_overlap=int(isolation.post_filter_parent_overlap.max()),
            max_post_filter_scaffold_overlap=int(isolation.post_filter_scaffold_overlap.max()),
            routing_active_paths_nonzero=bool(routing.active_path_nonzero.all()),
            routing_inactive_paths_zero=bool(routing.inactive_path_zero.all()),
            both_encoders_updated=bool(updates.loc[updates.module_type.eq("encoder"), "max_parameter_change"].gt(0).all()),
            all_private_heads_updated=bool(updates.loc[updates.module_type.eq("private_head"), "max_parameter_change"].gt(0).all()),
            training_phase_task_coverage_nonzero=bool(coverage_frame.updates.gt(0).all()),
            predictions_finite=bool(np.isfinite(predictions.predicted_primary_target).all()),
            bounded_predictions_within_unit_interval=bool(bounded_predictions.between(0, 1).all()),
            reload_max_abs_difference=reload_difference,
            evaluation_labels_read=False, evaluation_labels_used_during_fit=False,
            validation_target_file_opened=False, test_labels_read=False,
            architecture_selection_authorized=False, model_fitted=True,
            partial=True, device=args.device,
            cuda_device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "not_used",
        )
    print(f"Two-group multitask smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
