#!/usr/bin/env python3
"""Run the frozen train-CV-only fully-private versus shared/private ablation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dmpk_toolkit import featurize
from gate1b_torch_common import fit_feature_scaler, scale_features
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from run_vdss_cross_species_transfer_traincv import load_labels_only, parse_int_list, seed_everything, stable_limit


ROUTES = ["fully_private_task_networks", "shared_encoder_private_heads"]
META_COLUMNS = [
    "row_id", "task_id", "split", "endpoint", "smiles", "parent_id", "scaffold_group", "doc_id",
]


class TaskBlock(nn.Module):
    def __init__(self, n_features: int, width: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, latent), nn.GELU(), nn.LayerNorm(latent),
        )
        self.head = nn.Sequential(nn.Linear(latent, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)


class FullyPrivateNet(nn.Module):
    def __init__(self, n_features: int, n_tasks: int, width: int, latent: int, bounded: list[bool]):
        super().__init__()
        self.blocks = nn.ModuleList([TaskBlock(n_features, width, latent) for _ in range(n_tasks)])
        self.bounded = list(map(bool, bounded))

    def forward(self, x: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        output = torch.empty(len(x), dtype=x.dtype, device=x.device)
        for task in task_index.unique().tolist():
            take = task_index.eq(int(task))
            value = self.blocks[int(task)](x[take])
            output[take] = torch.sigmoid(value) if self.bounded[int(task)] else value
        return output


class SharedPrivateNet(nn.Module):
    def __init__(self, n_features: int, n_tasks: int, width: int, latent: int, bounded: list[bool]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, latent), nn.GELU(), nn.LayerNorm(latent),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(n_tasks)
        ])
        self.bounded = list(map(bool, bounded))

    def forward(self, x: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        output = torch.empty(len(x), dtype=x.dtype, device=x.device)
        for task in task_index.unique().tolist():
            take = task_index.eq(int(task))
            value = self.heads[int(task)](hidden[take]).squeeze(1)
            output[take] = torch.sigmoid(value) if self.bounded[int(task)] else value
        return output


def aggregate_parents(records: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    local = records.merge(
        registry[["task_id", "parent_aggregation", "target_standardization", "model_output", "training_loss", "primary_metric"]],
        on="task_id", validate="many_to_one",
    )
    local["aggregate_target"] = local.target_value.to_numpy(float)
    fu = local.parent_aggregation.eq("mean_record_level_expit_physical").to_numpy()
    local.loc[fu, "aggregate_target"] = 1.0 / (1.0 + np.exp(-np.clip(local.loc[fu, "target_value"].to_numpy(float), -50, 50)))
    grouped = local.groupby(["task_id", "parent_id"], as_index=False).agg(
        endpoint=("endpoint", "first"), smiles=("smiles", "first"), scaffold_group=("scaffold_group", "first"),
        target_value=("aggregate_target", "mean"), source_records=("row_id", "size"),
        target_standardization=("target_standardization", "first"), model_output=("model_output", "first"),
        training_loss=("training_loss", "first"), primary_metric=("primary_metric", "first"),
    )
    if grouped.duplicated(["task_id", "parent_id"]).any():
        raise ValueError("Parent aggregation is not unique")
    bounded = grouped.model_output.eq("bounded_sigmoid_fraction")
    if not grouped.loc[bounded, "target_value"].between(0, 1).all():
        raise ValueError("Bounded physical parent target outside [0,1]")
    return grouped


def scaled_features(x: np.ndarray, frame: pd.DataFrame, route: str, task_order: list[str]) -> tuple[np.ndarray, dict]:
    scaled = np.empty_like(x, dtype=np.float32)
    scalers = {}
    for task in task_order:
        take = frame.task_id.eq(task).to_numpy()
        mean, scale = fit_feature_scaler(x[take])
        scaled[take] = scale_features(x[take], mean, scale)
        scalers[task] = {"mean": mean, "scale": scale}
    return scaled, {"kind": "task_specific", "tasks": scalers}


def apply_eval_scaler(x: np.ndarray, tasks: list[str], scaler: dict) -> np.ndarray:
    if scaler["kind"] == "shared":
        return scale_features(x, scaler["mean"], scaler["scale"])
    out = np.empty_like(x, dtype=np.float32)
    task_array = np.asarray(tasks, dtype=object)
    for task in sorted(set(tasks)):
        take = task_array == task
        spec = scaler["tasks"][task]
        out[take] = scale_features(x[take], spec["mean"], spec["scale"])
    return out


def train_phases(
    model: nn.Module, x: np.ndarray, frame: pd.DataFrame, task_order: list[str], human_tasks: list[str],
    phase1_probabilities: dict[str, float],
    *, phase1_epochs: int, phase2_epochs: int, batch_size: int, learning_rate: float,
    weight_decay: float, seed: int, device: str,
) -> tuple[list[dict], list[dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    task_tensor = torch.as_tensor(frame.task_index.to_numpy(int), dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(frame.model_target.to_numpy(float), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    coverage, losses = [], []
    for phase, tasks, epochs in [("phase1_all_tasks", task_order, phase1_epochs), ("phase2_human_refinement", human_tasks, phase2_epochs)]:
        ids = {task: frame.index[frame.task_id.eq(task)].to_numpy(int) for task in tasks}
        if any(len(value) == 0 for value in ids.values()):
            raise ValueError(f"{phase} lacks a registered task")
        seen = {task: set() for task in tasks}
        updates = {task: 0 for task in tasks}
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_tasks = list(tasks)
            if phase == "phase1_all_tasks":
                probability = np.asarray([phase1_probabilities[task] for task in tasks], dtype=float)
                probability = probability / probability.sum()
            else:
                probability = np.full(len(tasks), 1.0 / len(tasks))
            epoch_tasks.extend(rng.choice(tasks, size=len(tasks), replace=True, p=probability).tolist())
            rng.shuffle(epoch_tasks)
            for task in epoch_tasks:
                available = ids[task]
                chosen = rng.choice(available, size=min(batch_size, len(available)), replace=False)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_tensor[chosen], task_tensor[chosen])
                target = target_tensor[chosen]
                loss_name = str(frame.loc[chosen[0], "training_loss"])
                loss = (
                    torch.mean(torch.abs(prediction - target)) if loss_name == "physical_MAE"
                    else torch.nn.functional.huber_loss(prediction, target, delta=1.0)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                updates[task] += 1
                seen[task].update(map(int, chosen))
                losses.append({"phase": phase, "epoch": epoch, "task_id": task, "loss": float(loss.detach().cpu())})
        coverage.extend({
            "phase": phase, "task_id": task, "epochs": epochs, "updates": updates[task],
            "unique_parents_sampled": len(seen[task]), "available_parents": len(ids[task]),
        } for task in tasks)
    return coverage, losses


def cpu_predict(checkpoint: dict, x: np.ndarray, task_indices: np.ndarray) -> np.ndarray:
    cls = SharedPrivateNet if checkpoint["route_id"] == "shared_encoder_private_heads" else FullyPrivateNet
    model = cls(checkpoint["n_features"], len(checkpoint["task_order"]), checkpoint["width"], checkpoint["latent"], checkpoint["bounded"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(task_indices, dtype=torch.long)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/public_development/multitask_shared_private_protocol_v2")
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path, default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/multitask_shared_private_traincv_v2")
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="20260917,20260918,20260919")
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-parents-per-task", type=int, default=0)
    parser.add_argument("--max-eval-parents-per-task", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    folds, seeds = parse_int_list(args.outer_folds), parse_int_list(args.seeds)
    if not set(folds) <= set(range(5)) or min(args.phase1_epochs, args.phase2_epochs, args.batch_size, args.width, args.latent, args.threads) < 1:
        raise ValueError("Invalid configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records_path = args.interface / "training_records.csv"
    required = [args.protocol / "complete.json", args.protocol / "protocol.json", args.protocol / "route_registry.csv",
                args.protocol / "task_sampling_registry.csv",
                args.protocol / "task_target_registry.csv", args.interface / "complete.json", records_path,
                args.stl / "complete.json", args.stl / "benchmark_train_records.csv"]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_meta = verify_stage(args.protocol, "multitask_shared_private_protocol")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(m.get("test_labels_read", False) for m in [protocol_meta, interface_meta, stl_meta]):
        raise ValueError("Inputs must remain test-closed")
    protocol = json.loads((args.protocol / "protocol.json").read_text())
    task_order = list(map(str, protocol["tasks"]))
    human_tasks = list(map(str, protocol["human_evaluation_tasks"]))
    task_index = {task: i for i, task in enumerate(task_order)}
    registry = pd.read_csv(args.protocol / "task_target_registry.csv")
    if set(registry.task_id) != set(task_order) or set(protocol["routes"]) != set(ROUTES):
        raise ValueError("Protocol task/route registry mismatch")
    bounded = registry.set_index("task_id").loc[task_order].model_output.eq("bounded_sigmoid_fraction").tolist()
    sampling = pd.read_csv(args.protocol / "task_sampling_registry.csv")
    if set(sampling.task_id) != set(task_order) or sampling.task_id.duplicated().any():
        raise ValueError("Sampling registry does not uniquely cover all tasks")
    phase1_probabilities = (
        sampling.set_index("task_id").tier_loss_weight / sampling.set_index("task_id").tasks_in_tier
    ).to_dict()
    route_registry = pd.read_csv(args.protocol / "route_registry.csv")
    registered_seeds = [int(v) for v in route_registry.formal_seeds.iloc[0].split(";")]
    full_configuration = bool(
        folds == list(range(5)) and seeds == registered_seeds and args.phase1_epochs == 20 and args.phase2_epochs == 10
        and args.batch_size == 128 and args.width == 128 and args.latent == 64
        and np.isclose(args.learning_rate, 3e-4) and np.isclose(args.weight_decay, 1e-3)
        and args.max_parents_per_task == 0 and args.max_eval_parents_per_task == 0
    )
    metadata = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=META_COLUMNS)
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(task_order)].copy()
    stl = pd.read_csv(args.stl / "benchmark_train_records.csv", dtype=str, keep_default_na=False,
                      usecols=["row_id", "task_id", "parent_id", "scaffold_group", "inner_fold_id"])
    stl = stl.loc[stl.task_id.isin(human_tasks)].copy()
    stl["inner_fold_id"] = pd.to_numeric(stl.inner_fold_id, errors="raise").astype(int)
    if set(stl.task_id) != set(human_tasks):
        raise ValueError("Human outer-fold registry incomplete")
    if args.check_only:
        print(f"Shared/private train-CV ready: folds={folds} seeds={seeds} routes=2 tasks={len(task_order)} human_heads={len(human_tasks)} device={args.device} full_configuration={full_configuration}")
        return

    prediction_rows, isolation_rows, target_rows, coverage_rows, loss_rows, check_rows = [], [], [], [], [], []
    with stage_output(args.output) as out:
        model_dir = out / "models"; model_dir.mkdir()
        for fold in folds:
            eval_membership = stl.loc[stl.inner_fold_id.eq(fold)].copy()
            eval_union_parents, eval_union_scaffolds = set(eval_membership.parent_id), set(eval_membership.scaffold_group)
            fold_meta = metadata.copy()
            fold_meta["parent_conflict"] = fold_meta.parent_id.isin(eval_union_parents)
            fold_meta["scaffold_conflict"] = fold_meta.scaffold_group.isin(eval_union_scaffolds)
            retained = fold_meta.loc[~fold_meta.parent_conflict & ~fold_meta.scaffold_conflict].copy()
            if set(retained.task_id) != set(task_order):
                raise ValueError(f"Fold {fold} lost a registered task")
            for task, before in fold_meta.groupby("task_id", sort=True):
                after = retained.loc[retained.task_id.eq(task)]
                isolation_rows.append({
                    "outer_fold": fold, "task_id": task, "records_before": len(before), "records_after": len(after),
                    "parents_after": after.parent_id.nunique(), "scaffolds_after": after.scaffold_group.nunique(),
                    "post_filter_parent_overlap": int(after.parent_id.isin(eval_union_parents).sum()),
                    "post_filter_scaffold_overlap": int(after.scaffold_group.isin(eval_union_scaffolds).sum()),
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
            if (stats.loc[stats.target_standardization.ne("none_physical_fraction"), "target_std_population"] <= 0).any():
                raise ValueError("Invalid fold-local target standardization")
            stats["outer_fold"] = fold; target_rows.extend(stats.to_dict("records"))
            fitting = fitting.merge(stats.drop(columns="outer_fold"), on=["task_id", "target_standardization"], validate="many_to_one")
            fitting["model_target"] = np.where(
                fitting.target_standardization.eq("none_physical_fraction"), fitting.target_value,
                (fitting.target_value - fitting.target_mean) / fitting.target_std_population,
            )
            fitting["task_index"] = fitting.task_id.map(task_index).astype(int)

            eval_meta = fold_meta.loc[fold_meta.row_id.isin(set(eval_membership.row_id)) & fold_meta.task_id.isin(human_tasks)].copy()
            eval_parent = eval_meta.groupby(["task_id", "parent_id"], as_index=False).agg(
                endpoint=("endpoint", "first"), smiles=("smiles", "first"), scaffold_group=("scaffold_group", "first"),
                source_records=("row_id", "size"),
            )
            if args.max_eval_parents_per_task > 0:
                eval_parent = pd.concat([
                    stable_limit(part, args.max_eval_parents_per_task, 20261917 + fold)
                    for _, part in eval_parent.groupby("task_id", sort=True)
                ], ignore_index=True)
            selected_eval = eval_meta.loc[
                eval_meta.set_index(["task_id", "parent_id"]).index.isin(
                    eval_parent.set_index(["task_id", "parent_id"]).index
                )
            ]
            eval_parent["task_index"] = eval_parent.task_id.map(task_index).astype(int)
            unique_smiles = list(dict.fromkeys([*fitting.smiles.astype(str), *eval_parent.smiles.astype(str)]))
            feature_all, descriptor_names = featurize(unique_smiles, feature_set="rdkit2d", progress=False)
            lookup = {smiles: i for i, smiles in enumerate(unique_smiles)}
            x_fit_raw = feature_all[np.asarray([lookup[v] for v in fitting.smiles])]
            x_eval_raw = feature_all[np.asarray([lookup[v] for v in eval_parent.smiles])]
            fold_predictions = []
            for seed in seeds:
                init_seed = seed + fold * 100000
                for route in ROUTES:
                    x_fit, scaler = scaled_features(x_fit_raw, fitting, route, task_order)
                    x_eval = apply_eval_scaler(x_eval_raw, eval_parent.task_id.tolist(), scaler)
                    seed_everything(init_seed, args.threads)
                    cls = SharedPrivateNet if route == "shared_encoder_private_heads" else FullyPrivateNet
                    model = cls(x_fit.shape[1], len(task_order), args.width, args.latent, bounded).to(args.device)
                    initial = torch.cat([v.detach().cpu().reshape(-1) for v in model.parameters()])
                    coverage, losses = train_phases(
                        model, x_fit, fitting, task_order, human_tasks, phase1_probabilities,
                        phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
                        batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                        seed=init_seed + 101, device=args.device,
                    )
                    final = torch.cat([v.detach().cpu().reshape(-1) for v in model.parameters()])
                    parameter_change = float(torch.max(torch.abs(final - initial)))
                    if parameter_change <= 0:
                        raise ValueError("Model parameters did not update")
                    checkpoint = {
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "route_id": route, "task_order": task_order, "bounded": bounded,
                        "n_features": x_fit.shape[1], "width": args.width, "latent": args.latent,
                        "scaler": scaler, "seed": seed, "outer_fold": fold,
                    }
                    model_path = model_dir / f"fold{fold}__seed{seed}__{route}.pt"
                    torch.save(checkpoint, model_path)
                    first = cpu_predict(checkpoint, x_eval, eval_parent.task_index.to_numpy(int))
                    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
                    second = cpu_predict(loaded, x_eval, eval_parent.task_index.to_numpy(int))
                    difference = float(np.max(np.abs(first - second)))
                    if difference > 1e-7 or not np.isfinite(first).all():
                        raise ValueError("Prediction reload/finite audit failed")
                    route_prediction = eval_parent[["task_id", "endpoint", "parent_id", "scaffold_group", "source_records"]].copy()
                    route_prediction["outer_fold"] = fold; route_prediction["seed"] = seed; route_prediction["route_id"] = route
                    route_prediction["predicted_model_target"] = first
                    fold_predictions.append(route_prediction)
                    coverage_rows.extend({"outer_fold": fold, "seed": seed, "route_id": route, **row} for row in coverage)
                    loss_rows.extend({"outer_fold": fold, "seed": seed, "route_id": route, **row} for row in losses)
                    check_rows.append({
                        "outer_fold": fold, "seed": seed, "route_id": route,
                        "max_parameter_change": parameter_change, "finite_predictions": True,
                        "reload_max_abs_difference": difference, "evaluation_targets_read_during_fit": False,
                    })
            eval_labels = load_labels_only(records_path, set(selected_eval.row_id))
            selected_eval = selected_eval.merge(eval_labels, on="row_id", validate="one_to_one")
            selected_eval["target_value"] = pd.to_numeric(selected_eval.target_value, errors="raise")
            observed = aggregate_parents(selected_eval, registry)[["task_id", "parent_id", "target_value"]].rename(columns={"target_value": "observed_primary_target"})
            fold_prediction = pd.concat(fold_predictions, ignore_index=True).merge(observed, on=["task_id", "parent_id"], validate="many_to_one")
            fold_prediction = fold_prediction.merge(
                stats[["task_id", "target_mean", "target_std_population", "target_standardization"]], on="task_id", validate="many_to_one"
            )
            fold_prediction["predicted_primary_target"] = np.where(
                fold_prediction.target_standardization.eq("none_physical_fraction"), fold_prediction.predicted_model_target,
                fold_prediction.predicted_model_target * fold_prediction.target_std_population + fold_prediction.target_mean,
            )
            prediction_rows.append(fold_prediction)

        predictions = pd.concat(prediction_rows, ignore_index=True)
        if predictions.duplicated(["outer_fold", "seed", "route_id", "task_id", "parent_id"]).any():
            raise ValueError("Prediction keys are not unique")
        ensemble = predictions.groupby(
            ["route_id", "outer_fold", "task_id", "endpoint", "parent_id", "scaffold_group"], as_index=False
        ).agg(observed_primary_target=("observed_primary_target", "first"), ensemble_predicted_primary_target=("predicted_primary_target", "mean"))
        metric_rows = []
        for keys, part in ensemble.groupby(["route_id", "outer_fold", "task_id", "endpoint"], sort=True):
            route, fold, task, endpoint = keys
            metric_name = registry.set_index("task_id").loc[task, "primary_metric"]
            error = part.ensemble_predicted_primary_target - part.observed_primary_target
            score = float(np.mean(np.abs(error))) if metric_name == "physical_MAE" else float(np.sqrt(np.mean(error ** 2)))
            metric_rows.append({"route_id": route, "outer_fold": fold, "task_id": task, "endpoint": endpoint,
                                "parents": len(part), "primary_metric": metric_name, "primary_score": score})
        isolation = pd.DataFrame(isolation_rows); checks = pd.DataFrame(check_rows); coverage = pd.DataFrame(coverage_rows)
        bounded_predictions = predictions.loc[predictions.task_id.isin(registry.loc[registry.model_output.eq("bounded_sigmoid_fraction"), "task_id"]), "predicted_primary_target"]
        predictions.to_csv(out / "seed_level_oof_predictions.csv", index=False)
        ensemble.to_csv(out / "ensemble_oof_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out / "ensemble_fold_metrics.csv", index=False)
        isolation.to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(target_rows).to_csv(out / "fold_local_target_statistics.csv", index=False)
        coverage.to_csv(out / "training_coverage.csv", index=False)
        pd.DataFrame(loss_rows).to_csv(out / "training_losses.csv", index=False)
        checks.to_csv(out / "route_checks.csv", index=False)
        (out / "README.md").write_text(
            "# Multitask shared/private train-CV\n\nOuter labels are read after every route and seed has fit. "
            "All 45 tasks are globally purged by the union of six human evaluation parent/scaffold sets. Fixed validation/test remain closed.\n",
            encoding="utf-8",
        )
        expected_models = len(folds) * len(seeds) * len(ROUTES)
        finish_stage(
            out, "multitask_shared_private_traincv",
            inputs={"protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                    "interface_complete_sha256": sha256(args.interface / "complete.json"),
                    "stl_complete_sha256": sha256(args.stl / "complete.json")},
            routes=2, tasks=len(task_order), human_evaluation_heads=len(human_tasks), outer_folds=folds, seeds=seeds,
            expected_model_files=expected_models, model_files=len(list(model_dir.glob("*.pt"))),
            max_post_filter_parent_overlap=int(isolation.post_filter_parent_overlap.max()),
            max_post_filter_scaffold_overlap=int(isolation.post_filter_scaffold_overlap.max()),
            predictions_finite=bool(np.isfinite(predictions.predicted_primary_target).all()),
            bounded_predictions_within_unit_interval=bool(bounded_predictions.between(0, 1).all()),
            max_reload_abs_difference=float(checks.reload_max_abs_difference.max()),
            training_phase_task_coverage_nonzero=bool(coverage.updates.gt(0).all()),
            evaluation_labels_read_after_all_fold_models_fit=True, evaluation_labels_used_during_fit=False,
            validation_target_file_opened=False, test_labels_read=False,
            architecture_selection_authorized=full_configuration, fixed_validation_selection_authorized=False,
            model_fitted=True, partial=not full_configuration, full_configuration=full_configuration,
            device=args.device, cuda_device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "not_used",
        )
    print(f"Shared/private train-CV: {args.output}")


if __name__ == "__main__":
    run_cli(main)
