#!/usr/bin/env python3
"""Run a frozen, protocol-driven controlled cross-species train-CV comparison.

Outer-evaluation labels are loaded only after all route/seed models for that
fold have finished fitting and prediction. Fixed validation and test are never
opened. Partial configurations are engineering checks and cannot select a route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dmpk_toolkit import featurize
from gate1b_torch_common import fit_feature_scaler, scale_features
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


HUMAN_TASK = "VDss__human__steady_state_iv"
ANIMAL_TASKS = [
    "VDss__dog__steady_state_iv",
    "VDss__monkey__steady_state_iv",
    "VDss__mouse__steady_state_iv",
    "VDss__rat__steady_state_iv",
]
TASKS = [HUMAN_TASK, *ANIMAL_TASKS]
SPECIES_ORDER = ["human", "dog", "monkey", "mouse", "rat"]
ROUTES = [
    "human_only_neural_control",
    "cross_species_pretrain_human_finetune",
    "species_conditioned_joint_shared_encoder",
]
MEMBERSHIP_COLUMNS = [
    "row_id", "task_id", "split", "endpoint", "species", "system", "canonical_unit",
    "smiles", "parent_id", "scaffold_group", "doc_id",
]
POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


class SpeciesPrivateNet(nn.Module):
    def __init__(
        self, n_features: int, n_species: int, encoder_width: int, latent_width: int,
        bounded_output: bool = False,
    ):
        super().__init__()
        self.bounded_output = bool(bounded_output)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, encoder_width), nn.GELU(), nn.LayerNorm(encoder_width),
            nn.Linear(encoder_width, latent_width), nn.GELU(), nn.LayerNorm(latent_width),
        )
        self.species_embedding = nn.Embedding(n_species, latent_width)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_width, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_species)
        ])

    def forward(self, x: torch.Tensor, species_index: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x) + self.species_embedding(species_index)
        output = torch.empty(len(x), dtype=x.dtype, device=x.device)
        for species in species_index.unique().tolist():
            take = species_index.eq(int(species))
            value = self.heads[int(species)](hidden[take]).squeeze(1)
            output[take] = torch.sigmoid(value) if self.bounded_output else value
        return output


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError(f"Expected a non-empty unique integer list: {value}")
    return parsed


def stable_limit(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.copy()
    ranked = frame.assign(
        _rank=frame.parent_id.map(lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    ).sort_values("_rank")
    return ranked.head(maximum).drop(columns="_rank").copy()


def load_labels_only(path: Path, allowed_row_ids: set[str]) -> pd.DataFrame:
    membership = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["row_id"])
    keep = membership.row_id.isin(allowed_row_ids).to_numpy()

    def skip(line_number: int) -> bool:
        return line_number > 0 and not bool(keep[line_number - 1])

    labels = pd.read_csv(
        path,
        dtype={"row_id": str, "target_value": float},
        keep_default_na=False,
        usecols=["row_id", "target_value"],
        skiprows=skip,
    )
    if set(labels.row_id) != allowed_row_ids or labels.target_value.isna().any():
        raise ValueError("Label-selective read did not return exactly the requested rows")
    return labels


def parent_table(records: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    local = records.copy()
    if aggregation == "mean_record_level_expit_physical":
        values = np.clip(local.target_value.to_numpy(float), -50.0, 50.0)
        local["aggregation_target"] = 1.0 / (1.0 + np.exp(-values))
    elif aggregation == "mean_transformed_target_within_task_and_parent":
        local["aggregation_target"] = local.target_value.to_numpy(float)
    else:
        raise ValueError(f"Unsupported parent aggregation: {aggregation}")
    grouped = local.groupby(["task_id", "parent_id"], as_index=False).agg(
        target_value=("aggregation_target", "mean"),
        smiles=("smiles", "first"),
        scaffold_group=("scaffold_group", "first"),
        species=("species", "first"),
        source_records=("row_id", "size"),
    )
    if grouped.duplicated(["task_id", "parent_id"]).any():
        raise ValueError("Parent aggregation is not unique")
    return grouped


def parameter_vector(model: nn.Module, prefix: str) -> torch.Tensor:
    values = [value.detach().cpu().reshape(-1) for name, value in model.named_parameters() if name.startswith(prefix)]
    return torch.cat(values) if values else torch.empty(0)


def balanced_epoch_batches(ids: np.ndarray, steps: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    needed = steps * batch_size
    blocks = []
    while sum(len(block) for block in blocks) < needed:
        blocks.append(rng.permutation(ids))
    sequence = np.concatenate(blocks)[:needed]
    return [sequence[start:start + batch_size] for start in range(0, needed, batch_size)]


def train_task_balanced(
    model: SpeciesPrivateNet,
    x: np.ndarray,
    frame: pd.DataFrame,
    task_ids: list[str],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    steps_per_task: int,
    loss_name: str,
) -> tuple[dict, list[dict]]:
    use = frame.loc[frame.task_id.isin(task_ids)]
    if use.empty or set(use.task_id) != set(task_ids):
        raise ValueError("Training phase does not cover every registered task")
    ids_by_task = {task: use.index[use.task_id.eq(task)].to_numpy(int) for task in task_ids}
    if steps_per_task < 1:
        raise ValueError("steps_per_task must be positive")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    species_tensor = torch.as_tensor(frame.species_index.to_numpy(int), dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(frame.model_target.to_numpy(float), dtype=torch.float32, device=device)
    coverage = {task: {"updates": 0, "sampled": 0, "unique": set()} for task in task_ids}
    losses = []
    model.train()
    for epoch in range(epochs):
        batches = {
            task: balanced_epoch_batches(ids, steps_per_task, batch_size, rng)
            for task, ids in ids_by_task.items()
        }
        for step in range(steps_per_task):
            for task in rng.permutation(task_ids):
                batch = batches[str(task)][step]
                batch_index = torch.as_tensor(batch, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_tensor.index_select(0, batch_index), species_tensor.index_select(0, batch_index))
                target = target_tensor.index_select(0, batch_index)
                if loss_name == "physical_MAE":
                    loss = torch.mean(torch.abs(prediction - target))
                elif loss_name == "task_standardized_Huber":
                    loss = torch.nn.functional.huber_loss(prediction, target, reduction="mean", delta=1.0)
                else:
                    raise ValueError(f"Unsupported transfer training loss: {loss_name}")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite cross-species transfer loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                values = coverage[str(task)]
                values["updates"] += 1
                values["sampled"] += len(batch)
                values["unique"].update(map(int, batch))
                losses.append({"epoch": epoch + 1, "step": step + 1, "task_id": str(task),
                               "loss": float(loss.detach().cpu())})
    flat = {
        task: {
            "updates": values["updates"],
            "steps_per_task_per_epoch": steps_per_task,
            "sampled_instances_across_epochs": values["sampled"],
            "unique_parents_sampled": len(values["unique"]),
        }
        for task, values in coverage.items()
    }
    if len({values["updates"] for values in flat.values()}) != 1:
        raise ValueError("Equal-step task balancing failed")
    return flat, losses


def cpu_predict(checkpoint: dict, x: np.ndarray, species_index: np.ndarray) -> np.ndarray:
    model = SpeciesPrivateNet(
        checkpoint["n_features"], len(checkpoint["species_order"]),
        checkpoint["encoder_width"], checkpoint["latent_width"], checkpoint.get("bounded_output", False),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaled = scale_features(x, checkpoint["x_mean"], checkpoint["x_scale"])
    with torch.no_grad():
        return model(
            torch.as_tensor(scaled, dtype=torch.float32),
            torch.as_tensor(species_index, dtype=torch.long),
        ).numpy().astype(float)


def maximum_tanimoto(eval_ecfp: np.ndarray, train_ecfp: np.ndarray, batch_size: int = 32) -> np.ndarray:
    eval_bits = np.packbits(eval_ecfp.astype(np.uint8, copy=False), axis=1)
    train_bits = np.packbits(train_ecfp.astype(np.uint8, copy=False), axis=1)
    eval_count = POPCOUNT[eval_bits].sum(axis=1).astype(np.int32)
    train_count = POPCOUNT[train_bits].sum(axis=1).astype(np.int32)
    result = np.empty(len(eval_bits), dtype=float)
    for start in range(0, len(eval_bits), batch_size):
        stop = min(start + batch_size, len(eval_bits))
        intersection = POPCOUNT[np.bitwise_and(eval_bits[start:stop, None, :], train_bits[None, :, :])].sum(axis=2)
        union = eval_count[start:stop, None] + train_count[None, :] - intersection
        result[start:stop] = np.max(intersection / np.maximum(union, 1), axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "data/public_development/vdss_cross_species_transfer_protocol_v4")
    parser.add_argument("--interface", type=Path,
                        default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--stl", type=Path,
                        default=ROOT / "data/public_development/stl_train_only_protocol_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/benchmarks/vdss_cross_species_transfer_traincv_v1")
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="20260917,20260918,20260919")
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--human-or-joint-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--latent-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-parents-per-task", type=int, default=0,
                        help="Engineering-only deterministic cap; zero means full registered data")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    folds = parse_int_list(args.outer_folds)
    seeds = parse_int_list(args.seeds)
    if not set(folds) <= set(range(5)):
        raise ValueError("Outer folds must be drawn from 0..4")
    numeric = [args.pretrain_epochs, args.human_or_joint_epochs, args.batch_size,
               args.encoder_width, args.latent_width, args.learning_rate, args.threads]
    if min(numeric) <= 0 or args.weight_decay < 0 or args.max_parents_per_task < 0:
        raise ValueError("Invalid training configuration")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)")

    records_path = args.interface / "training_records.csv"
    stl_path = args.stl / "benchmark_train_records.csv"
    required = [args.protocol / "complete.json", args.protocol / "protocol.json",
                args.protocol / "route_registry.csv", args.interface / "complete.json",
                args.stl / "complete.json", records_path, stl_path]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_marker = json.loads((args.protocol / "complete.json").read_text())
    protocol_stage = protocol_marker.get("stage")
    if protocol_stage not in {"vdss_cross_species_transfer_protocol", "cross_species_transfer_protocol"}:
        raise ValueError(f"Unsupported transfer protocol stage: {protocol_stage}")
    protocol_meta = verify_stage(args.protocol, protocol_stage)
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(meta.get("test_labels_read", False) for meta in [protocol_meta, interface_meta, stl_meta]):
        raise ValueError("Formal inputs must remain test-closed")
    contract = json.loads((args.protocol / "protocol.json").read_text())
    human_task = str(contract["human_task"])
    animal_tasks = [str(value) for value in contract["animal_tasks"]]
    tasks = [human_task, *animal_tasks]
    species_order = [str(value) for value in contract["species_order"]]
    endpoint = str(contract.get("endpoint", human_task.split("__", 1)[0]))
    parent_aggregation = str(
        contract.get("parent_aggregation", "mean_transformed_target_within_task_and_parent")
    )
    if parent_aggregation == "mean transformed target within task and canonical parent":
        parent_aggregation = "mean_transformed_target_within_task_and_parent"
    target_standardization = str(
        contract.get("target_standardization", "taskwise_zscore_fit_on_retained_train_parents")
    )
    model_output = str(contract.get("model_output", "unbounded_standardized_target"))
    training_loss = str(contract.get("training_loss", "task_standardized_Huber"))
    primary_metric = str(contract.get("human_primary_metric", "transformed_RMSE"))
    bounded_physical = model_output == "bounded_sigmoid_fraction"
    expected_seed_aggregation = (
        "arithmetic mean of the three fixed-seed predictions in physical fraction space before primary scoring"
        if bounded_physical
        else "arithmetic mean of the three fixed-seed predictions in transformed target space before primary scoring"
    )
    if contract.get("seed_aggregation") != expected_seed_aggregation:
        raise ValueError("Formal runner requires a target-space-specific seed-aggregation contract")
    if bounded_physical:
        expected_physical = {
            "parent_aggregation": "mean_record_level_expit_physical",
            "target_standardization": "none_physical_fraction",
            "training_loss": "physical_MAE",
            "primary_metric": "physical_MAE",
        }
        actual_physical = {
            "parent_aggregation": parent_aggregation,
            "target_standardization": target_standardization,
            "training_loss": training_loss,
            "primary_metric": primary_metric,
        }
        if actual_physical != expected_physical or endpoint != "fu":
            raise ValueError(f"Invalid bounded physical-target contract: {actual_physical}")
    elif any([
        parent_aggregation != "mean_transformed_target_within_task_and_parent",
        target_standardization != "taskwise_zscore_fit_on_retained_train_parents",
        training_loss != "task_standardized_Huber",
        primary_metric != "transformed_RMSE",
    ]):
        raise ValueError("Unbounded transfer contract differs from the implemented transformed-target path")
    if len(animal_tasks) != 4 or species_order != ["human", "dog", "monkey", "mouse", "rat"]:
        raise ValueError("Formal runner requires the registered human plus four-species task layout")

    registry = pd.read_csv(args.protocol / "route_registry.csv")
    if set(registry.route_id) != set(ROUTES) or not registry.encoder.eq("MLP_128_64_GELU_layernorm").all():
        raise ValueError("Route registry differs from the implemented finite comparison")
    registered_seeds = [int(value) for value in str(registry.formal_seeds.iloc[0]).split(";")]
    full_configuration = bool(
        folds == list(range(5)) and seeds == registered_seeds and args.pretrain_epochs == 40
        and args.human_or_joint_epochs == 30 and args.batch_size == 128
        and args.encoder_width == 128 and args.latent_width == 64
        and np.isclose(args.learning_rate, 0.0003) and np.isclose(args.weight_decay, 0.001)
        and args.max_parents_per_task == 0
    )
    json.dumps({"full_configuration": full_configuration}, allow_nan=False)

    metadata = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=MEMBERSHIP_COLUMNS)
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(tasks)].copy()
    if set(metadata.task_id) != set(tasks):
        raise ValueError("Training interface does not contain the five registered tasks")
    stl_membership = pd.read_csv(
        stl_path, dtype=str, keep_default_na=False,
        usecols=["row_id", "task_id", "parent_id", "scaffold_group", "inner_fold_id"],
    )
    stl_membership = stl_membership.loc[stl_membership.task_id.eq(human_task)].copy()
    stl_membership["inner_fold_id"] = pd.to_numeric(stl_membership.inner_fold_id, errors="raise").astype(int)
    if set(stl_membership.inner_fold_id) != set(range(5)):
        raise ValueError(f"Frozen human {endpoint} folds are incomplete")

    unique_smiles = list(dict.fromkeys(metadata.smiles.astype(str)))
    feature_all, descriptor_names = featurize(unique_smiles, feature_set="ecfp4_rdkit2d", progress=False)
    ecfp_all = feature_all[:, :2048]
    rdkit_all = feature_all[:, 2048:]
    feature_lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
    if args.check_only:
        print(
            f"{endpoint} formal train-CV ready: folds={folds} seeds={seeds} routes=3 tasks=5 "
            f"structures={len(unique_smiles)} rdkit2d={rdkit_all.shape[1]} device={args.device} "
            f"full_configuration={full_configuration}"
        )
        return

    species_map = {species: index for index, species in enumerate(species_order)}
    human_index = species_map["human"]
    prediction_rows, isolation_rows, target_rows = [], [], []
    coverage_rows, loss_rows, check_rows = [], [], []
    evaluation_labels_read_after_fit = True
    with stage_output(args.output) as out:
        model_dir = out / "models"
        model_dir.mkdir()
        for outer_fold in folds:
            eval_membership = stl_membership.loc[stl_membership.inner_fold_id.eq(outer_fold)]
            evaluation_parents = set(eval_membership.parent_id)
            evaluation_scaffolds = set(eval_membership.scaffold_group)
            evaluation_rows = set(eval_membership.row_id)
            fold_meta = metadata.copy()
            fold_meta["parent_conflict"] = fold_meta.parent_id.isin(evaluation_parents)
            fold_meta["scaffold_conflict"] = fold_meta.scaffold_group.isin(evaluation_scaffolds)
            retained = fold_meta.loc[~fold_meta.parent_conflict & ~fold_meta.scaffold_conflict].copy()
            if set(retained.task_id) != set(tasks):
                raise ValueError(f"Fold {outer_fold} filtering removed a registered task")
            for task, before in fold_meta.groupby("task_id", sort=True):
                after = retained.loc[retained.task_id.eq(task)]
                isolation_rows.append({
                    "outer_fold": outer_fold, "task_id": task, "records_before": len(before),
                    "records_removed_parent": int(before.parent_conflict.sum()),
                    "records_removed_scaffold_only": int((before.scaffold_conflict & ~before.parent_conflict).sum()),
                    "records_after": len(after), "parents_after": after.parent_id.nunique(),
                    "scaffolds_after": after.scaffold_group.nunique(),
                    "post_filter_parent_overlap": int(after.parent_id.isin(evaluation_parents).sum()),
                    "post_filter_scaffold_overlap": int(after.scaffold_group.isin(evaluation_scaffolds).sum()),
                })
            fit_labels = load_labels_only(records_path, set(retained.row_id))
            retained = retained.merge(fit_labels, on="row_id", validate="one_to_one")
            retained["target_value"] = pd.to_numeric(retained.target_value, errors="raise")
            parents = parent_table(retained, parent_aggregation)
            fitting = pd.concat([
                stable_limit(group, args.max_parents_per_task, 20260917 + outer_fold)
                for _, group in parents.groupby("task_id", sort=True)
            ], ignore_index=True)
            stats = fitting.groupby("task_id").target_value.agg(["mean", lambda x: x.std(ddof=0)]).reset_index()
            stats.columns = ["task_id", "target_mean", "target_std_population"]
            if (stats.target_std_population <= 0).any():
                raise ValueError(f"Invalid fold-local target statistics in fold {outer_fold}")
            stats["outer_fold"] = outer_fold
            target_rows.extend(stats.to_dict("records"))
            fitting = fitting.merge(stats.drop(columns="outer_fold"), on="task_id", validate="many_to_one")
            fitting["model_target"] = (
                fitting.target_value
                if bounded_physical
                else (fitting.target_value - fitting.target_mean) / fitting.target_std_population
            )
            fitting["species_index"] = fitting.species.map(species_map)
            if fitting.species_index.isna().any():
                raise ValueError("Unregistered species")
            fitting.species_index = fitting.species_index.astype(int)
            x_fit_raw = rdkit_all[np.asarray([feature_lookup[value] for value in fitting.smiles])]

            eval_meta = fold_meta.loc[
                fold_meta.row_id.isin(evaluation_rows) & fold_meta.task_id.eq(human_task)
            ].copy()
            eval_parent = eval_meta.groupby("parent_id", as_index=False).agg(
                smiles=("smiles", "first"), scaffold_group=("scaffold_group", "first"),
                source_records=("row_id", "size"),
                document_ids=("doc_id", lambda values: ";".join(sorted({
                    str(value) for value in values if str(value)
                }))),
            )
            if args.max_parents_per_task > 0:
                eval_parent = stable_limit(eval_parent, args.max_parents_per_task, 20261917 + outer_fold)
                selected_eval_parents = set(eval_parent.parent_id)
                selected_eval_rows = set(eval_meta.loc[eval_meta.parent_id.isin(selected_eval_parents), "row_id"])
            else:
                selected_eval_rows = evaluation_rows
            x_eval_raw = rdkit_all[np.asarray([feature_lookup[value] for value in eval_parent.smiles])]
            eval_ecfp = ecfp_all[np.asarray([feature_lookup[value] for value in eval_parent.smiles])]
            human_fit = fitting.loc[fitting.task_id.eq(human_task)]
            human_reference_steps = math.ceil(len(human_fit) / args.batch_size)
            human_fit_ecfp = ecfp_all[np.asarray([feature_lookup[value] for value in human_fit.smiles])]
            eval_parent["max_human_train_ecfp4_tanimoto"] = maximum_tanimoto(eval_ecfp, human_fit_ecfp)
            human_train_docs = set(filter(None, retained.loc[retained.task_id.eq(human_task), "doc_id"].astype(str)))
            eval_parent["source_seen_in_human_train"] = eval_parent.document_ids.map(
                lambda value: any(doc in human_train_docs for doc in value.split(";") if doc)
            )

            fold_predictions = []
            for seed in seeds:
                initialization_seed = seed + 100000 * outer_fold
                for route in ROUTES:
                    route_tasks = [human_task] if route == "human_only_neural_control" else tasks
                    scaler_rows = fitting.task_id.isin(route_tasks).to_numpy()
                    x_mean, x_scale = fit_feature_scaler(x_fit_raw[scaler_rows])
                    x_fit = scale_features(x_fit_raw, x_mean, x_scale)
                    seed_everything(initialization_seed, args.threads)
                    model = SpeciesPrivateNet(
                        x_fit.shape[1], len(species_order), args.encoder_width, args.latent_width,
                        bounded_output=bounded_physical,
                    ).to(args.device)
                    encoder_initial = parameter_vector(model, "encoder")
                    human_head_initial = parameter_vector(model, f"heads.{human_index}")
                    if route == "human_only_neural_control":
                        phases = [("human_only", [human_task], args.human_or_joint_epochs, initialization_seed + 101)]
                    elif route == "cross_species_pretrain_human_finetune":
                        phases = [
                            ("animal_pretrain", animal_tasks, args.pretrain_epochs, initialization_seed + 201),
                            ("human_finetune", [human_task], args.human_or_joint_epochs, initialization_seed + 101),
                        ]
                    else:
                        phases = [("joint_task_balanced", tasks, args.human_or_joint_epochs, initialization_seed + 301)]
                    for phase, phase_tasks, epochs, phase_seed in phases:
                        coverage, losses = train_task_balanced(
                            model, x_fit, fitting, phase_tasks, epochs=epochs,
                            batch_size=args.batch_size, learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay, seed=phase_seed, device=args.device,
                            steps_per_task=human_reference_steps, loss_name=training_loss,
                        )
                        for task, values in coverage.items():
                            coverage_rows.append({"outer_fold": outer_fold, "seed": seed, "route_id": route,
                                                  "phase": phase, "task_id": task, **values})
                        loss_rows.extend([{"outer_fold": outer_fold, "seed": seed, "route_id": route,
                                           "phase": phase, **row} for row in losses])
                    encoder_delta = float(torch.max(torch.abs(parameter_vector(model, "encoder") - encoder_initial)))
                    human_head_delta = float(torch.max(torch.abs(
                        parameter_vector(model, f"heads.{human_index}") - human_head_initial
                    )))
                    if encoder_delta <= 0 or human_head_delta <= 0:
                        raise ValueError(f"Expected parameter updates: fold={outer_fold} seed={seed} route={route}")
                    checkpoint = {
                        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                        "n_features": int(x_fit.shape[1]), "species_order": species_order,
                        "encoder_width": args.encoder_width, "latent_width": args.latent_width,
                        "x_mean": x_mean, "x_scale": x_scale, "route_id": route,
                        "seed": seed, "outer_fold": outer_fold, "bounded_output": bounded_physical,
                    }
                    model_path = model_dir / f"fold{outer_fold}__seed{seed}__{route}.pt"
                    torch.save(checkpoint, model_path)
                    first = cpu_predict(checkpoint, x_eval_raw, np.full(len(x_eval_raw), human_index))
                    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
                    second = cpu_predict(loaded, x_eval_raw, np.full(len(x_eval_raw), human_index))
                    difference = float(np.max(np.abs(first - second)))
                    if difference > 1e-7 or not np.isfinite(first).all():
                        raise ValueError(f"Reload/finite check failed: fold={outer_fold} seed={seed} route={route}")
                    if bounded_physical:
                        primary_prediction = first
                    else:
                        human_stats = stats.loc[stats.task_id.eq(human_task)].iloc[0]
                        primary_prediction = (
                            first * float(human_stats.target_std_population) + float(human_stats.target_mean)
                        )
                    route_prediction = eval_parent[[
                        "parent_id", "scaffold_group", "source_records", "document_ids",
                        "source_seen_in_human_train", "max_human_train_ecfp4_tanimoto",
                    ]].copy()
                    route_prediction["outer_fold"] = outer_fold
                    route_prediction["seed"] = seed
                    route_prediction["route_id"] = route
                    prediction_column = (
                        "predicted_physical_target" if bounded_physical else "predicted_transformed_target"
                    )
                    route_prediction[prediction_column] = primary_prediction
                    fold_predictions.append(route_prediction)
                    check_rows.append({
                        "outer_fold": outer_fold, "seed": seed, "route_id": route,
                        "encoder_max_parameter_change": encoder_delta,
                        "human_head_max_parameter_change": human_head_delta,
                        "finite_predictions": bool(np.isfinite(first).all()),
                        "reload_max_abs_difference": difference,
                        "evaluation_parents": len(eval_parent),
                        "evaluation_targets_read_during_fit": False,
                    })

            # This is deliberately after every route and seed in the fold has fit and predicted.
            evaluation_labels = load_labels_only(records_path, selected_eval_rows)
            labeled_eval = eval_meta.loc[eval_meta.row_id.isin(selected_eval_rows)].merge(
                evaluation_labels, on="row_id", validate="one_to_one"
            )
            labeled_eval["target_value"] = pd.to_numeric(labeled_eval.target_value, errors="raise")
            if bounded_physical:
                label_values = np.clip(labeled_eval.target_value.to_numpy(float), -50.0, 50.0)
                labeled_eval["physical_target"] = 1.0 / (1.0 + np.exp(-label_values))
                observed = labeled_eval.groupby("parent_id", as_index=False).agg(
                    observed_physical_target=("physical_target", "mean")
                )
                observed_column = "observed_physical_target"
            else:
                observed = labeled_eval.groupby("parent_id", as_index=False).agg(
                    observed_transformed_target=("target_value", "mean")
                )
                observed_column = "observed_transformed_target"
            fold_prediction = pd.concat(fold_predictions, ignore_index=True).merge(
                observed, on="parent_id", validate="many_to_one"
            )
            if fold_prediction[observed_column].isna().any():
                raise ValueError(f"Missing post-fit evaluation labels in fold {outer_fold}")
            prediction_rows.append(fold_prediction)

        predictions = pd.concat(prediction_rows, ignore_index=True)
        expected_rows = sum(
            predictions.loc[predictions.outer_fold.eq(fold), "parent_id"].nunique()
            for fold in folds
        ) * len(seeds) * len(ROUTES)
        if len(predictions) != expected_rows or predictions.duplicated(
            ["outer_fold", "seed", "route_id", "parent_id"]
        ).any():
            raise ValueError("Formal prediction coverage is duplicate or incomplete")
        if bounded_physical:
            observed_column = "observed_physical_target"
            seed_prediction_column = "predicted_physical_target"
            ensemble_prediction_column = "ensemble_predicted_physical_target"
            metric_column = "physical_MAE"
            metric_function = lambda frame, prediction: float(np.mean(np.abs(
                frame[observed_column] - frame[prediction]
            )))
        else:
            observed_column = "observed_transformed_target"
            seed_prediction_column = "predicted_transformed_target"
            ensemble_prediction_column = "ensemble_predicted_transformed_target"
            metric_column = "transformed_RMSE"
            metric_function = lambda frame, prediction: float(np.sqrt(np.mean((
                frame[observed_column] - frame[prediction]
            ) ** 2)))
        seed_metrics = predictions.groupby(["route_id", "seed", "outer_fold"], as_index=False).apply(
            lambda frame: pd.Series({
                "parents": len(frame), metric_column: metric_function(frame, seed_prediction_column),
            }), include_groups=False
        ).reset_index(drop=True)
        ensemble_group_columns = [
            "route_id", "outer_fold", "parent_id", "scaffold_group", "source_records", "document_ids",
            "source_seen_in_human_train", "max_human_train_ecfp4_tanimoto", observed_column,
        ]
        ensemble = predictions.groupby(ensemble_group_columns, as_index=False)[seed_prediction_column].mean().rename(
            columns={seed_prediction_column: ensemble_prediction_column}
        )
        ensemble_metrics = ensemble.groupby(["route_id", "outer_fold"], as_index=False).apply(
            lambda frame: pd.Series({
                "parents": len(frame), metric_column: metric_function(frame, ensemble_prediction_column),
            }), include_groups=False
        ).reset_index(drop=True)
        isolation = pd.DataFrame(isolation_rows)
        coverage = pd.DataFrame(coverage_rows)
        checks = pd.DataFrame(check_rows)
        expected_model_files = len(folds) * len(seeds) * len(ROUTES)
        model_files = len(list(model_dir.iterdir()))
        if model_files != expected_model_files:
            raise ValueError(f"Expected {expected_model_files} model files, found {model_files}")
        overlap_columns = ["post_filter_parent_overlap", "post_filter_scaffold_overlap"]
        if isolation[overlap_columns].to_numpy().any():
            raise ValueError("Outer-fold parent/scaffold isolation audit failed")
        if not checks.finite_predictions.all() or checks.reload_max_abs_difference.max() > 1e-7:
            raise ValueError("Registered route prediction/reload audit failed")
        predictions_within_primary_bounds = bool(
            predictions[seed_prediction_column].between(0.0, 1.0, inclusive="both").all()
        ) if bounded_physical else True
        if bounded_physical and not predictions_within_primary_bounds:
            raise ValueError("Bounded physical predictions escaped the unit interval")
        if not checks.encoder_max_parameter_change.gt(0).all() \
                or not checks.human_head_max_parameter_change.gt(0).all():
            raise ValueError("A registered route did not update the expected parameters")
        if not coverage.updates.gt(0).all() or not coverage.unique_parents_sampled.gt(0).all():
            raise ValueError("A registered training phase lacks task coverage")
        isolation.to_csv(out / "fold_isolation_audit.csv", index=False)
        pd.DataFrame(target_rows).to_csv(out / "fold_local_target_statistics.csv", index=False)
        coverage.to_csv(out / "training_coverage.csv", index=False)
        pd.DataFrame(loss_rows).to_csv(out / "training_losses.csv", index=False)
        checks.to_csv(out / "route_checks.csv", index=False)
        predictions.to_csv(out / "seed_level_oof_predictions.csv", index=False)
        ensemble.to_csv(out / "ensemble_oof_predictions.csv", index=False)
        seed_metrics.to_csv(out / "seed_fold_metrics.csv", index=False)
        ensemble_metrics.to_csv(out / "ensemble_fold_metrics.csv", index=False)
        (out / "README.md").write_text(
            f"# {endpoint} controlled cross-species transfer train-CV\n\n"
            "Models use only retained outer-training labels. Each fold's evaluation targets are loaded only after "
            "all registered routes and seeds have fit and predicted. Primary scoring uses the arithmetic mean of "
            f"three seed predictions in {'physical fraction' if bounded_physical else 'transformed target'} space. "
            "Fixed validation/test remain closed.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "vdss_cross_species_transfer_traincv" if protocol_stage == "vdss_cross_species_transfer_protocol"
            else "cross_species_transfer_traincv",
            inputs={
                "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_train_only_complete_sha256": sha256(args.stl / "complete.json"),
            },
            endpoint=endpoint, human_task=human_task,
            routes=len(ROUTES), tasks=len(tasks), outer_folds=folds, seeds=seeds,
            pretrain_epochs=args.pretrain_epochs,
            human_or_joint_epochs=args.human_or_joint_epochs,
            batch_size=args.batch_size, encoder_width=args.encoder_width,
            latent_width=args.latent_width, feature_set="rdkit2d",
            descriptor_count=len(descriptor_names), device=args.device,
            cuda_device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "not_used",
            task_balancing="equal_optimizer_steps_per_task_per_epoch_using_human_parent_count",
            seed_aggregation=(
                "arithmetic_mean_prediction_in_physical_fraction_space"
                if bounded_physical else "arithmetic_mean_prediction_in_transformed_space"
            ),
            parent_aggregation=parent_aggregation, target_standardization=target_standardization,
            model_output=model_output, training_loss=training_loss, primary_metric=primary_metric,
            evaluation_labels_read_after_all_fold_models_fit=evaluation_labels_read_after_fit,
            evaluation_labels_used_during_fit=False,
            validation_target_file_opened=False, test_labels_read=False,
            train_cv_route_comparison_authorized=full_configuration,
            fixed_validation_selection_authorized=False,
            expected_model_files=expected_model_files, model_files=model_files,
            seed_level_predictions=len(predictions), ensemble_predictions=len(ensemble),
            max_post_filter_parent_overlap=int(isolation.post_filter_parent_overlap.max()),
            max_post_filter_scaffold_overlap=int(isolation.post_filter_scaffold_overlap.max()),
            predictions_finite=bool(checks.finite_predictions.all()),
            predictions_within_primary_bounds=predictions_within_primary_bounds,
            max_reload_abs_difference=float(checks.reload_max_abs_difference.max()),
            encoder_updates_nonzero=bool(checks.encoder_max_parameter_change.gt(0).all()),
            human_head_updates_nonzero=bool(checks.human_head_max_parameter_change.gt(0).all()),
            training_phase_task_coverage_nonzero=bool(
                coverage.updates.gt(0).all() and coverage.unique_parents_sampled.gt(0).all()
            ),
            model_fitted=True, partial=not full_configuration, full_configuration=full_configuration,
        )
    print(f"{endpoint} transfer train-CV: {args.output}")


if __name__ == "__main__":
    run_cli(main)
