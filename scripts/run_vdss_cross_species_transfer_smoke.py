#!/usr/bin/env python3
"""Engineering smoke for the frozen VDss controlled-transfer protocol.

The smoke fits one seed on human outer fold 0 with bounded parents/epochs. It
does not parse outer-evaluation, fixed-validation, or test targets and cannot
authorize architecture selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


class SpeciesPrivateNet(nn.Module):
    def __init__(self, n_features: int, n_species: int, encoder_width: int, latent_width: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, encoder_width), nn.GELU(), nn.LayerNorm(encoder_width),
            nn.Linear(encoder_width, latent_width), nn.GELU(), nn.LayerNorm(latent_width),
        )
        self.species_embedding = nn.Embedding(n_species, latent_width)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(latent_width, 32), nn.GELU(), nn.Linear(32, 1))
                                    for _ in range(n_species)])

    def forward(self, x: torch.Tensor, species_index: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x) + self.species_embedding(species_index)
        output = torch.empty(len(x), dtype=x.dtype, device=x.device)
        for species in species_index.unique().tolist():
            take = species_index.eq(int(species))
            output[take] = self.heads[int(species)](hidden[take]).squeeze(1)
        return output


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_limit(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    ranked = frame.assign(
        _rank=frame.parent_id.map(lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    ).sort_values("_rank")
    return ranked.head(maximum).drop(columns="_rank").copy()


def load_train_labels_only(path: Path, allowed_row_ids: set[str]) -> pd.DataFrame:
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
        raise ValueError("Label-selective read did not return exactly the retained fitting rows")
    return labels


def parent_table(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby(["task_id", "parent_id"], as_index=False).agg(
        target_value=("target_value", "mean"),
        smiles=("smiles", "first"),
        scaffold_group=("scaffold_group", "first"),
        species=("species", "first"),
        source_records=("row_id", "size"),
    )
    if grouped.groupby(["task_id", "parent_id"]).size().max() != 1:
        raise ValueError("Parent aggregation is not unique")
    return grouped


def parameter_vector(model: nn.Module, prefix: str) -> torch.Tensor:
    values = [value.detach().cpu().reshape(-1) for name, value in model.named_parameters() if name.startswith(prefix)]
    return torch.cat(values) if values else torch.empty(0)


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
) -> tuple[dict, list[dict]]:
    use = frame.loc[frame.task_id.isin(task_ids)].copy()
    if use.empty or set(use.task_id) != set(task_ids):
        raise ValueError("Training phase does not cover every registered task")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    coverage = {task: {"updates": 0, "sampled": 0, "unique": set()} for task in task_ids}
    losses = []
    model.train()
    for epoch in range(epochs):
        for task in rng.permutation(task_ids):
            ids = use.index[use.task_id.eq(task)].to_numpy(int)
            order = rng.permutation(ids)
            for batch in np.array_split(order, max(1, int(np.ceil(len(order) / batch_size)))):
                part = frame.loc[batch]
                optimizer.zero_grad(set_to_none=True)
                xx = torch.as_tensor(x[batch], dtype=torch.float32, device=device)
                species = torch.as_tensor(part.species_index.to_numpy(int), dtype=torch.long, device=device)
                yy = torch.as_tensor(part.standardized_target.to_numpy(float), dtype=torch.float32, device=device)
                prediction = model(xx, species)
                loss = torch.nn.functional.huber_loss(prediction, yy, reduction="mean", delta=1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite transfer smoke loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                coverage[str(task)]["updates"] += 1
                coverage[str(task)]["sampled"] += len(batch)
                coverage[str(task)]["unique"].update(map(int, batch))
                losses.append({"epoch": epoch + 1, "task_id": str(task), "loss": float(loss.detach().cpu())})
    flat = {
        task: {
            "updates": values["updates"],
            "sampled_instances_across_epochs": values["sampled"],
            "unique_parents_sampled": len(values["unique"]),
        }
        for task, values in coverage.items()
    }
    if any(values["updates"] < epochs or values["unique_parents_sampled"] == 0 for values in flat.values()):
        raise ValueError("Task-balanced phase failed mandatory task coverage")
    return flat, losses


def cpu_predict(checkpoint: dict, x: np.ndarray, species_index: np.ndarray) -> np.ndarray:
    model = SpeciesPrivateNet(
        checkpoint["n_features"],
        len(checkpoint["species_order"]),
        checkpoint["encoder_width"],
        checkpoint["latent_width"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaled = scale_features(x, checkpoint["x_mean"], checkpoint["x_scale"])
    with torch.no_grad():
        return model(
            torch.as_tensor(scaled, dtype=torch.float32),
            torch.as_tensor(species_index, dtype=torch.long),
        ).numpy().astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "data/public_development/vdss_cross_species_transfer_protocol_v1",
    )
    parser.add_argument(
        "--interface",
        type=Path,
        default=ROOT / "data/public_development/multitask_pk_training_interface_v5",
    )
    parser.add_argument(
        "--stl",
        type=Path,
        default=ROOT / "data/public_development/stl_train_only_protocol_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/benchmarks/vdss_cross_species_transfer_smoke_v3",
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--max-parents-per-task", type=int, default=64)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--latent-width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.outer_fold not in range(5) or min(
        args.max_parents_per_task,
        args.smoke_epochs,
        args.batch_size,
        args.encoder_width,
        args.latent_width,
        args.threads,
    ) < 1:
        raise ValueError("Invalid bounded smoke configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    records_path = args.interface / "training_records.csv"
    stl_path = args.stl / "benchmark_train_records.csv"
    required = [args.protocol / "complete.json", args.interface / "complete.json", args.stl / "complete.json",
                records_path, stl_path]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol_meta = verify_stage(args.protocol, "vdss_cross_species_transfer_protocol")
    interface_meta = verify_stage(args.interface, "multitask_pk_training_interface")
    stl_meta = verify_stage(args.stl, "stl_train_only_protocol")
    if any(meta.get("test_labels_read", False) for meta in [protocol_meta, interface_meta, stl_meta]):
        raise ValueError("Smoke inputs must remain test-closed")
    if protocol_meta.get("architecture_selection_authorized"):
        raise ValueError("Smoke protocol cannot authorize architecture selection")

    metadata = pd.read_csv(records_path, dtype=str, keep_default_na=False, usecols=MEMBERSHIP_COLUMNS)
    metadata = metadata.loc[metadata.split.eq("train") & metadata.task_id.isin(TASKS)].copy()
    stl_membership = pd.read_csv(
        stl_path,
        dtype=str,
        keep_default_na=False,
        usecols=["row_id", "task_id", "parent_id", "scaffold_group", "inner_fold_id"],
    )
    stl_membership = stl_membership.loc[stl_membership.task_id.eq(HUMAN_TASK)].copy()
    stl_membership["inner_fold_id"] = pd.to_numeric(stl_membership.inner_fold_id, errors="raise").astype(int)
    evaluation_membership = stl_membership.loc[stl_membership.inner_fold_id.eq(args.outer_fold)]
    evaluation_parents = set(evaluation_membership.parent_id)
    evaluation_scaffolds = set(evaluation_membership.scaffold_group)
    evaluation_rows = set(evaluation_membership.row_id)
    if not evaluation_parents or not evaluation_scaffolds:
        raise ValueError("Empty VDss evaluation fold")

    metadata["parent_conflict"] = metadata.parent_id.isin(evaluation_parents)
    metadata["scaffold_conflict"] = metadata.scaffold_group.isin(evaluation_scaffolds)
    retained = metadata.loc[~metadata.parent_conflict & ~metadata.scaffold_conflict].copy()
    if retained.parent_id.isin(evaluation_parents).any() or retained.scaffold_group.isin(evaluation_scaffolds).any():
        raise ValueError("Retained smoke fitting membership overlaps evaluation")
    if set(retained.task_id) != set(TASKS):
        raise ValueError("Fold filtering removed an entire VDss task")

    allowed_labels = set(retained.row_id)
    labels = load_train_labels_only(records_path, allowed_labels)
    retained = retained.merge(labels, on="row_id", validate="one_to_one")
    retained["target_value"] = pd.to_numeric(retained.target_value, errors="raise")
    parents = parent_table(retained)
    sampled = []
    for task, group in parents.groupby("task_id", sort=True):
        sampled.append(stable_limit(group, args.max_parents_per_task, args.seed))
    fitting = pd.concat(sampled, ignore_index=True)
    if set(fitting.task_id) != set(TASKS):
        raise ValueError("Bounded fitting view lost a VDss task")

    evaluation_meta = metadata.loc[metadata.row_id.isin(evaluation_rows) & metadata.task_id.eq(HUMAN_TASK)].copy()
    evaluation_parent = evaluation_meta.groupby("parent_id", as_index=False).agg(
        smiles=("smiles", "first"), scaffold_group=("scaffold_group", "first")
    )
    evaluation_parent = stable_limit(evaluation_parent, args.max_parents_per_task, args.seed + 1)
    if evaluation_parent.empty:
        raise ValueError("Empty bounded evaluation structure view")

    stats = fitting.groupby("task_id").target_value.agg(["mean", lambda x: x.std(ddof=0)]).reset_index()
    stats.columns = ["task_id", "target_mean", "target_std_population"]
    if (stats.target_std_population <= 0).any() or not np.isfinite(stats[["target_mean", "target_std_population"]]).all().all():
        raise ValueError("Invalid fold-local target statistics")
    fitting = fitting.merge(stats, on="task_id", validate="many_to_one")
    fitting["standardized_target"] = (fitting.target_value - fitting.target_mean) / fitting.target_std_population
    species_map = {species: index for index, species in enumerate(SPECIES_ORDER)}
    fitting["species_index"] = fitting.species.map(species_map)
    if fitting.species_index.isna().any():
        raise ValueError("Unregistered species")
    fitting.species_index = fitting.species_index.astype(int)

    unique_smiles = list(dict.fromkeys([*fitting.smiles.astype(str), *evaluation_parent.smiles.astype(str)]))
    features, descriptor_names = featurize(unique_smiles, feature_set="rdkit2d", progress=False)
    feature_lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
    x_fit_raw = features[np.asarray([feature_lookup[value] for value in fitting.smiles])]
    x_eval_raw = features[np.asarray([feature_lookup[value] for value in evaluation_parent.smiles])]

    if args.check_only:
        print(
            f"VDss transfer smoke ready: fit_parents={len(fitting)} eval_parents={len(evaluation_parent)} "
            f"features={x_fit_raw.shape[1]} device={args.device}"
        )
        return

    seed_everything(args.seed, args.threads)
    with stage_output(args.output) as out:
        isolation_rows = []
        for task, before in metadata.groupby("task_id", sort=True):
            after = retained.loc[retained.task_id.eq(task)]
            isolation_rows.append({
                "task_id": task,
                "records_before": len(before),
                "records_removed_parent": int(before.parent_conflict.sum()),
                "records_removed_scaffold_only": int((before.scaffold_conflict & ~before.parent_conflict).sum()),
                "records_after": len(after),
                "parents_after": after.parent_id.nunique(),
                "scaffolds_after": after.scaffold_group.nunique(),
                "post_filter_parent_overlap": int(after.parent_id.isin(evaluation_parents).sum()),
                "post_filter_scaffold_overlap": int(after.scaffold_group.isin(evaluation_scaffolds).sum()),
            })

        prediction_rows, check_rows, coverage_rows, loss_rows = [], [], [], []
        human_index = species_map["human"]
        human_stats = stats.loc[stats.task_id.eq(HUMAN_TASK)].iloc[0]
        for route_index, route in enumerate(ROUTES):
            route_seed = args.seed + 1000 * route_index
            seed_everything(route_seed, args.threads)
            route_tasks = [HUMAN_TASK] if route == "human_only_neural_control" else TASKS
            scaler_rows = fitting.task_id.isin(route_tasks).to_numpy()
            x_mean, x_scale = fit_feature_scaler(x_fit_raw[scaler_rows])
            x_fit = scale_features(x_fit_raw, x_mean, x_scale)
            model = SpeciesPrivateNet(
                x_fit.shape[1], len(SPECIES_ORDER), args.encoder_width, args.latent_width
            ).to(args.device)
            encoder_initial = parameter_vector(model, "encoder")
            human_head_initial = parameter_vector(model, f"heads.{human_index}")
            phases = []
            if route == "human_only_neural_control":
                phases = [("human_only", [HUMAN_TASK], args.smoke_epochs)]
            elif route == "cross_species_pretrain_human_finetune":
                phases = [
                    ("animal_pretrain", ANIMAL_TASKS, args.smoke_epochs),
                    ("human_finetune", [HUMAN_TASK], args.smoke_epochs),
                ]
            else:
                phases = [("joint_task_balanced", TASKS, args.smoke_epochs)]
            for phase_index, (phase, phase_tasks, epochs) in enumerate(phases):
                coverage, losses = train_task_balanced(
                    model, x_fit, fitting, phase_tasks, epochs=epochs, batch_size=args.batch_size,
                    learning_rate=3e-4, weight_decay=1e-3, seed=route_seed + phase_index,
                    device=args.device,
                )
                for task, values in coverage.items():
                    coverage_rows.append({"route_id": route, "phase": phase, "task_id": task, **values})
                loss_rows.extend([{"route_id": route, "phase": phase, **row} for row in losses])
            encoder_delta = float(torch.max(torch.abs(parameter_vector(model, "encoder") - encoder_initial)))
            human_head_delta = float(torch.max(torch.abs(parameter_vector(model, f"heads.{human_index}") - human_head_initial)))
            if encoder_delta <= 0 or human_head_delta <= 0:
                raise ValueError(f"Expected encoder and human head updates: {route}")
            checkpoint = {
                "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "n_features": int(x_fit.shape[1]),
                "species_order": SPECIES_ORDER,
                "encoder_width": args.encoder_width,
                "latent_width": args.latent_width,
                "x_mean": x_mean,
                "x_scale": x_scale,
                "route_id": route,
                "seed": route_seed,
            }
            model_path = out / f"{route}.pt"
            torch.save(checkpoint, model_path)
            first = cpu_predict(checkpoint, x_eval_raw, np.full(len(x_eval_raw), human_index))
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
            second = cpu_predict(loaded, x_eval_raw, np.full(len(x_eval_raw), human_index))
            difference = float(np.max(np.abs(first - second)))
            if difference > 1e-7 or not np.isfinite(first).all():
                raise ValueError(f"Reload or finite prediction failure: {route}")
            transformed_prediction = first * float(human_stats.target_std_population) + float(human_stats.target_mean)
            prediction_rows.append(pd.DataFrame({
                "route_id": route,
                "outer_fold": args.outer_fold,
                "parent_id": evaluation_parent.parent_id,
                "scaffold_group": evaluation_parent.scaffold_group,
                "predicted_standardized_target": first,
                "predicted_transformed_target": transformed_prediction,
                "evaluation_target_included": False,
            }))
            check_rows.append({
                "route_id": route,
                "encoder_max_parameter_change": encoder_delta,
                "human_head_max_parameter_change": human_head_delta,
                "finite_predictions": bool(np.isfinite(first).all()),
                "reload_max_abs_difference": difference,
                "evaluation_parents": len(evaluation_parent),
                "evaluation_targets_read": False,
            })

        prediction_frame = pd.concat(prediction_rows, ignore_index=True)
        if prediction_frame.groupby("route_id").parent_id.nunique().ne(len(evaluation_parent)).any():
            raise ValueError("Route prediction coverage differs")
        if prediction_frame.duplicated(["route_id", "parent_id"]).any():
            raise ValueError("Duplicate route-parent prediction")
        pd.DataFrame(isolation_rows).to_csv(out / "fold_isolation_audit.csv", index=False)
        stats.to_csv(out / "fold_local_target_statistics.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(out / "training_coverage.csv", index=False)
        pd.DataFrame(loss_rows).to_csv(out / "training_losses.csv", index=False)
        pd.DataFrame(check_rows).to_csv(out / "route_checks.csv", index=False)
        prediction_frame.to_csv(out / "evaluation_structure_predictions_no_targets.csv", index=False)
        (out / "README.md").write_text(
            "# VDss controlled-transfer engineering smoke\n\n"
            "One outer fold, one seed, bounded parents, and short epochs validate the three frozen routes. "
            "Predictions contain no evaluation labels. This smoke cannot select an architecture.\n",
            encoding="utf-8",
        )
        finish_stage(
            out,
            "vdss_cross_species_transfer_smoke",
            inputs={
                "protocol_complete_sha256": sha256(args.protocol / "complete.json"),
                "interface_complete_sha256": sha256(args.interface / "complete.json"),
                "stl_train_only_complete_sha256": sha256(args.stl / "complete.json"),
            },
            routes=3,
            tasks=5,
            outer_folds=[args.outer_fold],
            seeds=[args.seed],
            max_parents_per_task=args.max_parents_per_task,
            smoke_epochs=args.smoke_epochs,
            feature_set="rdkit2d",
            descriptor_count=len(descriptor_names),
            encoder_width=args.encoder_width,
            latent_width=args.latent_width,
            device=args.device,
            limited_smoke=True,
            evaluation_targets_read=False,
            validation_target_file_opened=False,
            test_labels_read=False,
            model_fitted=True,
            architecture_selection_authorized=False,
            partial=True,
        )
    print(f"VDss transfer smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
