#!/usr/bin/env python3
"""Run interface-governed STL and shared-encoder/private-tower PK ablations.

This program has no test input.  A limited run is an implementation smoke test,
not an architecture-selection result.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from dmpk_toolkit import featurize
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


REQUIRED = {"row_id", "smiles", "task_id", "split", "target_value", "endpoint", "evidence_role", "mask",
            "molecule_id", "parent_id", "scaffold_group", "eligible", "evidence_quality_tier", "translation_role"}


class SharedPrivateNet(nn.Module):
    def __init__(self, n_features: int, n_tasks: int, width: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, width), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(width, width), nn.ReLU())
        self.towers = nn.ModuleList([nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
                                     for _ in range(n_tasks)])

    def forward(self, x: torch.Tensor, task_index: torch.Tensor, is_f: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        # Evaluate only the private tower required by each record, rather than
        # making a dense molecule-by-task target matrix with implicit masks.
        output = torch.empty(len(x), dtype=x.dtype, device=x.device)
        for task in task_index.unique().tolist():
            take = task_index.eq(int(task))
            output[take] = self.towers[int(task)](hidden[take]).squeeze(1)
        return torch.where(is_f, torch.sigmoid(output), output)


def load_interface(interface: Path, limit_per_task: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify_stage(interface, "multitask_pk_training_interface")
    records = pd.read_csv(interface / "training_records.csv", dtype=str, keep_default_na=False)
    transforms = pd.read_csv(interface / "task_target_transforms_train_only.csv")
    schedule = pd.read_csv(interface / "tiered_task_loss_schedule.csv")
    if not REQUIRED <= set(records.columns) or transforms.task_id.duplicated().any() or schedule.task_id.duplicated().any():
        raise ValueError("Training interface schema mismatch")
    records["target_value"] = pd.to_numeric(records.target_value, errors="coerce")
    records["mask"] = pd.to_numeric(records["mask"], errors="coerce")
    records = records.merge(transforms, on="task_id", validate="many_to_one", suffixes=("", "_transform"))
    records = records.merge(schedule[["task_id", "tier_loss_weight", "tasks_in_tier", "phase2_human_refinement"]],
                            on="task_id", validate="many_to_one")
    if records.target_value.isna().any() or not np.isfinite(records.target_value).all() or not records["mask"].eq(1).all():
        raise ValueError("Interface records contain invalid labels or masks")
    if set(records.split) - {"train", "val"}:
        raise ValueError("Training interface must not contain test rows")
    if not records.molecule_id.eq(records.parent_id).all() or not records.eligible.astype(str).str.lower().eq("true").all():
        raise ValueError("Training interface contains a non-canonical or ineligible structure")
    for key in ["molecule_id", "parent_id", "scaffold_group"]:
        if (records.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Training interface contains cross-split {key} overlap")
    if limit_per_task:
        records = (records.sort_values("row_id").groupby(["task_id", "split"], group_keys=False)
                   .head(limit_per_task).reset_index(drop=True))
    return records, transforms, schedule


def feature_matrix(records: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    smiles = records.smiles.astype(str).tolist()
    unique = list(dict.fromkeys(smiles))
    x_unique, names = featurize(unique, feature_set=feature_set, progress=False)
    lookup = {value: i for i, value in enumerate(unique)}
    return x_unique[np.asarray([lookup[value] for value in smiles])], np.asarray(unique), names


def training_targets(records: pd.DataFrame) -> np.ndarray:
    value = records.target_value.to_numpy(float)
    is_f = records.endpoint.eq("F").to_numpy()
    mean = records.train_mean.to_numpy(float)
    std = records.train_std_population.to_numpy(float)
    target = np.where(is_f, value, (value - mean) / std)
    if not np.isfinite(target).all():
        raise ValueError("Non-finite interface-standardized target")
    return target.astype(np.float32)


def set_global_seed(seed: int, threads: int = 1) -> None:
    """Seed before model construction and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def target_spec(row: pd.Series) -> dict:
    return {
        "task_id": row.task_id,
        "endpoint": row.endpoint,
        "canonical_unit": row.canonical_unit,
        "target_transform": row.target_transform,
        "reporting_inverse": row.reporting_inverse,
        "standardization": row.standardization,
        "train_mean": float(row.train_mean),
        "train_std_population": float(row.train_std_population),
        "model_output": row.model_output,
    }


def interface_predictions_to_reporting(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undo interface z-scoring and the registered endpoint transform."""
    pred = np.asarray(prediction, dtype=float)
    is_f = frame.endpoint.eq("F").to_numpy()
    transformed = np.where(
        is_f, pred,
        pred * frame.train_std_population.to_numpy(float) + frame.train_mean.to_numpy(float),
    )
    reporting = np.empty(len(frame), dtype=float)
    inverses = frame.reporting_inverse.astype(str).to_numpy()
    for inverse in np.unique(inverses):
        take = inverses == inverse
        if inverse == "identity":
            reporting[take] = transformed[take]
        elif inverse == "power10":
            reporting[take] = np.power(10.0, np.clip(transformed[take], -300.0, 300.0))
        elif inverse == "expit":
            reporting[take] = expit(transformed[take])
        else:
            raise ValueError(f"Unknown reporting inverse: {inverse}")
    if not np.isfinite(transformed).all() or not np.isfinite(reporting).all():
        raise ValueError("Non-finite inverse-transformed prediction")
    return transformed, reporting


def prediction_table(frame: pd.DataFrame, prediction: np.ndarray, model_name: str) -> pd.DataFrame:
    transformed, reporting = interface_predictions_to_reporting(frame, prediction)
    observed_interface = training_targets(frame)
    _, observed_reporting = interface_predictions_to_reporting(frame, observed_interface)
    table = frame[["row_id", "task_id", "split", "endpoint", "canonical_unit"]].copy()
    table["observed_interface_target"] = observed_interface
    table["predicted_interface_target"] = prediction
    table["observed_transformed_target"] = frame.target_value.to_numpy(float)
    table["predicted_transformed_target"] = transformed
    table["observed_reporting_value"] = observed_reporting
    table["predicted_reporting_value"] = reporting
    table["model"] = model_name
    return table


def transformed_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    observed = training_targets(frame)
    error = np.asarray(prediction, float) - observed
    return {"n": int(len(frame)), "rmse_interface_space": float(np.sqrt(np.mean(error ** 2))),
            "mae_interface_space": float(np.mean(np.abs(error)))}


def prediction_metrics(table: pd.DataFrame) -> dict:
    error = table.predicted_interface_target.to_numpy(float) - table.observed_interface_target.to_numpy(float)
    return {"n": int(len(table)), "rmse_interface_space": float(np.sqrt(np.mean(error ** 2))),
            "mae_interface_space": float(np.mean(np.abs(error)))}


def run_stl(records: pd.DataFrame, x: np.ndarray, out: Path, seed: int, sensitivity: pd.DataFrame) -> dict:
    """Ridge STL reference for each authorized human head, no auxiliary labels."""
    result, predictions = {}, []
    selected = records.loc[records.head_selection_authorized & records.evidence_role.eq("strict_A_curated_development")]
    for task_id, group in selected.groupby("task_id", sort=True):
        train = group.split.eq("train").to_numpy()
        val = group.split.eq("val").to_numpy()
        indices = group.index.to_numpy()
        # x is indexed by the original records index, preserved below.
        prep = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
        xx_train = prep.fit_transform(x[indices[train]])
        model = Ridge(alpha=10.0, solver="lsqr", tol=1e-6, random_state=seed).fit(xx_train, training_targets(group.loc[train]))
        val_pred = model.predict(prep.transform(x[indices[val]]))
        result[task_id] = transformed_metrics(group.loc[val], val_pred)
        table = prediction_table(group.loc[val], val_pred, "stl_ridge")
        predictions.append(table)
        folder = out / task_id; folder.mkdir()
        bundle = {"model": model, "preprocessor": prep, "target_spec": target_spec(group.iloc[0]),
                  "feature_count": int(x.shape[1]), "seed": seed}
        joblib.dump(bundle, folder / "model.joblib", compress=3)
        reloaded = joblib.load(folder / "model.joblib")
        reload_pred = reloaded["model"].predict(reloaded["preprocessor"].transform(x[indices[val]]))
        if not np.allclose(val_pred, reload_pred, rtol=1e-7, atol=1e-8):
            raise ValueError(f"STL reload predictions differ for {task_id}")
    table = pd.concat(predictions, ignore_index=True).merge(
        sensitivity[["row_id", "sensitivity_subset"]], on="row_id", how="left", validate="one_to_one"
    )
    table.to_csv(out / "validation_predictions.csv", index=False)
    for task_id in result:
        result[task_id]["source_cluster_sensitivity"] = {
            subset: prediction_metrics(table.loc[(table.task_id.eq(task_id)) & table.sensitivity_subset.eq(subset)])
            for subset in sorted(table.sensitivity_subset.dropna().unique())
            if ((table.task_id.eq(task_id)) & table.sensitivity_subset.eq(subset)).any()
        }
    (out / "reload_check.json").write_text(json.dumps({
        "passed": True, "tasks_checked": sorted(result), "rtol": 1e-7, "atol": 1e-8,
    }, indent=2) + "\n", encoding="utf-8")
    return result


def fit_phase(model: nn.Module, x: np.ndarray, records: pd.DataFrame, phase: str, args, seed: int) -> dict:
    use = records.split.eq("train")
    if phase == "phase2":
        use &= records.phase2_human_refinement
    frame = records.loc[use].copy()
    if frame.empty:
        raise ValueError(f"{phase} has no train records")
    frame["record_index"] = frame.index
    frame["task_index"] = frame.task_id.map({task: i for i, task in enumerate(sorted(records.task_id.unique()))})
    frame["target"] = training_targets(frame)
    task_weights = frame.groupby("task_id").size().rename("n").reset_index()
    if phase == "phase1":
        registered = frame.groupby("task_id").tier_loss_weight.first() / frame.groupby("task_id").tasks_in_tier.first()
        task_weights["weight"] = task_weights.task_id.map(registered)
    else:
        n_tasks = frame.task_id.nunique()
        task_weights["weight"] = 1.0 / n_tasks
    task_weights["weight"] = task_weights.weight / task_weights.weight.sum()
    tasks = task_weights.task_id.tolist()
    probabilities = task_weights.weight.to_numpy(float)
    groups = {task: frame.loc[frame.task_id.eq(task)].record_index.to_numpy(int) for task in tasks}
    rng = np.random.default_rng(seed)
    orders = {task: rng.permutation(indices) for task, indices in groups.items()}
    cursors = {task: 0 for task in tasks}

    def next_batch(task: str) -> np.ndarray:
        order = orders[task]
        start = cursors[task]
        stop = min(start + args.batch_size, len(order))
        batch = order[start:stop]
        if stop == len(order):
            orders[task] = rng.permutation(groups[task])
            cursors[task] = 0
        else:
            cursors[task] = stop
        return batch

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    model.train()
    epochs = args.epochs if phase == "phase1" else args.phase2_epochs
    steps_per_epoch = max(len(tasks), int(np.ceil(len(frame) / args.batch_size)))
    sampled = {task: 0 for task in tasks}
    unique_sampled = {task: set() for task in tasks}
    optimizer_updates = 0
    for _ in range(epochs):
        # Mandatory coverage prevents low-probability tasks from disappearing
        # in short runs; remaining steps retain the registered tier weights.
        chosen_tasks = list(tasks)
        if steps_per_epoch > len(tasks):
            chosen_tasks.extend(rng.choice(tasks, size=steps_per_epoch - len(tasks), p=probabilities).tolist())
        rng.shuffle(chosen_tasks)
        for chosen_value in chosen_tasks:
            chosen = str(chosen_value)
            indices = next_batch(chosen)
            part = frame.loc[indices]
            optimizer.zero_grad(set_to_none=True)
            xx = torch.as_tensor(x[part.record_index.to_numpy()], dtype=torch.float32, device=args.device)
            task = torch.as_tensor(part.task_index.to_numpy(), dtype=torch.long, device=args.device)
            fmask = torch.as_tensor(part.endpoint.eq("F").to_numpy(), dtype=torch.bool, device=args.device)
            yy = torch.as_tensor(part.target.to_numpy(), dtype=torch.float32, device=args.device)
            loss = torch.nn.functional.huber_loss(model(xx, task, fmask), yy, reduction="mean", delta=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_updates += 1
            sampled[chosen] += len(part)
            unique_sampled[chosen].update(map(int, indices))
    return {
        "phase": phase, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
        "optimizer_updates": optimizer_updates, "sampling_seed": seed,
        "task_sampling_probability": dict(zip(tasks, probabilities.tolist())),
        "sampled_records_with_replacement": sampled,
        "unique_records_sampled": {task: len(values) for task, values in unique_sampled.items()},
        "all_tasks_sampled_each_epoch": True,
    }


def run_shared_private(records: pd.DataFrame, x: np.ndarray, out: Path, args) -> dict:
    task_order = sorted(records.task_id.unique())
    prep = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    train = records.split.eq("train").to_numpy()
    x_scaled = np.empty_like(x, dtype=np.float32)
    x_scaled[train] = prep.fit_transform(x[train])
    x_scaled[~train] = prep.transform(x[~train])
    set_global_seed(args.seed, args.threads)
    model = SharedPrivateNet(x.shape[1], len(task_order), args.width, args.dropout).to(args.device)
    phase_stats = [fit_phase(model, x_scaled, records, "phase1", args, args.seed),
                   fit_phase(model, x_scaled, records, "phase2", args, args.seed + 1)]
    val = records.loc[records.split.eq("val") & records.head_selection_authorized].copy()
    val["task_index"] = val.task_id.map({task: i for i, task in enumerate(task_order)})
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(x_scaled[val.index], dtype=torch.float32, device=args.device),
                           torch.as_tensor(val.task_index.to_numpy(), dtype=torch.long, device=args.device),
                           torch.as_tensor(val.endpoint.eq("F").to_numpy(), dtype=torch.bool, device=args.device)).cpu().numpy()
    table = prediction_table(val, prediction, "shared_encoder_private_towers")
    metrics = {task: transformed_metrics(val.loc[val.task_id.eq(task)], prediction[val.task_id.eq(task).to_numpy()])
               for task in sorted(val.task_id.unique())}
    sensitivity = pd.read_csv(args.interface / "source_cluster_sensitivity_validation_manifest.csv")
    table = table.merge(sensitivity[["row_id", "sensitivity_subset"]], on="row_id", how="left", validate="one_to_one")
    table.to_csv(out / "validation_predictions.csv", index=False)
    sensitivity_metrics = {}
    for task in sorted(val.task_id.unique()):
        for subset in sorted(table.sensitivity_subset.dropna().unique()):
            take = (val.task_id.to_numpy() == task) & (table.sensitivity_subset.to_numpy() == subset)
            if take.any():
                sensitivity_metrics[f"{task}|{subset}"] = transformed_metrics(val.iloc[np.flatnonzero(take)], prediction[take])
    specs = {task: target_spec(group.iloc[0]) for task, group in records.groupby("task_id", sort=True)}
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "task_order": task_order, "n_features": int(x.shape[1]), "width": args.width,
                "dropout": args.dropout, "target_specs": specs, "seed": args.seed}, out / "model.pt")
    joblib.dump(prep, out / "feature_preprocessor.joblib", compress=3)
    checkpoint = torch.load(out / "model.pt", map_location=args.device, weights_only=False)
    reload_model = SharedPrivateNet(checkpoint["n_features"], len(checkpoint["task_order"]),
                                    checkpoint["width"], checkpoint["dropout"]).to(args.device)
    reload_model.load_state_dict(checkpoint["state_dict"])
    reload_model.eval()
    reload_prep = joblib.load(out / "feature_preprocessor.joblib")
    reload_x = reload_prep.transform(x[val.index]).astype(np.float32)
    with torch.no_grad():
        reload_prediction = reload_model(
            torch.as_tensor(reload_x, dtype=torch.float32, device=args.device),
            torch.as_tensor(val.task_index.to_numpy(), dtype=torch.long, device=args.device),
            torch.as_tensor(val.endpoint.eq("F").to_numpy(), dtype=torch.bool, device=args.device),
        ).cpu().numpy()
    if not np.allclose(prediction, reload_prediction, rtol=1e-6, atol=1e-7):
        raise ValueError("Shared/private reload predictions differ")
    (out / "reload_check.json").write_text(json.dumps({
        "passed": True, "n_predictions": len(prediction), "rtol": 1e-6, "atol": 1e-7,
        "max_abs_difference": float(np.max(np.abs(prediction - reload_prediction))),
    }, indent=2) + "\n", encoding="utf-8")
    (out / "optimizer_updates.json").write_text(json.dumps(phase_stats, indent=2) + "\n", encoding="utf-8")
    return {"validation": metrics, "source_cluster_sensitivity": sensitivity_metrics,
            "training_phases": phase_stats, "reload_check_passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, default=ROOT / "data/public_development/multitask_pk_training_interface_v5")
    parser.add_argument("--output", type=Path, default=ROOT / "models/public_multitask_pk/ablation_v2")
    parser.add_argument("--modes", default="stl,shared_private")
    parser.add_argument("--feature-set", default="ecfp4_rdkit2d", choices=["ecfp4_rdkit2d", "ecfp4", "rdkit2d"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--limit-per-task", type=int, help="Smoke-test only; prevents architecture selection")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    modes = list(dict.fromkeys(args.modes.split(",")))
    if not modes or not set(modes) <= {"stl", "shared_private"} or min(args.epochs, args.phase2_epochs, args.width, args.batch_size, args.threads) < 1:
        raise ValueError("Invalid mode or positive training parameter")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records, transforms, schedule = load_interface(args.interface, args.limit_per_task)
    if args.check_only:
        print(f"PK ablation input valid: records={len(records)} tasks={records.task_id.nunique()} modes={','.join(modes)}")
        return
    startup_self_check([args.interface / "complete.json"], output=args.output)
    with stage_output(args.output) as out:
        x, _, names = feature_matrix(records, args.feature_set)
        sensitivity = pd.read_csv(args.interface / "source_cluster_sensitivity_validation_manifest.csv")
        outcomes = {}
        if "stl" in modes:
            folder = out / "stl_ridge"; folder.mkdir(); outcomes["stl_ridge"] = run_stl(records, x, folder, args.seed, sensitivity)
        if "shared_private" in modes:
            folder = out / "shared_private"; folder.mkdir(); outcomes["shared_private"] = run_shared_private(records, x, folder, args)
        (out / "metrics.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "feature_schema.json").write_text(json.dumps({"feature_set": args.feature_set, "descriptor_names": names}, ensure_ascii=False) + "\n", encoding="utf-8")
        (out / "README.md").write_text("# Interface-governed PK architecture ablations\n\n"
                                        "STL and shared/private-tower models use the same interface and do not load test labels. "
                                        "This remains an engineering reference until the strong single-task benchmark gate is complete; "
                                        "a full run is not automatically authorized for architecture selection.\n", encoding="utf-8")
        finish_stage(out, "public_multitask_pk_architecture_ablation", inputs={"interface_complete_sha256": sha256(args.interface / "complete.json")},
                     modes=modes, feature_set=args.feature_set, seed=args.seed, limited_smoke=bool(args.limit_per_task),
                     optimizer_step_policy="one_step_per_task-balanced_sampled_batch",
                     reload_prediction_check=True, target_inverse_parameters_saved=True,
                     architecture_selection_authorized=False,
                     model_selection_block_reason="strong_STL_and_transfer_gates_not_yet_completed",
                     test_evaluated=False, partial=bool(args.limit_per_task))
    print(f"PK architecture ablation: {args.output}")


if __name__ == "__main__":
    run_cli(main)
