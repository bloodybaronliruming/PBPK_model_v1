"""Deterministic, fold-contained PyTorch helpers for Gate 1B diagnostics.

This module deliberately keeps target fitting, feature scaling, and neural
early stopping inside an outer training fold.  It is not a general model
selection framework: callers must provide an already fixed outer split.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Set local pseudo-random sources; no global data-dependent state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def inner_scaffold_masks(scaffold_groups, seed: int, fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Choose a deterministic whole-scaffold internal validation partition.

    Every internal-validation scaffold is absent from the inner training set.
    The caller has already removed the outer evaluation fold, hence this never
    touches outer evaluation labels or structures.
    """
    groups = np.asarray(scaffold_groups, dtype=str)
    if len(groups) < 8 or not 0.05 <= fraction <= 0.35:
        raise ValueError("Invalid internal scaffold split request")
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < 2:
        raise ValueError("At least two scaffold groups are required for internal early stopping")
    target = max(2, int(round(len(groups) * fraction)))
    ordered = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: _stable_rank(item[0], seed))
    selected, selected_count = [], 0
    for group, count in ordered:
        # Keep at least two parents outside the early-stopping partition.
        if len(groups) - (selected_count + count) < 2:
            continue
        selected.append(group)
        selected_count += count
        if selected_count >= target:
            break
    if not selected:
        # The largest scaffold can be selected only when another group remains.
        selected = [ordered[0][0]]
    valid_mask = np.isin(groups, selected)
    train_mask = ~valid_mask
    if not valid_mask.any() or train_mask.sum() < 2:
        raise ValueError("Internal scaffold split is degenerate")
    if set(groups[valid_mask]) & set(groups[train_mask]):
        raise AssertionError("Internal scaffold groups overlap")
    return train_mask, valid_mask


def fit_feature_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # RDKit descriptors can contain very large but finite values.  Compute
    # moments in float64; float32 variance can overflow on small smoke folds.
    values = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    filled = np.where(np.isfinite(values), values, mean)
    scale = np.std(filled, axis=0, ddof=0)
    scale[(~np.isfinite(scale)) | (scale < 1e-7)] = 1.0
    return mean, scale


def scale_features(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (np.where(np.isfinite(x), x, mean).astype(np.float64) - mean) / scale
    # Fixed robust bound prevents a rare descriptor from dominating gradients;
    # it is fit-free and applied identically to inner/train/evaluation inputs.
    return np.clip(np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0).astype(np.float32)


class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.10,
                 bounded_output: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.GELU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.body = nn.Sequential(*layers)
        self.bounded_output = bounded_output

    def forward(self, x):
        result = self.body(x).squeeze(-1)
        return torch.sigmoid(result) if self.bounded_output else result


def _loss(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=0.05)
    raise ValueError(f"Unknown loss: {name}")


def fit_mlp(
    x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
    *, seed: int, bounded_output: bool, loss_name: str, monitor_name: str,
    hidden: tuple[int, ...] = (128, 64), dropout: float = 0.10, learning_rate: float = 3e-4,
    weight_decay: float = 1e-3, max_epochs: int = 250, patience: int = 25,
) -> tuple[dict, list[dict]]:
    """Fit one reproducible MLP and return a reloadable CPU checkpoint plus curves."""
    if len(x_train) < 2 or len(x_valid) < 1:
        raise ValueError("Neural fit needs nonempty inner train and validation sets")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_mean, x_scale = fit_feature_scaler(x_train)
    train_x = torch.as_tensor(scale_features(x_train, x_mean, x_scale), device=device)
    valid_x = torch.as_tensor(scale_features(x_valid, x_mean, x_scale), device=device)
    train_y = torch.as_tensor(np.asarray(y_train, dtype=np.float32), device=device)
    valid_y = torch.as_tensor(np.asarray(y_valid, dtype=np.float32), device=device)
    model = RegressionMLP(train_x.shape[1], hidden=hidden, dropout=dropout, bounded_output=bounded_output).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = _loss(loss_name)
    batch_size = min(128, len(train_x))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state, best_score, best_epoch, stale = None, float("inf"), 0, 0
    curves: list[dict] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator, device="cpu").to(device)
        batch_losses = []
        for ids in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[ids])
            loss = criterion(prediction, train_y[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_prediction = model(valid_x)
            if monitor_name == "mae":
                score = float(torch.mean(torch.abs(valid_prediction - valid_y)).cpu())
            elif monitor_name == "rmse":
                score = float(torch.sqrt(torch.mean((valid_prediction - valid_y) ** 2)).cpu())
            else:
                raise ValueError(f"Unknown monitor: {monitor_name}")
        curves.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "internal_validation_metric": score})
        if score < best_score - 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No neural checkpoint was selected")
    checkpoint = {
        "state_dict": best_state, "input_dim": int(train_x.shape[1]), "hidden": tuple(hidden),
        "dropout": float(dropout), "bounded_output": bool(bounded_output), "x_mean": x_mean,
        "x_scale": x_scale, "best_epoch": best_epoch, "best_internal_metric": best_score,
        "epochs_ran": len(curves), "device_used": str(device),
    }
    return checkpoint, curves


def fit_mlp_fixed_epochs(
    x_train: np.ndarray, y_train: np.ndarray, *, seed: int, bounded_output: bool, loss_name: str,
    epochs: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.10,
    learning_rate: float = 3e-4, weight_decay: float = 1e-3,
) -> dict:
    """Refit on the complete outer-training fold for a pre-selected epoch count.

    `epochs` must come from a scaffold-contained internal early-stopping fit.
    This function deliberately has no access to any validation/evaluation data.
    """
    if len(x_train) < 2 or epochs < 1:
        raise ValueError("Final neural refit needs at least two rows and one epoch")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_mean, x_scale = fit_feature_scaler(x_train)
    train_x = torch.as_tensor(scale_features(x_train, x_mean, x_scale), device=device)
    train_y = torch.as_tensor(np.asarray(y_train, dtype=np.float32), device=device)
    model = RegressionMLP(train_x.shape[1], hidden=hidden, dropout=dropout, bounded_output=bounded_output).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = _loss(loss_name)
    batch_size = min(128, len(train_x))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train_x), generator=generator, device="cpu").to(device)
        for ids in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[ids]), train_y[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return {
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "input_dim": int(train_x.shape[1]), "hidden": tuple(hidden), "dropout": float(dropout),
        "bounded_output": bool(bounded_output), "x_mean": x_mean, "x_scale": x_scale,
        "final_refit_epochs": int(epochs), "device_used": str(device),
    }


def predict_checkpoint(checkpoint: dict, x: np.ndarray) -> np.ndarray:
    model = RegressionMLP(checkpoint["input_dim"], hidden=tuple(checkpoint["hidden"]),
                          dropout=float(checkpoint["dropout"]), bounded_output=bool(checkpoint["bounded_output"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaled = scale_features(x, checkpoint["x_mean"], checkpoint["x_scale"])
    with torch.no_grad():
        return model(torch.as_tensor(scaled)).detach().cpu().numpy().astype(float)
