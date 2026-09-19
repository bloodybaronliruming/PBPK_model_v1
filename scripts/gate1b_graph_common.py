"""Shared, serializable graph+RDKit2D adapter for Gate 1B."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from neural_models import TorchRegressor


FORMAL_CANDIDATES = [
    {"width": 128, "depth": 3, "dropout": 0.15, "learning_rate": 1e-3,
     "epochs": 80, "batch_size": 64},
    {"width": 192, "depth": 4, "dropout": 0.20, "learning_rate": 3e-4,
     "epochs": 100, "batch_size": 64},
]
SMOKE_CANDIDATES = [
    {"width": 32, "depth": 2, "dropout": 0.10, "learning_rate": 1e-3,
     "epochs": 2, "batch_size": 16},
]


class GraphRDKitRegressor:
    """Fold-local RDKit2D preprocessing plus the project D-MPNN encoder."""

    def __init__(self, *, width: int, depth: int, dropout: float, learning_rate: float,
                 epochs: int, batch_size: int, seed: int, device: str, threads: int):
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.threads = threads

    def fit(self, x, y, smiles, sample_weight=None):
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Gate 1B D-MPNN but is unavailable; CPU fallback is prohibited")
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        transformed = self.scaler_.fit_transform(self.imputer_.fit_transform(x)).astype(np.float32)
        self.regressor_ = TorchRegressor(
            kind="dmpnn", width=self.width, depth=self.depth, dropout=self.dropout,
            learning_rate=self.learning_rate, epochs=self.epochs, batch_size=self.batch_size,
            seed=self.seed, device=self.device, threads=self.threads,
        )
        self.regressor_.fit(transformed, y, np.asarray(smiles, dtype=str), weights)
        self.device_used_ = self.device
        self.cuda_device_name_ = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        return self

    def predict(self, x, smiles):
        transformed = self.scaler_.transform(self.imputer_.transform(np.asarray(x, dtype=np.float32))).astype(np.float32)
        return self.regressor_.predict(transformed, np.asarray(smiles, dtype=str))


def candidates(profile: str) -> list[dict]:
    if profile == "smoke":
        return [dict(x) for x in SMOKE_CANDIDATES]
    if profile == "stage1":
        return [dict(x) for x in FORMAL_CANDIDATES]
    raise ValueError(f"Unknown D-MPNN profile: {profile}")
