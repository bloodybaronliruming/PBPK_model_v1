"""Shared components for the capacity-matched two-group multitask encoder."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


FU_GROUP = "fu_family"
NON_FU_GROUP = "non_fu_family"
GROUPS = [FU_GROUP, NON_FU_GROUP]
ROUTE_ID = "two_group_shared_encoder_private_heads"


def encoder_block(n_features: int, width: int, latent: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_features, width), nn.GELU(), nn.LayerNorm(width),
        nn.Linear(width, latent), nn.GELU(), nn.LayerNorm(latent),
    )


class TwoGroupSharedPrivateNet(nn.Module):
    """Use one encoder for fu tasks and another for all non-fu tasks."""

    def __init__(
        self, n_features: int, n_tasks: int, width: int, latent: int,
        bounded: list[bool], task_groups: list[str],
    ):
        super().__init__()
        if len(task_groups) != n_tasks or set(task_groups) != set(GROUPS):
            raise ValueError("Two-group model requires complete fu/non-fu task routing")
        self.encoders = nn.ModuleDict({group: encoder_block(n_features, width, latent) for group in GROUPS})
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(n_tasks)
        ])
        self.bounded = list(map(bool, bounded))
        self.task_groups = list(map(str, task_groups))

    def forward(self, x: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        output = torch.empty(len(x), dtype=x.dtype, device=x.device)
        for task in task_index.unique().tolist():
            task = int(task)
            take = task_index.eq(task)
            hidden = self.encoders[self.task_groups[task]](x[take])
            value = self.heads[task](hidden).squeeze(1)
            output[take] = torch.sigmoid(value) if self.bounded[task] else value
        return output


def cpu_predict(checkpoint: dict, x: np.ndarray, task_indices: np.ndarray) -> np.ndarray:
    model = TwoGroupSharedPrivateNet(
        checkpoint["n_features"], len(checkpoint["task_order"]), checkpoint["width"],
        checkpoint["latent"], checkpoint["bounded"], checkpoint["task_groups"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(task_indices, dtype=torch.long),
        ).numpy()


def encoder_parameter_count(n_features: int, width: int, latent: int) -> int:
    """Linear layers plus affine LayerNorm parameters for one encoder."""
    return n_features * width + width + 2 * width + width * latent + latent + 2 * latent


def private_head_parameter_count(latent: int, tasks: int) -> int:
    return tasks * (latent * 32 + 32 + 32 + 1)
