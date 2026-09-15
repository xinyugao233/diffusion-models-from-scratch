"""Frozen CIFAR pixel-nearest-neighbor memorization evaluator."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


@torch.no_grad()
def cifar_pixel_memorization_metrics(
    samples: Tensor, references: Tensor, *, chunk_size: int = 128
) -> dict[str, Any]:
    """Reproduce the established CIFAR-1K d1 < d2/3 evaluator exactly."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if samples.ndim != 4 or references.ndim != 4:
        raise ValueError("samples and references must be NCHW tensors.")
    if references.shape[0] < 2 or samples.shape[1:] != references.shape[1:]:
        raise ValueError("References must contain two shape-compatible images.")
    flat_references = references.flatten(1)
    nearest_distances = []
    second_distances = []
    nearest_indices = []
    for chunk in samples.split(chunk_size):
        distances = torch.cdist(chunk.flatten(1), flat_references)
        values, indices = distances.topk(2, largest=False, dim=1)
        nearest_distances.append(values[:, 0])
        second_distances.append(values[:, 1])
        nearest_indices.append(indices[:, 0])
    d1 = torch.cat(nearest_distances)
    d2 = torch.cat(second_distances)
    neighbors = torch.cat(nearest_indices)
    memorized = d1 < d2 / 3.0
    memorized_neighbors = neighbors[memorized]
    counts = torch.bincount(memorized_neighbors, minlength=references.shape[0])
    return {
        "sample_count": samples.shape[0],
        "memorized_count": int(memorized.sum().item()),
        "memorization_rate": float(memorized.float().mean().item()),
        "unique_training_neighbors_hit": int((counts > 0).sum().item()),
        "unique_training_neighbor_fraction": float((counts > 0).float().mean().item()),
        "maximum_duplicate_count": int(counts.max().item()),
        "mean_d1": float(d1.mean().item()),
        "mean_d2": float(d2.mean().item()),
    }
