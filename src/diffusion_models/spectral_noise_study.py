"""Shared configuration and evaluation helpers for the Study 2 experiment."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from diffusion_models.spectral_boundary import radial_shell_indices


def resolve_spectral_noise_study_configuration(
    config: Mapping[str, Any], *, condition: str, seed: int
) -> dict[str, Any]:
    """Resolve one condition and paired seed from the frozen study config."""
    resolved = copy.deepcopy(dict(config))
    conditions = resolved.pop("study_conditions")
    paired_seeds = resolved.pop("paired_seeds")
    if condition not in conditions:
        raise ValueError(f"Unknown Study 2 condition {condition!r}.")
    matches = [entry for entry in paired_seeds if int(entry["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one paired seed record for seed {seed}.")
    seed_record = matches[0]
    resolved["model"]["initialization_seed"] = int(
        seed_record["initialization_seed"]
    )
    resolved["data"]["data_order_seed"] = int(seed_record["data_order_seed"])
    resolved["training"]["training_noise_seed"] = int(
        seed_record["training_noise_seed"]
    )
    resolved["experiment"]["condition"] = condition
    resolved["experiment"]["seed"] = seed
    resolved["spectral_noise_condition"] = copy.deepcopy(conditions[condition])
    resolved["resolved_study"] = {
        "condition": condition,
        "seed": seed,
        "rng_seeds": copy.deepcopy(seed_record),
    }
    return resolved


def state_sha256(model: nn.Module) -> str:
    """Hash a model state deterministically for paired-initialization checks."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def full_fft_shell_mse(residual: Tensor, *, num_shells: int) -> Tensor:
    """Return per-example, per-shell epsilon-residual power on the full FFT plane."""
    if residual.ndim != 4 or not residual.is_floating_point():
        raise ValueError("residual must be a floating-point NCHW tensor.")
    if num_shells <= 0:
        raise ValueError("num_shells must be positive.")
    shells = radial_shell_indices(
        residual.shape[-2], residual.shape[-1], device=residual.device
    )
    if int(shells.max()) >= num_shells:
        raise ValueError("num_shells does not cover the residual FFT grid.")
    power = torch.fft.fft2(residual, norm="ortho").abs().square().mean(dim=1)
    output = residual.new_zeros((residual.shape[0], num_shells))
    indices = shells.flatten()[None, :].expand(residual.shape[0], -1)
    output.scatter_add_(1, indices, power.flatten(start_dim=1))
    counts = torch.bincount(shells.flatten(), minlength=num_shells).to(residual.dtype)
    return output / counts[None, :]
