"""Noise schedules and coefficient extraction for discrete DDPMs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DDPMSchedule:
    """Precomputed coefficients for a discrete DDPM noise schedule."""

    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor
    sqrt_alpha_bars: Tensor
    sqrt_one_minus_alpha_bars: Tensor

    @property
    def num_steps(self) -> int:
        """Return the number of discrete noising steps."""
        return self.betas.shape[0]


def linear_beta_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Construct linearly increasing beta values with shape ``[num_steps]``."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError(
            "Expected 0 < beta_start < beta_end < 1, "
            f"but received {beta_start=} and {beta_end=}."
        )
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("The schedule dtype must be floating point.")

    return torch.linspace(
        beta_start,
        beta_end,
        num_steps,
        device=device,
        dtype=dtype,
    )


def build_ddpm_schedule(betas: Tensor) -> DDPMSchedule:
    """Precompute forward-process coefficients from a one-dimensional beta tensor."""
    if betas.ndim != 1:
        raise ValueError(
            f"betas must have shape [T], but received {tuple(betas.shape)}."
        )
    if betas.numel() == 0:
        raise ValueError("betas cannot be empty.")
    if not betas.is_floating_point():
        raise TypeError("betas must be a floating-point tensor.")
    if not torch.isfinite(betas).all():
        raise ValueError("betas contains nonfinite values.")
    if torch.any(betas <= 0.0) or torch.any(betas >= 1.0):
        raise ValueError("Every beta must lie strictly between 0 and 1.")

    # Schedules are fixed constants, not trainable parameters.
    betas = betas.detach().clone()
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(torch.clamp(1.0 - alpha_bars, min=0.0))

    return DDPMSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )


def make_linear_ddpm_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> DDPMSchedule:
    """Construct the complete linear DDPM schedule."""
    return build_ddpm_schedule(
        linear_beta_schedule(
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            dtype=dtype,
        )
    )


def extract(
    values: Tensor,
    timesteps: Tensor,
    target_shape: Sequence[int],
) -> Tensor:
    """Select one schedule value per batch item and reshape for broadcasting."""
    if values.ndim != 1:
        raise ValueError(
            f"values must have shape [T], but received {tuple(values.shape)}."
        )
    if timesteps.ndim != 1:
        raise ValueError(
            f"timesteps must have shape [B], but received {tuple(timesteps.shape)}."
        )
    if timesteps.dtype != torch.long:
        raise TypeError(
            f"timesteps must use torch.long, but received {timesteps.dtype}."
        )
    if len(target_shape) < 1:
        raise ValueError("target_shape must contain a batch dimension.")
    if target_shape[0] != timesteps.shape[0]:
        raise ValueError(
            "Batch-size mismatch: "
            f"target_shape[0]={target_shape[0]}, "
            f"timesteps.shape[0]={timesteps.shape[0]}."
        )
    if values.device != timesteps.device:
        raise ValueError(
            "values and timesteps must be on the same device: "
            f"values.device={values.device} and "
            f"timesteps.device={timesteps.device}."
        )

    if timesteps.numel() > 0:
        minimum_t = int(timesteps.min().item())
        maximum_t = int(timesteps.max().item())
        if minimum_t < 0 or maximum_t >= values.shape[0]:
            raise IndexError(
                f"Timesteps must be in [0, {values.shape[0] - 1}], "
                f"but received range [{minimum_t}, {maximum_t}]."
            )

    selected = values.gather(dim=0, index=timesteps)
    broadcast_shape = (timesteps.shape[0], *([1] * (len(target_shape) - 1)))
    return selected.reshape(broadcast_shape)
