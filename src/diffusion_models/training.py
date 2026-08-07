"""Explicit epsilon-prediction training utilities for DDPM experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diffusion_models.diffusion.ddpm import q_sample
from diffusion_models.diffusion.schedules import DDPMSchedule


@dataclass(frozen=True)
class EpsilonTrainingBatch:
    """One noised input and the exact Gaussian-noise target that created it."""

    x_t: Tensor
    timesteps: Tensor
    target_noise: Tensor


@dataclass(frozen=True)
class OptimizationMetrics:
    """Scalar measurements from one optimizer update."""

    loss: float
    gradient_norm: float
    clipped_gradient_norm: float


def make_epsilon_training_batch(
    x_start: Tensor,
    timesteps: Tensor,
    schedule: DDPMSchedule,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> EpsilonTrainingBatch:
    """Construct ``x_t`` and retain the exact noise as its prediction target."""
    if noise is not None and generator is not None:
        raise ValueError("Provide either fixed noise or a generator, not both.")
    if noise is None:
        noise = torch.randn(
            x_start.shape,
            dtype=x_start.dtype,
            device=x_start.device,
            generator=generator,
        )
    x_t = q_sample(
        x_start=x_start,
        timesteps=timesteps,
        schedule=schedule,
        noise=noise,
    )
    return EpsilonTrainingBatch(
        x_t=x_t,
        timesteps=timesteps,
        target_noise=noise,
    )


def sample_epsilon_training_batch(
    x_start: Tensor,
    schedule: DDPMSchedule,
    *,
    generator: torch.Generator | None = None,
) -> EpsilonTrainingBatch:
    """Sample one timestep and one fresh Gaussian-noise target per image."""
    if x_start.ndim != 4:
        raise ValueError(
            f"Expected x_start shape [B, C, H, W], got {tuple(x_start.shape)}."
        )
    timesteps = torch.randint(
        0,
        schedule.num_steps,
        (x_start.shape[0],),
        device=x_start.device,
        generator=generator,
    )
    noise = torch.randn(
        x_start.shape,
        dtype=x_start.dtype,
        device=x_start.device,
        generator=generator,
    )
    return make_epsilon_training_batch(
        x_start,
        timesteps,
        schedule,
        noise=noise,
    )


def epsilon_prediction_loss(
    model: nn.Module,
    batch: EpsilonTrainingBatch,
) -> tuple[Tensor, Tensor]:
    """Return scalar MSE and predicted noise for a prepared training batch."""
    if batch.x_t.ndim != 4:
        raise ValueError(
            f"Expected x_t shape [B, C, H, W], got {tuple(batch.x_t.shape)}."
        )
    if batch.timesteps.ndim != 1:
        raise ValueError(
            f"timesteps must have shape [B], got {tuple(batch.timesteps.shape)}."
        )
    if batch.timesteps.shape[0] != batch.x_t.shape[0]:
        raise ValueError("x_t and timesteps must have the same batch size.")
    if batch.target_noise.shape != batch.x_t.shape:
        raise ValueError(
            "target_noise must have the same shape as x_t: "
            f"{tuple(batch.target_noise.shape)} versus {tuple(batch.x_t.shape)}."
        )
    if not (batch.x_t.device == batch.timesteps.device == batch.target_noise.device):
        raise ValueError("x_t, timesteps, and target_noise must share a device.")

    predicted_noise = model(batch.x_t, batch.timesteps)
    if predicted_noise.shape != batch.target_noise.shape:
        raise ValueError(
            "predicted noise and target noise shapes differ: "
            f"{tuple(predicted_noise.shape)} versus "
            f"{tuple(batch.target_noise.shape)}."
        )
    loss = F.mse_loss(predicted_noise, batch.target_noise)
    if loss.ndim != 0:
        raise RuntimeError(f"Expected scalar loss, got shape {tuple(loss.shape)}.")
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite epsilon-prediction loss detected.")
    return loss, predicted_noise


def gradient_norm(parameters: list[nn.Parameter]) -> Tensor:
    """Calculate the global L2 norm and reject absent or nonfinite gradients."""
    gradients = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("No gradients were produced for trainable parameters.")
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Nonfinite parameter gradient detected.")
    squared_norm = torch.stack(
        [torch.sum(gradient.to(dtype=torch.float64).square()) for gradient in gradients]
    ).sum()
    return torch.sqrt(squared_norm)


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: EpsilonTrainingBatch,
    *,
    max_gradient_norm: float | None = None,
) -> OptimizationMetrics:
    """Run one checked epsilon-prediction update and return scalar metrics."""
    if max_gradient_norm is not None and max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive or None.")

    optimizer.zero_grad(set_to_none=True)
    loss, _ = epsilon_prediction_loss(model, batch)
    loss.backward()

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    before_clipping = gradient_norm(parameters)
    if max_gradient_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    after_clipping = gradient_norm(parameters)
    optimizer.step()

    return OptimizationMetrics(
        loss=float(loss.detach().item()),
        gradient_norm=float(before_clipping.item()),
        clipped_gradient_norm=float(after_clipping.item()),
    )


class JsonlScalarLogger:
    """Append JSON scalar records while refusing to overwrite prior evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite existing log: {self.path}")

    def log(self, step: int, **scalars: float) -> None:
        if step < 0:
            raise ValueError("step must be nonnegative.")
        for name, value in scalars.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"Cannot log nonfinite scalar {name}={value}.")
        record: dict[str, Any] = {"step": step, **scalars}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
