"""DDPM posterior calculations and ancestral reverse sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from diffusion_models.diffusion.ddpm import (
    _validate_image_and_timesteps,
    _validate_noise,
    predict_x_start_from_noise,
)
from diffusion_models.diffusion.schedules import DDPMSchedule, extract


@dataclass(frozen=True)
class PosteriorOutput:
    """Mean and scalar-per-example variance of the exact DDPM posterior."""

    mean: Tensor
    variance: Tensor


@dataclass(frozen=True)
class ReversePrediction:
    """Model prediction converted to a DDPM reverse Gaussian."""

    mean: Tensor
    variance: Tensor
    predicted_x_start: Tensor
    predicted_noise: Tensor


@dataclass(frozen=True)
class SamplingResult:
    """Final reverse-chain sample and explicitly requested intermediate states."""

    sample: Tensor
    trajectory: dict[int, Tensor]


def _posterior_coefficients(
    schedule: DDPMSchedule,
) -> tuple[Tensor, Tensor, Tensor]:
    alpha_bar_previous = torch.cat(
        (torch.ones_like(schedule.alpha_bars[:1]), schedule.alpha_bars[:-1])
    )
    denominator = 1.0 - schedule.alpha_bars
    variance = schedule.betas * (1.0 - alpha_bar_previous) / denominator
    x_start_coefficient = schedule.betas * torch.sqrt(alpha_bar_previous) / denominator
    x_t_coefficient = (
        torch.sqrt(schedule.alphas) * (1.0 - alpha_bar_previous) / denominator
    )
    return variance.clamp(min=0.0), x_start_coefficient, x_t_coefficient


def q_posterior_mean_variance(
    x_start: Tensor,
    x_t: Tensor,
    timesteps: Tensor,
    schedule: DDPMSchedule,
) -> PosteriorOutput:
    """Return ``q(x_{t-1} | x_t, x_start)`` posterior parameters."""
    _validate_image_and_timesteps(x_start, timesteps, schedule)
    _validate_image_and_timesteps(x_t, timesteps, schedule)
    if x_start.shape != x_t.shape:
        raise ValueError(
            "x_start and x_t must have the same shape: "
            f"{tuple(x_start.shape)} versus {tuple(x_t.shape)}."
        )
    if x_start.dtype != x_t.dtype:
        raise TypeError("x_start and x_t must use the same dtype.")

    variance, x_start_coefficient, x_t_coefficient = _posterior_coefficients(schedule)
    variance_t = extract(variance, timesteps, x_t.shape).to(dtype=x_t.dtype)
    x_start_coefficient_t = extract(x_start_coefficient, timesteps, x_t.shape).to(
        dtype=x_t.dtype
    )
    x_t_coefficient_t = extract(x_t_coefficient, timesteps, x_t.shape).to(
        dtype=x_t.dtype
    )
    mean = x_start_coefficient_t * x_start + x_t_coefficient_t * x_t
    return PosteriorOutput(mean=mean, variance=variance_t)


def p_mean_variance(
    model: nn.Module,
    x_t: Tensor,
    timesteps: Tensor,
    schedule: DDPMSchedule,
    *,
    clip_x_start: bool = True,
) -> ReversePrediction:
    """Convert an epsilon prediction into a DDPM reverse Gaussian."""
    _validate_image_and_timesteps(x_t, timesteps, schedule)
    predicted_noise = model(x_t, timesteps)
    _validate_noise(predicted_noise, x_t)
    predicted_x_start = predict_x_start_from_noise(
        x_t,
        timesteps,
        predicted_noise,
        schedule,
    )
    if clip_x_start:
        predicted_x_start = predicted_x_start.clamp(-1.0, 1.0)
    posterior = q_posterior_mean_variance(
        predicted_x_start,
        x_t,
        timesteps,
        schedule,
    )
    return ReversePrediction(
        mean=posterior.mean,
        variance=posterior.variance,
        predicted_x_start=predicted_x_start,
        predicted_noise=predicted_noise,
    )


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x_t: Tensor,
    timesteps: Tensor,
    schedule: DDPMSchedule,
    *,
    reverse_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
    clip_x_start: bool = True,
) -> Tensor:
    """Draw one ancestral DDPM reverse step.

    Supplying ``reverse_noise`` makes the stochastic part explicit. Code
    timestep zero always returns the posterior mean without adding noise.
    """
    if reverse_noise is not None and generator is not None:
        raise ValueError("Provide reverse_noise or generator, not both.")
    if reverse_noise is not None:
        _validate_noise(reverse_noise, x_t)

    prediction = p_mean_variance(
        model,
        x_t,
        timesteps,
        schedule,
        clip_x_start=clip_x_start,
    )
    if torch.all(timesteps == 0):
        return prediction.mean

    if reverse_noise is None:
        reverse_noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
    nonzero_mask = (timesteps != 0).reshape(timesteps.shape[0], *([1] * (x_t.ndim - 1)))
    return (
        prediction.mean
        + nonzero_mask.to(dtype=x_t.dtype)
        * torch.sqrt(prediction.variance)
        * reverse_noise
    )


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    schedule: DDPMSchedule,
    shape: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    initial_noise: Tensor | None = None,
    clip_x_start: bool = True,
    capture_timesteps: Sequence[int] = (),
) -> SamplingResult:
    """Run all ``T`` ancestral steps from Gaussian noise to a model sample."""
    shape = tuple(shape)
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"shape must be positive NCHW dimensions, got {shape}.")
    device = schedule.betas.device if device is None else torch.device(device)
    dtype = schedule.betas.dtype if dtype is None else dtype
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("Sampling dtype must be floating point.")
    if device != schedule.betas.device:
        raise ValueError("The sampling device and schedule device must match.")

    if initial_noise is None:
        sample = torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError(
                "initial_noise and shape differ: "
                f"{tuple(initial_noise.shape)} versus {shape}."
            )
        if initial_noise.device != device:
            raise ValueError("initial_noise and sampling device must match.")
        if initial_noise.dtype != dtype:
            raise TypeError("initial_noise and sampling dtype must match.")
        sample = initial_noise.detach().clone()

    captures = set(capture_timesteps)
    if any(
        not isinstance(timestep, int) or timestep < 0 or timestep > schedule.num_steps
        for timestep in captures
    ):
        raise ValueError(f"capture_timesteps must lie in [0, {schedule.num_steps}].")
    trajectory: dict[int, Tensor] = {}
    if schedule.num_steps in captures:
        trajectory[schedule.num_steps] = sample.detach().clone()

    for code_timestep in reversed(range(schedule.num_steps)):
        timesteps = torch.full(
            (shape[0],),
            code_timestep,
            device=device,
            dtype=torch.long,
        )
        sample = p_sample(
            model,
            sample,
            timesteps,
            schedule,
            generator=generator,
            clip_x_start=clip_x_start,
        )
        if not torch.isfinite(sample).all():
            raise FloatingPointError(
                f"Reverse sample became nonfinite at code timestep {code_timestep}."
            )
        if code_timestep in captures:
            trajectory[code_timestep] = sample.detach().clone()

    return SamplingResult(sample=sample, trajectory=trajectory)
