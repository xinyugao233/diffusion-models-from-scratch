"""Deterministic DDIM sampling for an epsilon-prediction DDPM model."""

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
from diffusion_models.diffusion.sampling import SamplingResult
from diffusion_models.diffusion.schedules import DDPMSchedule, extract


@dataclass(frozen=True)
class DDIMStepOutput:
    """One deterministic DDIM transition and its model predictions."""

    sample: Tensor
    predicted_x_start: Tensor
    predicted_noise: Tensor


def ddim_timesteps(num_train_steps: int, num_inference_steps: int) -> tuple[int, ...]:
    """Select rounded, endpoint-preserving code timesteps in descending order."""
    if isinstance(num_train_steps, bool) or not isinstance(num_train_steps, int):
        raise TypeError("num_train_steps must be an integer.")
    if isinstance(num_inference_steps, bool) or not isinstance(
        num_inference_steps, int
    ):
        raise TypeError("num_inference_steps must be an integer.")
    if num_train_steps < 2:
        raise ValueError("num_train_steps must be at least two.")
    if not 2 <= num_inference_steps <= num_train_steps:
        raise ValueError("num_inference_steps must lie in [2, num_train_steps].")

    selected = (
        torch.linspace(
            0,
            num_train_steps - 1,
            num_inference_steps,
            dtype=torch.float64,
        )
        .round()
        .to(dtype=torch.long)
    )
    ascending = tuple(int(timestep) for timestep in selected.tolist())
    if len(set(ascending)) != num_inference_steps:
        raise RuntimeError("Rounded DDIM timesteps are not unique.")
    if ascending[0] != 0 or ascending[-1] != num_train_steps - 1:
        raise RuntimeError("DDIM timestep selection did not preserve both endpoints.")
    return tuple(reversed(ascending))


@torch.no_grad()
def ddim_step(
    model: nn.Module,
    x_t: Tensor,
    timesteps: Tensor,
    previous_timestep: int,
    schedule: DDPMSchedule,
    *,
    clip_x_start: bool = True,
) -> DDIMStepOutput:
    """Apply one deterministic DDIM (eta=0) transition from ``t`` to ``s``."""
    _validate_image_and_timesteps(x_t, timesteps, schedule)
    if isinstance(previous_timestep, bool) or not isinstance(previous_timestep, int):
        raise TypeError("previous_timestep must be an integer.")
    if not -1 <= previous_timestep < schedule.num_steps:
        raise ValueError(
            f"previous_timestep must lie in [-1, {schedule.num_steps - 1}]."
        )
    if not torch.all(timesteps == timesteps[0]):
        raise ValueError("Every batch item must use the same DDIM timestep.")
    current_timestep = int(timesteps[0].item())
    if previous_timestep >= current_timestep:
        raise ValueError(
            "previous_timestep must be strictly below the current timestep."
        )

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

    alpha_bar_t = extract(schedule.alpha_bars, timesteps, x_t.shape).to(dtype=x_t.dtype)
    if previous_timestep == -1:
        alpha_bar_previous = torch.ones_like(alpha_bar_t)
    else:
        alpha_bar_previous = schedule.alpha_bars[previous_timestep].to(
            device=x_t.device,
            dtype=x_t.dtype,
        )
    sample = (
        torch.sqrt(alpha_bar_previous) * predicted_x_start
        + torch.sqrt(1.0 - alpha_bar_previous) * predicted_noise
    )
    for name, tensor in (
        ("predicted noise", predicted_noise),
        ("predicted x_start", predicted_x_start),
        ("DDIM sample", sample),
    ):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains nonfinite values.")
    return DDIMStepOutput(
        sample=sample,
        predicted_x_start=predicted_x_start,
        predicted_noise=predicted_noise,
    )


@torch.no_grad()
def ddim_sample_loop(
    model: nn.Module,
    schedule: DDPMSchedule,
    shape: Sequence[int],
    *,
    num_inference_steps: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    initial_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
    clip_x_start: bool = True,
    capture_timesteps: Sequence[int] = (),
) -> SamplingResult:
    """Run an eta=0 DDIM trajectory with exactly ``num_inference_steps`` calls."""
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
        sample = torch.randn(shape, device=device, dtype=dtype, generator=generator)
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

    selected = ddim_timesteps(schedule.num_steps, num_inference_steps)
    captures = set(capture_timesteps)
    allowed_captures = {schedule.num_steps, *selected}
    if any(
        not isinstance(timestep, int) or timestep not in allowed_captures
        for timestep in captures
    ):
        raise ValueError("capture_timesteps must be selected DDIM states.")
    trajectory: dict[int, Tensor] = {}
    if schedule.num_steps in captures:
        trajectory[schedule.num_steps] = sample.detach().clone()

    for index, code_timestep in enumerate(selected):
        previous_timestep = selected[index + 1] if index + 1 < len(selected) else -1
        timesteps = torch.full(
            (shape[0],), code_timestep, device=device, dtype=torch.long
        )
        output = ddim_step(
            model,
            sample,
            timesteps,
            previous_timestep,
            schedule,
            clip_x_start=clip_x_start,
        )
        sample = output.sample
        if code_timestep in captures:
            trajectory[code_timestep] = sample.detach().clone()

    return SamplingResult(sample=sample, trajectory=trajectory)
