"""Closed-form DDPM forward noising and exact-noise reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion_models.diffusion.schedules import DDPMSchedule, extract


def _validate_image_and_timesteps(
    image: Tensor,
    timesteps: Tensor,
    schedule: DDPMSchedule,
) -> None:
    if image.ndim != 4:
        raise ValueError(
            f"Expected image shape [B, C, H, W], but received {tuple(image.shape)}."
        )
    if not image.is_floating_point():
        raise TypeError("Images must use a floating-point dtype.")
    if timesteps.ndim != 1:
        raise ValueError(
            f"timesteps must have shape [B], got {tuple(timesteps.shape)}."
        )
    if timesteps.shape[0] != image.shape[0]:
        raise ValueError(
            "Image and timestep batch sizes differ: "
            f"{image.shape[0]} versus {timesteps.shape[0]}."
        )
    if timesteps.dtype != torch.long:
        raise TypeError("timesteps must use torch.long.")
    if image.device != timesteps.device:
        raise ValueError("Images and timesteps must be on the same device.")
    if image.device != schedule.betas.device:
        raise ValueError(
            "The schedule and images must be on the same device. "
            f"Received schedule device {schedule.betas.device} and "
            f"image device {image.device}."
        )


def _validate_noise(noise: Tensor, image: Tensor) -> None:
    if noise.shape != image.shape:
        raise ValueError(
            "noise must have the same shape as the image: "
            f"{tuple(noise.shape)} versus {tuple(image.shape)}."
        )
    if noise.device != image.device:
        raise ValueError("noise and image must be on the same device.")
    if noise.dtype != image.dtype:
        raise TypeError("noise and image must use the same dtype.")


def q_sample(
    x_start: Tensor,
    timesteps: Tensor,
    schedule: DDPMSchedule,
    noise: Tensor | None = None,
) -> Tensor:
    """Sample ``x_t`` directly from ``q(x_t | x_start)``.

    Implements ``x_t = sqrt(alpha_bar_t) * x_start
    + sqrt(1 - alpha_bar_t) * noise``. Supplying ``noise`` makes the operation
    deterministic.
    """
    _validate_image_and_timesteps(x_start, timesteps, schedule)
    if noise is None:
        noise = torch.randn_like(x_start)
    _validate_noise(noise, x_start)

    sqrt_alpha_bar_t = extract(schedule.sqrt_alpha_bars, timesteps, x_start.shape).to(
        dtype=x_start.dtype
    )
    sqrt_one_minus_alpha_bar_t = extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x_start.shape
    ).to(dtype=x_start.dtype)
    return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise


def predict_x_start_from_noise(
    x_t: Tensor,
    timesteps: Tensor,
    noise: Tensor,
    schedule: DDPMSchedule,
) -> Tensor:
    """Algebraically reconstruct ``x_start`` when the exact noise is known."""
    _validate_image_and_timesteps(x_t, timesteps, schedule)
    _validate_noise(noise, x_t)

    sqrt_alpha_bar_t = extract(schedule.sqrt_alpha_bars, timesteps, x_t.shape).to(
        dtype=x_t.dtype
    )
    sqrt_one_minus_alpha_bar_t = extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x_t.shape
    ).to(dtype=x_t.dtype)
    return (x_t - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t
