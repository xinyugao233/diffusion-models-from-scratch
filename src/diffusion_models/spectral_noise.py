"""Frequency-dependent monotone forward corruption and matched reverse chain."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from diffusion_models.diffusion.ddpm import q_sample
from diffusion_models.diffusion.sampling import SamplingResult
from diffusion_models.diffusion.schedules import DDPMSchedule
from diffusion_models.spectral_boundary import RadialPower
from diffusion_models.training import EpsilonTrainingBatch


@dataclass(frozen=True)
class SpectralNoiseConfig:
    """Frozen hazard-redistribution parameters for weighted-noise diffusion."""

    tau: float
    weight_floor: float = 0.1
    rho: float = 1.0
    allocation: str = "boundary_density"
    fft_normalization: str = "ortho"

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive.")
        if not math.isfinite(self.weight_floor) or self.weight_floor < 0.0:
            raise ValueError("weight_floor must be finite and nonnegative.")
        if not math.isfinite(self.rho) or not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1].")
        if self.allocation not in {"boundary_density", "baseline_reweighted"}:
            raise ValueError("Unknown spectral-noise hazard allocation.")
        if self.fft_normalization != "ortho":
            raise ValueError("Only orthonormal FFTs are supported.")


@dataclass(frozen=True)
class SpectralNoiseSchedule:
    """Per-timestep, per-shell coefficients for a Gaussian Markov chain."""

    baseline: DDPMSchedule
    radial_power: RadialPower
    config: SpectralNoiseConfig
    boundary_emphasis: Tensor
    hazards: Tensor
    alphas: Tensor
    betas: Tensor
    alpha_bars: Tensor
    sqrt_alpha_bars: Tensor
    sqrt_one_minus_alpha_bars: Tensor

    @property
    def num_steps(self) -> int:
        return self.alphas.shape[0]

    @property
    def num_shells(self) -> int:
        return self.alphas.shape[1]


@dataclass(frozen=True)
class SpectralReversePrediction:
    """Spatial reverse mean plus the Fourier-diagonal posterior variance."""

    mean: Tensor
    variance_rfft: Tensor
    predicted_x_start: Tensor
    predicted_noise: Tensor


def rfft_shell_indices(height: int, width: int, *, device: torch.device) -> Tensor:
    """Return floor-radius shell indices for the unshifted real FFT plane."""
    if height <= 0 or width <= 0:
        raise ValueError("FFT dimensions must be positive.")
    fy = torch.fft.fftfreq(height, device=device) * height
    fx = torch.fft.rfftfreq(width, device=device) * width
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.floor(torch.sqrt(xx.square() + yy.square()) + 1e-12).long()


def build_spectral_noise_schedule(
    baseline: DDPMSchedule,
    radial_power: RadialPower,
    config: SpectralNoiseConfig,
) -> SpectralNoiseSchedule:
    """Redistribute total baseline hazard while preserving terminal attenuation."""
    dtype = torch.float64
    device = baseline.betas.device
    base_alpha = baseline.alphas.to(dtype=dtype)
    base_alpha_bar = baseline.alpha_bars.to(dtype=dtype)
    base_hazard = -torch.log(base_alpha)
    # Match the baseline's stored terminal attenuation.  In float32, deriving
    # this from the individual alphas can differ from alpha_bars[-1] because
    # the baseline cumulative product and the float64 log-sum round differently.
    total_hazard = -torch.log(base_alpha_bar[-1])
    sigma_squared = (1.0 - base_alpha_bar) / base_alpha_bar
    power = radial_power.power.to(device=device, dtype=dtype)
    log_distance = torch.log(power)[None, :] - torch.log(sigma_squared)[:, None]
    emphasis = config.weight_floor + torch.exp(
        -0.5 * (log_distance / config.tau).square()
    )
    if config.allocation == "baseline_reweighted":
        raw = base_hazard[:, None] * emphasis
    else:
        # The originally proposed base_hazard * emphasis is monotone, but for
        # high-frequency CIFAR shells its absolute hazard peaks at t=999 rather
        # than the information boundary. Allocating by emphasis itself is the
        # minimal correction that makes the redistributed rate boundary-local.
        raw = emphasis
    boundary_hazard = total_hazard * raw / raw.sum(dim=0, keepdim=True)
    if config.rho == 0.0:
        hazards = base_hazard[:, None].expand(-1, power.numel())
        alphas = baseline.alphas.to(dtype=dtype)[:, None].expand_as(hazards)
        betas = baseline.betas.to(dtype=dtype)[:, None].expand_as(hazards)
        alpha_bars = baseline.alpha_bars.to(dtype=dtype)[:, None].expand_as(hazards)
    else:
        hazards = (1.0 - config.rho) * base_hazard[
            :, None
        ] + config.rho * boundary_hazard
        alphas = torch.exp(-hazards)
        betas = -torch.expm1(-hazards)
        alpha_bars = torch.exp(-torch.cumsum(hazards, dim=0))
    if not all(
        torch.isfinite(value).all()
        for value in (emphasis, hazards, alphas, betas, alpha_bars)
    ):
        raise FloatingPointError("Spectral-noise schedule contains nonfinite values.")
    if torch.any(alphas <= 0.0) or torch.any(alphas > 1.0):
        raise ValueError("Spectral-noise alphas must lie in (0,1].")
    if torch.any(betas < 0.0) or torch.any(betas >= 1.0):
        raise ValueError("Spectral-noise betas must lie in [0,1).")
    if torch.any(alpha_bars[1:] > alpha_bars[:-1]):
        raise ValueError("Spectral-noise cumulative attenuation is not monotone.")
    terminal = baseline.alpha_bars[-1].to(dtype=dtype)
    if not torch.allclose(
        alpha_bars[-1], terminal.expand_as(alpha_bars[-1]), rtol=1e-7, atol=1e-15
    ):
        raise ValueError("Shell terminal attenuation differs from baseline.")
    return SpectralNoiseSchedule(
        baseline=baseline,
        radial_power=radial_power,
        config=config,
        boundary_emphasis=emphasis,
        hazards=hazards,
        alphas=alphas,
        betas=betas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(torch.clamp(1.0 - alpha_bars, min=0.0)),
    )


def _coefficient_map(values: Tensor, timesteps: Tensor, shape: Sequence[int]) -> Tensor:
    if timesteps.ndim != 1 or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be one-dimensional torch.long.")
    if len(shape) != 4 or shape[0] != timesteps.shape[0]:
        raise ValueError("shape and timestep batch size differ.")
    selected = values[timesteps]
    shells = rfft_shell_indices(shape[-2], shape[-1], device=timesteps.device)
    if int(shells.max()) >= values.shape[1]:
        raise ValueError("Radial schedule does not cover the requested FFT grid.")
    return selected[:, shells]


def _spectral_mix(
    first: Tensor, second: Tensor, first_scale: Tensor, second_scale: Tensor
) -> Tensor:
    first_hat = torch.fft.rfft2(first, norm="ortho")
    second_hat = torch.fft.rfft2(second, norm="ortho")
    mixed = first_scale[:, None] * first_hat + second_scale[:, None] * second_hat
    return torch.fft.irfft2(mixed, s=first.shape[-2:], norm="ortho")


def q_sample_spectral(
    x_start: Tensor,
    timesteps: Tensor,
    schedule: SpectralNoiseSchedule,
    noise: Tensor | None = None,
) -> Tensor:
    """Sample the shell-dependent forward marginal from real spatial noise."""
    if noise is None:
        noise = torch.randn_like(x_start)
    if noise.shape != x_start.shape or noise.device != x_start.device:
        raise ValueError("noise must match x_start shape and device.")
    if noise.dtype != x_start.dtype or not x_start.is_floating_point():
        raise TypeError("noise and x_start must share a floating-point dtype.")
    if schedule.config.rho == 0.0:
        return q_sample(x_start, timesteps, schedule.baseline, noise)
    first_scale = _coefficient_map(
        schedule.sqrt_alpha_bars, timesteps, x_start.shape
    ).to(x_start.dtype)
    second_scale = _coefficient_map(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x_start.shape
    ).to(x_start.dtype)
    result = _spectral_mix(x_start, noise, first_scale, second_scale)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Spectral forward sample contains nonfinite values.")
    return result


def make_spectral_epsilon_training_batch(
    x_start: Tensor,
    schedule: SpectralNoiseSchedule,
    *,
    generator: torch.Generator,
) -> EpsilonTrainingBatch:
    """Draw paired timesteps/noise and retain the underlying spatial epsilon target."""
    timesteps = torch.randint(
        0,
        schedule.num_steps,
        (x_start.shape[0],),
        device=x_start.device,
        generator=generator,
    )
    noise = torch.randn(
        x_start.shape, device=x_start.device, dtype=x_start.dtype, generator=generator
    )
    return EpsilonTrainingBatch(
        x_t=q_sample_spectral(x_start, timesteps, schedule, noise),
        timesteps=timesteps,
        target_noise=noise,
    )


def predict_x_start_spectral(
    x_t: Tensor,
    timesteps: Tensor,
    predicted_noise: Tensor,
    schedule: SpectralNoiseSchedule,
) -> Tensor:
    """Invert the shell-dependent marginal using a spatial epsilon prediction."""
    x_hat = torch.fft.rfft2(x_t, norm="ortho")
    epsilon_hat = torch.fft.rfft2(predicted_noise, norm="ortho")
    signal = _coefficient_map(schedule.sqrt_alpha_bars, timesteps, x_t.shape).to(
        x_t.dtype
    )
    noise = _coefficient_map(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x_t.shape
    ).to(x_t.dtype)
    clean_hat = (x_hat - noise[:, None] * epsilon_hat) / signal[:, None]
    return torch.fft.irfft2(clean_hat, s=x_t.shape[-2:], norm="ortho")


def _posterior_shell_coefficients(
    schedule: SpectralNoiseSchedule,
) -> tuple[Tensor, Tensor, Tensor]:
    previous = torch.cat(
        (torch.ones_like(schedule.alpha_bars[:1]), schedule.alpha_bars[:-1])
    )
    denominator = 1.0 - schedule.alpha_bars
    variance = schedule.betas * (1.0 - previous) / denominator
    clean_coefficient = schedule.betas * torch.sqrt(previous) / denominator
    current_coefficient = torch.sqrt(schedule.alphas) * (1.0 - previous) / denominator
    return variance.clamp(min=0.0), clean_coefficient, current_coefficient


def p_mean_variance_spectral(
    model: nn.Module,
    x_t: Tensor,
    timesteps: Tensor,
    schedule: SpectralNoiseSchedule,
    *,
    clip_x_start: bool = True,
) -> SpectralReversePrediction:
    """Convert spatial epsilon prediction to the matched Fourier posterior."""
    predicted_noise = model(x_t, timesteps)
    predicted_x_start = predict_x_start_spectral(
        x_t, timesteps, predicted_noise, schedule
    )
    if clip_x_start:
        predicted_x_start = predicted_x_start.clamp(-1.0, 1.0)
    variance, clean_coefficient, current_coefficient = _posterior_shell_coefficients(
        schedule
    )
    variance_map = _coefficient_map(variance, timesteps, x_t.shape).to(x_t.dtype)
    clean_map = _coefficient_map(clean_coefficient, timesteps, x_t.shape).to(x_t.dtype)
    current_map = _coefficient_map(current_coefficient, timesteps, x_t.shape).to(
        x_t.dtype
    )
    clean_hat = torch.fft.rfft2(predicted_x_start, norm="ortho")
    current_hat = torch.fft.rfft2(x_t, norm="ortho")
    mean_hat = clean_map[:, None] * clean_hat + current_map[:, None] * current_hat
    mean = torch.fft.irfft2(mean_hat, s=x_t.shape[-2:], norm="ortho")
    return SpectralReversePrediction(
        mean=mean,
        variance_rfft=variance_map,
        predicted_x_start=predicted_x_start,
        predicted_noise=predicted_noise,
    )


@torch.no_grad()
def p_sample_spectral(
    model: nn.Module,
    x_t: Tensor,
    timesteps: Tensor,
    schedule: SpectralNoiseSchedule,
    *,
    generator: torch.Generator,
    clip_x_start: bool = True,
) -> Tensor:
    """Draw one matched ancestral reverse step using real spatial white noise."""
    prediction = p_mean_variance_spectral(
        model, x_t, timesteps, schedule, clip_x_start=clip_x_start
    )
    if torch.all(timesteps == 0):
        return prediction.mean
    noise = torch.randn(
        x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator
    )
    noise_hat = torch.fft.rfft2(noise, norm="ortho")
    addition_hat = torch.sqrt(prediction.variance_rfft)[:, None] * noise_hat
    addition = torch.fft.irfft2(addition_hat, s=x_t.shape[-2:], norm="ortho")
    mask = (timesteps != 0).reshape(-1, 1, 1, 1).to(x_t.dtype)
    result = prediction.mean + mask * addition
    if not torch.isfinite(result).all():
        raise FloatingPointError("Spectral reverse sample contains nonfinite values.")
    return result


@torch.no_grad()
def p_sample_loop_spectral(
    model: nn.Module,
    schedule: SpectralNoiseSchedule,
    shape: Sequence[int],
    *,
    device: torch.device,
    generator: torch.Generator,
    initial_noise: Tensor | None = None,
    clip_x_start: bool = True,
) -> SamplingResult:
    """Run the complete matched shell-dependent ancestral reverse chain."""
    if initial_noise is None:
        sample = torch.randn(tuple(shape), device=device, generator=generator)
    else:
        if tuple(initial_noise.shape) != tuple(shape) or initial_noise.device != device:
            raise ValueError("initial_noise must match the requested shape and device.")
        sample = initial_noise.detach().clone()
    for code_timestep in reversed(range(schedule.num_steps)):
        timesteps = torch.full(
            (shape[0],), code_timestep, device=device, dtype=torch.long
        )
        sample = p_sample_spectral(
            model,
            sample,
            timesteps,
            schedule,
            generator=generator,
            clip_x_start=clip_x_start,
        )
    return SamplingResult(sample=sample, trajectory={})
