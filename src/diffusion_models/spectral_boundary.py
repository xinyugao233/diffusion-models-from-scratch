"""Optional radial Fourier loss weighting near an empirical SNR=1 boundary."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from diffusion_models.diffusion.schedules import DDPMSchedule


@dataclass(frozen=True)
class SpectralBoundaryConfig:
    """Configuration for coefficient-weighted radial boundary emphasis."""

    tau: float
    weight_floor: float
    normalization: str = "coefficient_mean"
    fft_normalization: str = "ortho"
    sigma_min: float | None = None
    sigma_max: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive.")
        if not math.isfinite(self.weight_floor) or self.weight_floor < 0.0:
            raise ValueError("weight_floor must be finite and nonnegative.")
        if self.normalization not in {"coefficient_mean", "none"}:
            raise ValueError("normalization must be 'coefficient_mean' or 'none'.")
        if self.fft_normalization != "ortho":
            raise ValueError("Only the E006-compatible 'ortho' FFT is supported.")
        if (self.sigma_min is None) != (self.sigma_max is None):
            raise ValueError("sigma_min and sigma_max must be set together.")
        if self.sigma_min is not None:
            if not 0.0 < self.sigma_min <= self.sigma_max:
                raise ValueError("Expected 0 < sigma_min <= sigma_max.")


@dataclass(frozen=True)
class RadialPower:
    """Dataset-level mean power and full-FFT multiplicity for radial shells."""

    power: Tensor
    coefficient_counts: Tensor


@dataclass(frozen=True)
class SpectralBoundaryDiagnostics:
    """Batch-level diagnostics for one weighted loss evaluation."""

    sigma_mean: Tensor
    sigma_min: Tensor
    sigma_max: Tensor
    active_fraction: Tensor
    peak_shell_mean: Tensor
    information_radius_mean: Tensor
    mean_weight: Tensor
    maximum_weight: Tensor
    effective_shell_count: Tensor


def load_radial_power(path: str | Path) -> RadialPower:
    """Load a contiguous, positive E006-style radial-power CSV."""
    rows: list[tuple[int, float, int]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    int(row["radius"]),
                    float(row["shell_power"]),
                    int(row["full_fft_sites_per_channel"]),
                )
            )
    if not rows:
        raise ValueError("Radial-power CSV is empty.")
    radii = [row[0] for row in rows]
    if radii != list(range(len(rows))):
        raise ValueError("Radial-power radii must be contiguous and start at zero.")
    powers = torch.tensor([row[1] for row in rows], dtype=torch.float64)
    counts = torch.tensor([row[2] for row in rows], dtype=torch.float64)
    if not torch.isfinite(powers).all() or torch.any(powers <= 0.0):
        raise ValueError("Every radial shell power must be finite and positive.")
    if not torch.isfinite(counts).all() or torch.any(counts <= 0.0):
        raise ValueError("Every radial shell count must be finite and positive.")
    return RadialPower(power=powers, coefficient_counts=counts)


def radial_shell_indices(height: int, width: int, *, device: torch.device) -> Tensor:
    """Return E006-compatible unshifted full-FFT floor-radius indices."""
    if height <= 0 or width <= 0:
        raise ValueError("FFT height and width must be positive.")
    fy = torch.fft.fftfreq(height, device=device) * height
    fx = torch.fft.fftfreq(width, device=device) * width
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.floor(torch.sqrt(xx.square() + yy.square()) + 1e-12).long()


def effective_additive_sigma(
    schedule: DDPMSchedule,
    timesteps: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Map VP-DDPM timesteps to sigma in ``x_0 + sigma * epsilon`` coordinates."""
    if timesteps.ndim != 1 or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be a one-dimensional torch.long tensor.")
    alpha_bar = schedule.alpha_bars.gather(0, timesteps).to(dtype=dtype)
    return torch.sqrt((1.0 - alpha_bar) / alpha_bar)


def compute_boundary_weights(
    radial_power: RadialPower,
    sigma: Tensor,
    config: SpectralBoundaryConfig,
) -> Tensor:
    """Return one radial-shell weight vector per effective additive sigma."""
    if sigma.ndim != 1 or not sigma.is_floating_point():
        raise ValueError("sigma must be a one-dimensional floating-point tensor.")
    if not torch.isfinite(sigma).all() or torch.any(sigma <= 0.0):
        raise ValueError("sigma values must be finite and positive.")
    power = radial_power.power.to(device=sigma.device, dtype=sigma.dtype)
    counts = radial_power.coefficient_counts.to(
        device=sigma.device, dtype=sigma.dtype
    )
    log_snr = torch.log(power)[None, :] - 2.0 * torch.log(sigma)[:, None]
    log_kernel = -0.5 * (log_snr / config.tau).square()
    if config.weight_floor == 0.0 and config.normalization == "coefficient_mean":
        # The common factor cancels during normalization. Removing it prevents
        # every shell underflowing to zero for very narrow kernels.
        log_kernel = log_kernel - log_kernel.amax(dim=1, keepdim=True)
    weights = config.weight_floor + torch.exp(log_kernel)
    if config.sigma_min is not None:
        active = (sigma >= config.sigma_min) & (sigma <= config.sigma_max)
        weights = torch.where(active[:, None], weights, torch.ones_like(weights))
    if config.normalization == "coefficient_mean":
        normalizer = (weights * counts[None, :]).sum(dim=1, keepdim=True) / counts.sum()
        if torch.any(normalizer <= 0.0):
            raise FloatingPointError("Spectral boundary weights have zero mean.")
        weights = weights / normalizer
    if not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite spectral boundary weights detected.")
    return weights


def radial_power_map(shell_weights: Tensor, shell_indices: Tensor) -> Tensor:
    """Map ``[batch, shell]`` weights onto unshifted full-FFT coefficients."""
    if shell_weights.ndim != 2 or shell_indices.ndim != 2:
        raise ValueError("Expected shell_weights [B,R] and shell_indices [H,W].")
    if int(shell_indices.max().item()) >= shell_weights.shape[1]:
        raise ValueError("Radial-power table does not cover every FFT coefficient.")
    return shell_weights[:, shell_indices]


def weighted_spectral_loss(
    residual: Tensor,
    timesteps: Tensor,
    schedule: DDPMSchedule,
    radial_power: RadialPower,
    config: SpectralBoundaryConfig,
) -> tuple[Tensor, SpectralBoundaryDiagnostics]:
    """Weight epsilon residual power by the moving empirical boundary.

    The reduction remains coefficient based. With coefficient-mean-normalized
    uniform weights, Parseval makes this exactly the ordinary pixel MSE up to
    floating-point FFT roundoff. This avoids silently adding equal-shell
    balancing as a second intervention.
    """
    if residual.ndim != 4 or not residual.is_floating_point():
        raise ValueError("residual must be a floating-point NCHW tensor.")
    if timesteps.shape != (residual.shape[0],):
        raise ValueError("timesteps must contain one value per residual image.")
    sigma = effective_additive_sigma(schedule, timesteps, dtype=residual.dtype)
    shell_weights = compute_boundary_weights(radial_power, sigma, config)
    shell_indices = radial_shell_indices(
        residual.shape[-2], residual.shape[-1], device=residual.device
    )
    coefficient_weights = radial_power_map(shell_weights, shell_indices)
    observed_counts = torch.bincount(
        shell_indices.flatten(), minlength=shell_weights.shape[1]
    ).to(device=residual.device, dtype=residual.dtype)
    expected_counts = radial_power.coefficient_counts.to(
        device=residual.device, dtype=residual.dtype
    )
    if not torch.equal(observed_counts, expected_counts):
        raise ValueError(
            "Radial-power multiplicities do not match the residual FFT grid."
        )
    spectrum = torch.fft.fft2(
        residual, dim=(-2, -1), norm=config.fft_normalization
    )
    per_example_loss = (
        spectrum.abs().square() * coefficient_weights[:, None]
    ).mean(dim=(1, 2, 3))
    loss = per_example_loss.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite weighted spectral loss detected.")

    power = radial_power.power.to(device=sigma.device, dtype=sigma.dtype)
    valid = power[None, :] >= sigma[:, None].square()
    radii = torch.arange(power.numel(), device=sigma.device)
    information_radius = torch.where(valid, radii[None, :], -1).amax(dim=1)
    peak_shell = shell_weights.argmax(dim=1)
    active = torch.ones_like(sigma, dtype=torch.bool)
    if config.sigma_min is not None:
        active = (sigma >= config.sigma_min) & (sigma <= config.sigma_max)
    active_peak_mean = (
        peak_shell[active].to(residual.dtype).mean()
        if torch.any(active)
        else residual.new_tensor(-1.0)
    )
    effective_shells = shell_weights.sum(dim=1).square() / shell_weights.square().sum(
        dim=1
    )
    diagnostics = SpectralBoundaryDiagnostics(
        sigma_mean=sigma.mean(),
        sigma_min=sigma.amin(),
        sigma_max=sigma.amax(),
        active_fraction=active.to(residual.dtype).mean(),
        peak_shell_mean=active_peak_mean,
        information_radius_mean=information_radius.to(residual.dtype).mean(),
        mean_weight=coefficient_weights.mean(),
        maximum_weight=shell_weights.amax(dim=1).mean(),
        effective_shell_count=effective_shells.mean(),
    )
    return loss, diagnostics
