from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from diffusion_models.diffusion import make_linear_ddpm_schedule, q_sample
from diffusion_models.diffusion.sampling import _posterior_coefficients
from diffusion_models.spectral_boundary import load_radial_power
from diffusion_models.spectral_noise import (
    SpectralNoiseConfig,
    _posterior_shell_coefficients,
    build_spectral_noise_schedule,
    make_spectral_epsilon_training_batch,
    p_mean_variance_spectral,
    predict_x_start_spectral,
    q_sample_spectral,
    rfft_shell_indices,
)

POWER_PATH = Path(__file__).parents[1] / "configs" / "cifar10_50k_radial_power.csv"


def make_schedule(
    tau: float = math.log(2),
    *,
    rho: float = 1.0,
    allocation: str = "boundary_density",
):
    baseline = make_linear_ddpm_schedule(dtype=torch.float64)
    return build_spectral_noise_schedule(
        baseline,
        load_radial_power(POWER_PATH),
        SpectralNoiseConfig(tau=tau, weight_floor=0.1, rho=rho, allocation=allocation),
    )


class FixedNoise(nn.Module):
    def __init__(self, noise: torch.Tensor) -> None:
        super().__init__()
        self.noise = noise

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.noise.expand_as(image)


def test_rho_zero_schedule_and_forward_are_exact_baseline() -> None:
    schedule = make_schedule(rho=0.0)
    baseline = schedule.baseline
    torch.testing.assert_close(
        schedule.alphas,
        baseline.alphas[:, None].expand_as(schedule.alphas),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        schedule.betas,
        baseline.betas[:, None].expand_as(schedule.betas),
        rtol=0,
        atol=0,
    )
    x = torch.randn(
        (3, 2, 32, 32), generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    noise = torch.randn(
        x.shape, generator=torch.Generator().manual_seed(2), dtype=x.dtype
    )
    timesteps = torch.tensor([0, 411, 999])
    assert torch.equal(
        q_sample_spectral(x, timesteps, schedule, noise),
        q_sample(x, timesteps, baseline, noise),
    )


def test_schedule_ranges_monotonicity_terminal_equality_and_finiteness() -> None:
    for tau in (math.log(2), math.log(4)):
        schedule = make_schedule(tau)
        assert all(
            torch.isfinite(value).all()
            for value in (
                schedule.hazards,
                schedule.alphas,
                schedule.betas,
                schedule.alpha_bars,
            )
        )
        assert torch.all((schedule.alphas > 0) & (schedule.alphas <= 1))
        assert torch.all((schedule.betas >= 0) & (schedule.betas < 1))
        assert torch.all(schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1])
        torch.testing.assert_close(
            schedule.alpha_bars[-1],
            schedule.baseline.alpha_bars[-1].expand(schedule.num_shells),
            rtol=1e-12,
            atol=1e-15,
        )


def test_corrected_hazard_peaks_at_boundary_but_literal_proposal_does_not() -> None:
    corrected = make_schedule()
    literal = make_schedule(allocation="baseline_reweighted")
    sigma_squared = (1 - corrected.baseline.alpha_bars) / corrected.baseline.alpha_bars
    power = corrected.radial_power.power
    crossing = (
        (torch.log(sigma_squared)[:, None] - torch.log(power)[None, :])
        .abs()
        .argmin(dim=0)
    )
    assert torch.equal(corrected.hazards.argmax(dim=0), crossing)
    literal_peaks = literal.hazards.argmax(dim=0)
    assert torch.all(literal_peaks[8:] == 999)


def test_narrow_boundary_component_is_tighter_than_broad() -> None:
    narrow = make_schedule(math.log(2))
    broad = make_schedule(math.log(4))
    positions = torch.arange(1000, dtype=torch.float64)[:, None]

    def variance(schedule):
        mass = schedule.boundary_emphasis - schedule.config.weight_floor
        probability = mass / mass.sum(dim=0, keepdim=True)
        mean = (probability * positions).sum(dim=0)
        return (probability * (positions - mean) ** 2).sum(dim=0)

    assert torch.all(variance(narrow) < variance(broad))


def test_forward_is_real_invertible_and_deterministic() -> None:
    schedule = make_schedule()
    x = torch.randn(
        (2, 3, 32, 32), generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    noise = torch.randn(
        x.shape, generator=torch.Generator().manual_seed(4), dtype=x.dtype
    )
    timesteps = torch.tensor([0, 999])
    first = q_sample_spectral(x, timesteps, schedule, noise)
    second = q_sample_spectral(x, timesteps, schedule, noise)
    reconstructed = predict_x_start_spectral(first, timesteps, noise, schedule)
    assert not first.is_complex()
    assert torch.equal(first, second)
    torch.testing.assert_close(reconstructed, x, rtol=1e-9, atol=1e-9)


def test_empirical_forward_fourier_variance_matches_schedule() -> None:
    schedule = make_schedule()
    count = 4096
    timestep = 500
    clean = torch.zeros((count, 1, 32, 32), dtype=torch.float64)
    noise = torch.randn(
        clean.shape, generator=torch.Generator().manual_seed(5), dtype=clean.dtype
    )
    result = q_sample_spectral(clean, torch.full((count,), timestep), schedule, noise)
    spectrum = torch.fft.rfft2(result, norm="ortho")
    shells = rfft_shell_indices(32, 32, device=torch.device("cpu"))
    for radius in (0, 5, 11, 22):
        observed = spectrum[:, 0, shells == radius].abs().square().mean()
        expected = 1.0 - schedule.alpha_bars[timestep, radius]
        torch.testing.assert_close(observed, expected, rtol=0.08, atol=0.01)


def test_reverse_coefficients_match_scalar_ddpm_shell_by_shell() -> None:
    schedule = make_schedule()
    variance, clean, current = _posterior_shell_coefficients(schedule)
    for radius in (0, 7, 15, 22):
        scalar = make_linear_ddpm_schedule(dtype=torch.float64)
        scalar = type(scalar)(
            betas=schedule.betas[:, radius],
            alphas=schedule.alphas[:, radius],
            alpha_bars=schedule.alpha_bars[:, radius],
            sqrt_alpha_bars=schedule.sqrt_alpha_bars[:, radius],
            sqrt_one_minus_alpha_bars=schedule.sqrt_one_minus_alpha_bars[:, radius],
        )
        expected = _posterior_coefficients(scalar)
        torch.testing.assert_close(variance[:, radius], expected[0])
        torch.testing.assert_close(clean[:, radius], expected[1])
        torch.testing.assert_close(current[:, radius], expected[2])
    assert torch.all(variance[0] == 0)


def test_exact_epsilon_reverse_recovers_posterior_with_valid_endpoints() -> None:
    schedule = make_schedule()
    clean = torch.randn(
        (2, 3, 32, 32), generator=torch.Generator().manual_seed(6), dtype=torch.float64
    )
    epsilon = torch.randn(
        clean.shape, generator=torch.Generator().manual_seed(7), dtype=clean.dtype
    )
    timesteps = torch.tensor([0, 999])
    noisy = q_sample_spectral(clean, timesteps, schedule, epsilon)
    prediction = p_mean_variance_spectral(
        FixedNoise(epsilon), noisy, timesteps, schedule, clip_x_start=False
    )
    torch.testing.assert_close(
        prediction.predicted_x_start, clean, rtol=1e-9, atol=1e-9
    )
    assert torch.all(prediction.variance_rfft[0] == 0)
    assert torch.isfinite(prediction.mean).all()
    assert prediction.mean.shape == clean.shape


def test_fixed_generator_reproduces_training_batch() -> None:
    schedule = make_schedule()
    images = torch.randn(
        (4, 3, 32, 32), generator=torch.Generator().manual_seed(8), dtype=torch.float64
    )
    first = make_spectral_epsilon_training_batch(
        images, schedule, generator=torch.Generator().manual_seed(9)
    )
    second = make_spectral_epsilon_training_batch(
        images, schedule, generator=torch.Generator().manual_seed(9)
    )
    assert torch.equal(first.timesteps, second.timesteps)
    assert torch.equal(first.target_noise, second.target_noise)
    assert torch.equal(first.x_t, second.x_t)
