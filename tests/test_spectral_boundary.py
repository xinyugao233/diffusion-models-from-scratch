from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_models.diffusion import make_linear_ddpm_schedule
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.full_training import production_train_step
from diffusion_models.spectral_boundary import (
    SpectralBoundaryConfig,
    compute_boundary_weights,
    effective_additive_sigma,
    load_radial_power,
    radial_shell_indices,
    weighted_spectral_loss,
)

POWER_PATH = Path(__file__).parents[1] / "configs" / "cifar10_e006_radial_power.csv"


class TinyNoisePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.convolution(image) + timesteps[:, None, None, None] / 1000.0


def test_e006_power_matches_the_full_fft_shell_grid() -> None:
    radial_power = load_radial_power(POWER_PATH)
    shells = radial_shell_indices(32, 32, device=torch.device("cpu"))
    counts = torch.bincount(shells.flatten(), minlength=radial_power.power.numel())

    assert radial_power.power.shape == radial_power.coefficient_counts.shape == (23,)
    torch.testing.assert_close(
        counts.to(torch.float64), radial_power.coefficient_counts, rtol=0.0, atol=0.0
    )
    assert int(counts.sum()) == 32 * 32


def test_effective_sigma_uses_additive_coordinates() -> None:
    schedule = make_linear_ddpm_schedule(num_steps=10)
    timesteps = torch.tensor([0, 9], dtype=torch.long)
    sigma = effective_additive_sigma(schedule, timesteps, dtype=torch.float64)
    expected = torch.sqrt(
        (1.0 - schedule.alpha_bars[timesteps].double())
        / schedule.alpha_bars[timesteps].double()
    )

    torch.testing.assert_close(sigma, expected, rtol=0.0, atol=0.0)
    assert sigma[1] > sigma[0] > 0.0


def test_boundary_peak_moves_outward_as_sigma_decreases() -> None:
    radial_power = load_radial_power(POWER_PATH)
    config = SpectralBoundaryConfig(tau=0.3, weight_floor=0.05)
    weights = compute_boundary_weights(radial_power, torch.tensor([4.0, 0.02]), config)

    assert int(weights[1].argmax()) > int(weights[0].argmax())


def test_wild_middle_range_leaves_outside_examples_uniform() -> None:
    radial_power = load_radial_power(POWER_PATH)
    config = SpectralBoundaryConfig(
        tau=0.5,
        weight_floor=0.05,
        sigma_min=0.14,
        sigma_max=8.4,
    )
    weights = compute_boundary_weights(
        radial_power, torch.tensor([0.1, 1.0, 12.0]), config
    )

    torch.testing.assert_close(weights[0], torch.ones_like(weights[0]))
    assert not torch.equal(weights[1], torch.ones_like(weights[1]))
    torch.testing.assert_close(weights[2], torch.ones_like(weights[2]))


def test_uniform_boundary_weights_recover_pixel_mse_by_parseval() -> None:
    generator = torch.Generator().manual_seed(41)
    residual = torch.randn((3, 3, 32, 32), generator=generator, dtype=torch.float64)
    schedule = make_linear_ddpm_schedule(num_steps=10, dtype=torch.float64)
    timesteps = torch.tensor([0, 4, 9], dtype=torch.long)
    loss, diagnostics = weighted_spectral_loss(
        residual,
        timesteps,
        schedule,
        load_radial_power(POWER_PATH),
        SpectralBoundaryConfig(tau=1e300, weight_floor=0.0),
    )

    torch.testing.assert_close(loss, residual.square().mean(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        diagnostics.mean_weight, torch.ones_like(diagnostics.mean_weight)
    )


@pytest.mark.parametrize("tau", [1e-3, 0.25, 1e6])
@pytest.mark.parametrize("weight_floor", [0.0, 0.1])
def test_extreme_weight_parameters_remain_finite(
    tau: float, weight_floor: float
) -> None:
    radial_power = load_radial_power(POWER_PATH)
    config = SpectralBoundaryConfig(tau=tau, weight_floor=weight_floor)
    sigma = torch.tensor([0.002, 0.14, 8.4, 80.0], dtype=torch.float64)

    weights = compute_boundary_weights(radial_power, sigma, config)
    assert torch.isfinite(weights).all()


def test_disabled_production_path_is_bitwise_identical() -> None:
    schedule = make_linear_ddpm_schedule(num_steps=10)
    images = (
        torch.rand((2, 3, 4, 4), generator=torch.Generator().manual_seed(51)) * 2.0
        - 1.0
    )

    def build():
        torch.manual_seed(52)
        model = TinyNoisePredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ema = ExponentialMovingAverage(model, decay=0.9)
        generator = torch.Generator().manual_seed(53)
        return model, optimizer, ema, generator

    first_model, first_optimizer, first_ema, first_generator = build()
    second_model, second_optimizer, second_ema, second_generator = build()
    first = production_train_step(
        first_model,
        first_optimizer,
        first_ema,
        images,
        schedule,
        generator=first_generator,
        max_gradient_norm=1.0,
    )
    second = production_train_step(
        second_model,
        second_optimizer,
        second_ema,
        images,
        schedule,
        generator=second_generator,
        max_gradient_norm=1.0,
        spectral_boundary=None,
    )

    assert first == second
    assert all(
        torch.equal(first_value, second_model.state_dict()[name])
        for name, first_value in first_model.state_dict().items()
    )
    assert first.loss == first.unweighted_loss


def test_weighted_loss_changes_only_the_residual_objective() -> None:
    residual = torch.randn((2, 3, 32, 32), generator=torch.Generator().manual_seed(61))
    schedule = make_linear_ddpm_schedule(num_steps=1000)
    timesteps = torch.tensor([100, 800], dtype=torch.long)
    weighted, diagnostics = weighted_spectral_loss(
        residual,
        timesteps,
        schedule,
        load_radial_power(POWER_PATH),
        SpectralBoundaryConfig(tau=0.5, weight_floor=0.05),
    )

    assert torch.isfinite(weighted)
    assert not torch.isclose(weighted, F.mse_loss(residual, torch.zeros_like(residual)))
    torch.testing.assert_close(
        diagnostics.mean_weight,
        torch.ones_like(diagnostics.mean_weight),
        atol=1e-6,
        rtol=1e-6,
    )
