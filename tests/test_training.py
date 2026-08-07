from __future__ import annotations

import pytest
import torch
from torch import nn

from diffusion_models.diffusion.schedules import make_linear_ddpm_schedule
from diffusion_models.training import (
    EpsilonTrainingBatch,
    epsilon_prediction_loss,
    make_epsilon_training_batch,
    optimizer_step,
)


class TinyNoisePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_bias = timesteps.to(image.dtype)[:, None, None, None] / 1000.0
        return self.convolution(image) + time_bias


def _fixed_batch() -> EpsilonTrainingBatch:
    generator = torch.Generator().manual_seed(31)
    x_start = torch.rand((2, 3, 4, 4), generator=generator) * 2.0 - 1.0
    noise = torch.randn(x_start.shape, generator=generator)
    timesteps = torch.tensor([0, 9], dtype=torch.long)
    schedule = make_linear_ddpm_schedule(num_steps=10)
    return make_epsilon_training_batch(
        x_start,
        timesteps,
        schedule,
        noise=noise,
    )


def test_epsilon_loss_is_scalar_and_shapes_match() -> None:
    torch.manual_seed(32)
    batch = _fixed_batch()
    loss, prediction = epsilon_prediction_loss(TinyNoisePredictor(), batch)

    assert loss.shape == ()
    assert prediction.shape == batch.target_noise.shape == batch.x_t.shape
    assert torch.isfinite(loss)


def test_fixed_supplied_noise_reproduces_noised_input_and_target() -> None:
    generator = torch.Generator().manual_seed(33)
    x_start = torch.randn((2, 3, 4, 4), generator=generator)
    noise = torch.randn(x_start.shape, generator=generator)
    timesteps = torch.tensor([2, 7], dtype=torch.long)
    schedule = make_linear_ddpm_schedule(num_steps=10)

    first = make_epsilon_training_batch(x_start, timesteps, schedule, noise=noise)
    second = make_epsilon_training_batch(x_start, timesteps, schedule, noise=noise)

    torch.testing.assert_close(first.x_t, second.x_t, rtol=0.0, atol=0.0)
    assert first.target_noise.data_ptr() == noise.data_ptr()
    assert second.target_noise.data_ptr() == noise.data_ptr()


def test_optimizer_step_changes_a_parameter() -> None:
    torch.manual_seed(34)
    model = TinyNoisePredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = optimizer_step(model, optimizer, _fixed_batch())

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
    assert metrics.loss >= 0.0
    assert torch.isfinite(torch.tensor(metrics.gradient_norm))


@pytest.mark.parametrize(
    "batch,match",
    [
        (
            EpsilonTrainingBatch(
                torch.zeros(2, 3, 4),
                torch.zeros(2, dtype=torch.long),
                torch.zeros(2, 3, 4),
            ),
            r"shape \[B, C, H, W\]",
        ),
        (
            EpsilonTrainingBatch(
                torch.zeros(2, 3, 4, 4),
                torch.zeros(2, 1, dtype=torch.long),
                torch.zeros(2, 3, 4, 4),
            ),
            r"shape \[B\]",
        ),
        (
            EpsilonTrainingBatch(
                torch.zeros(2, 3, 4, 4),
                torch.zeros(2, dtype=torch.long),
                torch.zeros(2, 3, 3, 3),
            ),
            "same shape",
        ),
    ],
)
def test_epsilon_loss_rejects_invalid_shapes(
    batch: EpsilonTrainingBatch,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        epsilon_prediction_loss(TinyNoisePredictor(), batch)


def test_nonfinite_loss_is_detected() -> None:
    class NonfiniteModel(TinyNoisePredictor):
        def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return torch.full_like(image, float("nan")) + self.convolution(image) * 0.0

    with pytest.raises(FloatingPointError, match="loss"):
        epsilon_prediction_loss(NonfiniteModel(), _fixed_batch())


def test_nonfinite_gradient_is_detected() -> None:
    torch.manual_seed(35)
    model = TinyNoisePredictor()
    model.convolution.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    with pytest.raises(FloatingPointError, match="gradient"):
        optimizer_step(model, optimizer, _fixed_batch())


def test_gradient_clipping_returns_finite_bounded_norm() -> None:
    torch.manual_seed(36)
    model = TinyNoisePredictor()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    metrics = optimizer_step(
        model,
        optimizer,
        _fixed_batch(),
        max_gradient_norm=0.05,
    )

    assert torch.isfinite(torch.tensor(metrics.gradient_norm))
    assert torch.isfinite(torch.tensor(metrics.clipped_gradient_norm))
    assert metrics.clipped_gradient_norm <= 0.050001
