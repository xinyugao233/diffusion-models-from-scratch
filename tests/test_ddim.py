from __future__ import annotations

from itertools import pairwise

import pytest
import torch
from torch import nn

from diffusion_models.diffusion import (
    build_ddpm_schedule,
    ddim_sample_loop,
    ddim_step,
    ddim_timesteps,
    make_linear_ddpm_schedule,
)


class ConstantNoiseModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        self.calls += 1
        return torch.full_like(image, self.value)


def tiny_schedule():
    return build_ddpm_schedule(torch.tensor([0.1, 0.2], dtype=torch.float64))


def test_ddim_timesteps_preserve_endpoints_count_and_order() -> None:
    selected = ddim_timesteps(10, 4)

    assert selected == (9, 6, 3, 0)
    assert len(selected) == 4
    assert all(left > right for left, right in pairwise(selected))
    assert ddim_timesteps(5, 5) == (4, 3, 2, 1, 0)


@pytest.mark.parametrize(
    ("train_steps", "inference_steps", "error"),
    [
        (1, 1, ValueError),
        (10, 1, ValueError),
        (10, 11, ValueError),
        (True, 2, TypeError),
    ],
)
def test_ddim_timesteps_reject_invalid_counts(
    train_steps: int, inference_steps: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        ddim_timesteps(train_steps, inference_steps)


def test_ddim_step_matches_independent_eta_zero_calculation() -> None:
    schedule = tiny_schedule()
    x_t = torch.tensor([[[[0.4]]]], dtype=torch.float64)
    predicted_noise = 0.25

    output = ddim_step(
        ConstantNoiseModel(predicted_noise),
        x_t,
        torch.tensor([1]),
        0,
        schedule,
        clip_x_start=False,
    )

    alpha_bar_t = 0.9 * 0.8
    alpha_bar_previous = 0.9
    predicted_x_start = (
        0.4 - (1.0 - alpha_bar_t) ** 0.5 * predicted_noise
    ) / alpha_bar_t**0.5
    expected = (
        alpha_bar_previous**0.5 * predicted_x_start
        + (1.0 - alpha_bar_previous) ** 0.5 * predicted_noise
    )
    torch.testing.assert_close(
        output.predicted_x_start,
        torch.tensor([[[[predicted_x_start]]]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        output.sample,
        torch.tensor([[[[expected]]]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_ddim_final_step_returns_clipped_clean_estimate() -> None:
    output = ddim_step(
        ConstantNoiseModel(),
        torch.full((2, 3, 2, 2), 10.0, dtype=torch.float64),
        torch.zeros(2, dtype=torch.long),
        -1,
        tiny_schedule(),
    )

    assert torch.equal(output.sample, torch.ones_like(output.sample))
    assert torch.equal(output.sample, output.predicted_x_start)


def test_ddim_loop_is_deterministic_and_calls_model_exactly_once_per_step() -> None:
    schedule = make_linear_ddpm_schedule(num_steps=10)
    initial_noise = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(7))
    model = ConstantNoiseModel(value=0.1)

    first = ddim_sample_loop(
        model,
        schedule,
        initial_noise.shape,
        num_inference_steps=4,
        initial_noise=initial_noise,
        capture_timesteps=(10, 6, 0),
    )
    repeated = ddim_sample_loop(
        ConstantNoiseModel(value=0.1),
        schedule,
        initial_noise.shape,
        num_inference_steps=4,
        initial_noise=initial_noise,
    )

    assert model.calls == 4
    assert torch.equal(first.sample, repeated.sample)
    assert list(first.trajectory) == [10, 6, 0]
    assert torch.equal(first.trajectory[10], initial_noise)
    assert torch.equal(first.trajectory[0], first.sample)
    assert torch.isfinite(first.sample).all()


def test_ddim_loop_changes_with_initial_noise_and_rejects_unselected_capture() -> None:
    schedule = make_linear_ddpm_schedule(num_steps=10)
    first = ddim_sample_loop(
        ConstantNoiseModel(),
        schedule,
        (1, 3, 2, 2),
        num_inference_steps=4,
        initial_noise=torch.zeros(1, 3, 2, 2),
    ).sample
    changed = ddim_sample_loop(
        ConstantNoiseModel(),
        schedule,
        (1, 3, 2, 2),
        num_inference_steps=4,
        initial_noise=torch.ones(1, 3, 2, 2),
    ).sample

    assert not torch.equal(first, changed)
    with pytest.raises(ValueError, match="selected DDIM states"):
        ddim_sample_loop(
            ConstantNoiseModel(),
            schedule,
            (1, 3, 2, 2),
            num_inference_steps=4,
            initial_noise=torch.zeros(1, 3, 2, 2),
            capture_timesteps=(5,),
        )


def test_ddim_step_rejects_nonfinite_model_output() -> None:
    with pytest.raises((ValueError, FloatingPointError)):
        ddim_step(
            ConstantNoiseModel(float("nan")),
            torch.zeros(1, 3, 2, 2, dtype=torch.float64),
            torch.tensor([1]),
            0,
            tiny_schedule(),
        )
