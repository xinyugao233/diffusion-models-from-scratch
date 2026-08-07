from __future__ import annotations

import torch
from torch import nn

from diffusion_models.diffusion import (
    build_ddpm_schedule,
    make_linear_ddpm_schedule,
    p_mean_variance,
    p_sample,
    p_sample_loop,
    q_posterior_mean_variance,
)
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.models import CIFAR10UNet, smoke_unet_config


class ConstantNoiseModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        self.calls += 1
        return torch.full_like(image, self.value)


def tiny_schedule(*, dtype: torch.dtype = torch.float64):
    return build_ddpm_schedule(torch.tensor([0.1, 0.2], dtype=dtype))


def test_q_posterior_matches_independent_two_step_calculation() -> None:
    schedule = tiny_schedule()
    x_start = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    x_t = torch.tensor([[[[3.0]]]], dtype=torch.float64)
    timesteps = torch.tensor([1], dtype=torch.long)

    posterior = q_posterior_mean_variance(x_start, x_t, timesteps, schedule)

    expected_variance = 0.2 * (1.0 - 0.9) / (1.0 - 0.72)
    expected_x_start_coefficient = (0.9**0.5) * 0.2 / (1.0 - 0.72)
    expected_x_t_coefficient = (0.8**0.5) * (1.0 - 0.9) / (1.0 - 0.72)
    expected_mean = expected_x_start_coefficient * 2.0 + expected_x_t_coefficient * 3.0
    torch.testing.assert_close(
        posterior.variance,
        torch.tensor([[[[expected_variance]]]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        posterior.mean,
        torch.tensor([[[[expected_mean]]]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_q_posterior_timestep_zero_is_safe_and_deterministic() -> None:
    schedule = tiny_schedule()
    x_start = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    x_t = torch.randn_like(x_start)
    timesteps = torch.zeros(2, dtype=torch.long)

    posterior = q_posterior_mean_variance(x_start, x_t, timesteps, schedule)

    torch.testing.assert_close(posterior.mean, x_start, rtol=1e-12, atol=1e-12)
    assert posterior.variance.shape == (2, 1, 1, 1)
    assert torch.equal(posterior.variance, torch.zeros_like(posterior.variance))
    assert torch.isfinite(posterior.mean).all()


def test_q_posterior_broadcasts_finite_nonnegative_variance() -> None:
    schedule = tiny_schedule(dtype=torch.float32)
    x_start = torch.randn(2, 3, 5, 7)
    x_t = torch.randn_like(x_start)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    posterior = q_posterior_mean_variance(x_start, x_t, timesteps, schedule)

    assert posterior.mean.shape == x_t.shape
    assert posterior.variance.shape == (2, 1, 1, 1)
    assert torch.isfinite(posterior.variance).all()
    assert torch.all(posterior.variance >= 0.0)


def test_p_mean_variance_converts_known_epsilon_prediction() -> None:
    schedule = tiny_schedule()
    model = ConstantNoiseModel(value=0.25)
    x_t = torch.tensor([[[[0.4]]]], dtype=torch.float64)
    timesteps = torch.tensor([1], dtype=torch.long)

    prediction = p_mean_variance(
        model,
        x_t,
        timesteps,
        schedule,
        clip_x_start=False,
    )

    alpha_bar = 0.72
    predicted_x_start = (0.4 - (1.0 - alpha_bar) ** 0.5 * 0.25) / alpha_bar**0.5
    coefficient_x_start = 0.9**0.5 * 0.2 / (1.0 - alpha_bar)
    coefficient_x_t = 0.8**0.5 * 0.1 / (1.0 - alpha_bar)
    expected_mean = coefficient_x_start * predicted_x_start + coefficient_x_t * 0.4
    torch.testing.assert_close(
        prediction.predicted_x_start,
        torch.tensor([[[[predicted_x_start]]]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        prediction.mean,
        torch.tensor([[[[expected_mean]]]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert prediction.predicted_noise.shape == x_t.shape


def test_p_mean_variance_clips_predicted_x_start_by_default() -> None:
    schedule = tiny_schedule()
    prediction = p_mean_variance(
        ConstantNoiseModel(),
        torch.full((1, 3, 2, 2), 10.0, dtype=torch.float64),
        torch.tensor([1], dtype=torch.long),
        schedule,
    )

    assert torch.equal(
        prediction.predicted_x_start,
        torch.ones_like(prediction.predicted_x_start),
    )


def test_p_sample_matches_manual_supplied_noise_step() -> None:
    schedule = tiny_schedule()
    model = ConstantNoiseModel(value=0.1)
    x_t = torch.full((1, 3, 2, 2), 0.2, dtype=torch.float64)
    timesteps = torch.tensor([1], dtype=torch.long)
    reverse_noise = torch.full_like(x_t, 0.3)
    prediction = p_mean_variance(model, x_t, timesteps, schedule)

    sampled = p_sample(
        model,
        x_t,
        timesteps,
        schedule,
        reverse_noise=reverse_noise,
    )

    expected = prediction.mean + torch.sqrt(prediction.variance) * reverse_noise
    torch.testing.assert_close(sampled, expected, rtol=1e-12, atol=1e-12)
    assert sampled.shape == x_t.shape


def test_p_sample_timestep_zero_ignores_noise_and_rng() -> None:
    schedule = tiny_schedule()
    model = ConstantNoiseModel(value=0.1)
    x_t = torch.randn(2, 3, 2, 2, dtype=torch.float64)
    timesteps = torch.zeros(2, dtype=torch.long)
    generator = torch.Generator().manual_seed(19)
    initial_generator_state = generator.get_state().clone()

    first = p_sample(
        model,
        x_t,
        timesteps,
        schedule,
        reverse_noise=torch.randn_like(x_t),
    )
    second = p_sample(
        model,
        x_t,
        timesteps,
        schedule,
        reverse_noise=torch.randn_like(x_t),
    )
    generated = p_sample(model, x_t, timesteps, schedule, generator=generator)
    expected = p_mean_variance(model, x_t, timesteps, schedule).mean

    assert torch.equal(first, second)
    assert torch.equal(generated, expected)
    assert torch.equal(generator.get_state(), initial_generator_state)


def test_p_sample_masks_noise_for_mixed_zero_timestep_batch() -> None:
    schedule = tiny_schedule()
    model = ConstantNoiseModel()
    x_t = torch.randn(2, 3, 2, 2, dtype=torch.float64)
    timesteps = torch.tensor([0, 1], dtype=torch.long)
    prediction = p_mean_variance(model, x_t, timesteps, schedule)
    reverse_noise = torch.ones_like(x_t)

    sampled = p_sample(
        model,
        x_t,
        timesteps,
        schedule,
        reverse_noise=reverse_noise,
    )

    assert torch.equal(sampled[0], prediction.mean[0])
    assert not torch.equal(sampled[1], prediction.mean[1])


def test_p_sample_loop_calls_model_once_per_timestep_and_captures_states() -> None:
    schedule = make_linear_ddpm_schedule(num_steps=5)
    model = ConstantNoiseModel()
    initial_noise = torch.zeros(2, 3, 4, 4)

    result = p_sample_loop(
        model,
        schedule,
        initial_noise.shape,
        initial_noise=initial_noise,
        generator=torch.Generator().manual_seed(23),
        capture_timesteps=(5, 3, 0),
    )

    assert model.calls == 5
    assert result.sample.shape == initial_noise.shape
    assert list(result.trajectory) == [5, 3, 0]
    assert torch.equal(result.trajectory[5], initial_noise)
    assert torch.equal(result.trajectory[0], result.sample)


def test_p_sample_loop_is_seeded_and_stochastic() -> None:
    schedule = make_linear_ddpm_schedule(num_steps=4)

    def sample(seed: int) -> torch.Tensor:
        return p_sample_loop(
            ConstantNoiseModel(),
            schedule,
            (2, 3, 4, 4),
            generator=torch.Generator().manual_seed(seed),
        ).sample

    first = sample(29)
    repeated = sample(29)
    changed = sample(31)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)


def test_random_smoke_unet_reverse_loop_is_finite() -> None:
    torch.manual_seed(37)
    model = CIFAR10UNet(smoke_unet_config()).eval()
    schedule = make_linear_ddpm_schedule(num_steps=3)

    result = p_sample_loop(
        model,
        schedule,
        (2, 3, 32, 32),
        generator=torch.Generator().manual_seed(41),
    )

    assert result.sample.shape == (2, 3, 32, 32)
    assert torch.isfinite(result.sample).all()


def test_ema_weighted_model_uses_same_sampling_interface() -> None:
    torch.manual_seed(43)
    training_model = CIFAR10UNet(smoke_unet_config())
    ema = ExponentialMovingAverage(training_model, decay=0.99)
    ema_model = CIFAR10UNet(smoke_unet_config()).eval()
    ema.copy_to(ema_model)
    schedule = make_linear_ddpm_schedule(num_steps=2)

    result = p_sample_loop(
        ema_model,
        schedule,
        (1, 3, 32, 32),
        generator=torch.Generator().manual_seed(47),
    )

    assert result.sample.shape == (1, 3, 32, 32)
    assert torch.isfinite(result.sample).all()


def test_sampling_rejects_invalid_timestep_and_noise_shape() -> None:
    schedule = tiny_schedule()
    model = ConstantNoiseModel()
    x_t = torch.randn(1, 3, 2, 2, dtype=torch.float64)

    try:
        p_sample(
            model,
            x_t,
            torch.tensor([2], dtype=torch.long),
            schedule,
        )
    except IndexError as error:
        assert "Timesteps must be in [0, 1]" in str(error)
    else:
        raise AssertionError("Expected an invalid timestep to fail.")

    try:
        p_sample(
            model,
            x_t,
            torch.tensor([1], dtype=torch.long),
            schedule,
            reverse_noise=torch.randn(1, 3, 2, 3, dtype=torch.float64),
        )
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError("Expected an invalid reverse-noise shape to fail.")
