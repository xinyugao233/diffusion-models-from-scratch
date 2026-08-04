import torch

from diffusion_models.diffusion.ddpm import predict_x_start_from_noise, q_sample
from diffusion_models.diffusion.schedules import (
    build_ddpm_schedule,
    make_linear_ddpm_schedule,
)


def test_schedule_matches_hand_computed_values() -> None:
    betas = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    schedule = build_ddpm_schedule(betas)

    expected_alphas = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
    expected_alpha_bars = torch.tensor([0.9, 0.72, 0.504], dtype=torch.float64)
    torch.testing.assert_close(schedule.alphas, expected_alphas, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        schedule.alpha_bars, expected_alpha_bars, rtol=0.0, atol=1e-12
    )
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])


def test_q_sample_matches_hand_computed_formula() -> None:
    dtype = torch.float64
    schedule = build_ddpm_schedule(torch.tensor([0.1, 0.2, 0.3], dtype=dtype))
    x_start = torch.ones((2, 1, 2, 2), dtype=dtype)
    noise = torch.full_like(x_start, fill_value=2.0)
    timesteps = torch.tensor([0, 2], dtype=torch.long)

    actual_x_t = q_sample(x_start, timesteps, schedule, noise)

    expected_alpha_bars = torch.tensor([0.9, 0.504], dtype=dtype).reshape(2, 1, 1, 1)
    expected_x_t = (
        torch.sqrt(expected_alpha_bars) * x_start
        + torch.sqrt(1.0 - expected_alpha_bars) * noise
    )
    torch.testing.assert_close(actual_x_t, expected_x_t, rtol=0.0, atol=1e-12)


def test_q_sample_can_be_algebraically_inverted() -> None:
    generator = torch.Generator().manual_seed(7)
    dtype = torch.float64
    schedule = make_linear_ddpm_schedule(
        num_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        dtype=dtype,
    )
    x_start = torch.randn((8, 3, 32, 32), generator=generator, dtype=dtype)
    known_noise = torch.randn(
        x_start.shape,
        generator=generator,
        dtype=dtype,
    )
    timesteps = torch.tensor([0, 1, 10, 100, 250, 500, 750, 999], dtype=torch.long)

    x_t = q_sample(x_start, timesteps, schedule, known_noise)
    reconstructed = predict_x_start_from_noise(x_t, timesteps, known_noise, schedule)

    torch.testing.assert_close(reconstructed, x_start, rtol=1e-10, atol=1e-10)
