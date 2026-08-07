from __future__ import annotations

import copy
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from diffusion_models.checkpointing import (
    CheckpointError,
    load_training_checkpoint,
    save_training_checkpoint,
    states_exactly_equal,
)
from diffusion_models.diffusion.schedules import make_linear_ddpm_schedule
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.training import optimizer_step, sample_epsilon_training_batch


class TinyNoisePredictor(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_bias = timesteps.to(image.dtype)[:, None, None, None] / 10.0
        return self.convolution(image) + time_bias


def _assert_nested_equal(first: Any, second: Any) -> None:
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second, strict=True):
            _assert_nested_equal(first_item, second_item)
    else:
        assert first == second


def _components(
    seed: int = 61,
) -> tuple[
    TinyNoisePredictor,
    torch.optim.AdamW,
    ExponentialMovingAverage,
]:
    torch.manual_seed(seed)
    model = TinyNoisePredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    return model, optimizer, ema


def test_exact_state_comparison_accepts_equivalent_mapping_subclasses() -> None:
    tensor = torch.tensor([1.0, 2.0])
    ordered = OrderedDict([("tensor", tensor), ("nested", {"step": 3})])
    plain = {"tensor": tensor.clone(), "nested": {"step": 3}}

    assert states_exactly_equal(ordered, plain)
    plain["tensor"][0] = -1.0
    assert not states_exactly_equal(ordered, plain)


def _train_steps(
    model: TinyNoisePredictor,
    optimizer: torch.optim.AdamW,
    ema: ExponentialMovingAverage,
    x_start: torch.Tensor,
    steps: int,
) -> float:
    schedule = make_linear_ddpm_schedule(num_steps=10)
    last_loss = -1.0
    for _ in range(steps):
        batch = sample_epsilon_training_batch(x_start, schedule)
        metrics = optimizer_step(model, optimizer, batch, max_gradient_norm=1.0)
        ema.update(model)
        last_loss = metrics.loss
    return last_loss


def test_checkpoint_round_trip_restores_all_state_and_rng(tmp_path: Path) -> None:
    model, optimizer, ema = _components()
    x_start = torch.randn(2, 3, 4, 4)
    _train_steps(model, optimizer, ema, x_start, 1)
    saved_model = copy.deepcopy(model.state_dict())
    saved_ema = ema.state_dict()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    configuration = {"learning_rate": 0.01, "name": "test"}
    random.seed(67)
    torch.manual_seed(71)
    checkpoint = tmp_path / "round_trip.pt"

    save_training_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        global_step=1,
        configuration=configuration,
        metadata={"commit": "abc123"},
    )
    expected_python = random.random()
    expected_torch = torch.randn(5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    random.seed(73)
    torch.manual_seed(79)

    loaded = load_training_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        expected_configuration=configuration,
    )

    _assert_nested_equal(model.state_dict(), saved_model)
    _assert_nested_equal(ema.state_dict(), saved_ema)
    _assert_nested_equal(optimizer.state_dict(), saved_optimizer)
    assert loaded.global_step == 1
    assert loaded.configuration == configuration
    assert loaded.metadata == {"commit": "abc123"}
    assert random.random() == expected_python
    assert torch.equal(torch.randn(5), expected_torch)


def test_checkpoint_rejects_incomplete_corrupt_and_configuration_mismatch(
    tmp_path: Path,
) -> None:
    model, optimizer, ema = _components()
    incomplete = tmp_path / "incomplete.pt"
    torch.save({"checkpoint_version": 1}, incomplete)
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")

    for path in (incomplete, corrupt):
        with pytest.raises(CheckpointError):
            load_training_checkpoint(
                path,
                model=model,
                ema=ema,
                optimizer=optimizer,
            )

    valid = tmp_path / "valid.pt"
    save_training_checkpoint(
        valid,
        model=model,
        ema=ema,
        optimizer=optimizer,
        global_step=0,
        configuration={"version": 1},
    )
    with pytest.raises(CheckpointError, match="configuration"):
        load_training_checkpoint(
            valid,
            model=model,
            ema=ema,
            optimizer=optimizer,
            expected_configuration={"version": 2},
        )


def test_checkpoint_rejects_incompatible_model(tmp_path: Path) -> None:
    model, optimizer, ema = _components()
    checkpoint = tmp_path / "model.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        global_step=0,
        configuration={},
    )
    incompatible_model = TinyNoisePredictor(channels=1)
    incompatible_optimizer = torch.optim.AdamW(incompatible_model.parameters(), lr=1e-2)
    incompatible_ema = ExponentialMovingAverage(incompatible_model, decay=0.9)

    with pytest.raises(CheckpointError, match="incompatible"):
        load_training_checkpoint(
            checkpoint,
            model=incompatible_model,
            ema=incompatible_ema,
            optimizer=incompatible_optimizer,
        )


def test_short_uninterrupted_and_resumed_trajectories_are_exact(
    tmp_path: Path,
) -> None:
    generator = torch.Generator().manual_seed(83)
    x_start = torch.randn((2, 3, 4, 4), generator=generator)
    configuration = {"total_steps": 4, "interruption_step": 2}

    model_a, optimizer_a, ema_a = _components(seed=89)
    random.seed(97)
    torch.manual_seed(101)
    loss_a = _train_steps(model_a, optimizer_a, ema_a, x_start, 4)
    final_model_a = copy.deepcopy(model_a.state_dict())
    final_ema_a = ema_a.state_dict()
    final_optimizer_a = copy.deepcopy(optimizer_a.state_dict())
    next_draw_a = torch.randn(5)

    model_b, optimizer_b, ema_b = _components(seed=89)
    random.seed(97)
    torch.manual_seed(101)
    _train_steps(model_b, optimizer_b, ema_b, x_start, 2)
    checkpoint = tmp_path / "resume.pt"
    save_training_checkpoint(
        checkpoint,
        model=model_b,
        ema=ema_b,
        optimizer=optimizer_b,
        global_step=2,
        configuration=configuration,
    )
    del model_b, optimizer_b, ema_b

    model_b, optimizer_b, ema_b = _components(seed=89)
    loaded = load_training_checkpoint(
        checkpoint,
        model=model_b,
        ema=ema_b,
        optimizer=optimizer_b,
        expected_configuration=configuration,
    )
    loss_b = _train_steps(model_b, optimizer_b, ema_b, x_start, 2)
    next_draw_b = torch.randn(5)

    assert loaded.global_step == 2
    assert loss_a == loss_b
    _assert_nested_equal(model_b.state_dict(), final_model_a)
    _assert_nested_equal(ema_b.state_dict(), final_ema_a)
    _assert_nested_equal(optimizer_b.state_dict(), final_optimizer_a)
    assert torch.equal(next_draw_a, next_draw_b)
