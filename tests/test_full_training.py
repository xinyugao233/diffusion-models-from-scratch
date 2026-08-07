from __future__ import annotations

import json
import math

import pytest
import torch

from diffusion_models.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from diffusion_models.diffusion import make_linear_ddpm_schedule
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.full_training import (
    AppendOnlyJsonlLogger,
    DeterministicStepBatchSampler,
    checkpoint_steps,
    fixed_initial_noise,
    production_train_step,
    require_slurm_environment,
    sampling_steps,
)
from diffusion_models.models import CIFAR10UNet, smoke_unet_config


def test_step_batch_sampler_resume_matches_uninterrupted_order() -> None:
    complete = list(
        DeterministicStepBatchSampler(
            dataset_size=10,
            batch_size=4,
            data_seed=7,
            start_step=0,
            end_step=7,
        )
    )
    first = list(
        DeterministicStepBatchSampler(
            dataset_size=10,
            batch_size=4,
            data_seed=7,
            start_step=0,
            end_step=3,
        )
    )
    resumed = list(
        DeterministicStepBatchSampler(
            dataset_size=10,
            batch_size=4,
            data_seed=7,
            start_step=3,
            end_step=7,
        )
    )

    assert complete == first + resumed
    assert len(complete) == 7
    assert all(len(batch) == 4 and len(set(batch)) == 4 for batch in complete)


def test_production_step_updates_model_then_ema() -> None:
    torch.manual_seed(11)
    model = CIFAR10UNet(smoke_unet_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    schedule = make_linear_ddpm_schedule(num_steps=10)
    images = torch.rand(2, 3, 32, 32) * 2.0 - 1.0
    before = next(model.parameters()).detach().clone()
    events: list[str] = []

    metrics = production_train_step(
        model,
        optimizer,
        ema,
        images,
        schedule,
        generator=torch.Generator().manual_seed(13),
        max_gradient_norm=1.0,
        event_callback=events.append,
    )

    assert events == ["zero_grad", "backward", "optimizer_step", "ema_update"]
    assert not torch.equal(next(model.parameters()), before)
    assert ema.num_updates == 1
    assert metrics.ema_num_updates == 1
    assert all(
        math.isfinite(value)
        for value in (
            metrics.loss,
            metrics.gradient_norm,
            metrics.clipped_gradient_norm,
        )
    )
    assert metrics.clipped_gradient_norm <= 1.0 + 1e-6


def test_checkpoint_and_sampling_cadence() -> None:
    full = {"max_steps": 50_000, "checkpoint_every": 5_000, "sample_count": 16}
    gate = {
        "max_steps": 500,
        "checkpoint_steps": [250, 500],
        "sample_steps": [500],
        "sample_count": 4,
    }
    evaluation = {"sample_steps": [10_000, 25_000, 50_000]}

    assert checkpoint_steps(full) == set(range(5_000, 50_001, 5_000))
    assert checkpoint_steps(gate) == {250, 500}
    assert sampling_steps("full", full, evaluation) == {10_000, 25_000, 50_000}
    assert sampling_steps("gate_b", gate, evaluation) == {500}


def test_fixed_initial_noise_preserves_each_seed() -> None:
    first = fixed_initial_noise([17, 19], image_shape=(3, 4, 4))
    repeated = fixed_initial_noise([17, 19], image_shape=(3, 4, 4))
    changed = fixed_initial_noise([17, 23], image_shape=(3, 4, 4))

    assert torch.equal(first, repeated)
    assert torch.equal(first[0], changed[0])
    assert not torch.equal(first[1], changed[1])


def test_append_only_logger_validates_resume_boundary(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = AppendOnlyJsonlLogger(path, expected_last_step=0)
    logger.log(1, loss=1.0)
    logger.log(2, loss=0.5)

    resumed = AppendOnlyJsonlLogger(path, expected_last_step=2)
    resumed.log(3, loss=0.25)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3]

    with pytest.raises(ValueError, match="resume boundary"):
        AppendOnlyJsonlLogger(path, expected_last_step=1)


def test_resume_wiring_restores_step_config_and_continues(tmp_path) -> None:
    config = {"experiment": "test", "batch_size": 2}
    schedule = make_linear_ddpm_schedule(num_steps=10)
    images = torch.rand(2, 3, 32, 32) * 2.0 - 1.0

    def build():
        torch.manual_seed(29)
        model = CIFAR10UNet(smoke_unet_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ema = ExponentialMovingAverage(model, decay=0.9)
        generator = torch.Generator().manual_seed(31)
        return model, optimizer, ema, generator

    model, optimizer, ema, generator = build()
    production_train_step(
        model,
        optimizer,
        ema,
        images,
        schedule,
        generator=generator,
        max_gradient_norm=1.0,
    )
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        ema=ema,
        global_step=1,
        configuration=config,
        generators={"training": generator},
    )

    model, optimizer, ema, generator = build()
    loaded = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        ema=ema,
        expected_configuration=config,
        generators={"training": generator},
    )
    metrics = production_train_step(
        model,
        optimizer,
        ema,
        images,
        schedule,
        generator=generator,
        max_gradient_norm=1.0,
    )

    assert loaded.global_step == 1
    assert loaded.configuration == config
    assert metrics.ema_num_updates == 2


def test_production_entrypoint_requires_slurm() -> None:
    with pytest.raises(RuntimeError, match="outside Slurm"):
        require_slurm_environment({})
    assert require_slurm_environment({"SLURM_JOB_ID": "12345"}) == "12345"
