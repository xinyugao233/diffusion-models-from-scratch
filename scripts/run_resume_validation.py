#!/usr/bin/env python3
"""Run the frozen exact uninterrupted-versus-resumed training validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from _experiment_utils import (
    REPOSITORY_ROOT,
    environment_identity,
    git_identity,
    load_json,
    repository_path,
    sha256_file,
    write_json_exclusive,
)

from diffusion_models.checkpointing import (
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
    states_exactly_equal,
)
from diffusion_models.diffusion.schedules import DDPMSchedule, make_linear_ddpm_schedule
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.models import CIFAR10UNet, smoke_unet_config
from diffusion_models.training import (
    epsilon_prediction_loss,
    optimizer_step,
    sample_epsilon_training_batch,
)


@dataclass
class TrainingObjects:
    model: CIFAR10UNet
    optimizer: torch.optim.AdamW
    ema: ExponentialMovingAverage
    schedule: DDPMSchedule


def build_training_objects(config: dict[str, Any]) -> TrainingObjects:
    torch.manual_seed(config["seeds"]["model"])
    model = CIFAR10UNet(smoke_unet_config())
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
    ema = ExponentialMovingAverage(model, decay=config["ema_decay"])
    schedule = make_linear_ddpm_schedule(**config["schedule"])
    return TrainingObjects(model, optimizer, ema, schedule)


def fixed_synthetic_images(config: dict[str, Any]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config["seeds"]["synthetic_data"])
    return (
        torch.rand(
            (config["batch_size"], 3, 32, 32),
            generator=generator,
        )
        * 2.0
        - 1.0
    )


def seed_training_rng(config: dict[str, Any]) -> None:
    random.seed(config["seeds"]["python"])
    torch.manual_seed(config["seeds"]["torch_training"])


def train_steps(
    objects: TrainingObjects,
    x_start: torch.Tensor,
    *,
    steps: int,
    gradient_clip: float,
) -> tuple[float, list[float]]:
    losses: list[float] = []
    for _ in range(steps):
        random.random()
        batch = sample_epsilon_training_batch(x_start, objects.schedule)
        metrics = optimizer_step(
            objects.model,
            objects.optimizer,
            batch,
            max_gradient_norm=gradient_clip,
        )
        objects.ema.update(objects.model)
        losses.append(metrics.loss)
    return losses[-1], losses


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def next_probe(
    objects: TrainingObjects,
    x_start: torch.Tensor,
) -> dict[str, Any]:
    python_draw = random.random()
    batch = sample_epsilon_training_batch(x_start, objects.schedule)
    loss, _ = epsilon_prediction_loss(objects.model, batch)
    return {
        "python_draw": python_draw,
        "timesteps": batch.timesteps.detach().clone(),
        "noise": batch.target_noise.detach().clone(),
        "loss": float(loss.item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = repository_path(args.config)
    config = load_json(config_path)
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("EXP002 is frozen to CPU float32.")
    if config["model"] != "smoke_unet":
        raise ValueError("EXP002 is frozen to the smoke U-Net.")
    if not 0 < config["interruption_step"] < config["total_steps"]:
        raise ValueError("interruption_step must lie strictly within total_steps.")
    git = git_identity()
    if git["commit"] != config["baseline_git_commit"]:
        raise RuntimeError("Current HEAD differs from the frozen baseline commit.")

    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    x_start = fixed_synthetic_images(config)

    path_a = build_training_objects(config)
    seed_training_rng(config)
    path_a_start = time.perf_counter()
    final_loss_a, losses_a = train_steps(
        path_a,
        x_start,
        steps=config["total_steps"],
        gradient_clip=config["gradient_clip"],
    )
    path_a_duration = time.perf_counter() - path_a_start
    model_state_a = clone_model_state(path_a.model)
    ema_state_a = path_a.ema.state_dict()
    optimizer_state_a = copy.deepcopy(path_a.optimizer.state_dict())
    next_probe_a = next_probe(path_a, x_start)

    path_b = build_training_objects(config)
    seed_training_rng(config)
    path_b_start = time.perf_counter()
    _, first_half_losses = train_steps(
        path_b,
        x_start,
        steps=config["interruption_step"],
        gradient_clip=config["gradient_clip"],
    )
    checkpoint_path = repository_path(config["output"]["checkpoint"])
    save_training_checkpoint(
        checkpoint_path,
        model=path_b.model,
        ema=path_b.ema,
        optimizer=path_b.optimizer,
        global_step=config["interruption_step"],
        configuration=config,
        metadata={
            "baseline_git_commit": git["commit"],
            "created_by": Path(__file__).name,
            "numpy_randomness_used": False,
        },
    )

    checkpoint_rng = capture_rng_state()
    expected_after_checkpoint = next_probe(path_b, x_start)
    restore_rng_state(checkpoint_rng)
    del path_b

    no_rng_restore = build_training_objects(config)
    load_training_checkpoint(
        checkpoint_path,
        model=no_rng_restore.model,
        ema=no_rng_restore.ema,
        optimizer=no_rng_restore.optimizer,
        expected_configuration=config,
        restore_rng=False,
    )
    omitted_rng_probe = next_probe(no_rng_restore, x_start)
    rng_omission_changes_next_draw = not (
        torch.equal(
            expected_after_checkpoint["timesteps"],
            omitted_rng_probe["timesteps"],
        )
        and torch.equal(
            expected_after_checkpoint["noise"],
            omitted_rng_probe["noise"],
        )
    )
    del no_rng_restore

    path_b = build_training_objects(config)
    loaded = load_training_checkpoint(
        checkpoint_path,
        model=path_b.model,
        ema=path_b.ema,
        optimizer=path_b.optimizer,
        expected_configuration=config,
        restore_rng=True,
    )
    remaining_steps = config["total_steps"] - loaded.global_step
    final_loss_b, second_half_losses = train_steps(
        path_b,
        x_start,
        steps=remaining_steps,
        gradient_clip=config["gradient_clip"],
    )
    path_b_duration = time.perf_counter() - path_b_start
    next_probe_b = next_probe(path_b, x_start)

    comparisons = {
        "model_state_exact": states_exactly_equal(
            path_b.model.state_dict(), model_state_a
        ),
        "ema_state_exact": states_exactly_equal(path_b.ema.state_dict(), ema_state_a),
        "optimizer_state_exact": states_exactly_equal(
            path_b.optimizer.state_dict(), optimizer_state_a
        ),
        "global_step_exact": loaded.global_step + remaining_steps
        == config["total_steps"],
        "ema_update_count_exact": path_a.ema.num_updates
        == path_b.ema.num_updates
        == config["total_steps"],
        "final_loss_exact": final_loss_a == final_loss_b,
        "loss_trajectory_exact": losses_a == first_half_losses + second_half_losses,
        "next_python_draw_exact": next_probe_a["python_draw"]
        == next_probe_b["python_draw"],
        "next_timesteps_exact": torch.equal(
            next_probe_a["timesteps"], next_probe_b["timesteps"]
        ),
        "next_noise_exact": torch.equal(next_probe_a["noise"], next_probe_b["noise"]),
        "next_loss_exact": next_probe_a["loss"] == next_probe_b["loss"],
        "rng_omission_changes_next_draw": rng_omission_changes_next_draw,
    }
    passed = all(comparisons.values())
    summary = {
        "checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "global_step": loaded.global_step,
            "sha256": sha256_file(checkpoint_path),
        },
        "command": (
            f".venv/bin/python scripts/{Path(__file__).name} --config {args.config}"
        ),
        "comparisons": comparisons,
        "config_sha256": sha256_file(config_path),
        "environment": environment_identity(),
        "final_loss": final_loss_a,
        "git": git,
        "next_loss": next_probe_a["loss"],
        "next_noise_sha256": tensor_sha256(next_probe_a["noise"]),
        "next_timesteps": next_probe_a["timesteps"].tolist(),
        "numpy_randomness_used": False,
        "passed": passed,
        "path_a_duration_seconds": path_a_duration,
        "path_b_duration_seconds": path_b_duration,
        "record_count": 1,
        "source_hashes": {
            "checkpointing": sha256_file(
                REPOSITORY_ROOT / "src/diffusion_models/checkpointing.py"
            ),
            "ema": sha256_file(REPOSITORY_ROOT / "src/diffusion_models/ema.py"),
            "script": sha256_file(Path(__file__)),
        },
        "steps": {
            "interruption": config["interruption_step"],
            "total": config["total_steps"],
        },
    }
    write_json_exclusive(config["output"]["summary"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
