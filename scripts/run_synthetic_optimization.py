#!/usr/bin/env python3
"""Run the frozen deterministic synthetic epsilon-prediction gate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

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

from diffusion_models.diffusion.schedules import make_linear_ddpm_schedule
from diffusion_models.models import CIFAR10UNet, SelfAttention2d, smoke_unet_config
from diffusion_models.training import (
    JsonlScalarLogger,
    epsilon_prediction_loss,
    make_epsilon_training_batch,
    optimizer_step,
)


def attention_gradient_diagnostic() -> dict[str, bool]:
    """Check the expected two-step gradient transition of zero-init attention."""
    torch.manual_seed(17)
    attention = SelfAttention2d(32)
    optimizer = torch.optim.SGD(attention.parameters(), lr=0.1)
    image = torch.randn(2, 32, 4, 4)

    optimizer.zero_grad(set_to_none=True)
    attention(image).square().mean().backward()
    first_qkv_nonzero = bool(torch.count_nonzero(attention.qkv.weight.grad).item())
    first_output_nonzero = bool(
        torch.count_nonzero(attention.output_projection.weight.grad).item()
    )
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    attention(image).square().mean().backward()
    second_qkv_nonzero = bool(torch.count_nonzero(attention.qkv.weight.grad).item())
    return {
        "first_qkv_gradient_nonzero": first_qkv_nonzero,
        "first_output_projection_gradient_nonzero": first_output_nonzero,
        "second_qkv_gradient_nonzero": second_qkv_nonzero,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = repository_path(args.config)
    config = load_json(config_path)
    if config["device"] != "cpu" or config["model"] != "smoke_unet":
        raise ValueError("Gate A is frozen to the smoke U-Net on CPU.")

    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["model_seed"])
    model = CIFAR10UNet(smoke_unet_config())
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    schedule_config = config["schedule"]
    schedule = make_linear_ddpm_schedule(**schedule_config)
    generator = torch.Generator().manual_seed(config["synthetic_seed"])
    x_start = (
        torch.rand(
            (config["batch_size"], 3, 32, 32),
            generator=generator,
        )
        * 2.0
        - 1.0
    )
    noise = torch.randn(x_start.shape, generator=generator)
    timesteps = torch.tensor(config["timesteps"], dtype=torch.long)
    batch = make_epsilon_training_batch(
        x_start,
        timesteps,
        schedule,
        noise=noise,
    )

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
    logger = JsonlScalarLogger(repository_path(config["output"]["metrics"]))
    losses: list[float] = []
    gradients: list[float] = []
    start = time.perf_counter()
    for step in range(config["steps"]):
        metrics = optimizer_step(
            model,
            optimizer,
            batch,
            max_gradient_norm=config["gradient_clip"],
        )
        losses.append(metrics.loss)
        gradients.append(metrics.gradient_norm)
        logger.log(
            step,
            loss=metrics.loss,
            gradient_norm=metrics.gradient_norm,
            clipped_gradient_norm=metrics.clipped_gradient_norm,
        )
    duration = time.perf_counter() - start

    final_loss_tensor, _ = epsilon_prediction_loss(model, batch)
    final_loss = float(final_loss_tensor.item())
    parameter_changed = any(
        not torch.equal(initial_parameters[name], parameter)
        for name, parameter in model.named_parameters()
    )
    window = config["acceptance"]["window_size"]
    early_mean = statistics.fmean(losses[:window])
    late_mean = statistics.fmean(losses[-window:])
    ratio = late_mean / early_mean
    attention = attention_gradient_diagnostic()
    passed = (
        parameter_changed
        and ratio <= config["acceptance"]["late_to_early_max_ratio"]
        and not attention["first_qkv_gradient_nonzero"]
        and attention["first_output_projection_gradient_nonzero"]
        and attention["second_qkv_gradient_nonzero"]
    )
    summary = {
        "acceptance": config["acceptance"],
        "attention_diagnostic": attention,
        "attention_in_smoke_model": False,
        "command": f".venv/bin/python scripts/{Path(__file__).name} --config {args.config}",
        "config_sha256": sha256_file(config_path),
        "duration_seconds": duration,
        "early_window_mean": early_mean,
        "environment": environment_identity(),
        "final_gradient_norm": gradients[-1],
        "final_loss": final_loss,
        "first_logged_loss": losses[0],
        "git": git_identity(),
        "initial_gradient_norm": gradients[0],
        "late_to_early_ratio": ratio,
        "late_window_mean": late_mean,
        "model_parameter_count": model.trainable_parameter_count,
        "parameter_changed": parameter_changed,
        "passed": passed,
        "record_count": len(losses),
        "relative_loss_decrease": 1.0 - ratio,
        "repository": str(REPOSITORY_ROOT),
        "training_source_sha256": sha256_file(
            REPOSITORY_ROOT / "src/diffusion_models/training.py"
        ),
    }
    write_json_exclusive(config["output"]["summary"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
