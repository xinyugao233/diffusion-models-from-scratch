#!/usr/bin/env python3
"""Run the frozen random-model DDPM reverse-sampling smoke experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from _experiment_utils import (
    environment_identity,
    git_identity,
    load_json,
    repository_path,
    sha256_file,
    write_json_exclusive,
)
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.transforms.functional import to_pil_image

from diffusion_models.data import inverse_normalize
from diffusion_models.diffusion import make_linear_ddpm_schedule, p_sample_loop
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.models import CIFAR10UNet, smoke_unet_config


class CountingModel(nn.Module):
    """Count calls while preserving an ordinary model sampling interface."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.model(image, timesteps)


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    return {
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "standard_deviation": float(tensor.std().item()),
        "sha256": tensor_sha256(tensor),
    }


def save_tensor_exclusive(path: Path, tensor: torch.Tensor) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing tensor: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def trajectory_label(timestep: int, total_steps: int) -> str:
    return "x_T" if timestep == total_steps else f"x_{timestep}"


def save_trajectory_exclusive(
    path: Path,
    trajectory: dict[int, torch.Tensor],
    capture_timesteps: list[int],
    *,
    total_steps: int,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {path}")
    if list(trajectory) != capture_timesteps:
        raise ValueError(
            "Trajectory keys differ from the frozen order: "
            f"{list(trajectory)} versus {capture_timesteps}."
        )

    tile_size = 96
    padding = 8
    header_height = 30
    batch_size = next(iter(trajectory.values())).shape[0]
    width = padding + len(capture_timesteps) * (tile_size + padding)
    height = header_height + padding + batch_size * (tile_size + padding)
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)

    for column, timestep in enumerate(capture_timesteps):
        x_position = padding + column * (tile_size + padding)
        label = trajectory_label(timestep, total_steps)
        draw.text((x_position + 4, 7), label, fill=(20, 20, 20), font=font)
        displayed = inverse_normalize(trajectory[timestep], clamp=True).cpu()
        for row, image in enumerate(displayed):
            tile = to_pil_image(image).resize(
                (tile_size, tile_size),
                resample=Image.Resampling.NEAREST,
            )
            y_position = header_height + row * (tile_size + padding)
            canvas.paste(tile, (x_position, y_position))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def validate_config(config: dict[str, Any]) -> None:
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("EXP003/try01 is frozen to CPU float32.")
    if config["model"] != "smoke_unet":
        raise ValueError("EXP003/try01 is frozen to the smoke U-Net.")
    if config["weights"] != "ema_initialized_from_random_model":
        raise ValueError("EXP003/try01 is frozen to an EMA-initialized model copy.")
    expected_captures = [
        config["schedule"]["num_steps"],
        750,
        500,
        250,
        100,
        0,
    ]
    if config["capture_timesteps"] != expected_captures:
        raise ValueError(f"Expected capture_timesteps={expected_captures}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = repository_path(args.config)
    config = load_json(config_path)
    validate_config(config)
    git = git_identity()
    if git["commit"] != config["baseline_git_commit"]:
        raise RuntimeError("Current HEAD differs from the frozen baseline commit.")

    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seeds"]["model"])
    training_model = CIFAR10UNet(smoke_unet_config())
    ema = ExponentialMovingAverage(training_model, decay=0.9999)
    ema_model = CIFAR10UNet(smoke_unet_config()).eval()
    ema.copy_to(ema_model)
    model = CountingModel(ema_model).eval()
    schedule = make_linear_ddpm_schedule(**config["schedule"])
    generator = torch.Generator(device="cpu").manual_seed(config["seeds"]["sampling"])
    shape = (config["batch_size"], 3, 32, 32)

    start = time.perf_counter()
    result = p_sample_loop(
        model,
        schedule,
        shape,
        generator=generator,
        clip_x_start=config["clip_x_start"],
        capture_timesteps=config["capture_timesteps"],
    )
    duration_seconds = time.perf_counter() - start
    expected_calls = config["schedule"]["num_steps"]
    if model.calls != expected_calls:
        raise RuntimeError(f"Expected {expected_calls} model calls, got {model.calls}.")
    if not torch.isfinite(result.sample).all():
        raise FloatingPointError("Final sample contains nonfinite values.")

    final_path = repository_path(config["output"]["final_sample"])
    figure_path = repository_path(config["output"]["figure"])
    summary_path = repository_path(config["output"]["summary"])
    save_tensor_exclusive(final_path, result.sample)
    save_trajectory_exclusive(
        figure_path,
        result.trajectory,
        config["capture_timesteps"],
        total_steps=schedule.num_steps,
    )

    summary = {
        "schema_version": 1,
        "experiment": "EXP003/try01",
        "passed": True,
        "interpretation": (
            "Structural random-model sampling smoke only; no image-quality claim."
        ),
        "git": git,
        "environment": environment_identity(),
        "configuration_path": str(config_path.relative_to(repository_path("."))),
        "configuration_sha256": sha256_file(config_path),
        "model": {
            "name": config["model"],
            "weight_source": config["weights"],
            "parameter_count": sum(
                parameter.numel() for parameter in ema_model.parameters()
            ),
            "model_seed": config["seeds"]["model"],
        },
        "sampling": {
            "seed": config["seeds"]["sampling"],
            "model_calls": model.calls,
            "expected_model_calls": expected_calls,
            "duration_seconds": duration_seconds,
            "clip_x_start": config["clip_x_start"],
            "capture_timesteps": config["capture_timesteps"],
        },
        "final_sample": tensor_summary(result.sample),
        "trajectory": {
            str(timestep): tensor_summary(result.trajectory[timestep])
            for timestep in config["capture_timesteps"]
        },
        "artifacts": {
            "final_sample": str(final_path.relative_to(repository_path("."))),
            "final_sample_file_sha256": sha256_file(final_path),
            "figure": str(figure_path.relative_to(repository_path("."))),
            "figure_sha256": sha256_file(figure_path),
        },
        "source_sha256": {
            "sampling": sha256_file(
                repository_path("src/diffusion_models/diffusion/sampling.py")
            ),
            "script": sha256_file(Path(__file__)),
        },
    }
    write_json_exclusive(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
