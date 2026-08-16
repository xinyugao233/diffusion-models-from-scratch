#!/usr/bin/env python3
"""Run the frozen EXP005 DDPM/DDIM checkpoint comparison under Slurm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from _experiment_utils import (
    git_identity,
    load_json,
    repository_path,
    sha256_file,
    write_json_exclusive,
)
from PIL import Image, ImageDraw
from torch import nn
from torchvision.utils import save_image

from diffusion_models.checkpointing import load_training_checkpoint
from diffusion_models.data import inverse_normalize
from diffusion_models.diffusion import (
    ddim_sample_loop,
    ddim_timesteps,
    make_linear_ddpm_schedule,
    p_sample_loop,
)
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.full_training import (
    fixed_initial_noise,
    require_slurm_environment,
)
from diffusion_models.models import CIFAR10UNet, primary_unet_config


class CountingModel(nn.Module):
    """Count denoiser evaluations without changing the wrapped model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.model(image, timesteps)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def save_tensor_exclusive(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite tensor: {path}")
    with path.open("xb") as handle:
        torch.save(tensor.detach().cpu(), handle)


def validate_config(config: dict[str, Any]) -> None:
    expected = [
        ("ddpm_1000", "ddpm", 1000),
        ("ddim_100", "ddim", 100),
        ("ddim_50", "ddim", 50),
        ("ddim_25", "ddim", 25),
    ]
    actual = [
        (setting["name"], setting["sampler"], setting["model_evaluations"])
        for setting in config["settings"]
    ]
    if actual != expected:
        raise ValueError(f"Comparison settings changed: {actual!r}.")
    evaluation = config["evaluation"]
    if evaluation["fixed_initial_noise_seeds"] != list(range(1000, 1016)):
        raise ValueError("Fixed initial-noise seeds changed.")
    if evaluation["timed_repeats"] != 3 or evaluation["sample_count"] != 16:
        raise ValueError("EXP005 requires 16 samples and three repeats.")
    if config["diffusion"] != {
        "beta_end": 0.02,
        "beta_start": 0.0001,
        "num_steps": 1000,
        "schedule": "linear",
    }:
        raise ValueError("Training diffusion schedule changed.")


def system_identity(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_index": index,
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_total_memory_bytes": properties.total_memory,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def make_headline_figure(path: Path, grid_paths: list[Path], labels: list[str]) -> None:
    grids = [Image.open(grid_path).convert("RGB") for grid_path in grid_paths]
    scale = 2
    grids = [grid.resize((grid.width * scale, grid.height * scale)) for grid in grids]
    label_height = 40
    canvas = Image.new(
        "RGB",
        (sum(grid.width for grid in grids), grids[0].height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for grid, label in zip(grids, labels, strict=True):
        draw.text((x + 10, 12), label, fill="black")
        canvas.paste(grid, (x, label_height))
        x += grid.width
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {path}")
    canvas.save(path)


def make_runtime_figure(path: Path, records: list[dict[str, Any]]) -> None:
    width, height = 1000, 420
    left, right, top, row_height = 180, 50, 60, 75
    maximum = max(record["median_seconds"] for record in records)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), "Median synchronized trajectory runtime", fill="black")
    for index, record in enumerate(records):
        y = top + index * row_height
        bar_width = (width - left - right) * record["median_seconds"] / maximum
        draw.text((20, y + 10), record["name"], fill="black")
        draw.rectangle((left, y, left + bar_width, y + 35), fill="#2563eb")
        label = f"{record['median_seconds']:.3f}s | {record['model_evaluations']} evals"
        label_box = draw.textbbox((0, 0), label)
        label_width = label_box[2] - label_box[0]
        label_x = left + bar_width + 8
        label_fill = "black"
        if label_x + label_width > width - right:
            label_x = left + bar_width - label_width - 8
            label_fill = "white"
        draw.text(
            (label_x, y + 10),
            label,
            fill=label_fill,
        )
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {path}")
    image.save(path)


def main() -> int:
    require_slurm_environment(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config_path = repository_path(args.config)
    config = load_json(config_path)
    validate_config(config)
    training_config_path = repository_path(config["training_config"]["path"])
    if (
        sha256_file(training_config_path)
        != config["training_config"]["expected_sha256"]
    ):
        raise RuntimeError("Full-training configuration hash mismatch.")
    checkpoint_path = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint_path) != config["checkpoint"]["expected_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch.")

    git = git_identity()
    run_commit = os.environ.get("RUN_COMMIT", "")
    if not run_commit or git["commit"] != run_commit or git["working_tree_dirty"]:
        raise RuntimeError("Execution checkout must be clean and equal RUN_COMMIT.")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP005 requires a Slurm-assigned CUDA GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    system = system_identity(device)
    if config["hardware"]["required_gpu_substring"] not in system["gpu_name"]:
        raise RuntimeError(f"EXP005 requires an H100, found {system['gpu_name']}.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    training_config = load_json(training_config_path)
    model = CIFAR10UNet(primary_unet_config())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != config["model"]["expected_parameter_count"]:
        raise RuntimeError(f"Unexpected model parameter count: {parameter_count}.")
    optimizer_config = training_config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
    ema = ExponentialMovingAverage(
        model, decay=training_config["training"]["ema_decay"]
    )
    training_generator = torch.Generator().manual_seed(
        training_config["training"]["training_noise_seed"]
    )
    loaded = load_training_checkpoint(
        checkpoint_path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        expected_configuration={"config": training_config, "stage": "full"},
        generators={"training": training_generator},
        map_location="cpu",
        restore_rng=False,
    )
    if loaded.global_step != config["checkpoint"]["expected_global_step"]:
        raise RuntimeError(f"Unexpected checkpoint step: {loaded.global_step}.")
    ema.copy_to(model)
    del ema, optimizer, training_generator
    model = model.to(device).eval()
    counted_model = CountingModel(model).eval()

    diffusion = config["diffusion"]
    schedule = make_linear_ddpm_schedule(
        num_steps=diffusion["num_steps"],
        beta_start=diffusion["beta_start"],
        beta_end=diffusion["beta_end"],
        device=device,
    )
    evaluation = config["evaluation"]
    initial_noise = fixed_initial_noise(evaluation["fixed_initial_noise_seeds"]).to(
        device
    )
    output_dir = Path(args.output_dir).resolve()
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        counted_model(
            initial_noise, torch.full((16,), 999, device=device, dtype=torch.long)
        )
    torch.cuda.synchronize(device)
    counted_model.calls = 0

    records: list[dict[str, Any]] = []
    grid_paths: list[Path] = []
    for setting in config["settings"]:
        timings: list[float] = []
        output_hashes: list[str] = []
        first_output: torch.Tensor | None = None
        call_counts: list[int] = []
        for _ in range(evaluation["timed_repeats"]):
            counted_model.calls = 0
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if setting["sampler"] == "ddpm":
                generator = torch.Generator(device=device).manual_seed(
                    evaluation["ddpm_reverse_noise_seed"]
                )
                result = p_sample_loop(
                    counted_model,
                    schedule,
                    initial_noise.shape,
                    initial_noise=initial_noise,
                    generator=generator,
                    clip_x_start=evaluation["clip_x_start"],
                )
            else:
                result = ddim_sample_loop(
                    counted_model,
                    schedule,
                    initial_noise.shape,
                    num_inference_steps=setting["model_evaluations"],
                    initial_noise=initial_noise,
                    clip_x_start=evaluation["clip_x_start"],
                )
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - started)
            call_counts.append(counted_model.calls)
            output_hashes.append(tensor_sha256(result.sample))
            if first_output is None:
                first_output = result.sample.detach().cpu()
        expected_calls = setting["model_evaluations"]
        if call_counts != [expected_calls] * evaluation["timed_repeats"]:
            raise RuntimeError(f"Unexpected model call counts for {setting['name']}.")
        if len(set(output_hashes)) != 1:
            raise RuntimeError(f"Repeated outputs differ for {setting['name']}.")
        if first_output is None or tuple(first_output.shape) != (16, 3, 32, 32):
            raise RuntimeError(f"Unexpected sample shape for {setting['name']}.")
        if not torch.isfinite(first_output).all():
            raise FloatingPointError(f"Nonfinite sample for {setting['name']}.")

        tensor_path = result_dir / f"{setting['name']}_samples.pt"
        grid_path = figure_dir / f"{setting['name']}_grid.png"
        save_tensor_exclusive(tensor_path, first_output)
        save_image(
            inverse_normalize(first_output).clamp(0.0, 1.0),
            grid_path,
            nrow=4,
            padding=2,
        )
        median = statistics.median(timings)
        records.append(
            {
                "all_outputs_bitwise_equal": True,
                "finite": True,
                "images_per_second": evaluation["sample_count"] / median,
                "median_seconds": median,
                "model_evaluations": expected_calls,
                "model_evaluations_per_repeat": call_counts,
                "name": setting["name"],
                "output_sha256": output_hashes[0],
                "sample_shape": list(first_output.shape),
                "sampler": setting["sampler"],
                "timed_seconds": timings,
                "timesteps": (
                    list(ddim_timesteps(schedule.num_steps, expected_calls))
                    if setting["sampler"] == "ddim"
                    else list(reversed(range(schedule.num_steps)))
                ),
            }
        )
        grid_paths.append(grid_path)

    baseline_median = records[0]["median_seconds"]
    for record in records:
        record["speedup_vs_ddpm_1000"] = baseline_median / record["median_seconds"]
    if any(record["median_seconds"] >= baseline_median for record in records[1:]):
        raise RuntimeError("At least one DDIM median was not faster than DDPM-1000.")

    make_headline_figure(
        figure_dir / "ddpm_ddim_headline.png",
        grid_paths,
        ["DDPM-1000", "DDIM-100", "DDIM-50", "DDIM-25"],
    )
    make_runtime_figure(figure_dir / "runtime_comparison.png", records)
    comparison = {
        "checkpoint_global_step": loaded.global_step,
        "checkpoint_sha256": config["checkpoint"]["expected_sha256"],
        "config_sha256": sha256_file(config_path),
        "initial_noise_seeds": evaluation["fixed_initial_noise_seeds"],
        "records": records,
        "run_commit": run_commit,
        "system": system,
        "training_config_sha256": sha256_file(training_config_path),
    }
    comparison_path = write_json_exclusive(result_dir / "comparison.json", comparison)
    manifest = {
        "artifacts": {
            str(path.relative_to(output_dir)): sha256_file(path)
            for path in sorted([*result_dir.glob("*"), *figure_dir.glob("*")])
            if path != comparison_path
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": config["checkpoint"]["expected_sha256"],
        "comparison_sha256": sha256_file(comparison_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "run_commit": run_commit,
        "system": system,
    }
    write_json_exclusive(output_dir / "manifest.json", manifest)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
