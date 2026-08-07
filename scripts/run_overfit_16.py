#!/usr/bin/env python3
"""Run the frozen 16-image CIFAR-10 epsilon-prediction overfit gate."""

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
    md5_file,
    repository_path,
    save_loss_curve,
    sha256_file,
    write_json_exclusive,
    write_loss_csv,
)

from diffusion_models.data import load_cifar10
from diffusion_models.diffusion.schedules import make_linear_ddpm_schedule
from diffusion_models.models import CIFAR10UNet, smoke_unet_config
from diffusion_models.training import (
    JsonlScalarLogger,
    optimizer_step,
    sample_epsilon_training_batch,
)


def prepare_manifest(config: dict, *, download: bool) -> tuple[torch.Tensor, dict]:
    data_root = repository_path(config["data_root"])
    dataset = load_cifar10(data_root, train=True, download=download)
    indices = config["subset_indices"]
    if len(indices) != 16 or len(set(indices)) != 16:
        raise ValueError("The frozen subset must contain exactly 16 unique indices.")

    examples = [dataset[index] for index in indices]
    images = torch.stack([image for image, _ in examples])
    labels = [int(label) for _, label in examples]
    if images.shape != (16, 3, 32, 32):
        raise ValueError(f"Unexpected fixed-subset shape: {tuple(images.shape)}")
    if float(images.min()) < -1.0 or float(images.max()) > 1.0:
        raise ValueError("Normalized CIFAR-10 values must remain in [-1, 1].")

    archive = data_root / "cifar-10-python.tar.gz"
    manifest = {
        "archive_md5": md5_file(archive) if archive.exists() else None,
        "augmentation": "none",
        "class_names": list(dataset.classes),
        "dataset": "CIFAR-10",
        "download_requested": download,
        "examples": [
            {
                "class_name": dataset.classes[label],
                "index": index,
                "label": label,
            }
            for index, label in zip(indices, labels, strict=True)
        ],
        "normalized_max": float(images.max()),
        "normalized_min": float(images.min()),
        "split": "train",
    }
    manifest_path = repository_path(config["manifest"])
    if manifest_path.exists():
        if load_json(manifest_path) != manifest:
            raise RuntimeError("Existing manifest differs from the frozen subset.")
    else:
        write_json_exclusive(manifest_path, manifest)
    return images, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config_path = repository_path(args.config)
    config = load_json(config_path)
    if config["device"] != "cpu" or config["model"] != "smoke_unet":
        raise ValueError("Gate B is frozen to the smoke U-Net on CPU.")
    current_git = git_identity()
    if current_git["commit"] != config["baseline_git_commit"]:
        raise RuntimeError("Current HEAD differs from the frozen baseline commit.")

    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    x_start, manifest = prepare_manifest(config, download=args.download)

    torch.manual_seed(config["model_seed"])
    model = CIFAR10UNet(smoke_unet_config())
    schedule = make_linear_ddpm_schedule(**config["schedule"])
    generator = torch.Generator().manual_seed(config["sampling_seed"])
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
    clipped_gradients: list[float] = []
    start = time.perf_counter()
    for step in range(config["steps"]):
        batch = sample_epsilon_training_batch(
            x_start,
            schedule,
            generator=generator,
        )
        metrics = optimizer_step(
            model,
            optimizer,
            batch,
            max_gradient_norm=config["gradient_clip"],
        )
        losses.append(metrics.loss)
        gradients.append(metrics.gradient_norm)
        clipped_gradients.append(metrics.clipped_gradient_norm)
        logger.log(
            step,
            loss=metrics.loss,
            gradient_norm=metrics.gradient_norm,
            clipped_gradient_norm=metrics.clipped_gradient_norm,
            timestep_min=int(batch.timesteps.min().item()),
            timestep_max=int(batch.timesteps.max().item()),
        )
    duration = time.perf_counter() - start

    window = config["acceptance"]["window_size"]
    early_mean = statistics.fmean(losses[:window])
    late_mean = statistics.fmean(losses[-window:])
    ratio = late_mean / early_mean
    passed = ratio <= config["acceptance"]["late_to_early_max_ratio"]
    loss_csv = write_loss_csv(config["output"]["loss_csv"], losses)
    curve = save_loss_curve(config["output"]["curve"], losses)
    summary = {
        "acceptance": config["acceptance"],
        "command": (
            f".venv/bin/python scripts/{Path(__file__).name} "
            f"--config {args.config} --download"
        ),
        "completed_normally": True,
        "config_sha256": sha256_file(config_path),
        "curve": str(curve.relative_to(REPOSITORY_ROOT)),
        "duration_seconds": duration,
        "early_window_mean": early_mean,
        "environment": environment_identity(),
        "final_gradient_norm": gradients[-1],
        "final_loss": losses[-1],
        "git": current_git,
        "gradient_norm_max": max(gradients),
        "gradient_norm_mean": statistics.fmean(gradients),
        "gradient_norm_min": min(gradients),
        "initial_gradient_norm": gradients[0],
        "initial_loss": losses[0],
        "instability_or_nan": False,
        "late_to_early_ratio": ratio,
        "late_window_mean": late_mean,
        "loss_csv": str(loss_csv.relative_to(REPOSITORY_ROOT)),
        "manifest": manifest,
        "model_parameter_count": model.trainable_parameter_count,
        "passed": passed,
        "record_count": len(losses),
        "relative_loss_decrease": 1.0 - ratio,
        "training_source_sha256": sha256_file(
            REPOSITORY_ROOT / "src/diffusion_models/training.py"
        ),
    }
    write_json_exclusive(config["output"]["summary"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
