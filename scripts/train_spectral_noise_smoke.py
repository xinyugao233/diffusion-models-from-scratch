#!/usr/bin/env python3
"""Run one engineering-only weighted-noise smoke condition under Slurm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

import torch
from _experiment_utils import git_identity, load_json, repository_path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from diffusion_models.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from diffusion_models.data import inverse_normalize, load_cifar10
from diffusion_models.diffusion import make_linear_ddpm_schedule, p_sample_loop
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.full_training import (
    DeterministicStepBatchSampler,
    require_slurm_environment,
)
from diffusion_models.models import CIFAR10UNet, primary_unet_config
from diffusion_models.spectral_boundary import load_radial_power
from diffusion_models.spectral_noise import (
    SpectralNoiseConfig,
    build_spectral_noise_schedule,
    make_spectral_epsilon_training_batch,
    p_sample_loop_spectral,
)
from diffusion_models.training import (
    epsilon_prediction_loss,
    gradient_norm,
    sample_epsilon_training_batch,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    job_id = require_slurm_environment(os.environ)
    config_path = repository_path(args.config)
    config = load_json(config_path)
    if args.condition not in config["conditions"]:
        raise ValueError("Unknown smoke condition.")
    git = git_identity()
    if git["working_tree_dirty"] or git["commit"] != os.environ.get("RUN_COMMIT"):
        raise RuntimeError("Smoke training requires the exact clean RUN_COMMIT.")
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke training requires a Slurm-assigned CUDA GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(config["model"]["initialization_seed"])
    torch.manual_seed(config["model"]["initialization_seed"])
    torch.cuda.manual_seed_all(config["training"]["training_noise_seed"] + 1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    dataset = load_cifar10(config["data"]["root"], train=True, download=False)
    if len(dataset) != config["data"]["expected_train_size"]:
        raise RuntimeError("Unexpected CIFAR-10 training size.")
    sampler = DeterministicStepBatchSampler(
        dataset_size=len(dataset),
        batch_size=config["data"]["batch_size"],
        data_seed=config["data"]["data_order_seed"],
        start_step=0,
        end_step=config["training"]["steps"],
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["data"]["num_workers"] > 0,
    )
    model = CIFAR10UNet(primary_unet_config()).to(device)
    initial_model_sha256 = state_sha256(model)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != config["model"]["expected_parameter_count"]
    ):
        raise RuntimeError("Unexpected model parameter count.")
    optimizer_values = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_values["learning_rate"],
        betas=tuple(optimizer_values["betas"]),
        eps=optimizer_values["eps"],
        weight_decay=optimizer_values["weight_decay"],
    )
    ema = ExponentialMovingAverage(model, decay=config["training"]["ema_decay"])
    baseline = make_linear_ddpm_schedule(
        num_steps=config["diffusion"]["num_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )
    condition = config["conditions"][args.condition]
    spectral_schedule = None
    if condition["enabled"]:
        values = config["weighted_noise"]
        spectral_schedule = build_spectral_noise_schedule(
            baseline,
            load_radial_power(repository_path(config["spectrum"])),
            SpectralNoiseConfig(
                tau=condition["tau"],
                weight_floor=values["weight_floor"],
                rho=values["rho"],
                allocation=values["allocation"],
                fft_normalization=values["fft_normalization"],
            ),
        )
    generator = torch.Generator(device=device).manual_seed(
        config["training"]["training_noise_seed"]
    )
    metrics_path = output / "metrics.jsonl"
    step_times = []
    losses = []
    gradient_norms = []
    started = time.perf_counter()
    for step, (images, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        if spectral_schedule is None:
            batch = sample_epsilon_training_batch(images, baseline, generator=generator)
        else:
            batch = make_spectral_epsilon_training_batch(
                images, spectral_schedule, generator=generator
            )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = epsilon_prediction_loss(model, batch)
        loss.backward()
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        before_clip = gradient_norm(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"])
        after_clip = gradient_norm(parameters)
        optimizer.step()
        ema.update(model)
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        values = {
            "step": step,
            "condition": args.condition,
            "ordinary_epsilon_mse": float(loss.detach()),
            "gradient_norm": float(before_clip),
            "clipped_gradient_norm": float(after_clip),
            "step_seconds": step_seconds,
            "wall_clock_seconds": time.perf_counter() - started,
            "examples_seen": step * images.shape[0],
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")
        losses.append(values["ordinary_epsilon_mse"])
        gradient_norms.append(values["gradient_norm"])
        step_times.append(step_seconds)
        if step == 1 or step % 10 == 0:
            print(json.dumps(values, sort_keys=True), flush=True)
    training_wall_clock_seconds = time.perf_counter() - started
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("Smoke model contains nonfinite values.")

    checkpoint_config = {"config": config, "condition": args.condition}
    checkpoint = output / f"checkpoint_step_{config['training']['steps']:06d}.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        global_step=config["training"]["steps"],
        configuration=checkpoint_config,
        metadata={"git_commit": git["commit"], "slurm_job_id": job_id},
        generators={"training": generator},
        cuda_devices=[device.index or 0],
    )
    reload_model = CIFAR10UNet(primary_unet_config()).to(device)
    reload_optimizer = torch.optim.AdamW(reload_model.parameters(), lr=1.0)
    reload_ema = ExponentialMovingAverage(
        reload_model, decay=config["training"]["ema_decay"]
    )
    loaded = load_training_checkpoint(
        checkpoint,
        model=reload_model,
        ema=reload_ema,
        optimizer=reload_optimizer,
        expected_configuration=checkpoint_config,
        map_location=device,
        restore_rng=False,
    )
    if loaded.global_step != config["training"]["steps"] or state_sha256(
        reload_model
    ) != state_sha256(model):
        raise RuntimeError("Checkpoint exact reload validation failed.")

    evaluation_model = copy.deepcopy(model).eval()
    ema.copy_to(evaluation_model)
    sample_generator = torch.Generator(device=device).manual_seed(
        config["sampling"]["seed"]
    )
    shape = (config["sampling"]["count"], 3, 32, 32)
    sample_started = time.perf_counter()
    if spectral_schedule is None:
        samples = p_sample_loop(
            evaluation_model,
            baseline,
            shape,
            device=device,
            generator=sample_generator,
            clip_x_start=config["sampling"]["clip_x_start"],
        ).sample
    else:
        samples = p_sample_loop_spectral(
            evaluation_model,
            spectral_schedule,
            shape,
            device=device,
            generator=sample_generator,
            clip_x_start=config["sampling"]["clip_x_start"],
        ).sample
    torch.cuda.synchronize(device)
    sample_seconds = time.perf_counter() - sample_started
    if samples.shape != shape or not torch.isfinite(samples).all():
        raise RuntimeError("Smoke sampler returned invalid samples.")
    if float(samples.min()) < -1.00001 or float(samples.max()) > 1.00001:
        raise RuntimeError("Final clipped smoke samples lie outside [-1,1].")
    sample_path = output / "samples_step_000100.png"
    save_image(inverse_normalize(samples.cpu(), clamp=True), sample_path, nrow=2)
    warmup = config["training"]["timing_warmup_steps"]
    summary = {
        "schema_version": 1,
        "engineering_only": True,
        "passed": True,
        "condition": args.condition,
        "git": git,
        "slurm_job_id": job_id,
        "system": {
            "gpu": torch.cuda.get_device_name(device),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "initial_model_sha256": initial_model_sha256,
        "steps": config["training"]["steps"],
        "examples_seen": config["training"]["steps"] * config["data"]["batch_size"],
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "all_gradients_finite": all(math.isfinite(value) for value in gradient_norms),
        "final_loss": losses[-1],
        "maximum_gradient_norm": max(gradient_norms),
        "mean_step_seconds_after_warmup": statistics.fmean(step_times[warmup:]),
        "median_step_seconds_after_warmup": statistics.median(step_times[warmup:]),
        "training_wall_clock_seconds": training_wall_clock_seconds,
        "sampling_seconds": sample_seconds,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_reload_exact": True,
        "sample_path": str(sample_path),
        "sample_sha256": file_sha256(sample_path),
        "sample_min": float(samples.min()),
        "sample_max": float(samples.max()),
    }
    if spectral_schedule is not None:
        summary["spectral_schedule"] = {
            "tau": spectral_schedule.config.tau,
            "rho": spectral_schedule.config.rho,
            "allocation": spectral_schedule.config.allocation,
            "beta_min": float(spectral_schedule.betas.min()),
            "beta_max": float(spectral_schedule.betas.max()),
        }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
