#!/usr/bin/env python3
"""Train one paired Study 2 condition with resumable, audited checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from _experiment_utils import git_identity, load_json, repository_path, sha256_file
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
    AppendOnlyJsonlLogger,
    DeterministicStepBatchSampler,
    ema_tensors_are_finite,
    model_tensors_are_finite,
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
from diffusion_models.spectral_noise_study import (
    resolve_spectral_noise_study_configuration,
    state_sha256,
)
from diffusion_models.training import (
    epsilon_prediction_loss,
    gradient_norm,
    sample_epsilon_training_batch,
)


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def build_schedules(config: dict[str, Any], device: torch.device):
    baseline = make_linear_ddpm_schedule(
        num_steps=config["diffusion"]["num_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )
    condition = config["spectral_noise_condition"]
    if not condition["enabled"]:
        return baseline, None
    weighted = config["weighted_noise"]
    spectral = build_spectral_noise_schedule(
        baseline,
        load_radial_power(repository_path(config["spectrum"])),
        SpectralNoiseConfig(
            tau=float(condition["tau"]),
            weight_floor=float(weighted["weight_floor"]),
            rho=float(weighted["rho"]),
            allocation=weighted["allocation"],
            fft_normalization=weighted["fft_normalization"],
        ),
    )
    return baseline, spectral


@torch.no_grad()
def validation_sample(
    model: CIFAR10UNet,
    ema: ExponentialMovingAverage,
    baseline,
    spectral,
    *,
    seed: int,
    device: torch.device,
    destination: Path,
) -> None:
    evaluation_model = copy.deepcopy(model).eval()
    ema.copy_to(evaluation_model)
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (1, 3, 32, 32)
    if spectral is None:
        samples = p_sample_loop(
            evaluation_model,
            baseline,
            shape,
            device=device,
            generator=generator,
            clip_x_start=True,
        ).sample
    else:
        samples = p_sample_loop_spectral(
            evaluation_model,
            spectral,
            shape,
            device=device,
            generator=generator,
            clip_x_start=True,
        ).sample
    if samples.shape != shape or not torch.isfinite(samples).all():
        raise RuntimeError("Validation sampler returned invalid samples.")
    save_image(inverse_normalize(samples.cpu(), clamp=True), destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-end", required=True, type=int)
    parser.add_argument("--resume")
    parser.add_argument("--validation-sample", action="store_true")
    args = parser.parse_args()
    job_id = require_slurm_environment(os.environ)
    config_path = repository_path(args.config)
    source = load_json(config_path)
    config = resolve_spectral_noise_study_configuration(
        source, condition=args.condition, seed=args.seed
    )
    total_steps = int(config["training"]["steps"])
    if not 0 < args.segment_end <= total_steps:
        raise ValueError("segment-end must lie within the frozen training horizon.")

    git = git_identity()
    submitted_commit = os.environ.get("RUN_COMMIT", "")
    if git["working_tree_dirty"] or git["commit"] != submitted_commit:
        raise RuntimeError("Training requires the exact clean RUN_COMMIT.")
    if not torch.cuda.is_available():
        raise RuntimeError("Study 2 training requires a Slurm-assigned CUDA GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(
        bool(config["training"]["deterministic_algorithms"])
    )
    torch.backends.cudnn.benchmark = False
    random.seed(config["model"]["initialization_seed"])
    torch.manual_seed(config["model"]["initialization_seed"])
    torch.cuda.manual_seed_all(config["training"]["training_noise_seed"] + 1)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    result_dir = output / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    dataset = load_cifar10(config["data"]["root"], train=True, download=False)
    if len(dataset) != int(config["data"]["expected_train_size"]):
        raise RuntimeError("Unexpected CIFAR-10 training size.")

    model = CIFAR10UNet(primary_unet_config()).to(device)
    initial_model_sha256 = state_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["model"]["expected_parameter_count"]):
        raise RuntimeError("Unexpected model parameter count.")
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
    ema = ExponentialMovingAverage(model, decay=config["training"]["ema_decay"])
    baseline, spectral = build_schedules(config, device)
    generator = torch.Generator(device=device).manual_seed(
        config["training"]["training_noise_seed"]
    )
    checkpoint_configuration = {
        "config": config,
        "condition": args.condition,
        "seed": args.seed,
    }
    start_step = 0
    resume_metadata = None
    if args.resume:
        loaded = load_training_checkpoint(
            args.resume,
            model=model,
            ema=ema,
            optimizer=optimizer,
            expected_configuration=checkpoint_configuration,
            generators={"training": generator},
            map_location=device,
            restore_rng=True,
        )
        start_step = loaded.global_step
        resume_metadata = loaded.metadata
    if not start_step < args.segment_end:
        raise ValueError("Resume step must be below segment-end.")

    if start_step and metrics_path.exists():
        metric_lines = [
            line
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(metric_lines) < start_step:
            raise RuntimeError("Metric log ends before the resume checkpoint.")
        if len(metric_lines) > start_step:
            interrupted_path = output / (
                f"metrics_interrupted_through_{len(metric_lines):06d}_"
                f"job_{job_id}.jsonl"
            )
            metrics_path.replace(interrupted_path)
            metrics_path.write_text(
                "\n".join(metric_lines[:start_step]) + "\n", encoding="utf-8"
            )

    logger = AppendOnlyJsonlLogger(metrics_path, expected_last_step=start_step)
    sampler = DeterministicStepBatchSampler(
        dataset_size=len(dataset),
        batch_size=config["data"]["batch_size"],
        data_seed=config["data"]["data_order_seed"],
        start_step=start_step,
        end_step=args.segment_end,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["data"]["num_workers"] > 0,
    )
    due_checkpoints = {
        int(step) for step in config["training"]["checkpoint_steps"]
    }
    losses: list[float] = []
    gradients: list[float] = []
    step_times: list[float] = []
    checkpoint_records = []
    started = time.perf_counter()
    model.train()
    for batch_index, (images, _) in enumerate(loader, start=1):
        step = start_step + batch_index
        images = images.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        if spectral is None:
            batch = sample_epsilon_training_batch(images, baseline, generator=generator)
        else:
            batch = make_spectral_epsilon_training_batch(
                images, spectral, generator=generator
            )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = epsilon_prediction_loss(model, batch)
        loss.backward()
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        before_clip = gradient_norm(parameters)
        torch.nn.utils.clip_grad_norm_(
            parameters, float(config["training"]["gradient_clip"])
        )
        after_clip = gradient_norm(parameters)
        optimizer.step()
        ema.update(model)
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        record = {
            "condition": args.condition,
            "seed": args.seed,
            "ordinary_epsilon_mse": float(loss.detach()),
            "gradient_norm": float(before_clip),
            "clipped_gradient_norm": float(after_clip),
            "step_seconds": step_seconds,
            "wall_clock_seconds": time.perf_counter() - started,
            "examples_seen": step * images.shape[0],
        }
        logger.log(step, **record)
        losses.append(record["ordinary_epsilon_mse"])
        gradients.append(record["gradient_norm"])
        step_times.append(step_seconds)
        if step == 1 or step % 100 == 0 or step == args.segment_end:
            print(json.dumps({"step": step, **record}, sort_keys=True), flush=True)
        if step in due_checkpoints or step == args.segment_end:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                global_step=step,
                configuration=checkpoint_configuration,
                metadata={
                    "git_commit": git["commit"],
                    "slurm_job_id": job_id,
                    "configuration_sha256": sha256_file(config_path),
                    "initial_model_sha256": initial_model_sha256,
                },
                generators={"training": generator},
                cuda_devices=[device.index or 0],
            )
            checkpoint_records.append(
                {
                    "step": step,
                    "path": str(checkpoint_path),
                    "sha256": file_sha256(checkpoint_path),
                }
            )

    final_checkpoint = checkpoint_dir / f"checkpoint_step_{args.segment_end:06d}.pt"
    reload_model = CIFAR10UNet(primary_unet_config()).to(device)
    reload_optimizer = torch.optim.AdamW(reload_model.parameters(), lr=1.0)
    reload_ema = ExponentialMovingAverage(
        reload_model, decay=config["training"]["ema_decay"]
    )
    loaded = load_training_checkpoint(
        final_checkpoint,
        model=reload_model,
        ema=reload_ema,
        optimizer=reload_optimizer,
        expected_configuration=checkpoint_configuration,
        map_location=device,
        restore_rng=False,
    )
    checkpoint_reload_exact = (
        loaded.global_step == args.segment_end
        and state_sha256(reload_model) == state_sha256(model)
    )
    if not checkpoint_reload_exact:
        raise RuntimeError("Final checkpoint exact reload validation failed.")
    sample_path = None
    if args.validation_sample:
        sample_path = result_dir / f"validation_sample_step_{args.segment_end:06d}.png"
        validation_sample(
            model,
            ema,
            baseline,
            spectral,
            seed=40000,
            device=device,
            destination=sample_path,
        )
    if not model_tensors_are_finite(model) or not ema_tensors_are_finite(ema):
        raise FloatingPointError("Model or EMA state contains nonfinite values.")
    warmup = min(int(config["training"]["timing_warmup_steps"]), len(step_times) - 1)
    measured_times = step_times[warmup:]
    spectral_summary = None
    if spectral is not None:
        spectral_summary = {
            "tau": spectral.config.tau,
            "rho": spectral.config.rho,
            "beta_min": float(spectral.betas.min()),
            "beta_max": float(spectral.betas.max()),
            "terminal_max_abs_difference": float(
                (spectral.alpha_bars[-1] - baseline.alpha_bars[-1]).abs().max()
            ),
            "monotone": bool(
                torch.all(spectral.alpha_bars[1:] <= spectral.alpha_bars[:-1])
            ),
        }
    summary = {
        "schema_version": 1,
        "passed": True,
        "experiment": config["experiment"],
        "condition": args.condition,
        "seed": args.seed,
        "git": git,
        "submitted_commit": submitted_commit,
        "slurm_job_id": job_id,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output_dir": str(output),
        "segment": {"start_step": start_step, "end_step": args.segment_end},
        "resume_checkpoint": args.resume,
        "resume_metadata": resume_metadata,
        "initial_model_sha256": initial_model_sha256,
        "parameter_count": parameter_count,
        "steps_completed": args.segment_end,
        "examples_seen_total": args.segment_end * config["data"]["batch_size"],
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "all_gradients_finite": all(math.isfinite(value) for value in gradients),
        "final_segment_loss": losses[-1],
        "maximum_segment_gradient_norm": max(gradients),
        "mean_step_seconds_after_warmup": statistics.fmean(measured_times),
        "median_step_seconds_after_warmup": statistics.median(measured_times),
        "segment_wall_clock_seconds": time.perf_counter() - started,
        "checkpoint_reload_exact": checkpoint_reload_exact,
        "checkpoints_written": checkpoint_records,
        "validation_sample": str(sample_path) if sample_path else None,
        "spectral_schedule": spectral_summary,
        "system": {
            "gpu": torch.cuda.get_device_name(device),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    summary_path = result_dir / (
        f"segment_summary_{start_step:06d}_{args.segment_end:06d}.json"
    )
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
