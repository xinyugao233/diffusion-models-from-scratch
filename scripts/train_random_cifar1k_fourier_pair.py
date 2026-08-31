#!/usr/bin/env python3
"""Train one paired spatial/Fourier CIFAR-1K memorization comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import torch
from _experiment_utils import git_identity, load_json, repository_path, sha256_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from diffusion_models.data import inverse_normalize, load_cifar10
from diffusion_models.diffusion import make_linear_ddpm_schedule
from diffusion_models.diffusion.schedules import DDPMSchedule, extract
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.fourier import (
    fourier_channels_to_image,
    image_to_fourier_channels,
    inverse_imaginary_residual,
    parseval_equivalent_mse,
    project_hermitian,
)
from diffusion_models.full_training import (
    DeterministicStepBatchSampler,
    require_slurm_environment,
)
from diffusion_models.models import CIFAR10UNet, UNetConfig, primary_unet_config


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def tensor_sha256(tensor: Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def make_models(seed: int) -> tuple[CIFAR10UNet, CIFAR10UNet, dict[str, Any]]:
    spatial_config = primary_unet_config()
    torch.manual_seed(10_000 + seed)
    spatial = CIFAR10UNet(spatial_config)
    fourier_config = UNetConfig(
        image_size=spatial_config.image_size,
        in_channels=6,
        out_channels=6,
        base_channels=spatial_config.base_channels,
        channel_multipliers=spatial_config.channel_multipliers,
        residual_blocks_per_level=spatial_config.residual_blocks_per_level,
        normalization=spatial_config.normalization,
        group_norm_groups=spatial_config.group_norm_groups,
        activation=spatial_config.activation,
        dropout=spatial_config.dropout,
        attention_resolutions=spatial_config.attention_resolutions,
        time_embedding_dim=spatial_config.time_embedding_dim,
        downsampling=spatial_config.downsampling,
        upsampling=spatial_config.upsampling,
        parameter_budget_max=spatial_config.parameter_budget_max,
    )
    torch.manual_seed(20_000 + seed)
    fourier = CIFAR10UNet(fourier_config)
    spatial_state = spatial.state_dict()
    fourier_state = fourier.state_dict()
    shared_names = []
    boundary_names = []
    for name, target in fourier_state.items():
        source = spatial_state.get(name)
        if source is not None and source.shape == target.shape:
            target.copy_(source)
            shared_names.append(name)
        else:
            boundary_names.append(name)
    fourier.load_state_dict(fourier_state)
    return spatial, fourier, {
        "spatial_parameter_count": spatial.trainable_parameter_count,
        "fourier_parameter_count": fourier.trainable_parameter_count,
        "shared_state_tensor_count": len(shared_names),
        "shape_specific_state_tensors": boundary_names,
    }


def optimizer_for(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    spec = config["training"]["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=spec["learning_rate"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )


def paired_training_batch(
    images: Tensor,
    schedule: DDPMSchedule,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    timesteps = torch.randint(
        0,
        schedule.num_steps,
        (images.shape[0],),
        device=images.device,
        generator=generator,
    )
    spatial_noise = torch.randn(
        images.shape,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    alpha = extract(schedule.sqrt_alpha_bars, timesteps, images.shape).to(images.dtype)
    sigma = extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, images.shape
    ).to(images.dtype)
    spatial_x_t = alpha * images + sigma * spatial_noise
    fourier_clean = image_to_fourier_channels(images)
    fourier_noise = image_to_fourier_channels(spatial_noise)
    fourier_x_t = (
        alpha * fourier_clean + sigma * fourier_noise
    )
    commutation_error = (
        image_to_fourier_channels(spatial_x_t) - fourier_x_t
    ).abs().amax()
    return timesteps, spatial_noise, spatial_x_t, fourier_noise, fourier_x_t, commutation_error


def optimize_condition(
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    x_t: Tensor,
    timesteps: Tensor,
    target: Tensor,
    gradient_clip: float,
) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x_t, timesteps)
    if name == "fourier":
        prediction = project_hermitian(prediction)
        loss = parseval_equivalent_mse(prediction, target)
    else:
        loss = F.mse_loss(prediction, target)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite {name} loss.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    ema.update(model)
    end.record()
    end.synchronize()
    return float(loss.detach().item()), float(start.elapsed_time(end))


def paired_initial_noise(
    count: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((count, 3, 32, 32), generator=generator, device=device)


@torch.no_grad()
def spatial_ddim(
    model: nn.Module,
    schedule: DDPMSchedule,
    initial_noise: Tensor,
    inference_steps: int,
) -> Tensor:
    from diffusion_models.diffusion import ddim_sample_loop

    return ddim_sample_loop(
        model,
        schedule,
        initial_noise.shape,
        num_inference_steps=inference_steps,
        initial_noise=initial_noise,
        clip_x_start=True,
    ).sample


def selected_ddim_timesteps(total: int, count: int) -> tuple[int, ...]:
    selected = torch.linspace(0, total - 1, count, dtype=torch.float64)
    return tuple(reversed(selected.round().to(torch.long).tolist()))


@torch.no_grad()
def fourier_ddim(
    model: nn.Module,
    schedule: DDPMSchedule,
    initial_spatial_noise: Tensor,
    inference_steps: int,
) -> Tensor:
    sample = image_to_fourier_channels(initial_spatial_noise)
    selected = selected_ddim_timesteps(schedule.num_steps, inference_steps)
    for index, code_timestep in enumerate(selected):
        previous = selected[index + 1] if index + 1 < len(selected) else -1
        timesteps = torch.full(
            (sample.shape[0],), code_timestep, device=sample.device, dtype=torch.long
        )
        predicted_noise = project_hermitian(model(sample, timesteps))
        alpha_t = extract(schedule.sqrt_alpha_bars, timesteps, sample.shape).to(
            sample.dtype
        )
        sigma_t = extract(
            schedule.sqrt_one_minus_alpha_bars, timesteps, sample.shape
        ).to(sample.dtype)
        predicted_clean = project_hermitian(
            (sample - sigma_t * predicted_noise) / alpha_t
        )
        clipped_image = fourier_channels_to_image(predicted_clean).clamp(-1.0, 1.0)
        predicted_clean = image_to_fourier_channels(clipped_image)
        if previous == -1:
            alpha_previous = torch.ones(
                (), device=sample.device, dtype=sample.dtype
            )
        else:
            alpha_previous = schedule.alpha_bars[previous].to(
                device=sample.device, dtype=sample.dtype
            )
        sample = project_hermitian(
            torch.sqrt(alpha_previous) * predicted_clean
            + torch.sqrt(1.0 - alpha_previous) * predicted_noise
        )
    return fourier_channels_to_image(sample)


@torch.no_grad()
def memorization_metrics(samples: Tensor, references: Tensor) -> dict[str, Any]:
    flat_references = references.flatten(1)
    nearest_distances = []
    second_distances = []
    nearest_indices = []
    for chunk in samples.split(128):
        distances = torch.cdist(chunk.flatten(1), flat_references)
        values, indices = distances.topk(2, largest=False, dim=1)
        nearest_distances.append(values[:, 0])
        second_distances.append(values[:, 1])
        nearest_indices.append(indices[:, 0])
    d1 = torch.cat(nearest_distances)
    d2 = torch.cat(second_distances)
    neighbors = torch.cat(nearest_indices)
    memorized = d1 < d2 / 3.0
    memorized_neighbors = neighbors[memorized]
    counts = torch.bincount(memorized_neighbors, minlength=references.shape[0])
    return {
        "sample_count": samples.shape[0],
        "memorized_count": int(memorized.sum().item()),
        "memorization_rate": float(memorized.float().mean().item()),
        "unique_training_neighbors_hit": int((counts > 0).sum().item()),
        "unique_training_neighbor_fraction": float(
            (counts > 0).float().mean().item()
        ),
        "maximum_duplicate_count": int(counts.max().item()),
        "mean_d1": float(d1.mean().item()),
        "mean_d2": float(d2.mean().item()),
    }


def system_identity(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


@torch.no_grad()
def evaluate(
    models: dict[str, nn.Module],
    emas: dict[str, ExponentialMovingAverage],
    schedule: DDPMSchedule,
    references: Tensor,
    config: dict[str, Any],
    seed: int,
    step: int,
    output_dir: Path,
    *,
    fresh: bool,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    sample_count = (
        evaluation["fresh_confirmation_sample_count"]
        if fresh
        else evaluation["sample_count"]
    )
    seed_offset = (
        evaluation["fresh_sample_seed_offset"]
        if fresh
        else evaluation["fixed_sample_seed_offset"]
    )
    device = references.device
    initial_noise = paired_initial_noise(sample_count, seed_offset + seed, device)
    result: dict[str, Any] = {"step": step, "fresh_panel": fresh}
    for name in ("spatial", "fourier"):
        evaluation_model = copy.deepcopy(models[name]).to(device).eval()
        emas[name].copy_to(evaluation_model)
        batches = []
        start = time.perf_counter()
        for noise_batch in initial_noise.split(evaluation["sample_batch_size"]):
            if name == "spatial":
                generated = spatial_ddim(
                    evaluation_model,
                    schedule,
                    noise_batch,
                    evaluation["ddim_steps"],
                )
            else:
                generated = fourier_ddim(
                    evaluation_model,
                    schedule,
                    noise_batch,
                    evaluation["ddim_steps"],
                )
            batches.append(generated)
        torch.cuda.synchronize(device)
        samples = torch.cat(batches).clamp(-1.0, 1.0)
        record = memorization_metrics(samples, references)
        record["evaluation_seconds"] = time.perf_counter() - start
        record["sample_sha256"] = tensor_sha256(samples)
        result[name] = record
        if not fresh:
            figure_dir = output_dir / "figures"
            figure_dir.mkdir(parents=True, exist_ok=True)
            figure_path = figure_dir / f"{name}_samples_step_{step:06d}.png"
            if figure_path.exists():
                raise FileExistsError(f"Refusing to overwrite {figure_path}.")
            save_image(
                inverse_normalize(
                    samples[: evaluation["grid_count"]].cpu(), clamp=True
                ),
                figure_path,
                nrow=8,
                padding=2,
            )
        del evaluation_model, batches, samples
    return result


def main() -> int:
    require_slurm_environment(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = repository_path(args.config)
    config = load_json(config_path)
    if args.seed not in config["training"]["paired_seeds"]:
        raise ValueError("Seed is not in the frozen paired-seed list.")
    git = git_identity()
    run_commit = os.environ.get("RUN_COMMIT", "")
    if git["working_tree_dirty"] or git["commit"] != run_commit:
        raise RuntimeError("Execution checkout must be clean and equal RUN_COMMIT.")
    if not torch.cuda.is_available():
        raise RuntimeError("A Slurm-assigned CUDA GPU is required.")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.use_deterministic_algorithms(config["training"]["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = output_dir / "COMPLETED"
    if completed.exists():
        raise FileExistsError(f"Refusing to overwrite completed run {output_dir}.")

    data_config = config["data"]
    dataset = load_cifar10(data_config["root"], train=True, download=False)
    subset_indices = list(
        range(
            data_config["subset_start"],
            data_config["subset_start"] + data_config["subset_count"],
        )
    )
    subset = Subset(dataset, subset_indices)
    references = torch.stack([dataset[index][0] for index in subset_indices]).to(device)
    models_tuple = make_models(args.seed)
    spatial_model, fourier_model, initialization = models_tuple
    models = {
        "spatial": spatial_model.to(device),
        "fourier": fourier_model.to(device),
    }
    optimizers = {
        name: optimizer_for(model, config) for name, model in models.items()
    }
    emas = {
        name: ExponentialMovingAverage(model, decay=config["training"]["ema_decay"])
        for name, model in models.items()
    }
    diffusion = config["diffusion"]
    schedule = make_linear_ddpm_schedule(
        num_steps=diffusion["num_steps"],
        beta_start=diffusion["beta_start"],
        beta_end=diffusion["beta_end"],
        device=device,
    )
    maximum_steps = config["training"]["maximum_steps"]
    batch_sampler = DeterministicStepBatchSampler(
        dataset_size=len(subset),
        batch_size=data_config["batch_size"],
        data_seed=20_000 + args.seed,
        start_step=0,
        end_step=maximum_steps,
    )
    loader = DataLoader(
        subset,
        batch_sampler=batch_sampler,
        num_workers=data_config["num_workers"],
        pin_memory=True,
        persistent_workers=data_config["num_workers"] > 0,
    )
    training_generator = torch.Generator(device=device).manual_seed(30_000 + args.seed)
    metrics_path = output_dir / "metrics.jsonl"
    evaluations_path = output_dir / "evaluations.jsonl"
    if metrics_path.exists() or evaluations_path.exists():
        raise FileExistsError("Refusing to append to a prior incomplete try.")
    threshold = config["evaluation"]["memorization_rate_threshold"]
    consecutive_required = config["evaluation"]["consecutive_checkpoints"]
    consecutive = {"spatial": 0, "fourier": 0}
    threshold_steps: dict[str, int | None] = {"spatial": None, "fourier": None}
    confirmed = {"spatial": False, "fourier": False}
    training_ms = {"spatial": 0.0, "fourier": 0.0}
    maximum_commutation_error = 0.0
    wall_start = time.perf_counter()
    for batch_number, (images, _) in enumerate(loader, start=1):
        step = batch_number
        images = images.to(device, non_blocking=True)
        (
            timesteps,
            spatial_noise,
            spatial_x_t,
            fourier_noise,
            fourier_x_t,
            commutation_error,
        ) = paired_training_batch(images, schedule, training_generator)
        maximum_commutation_error = max(
            maximum_commutation_error, float(commutation_error.item())
        )
        payload = {
            "spatial": (spatial_x_t, spatial_noise),
            "fourier": (fourier_x_t, fourier_noise),
        }
        order = ("spatial", "fourier") if step % 2 else ("fourier", "spatial")
        step_record: dict[str, Any] = {"step": step, "condition_order": order}
        for name in order:
            loss, elapsed_ms = optimize_condition(
                name,
                models[name],
                optimizers[name],
                emas[name],
                payload[name][0],
                timesteps,
                payload[name][1],
                config["training"]["gradient_clip"],
            )
            training_ms[name] += elapsed_ms
            step_record[f"{name}_loss"] = loss
            step_record[f"{name}_cuda_ms"] = elapsed_ms
            step_record[f"{name}_cumulative_cuda_ms"] = training_ms[name]
        step_record["forward_commutation_max_abs_error"] = float(
            commutation_error.item()
        )
        append_jsonl(metrics_path, step_record)
        if step == 1 or step % 100 == 0:
            print(json.dumps(step_record, sort_keys=True), flush=True)
        if step % config["evaluation"]["checkpoint_every"] == 0:
            evaluation = evaluate(
                models,
                emas,
                schedule,
                references,
                config,
                args.seed,
                step,
                output_dir,
                fresh=False,
            )
            append_jsonl(evaluations_path, evaluation)
            for name in ("spatial", "fourier"):
                if evaluation[name]["memorization_rate"] >= threshold:
                    if not confirmed[name]:
                        fresh = evaluate(
                            models,
                            emas,
                            schedule,
                            references,
                            config,
                            args.seed,
                            step,
                            output_dir,
                            fresh=True,
                        )
                        append_jsonl(evaluations_path, fresh)
                        confirmed[name] = fresh[name]["memorization_rate"] >= threshold
                    consecutive[name] = consecutive[name] + 1 if confirmed[name] else 0
                else:
                    consecutive[name] = 0
                if (
                    threshold_steps[name] is None
                    and consecutive[name] >= consecutive_required
                ):
                    threshold_steps[name] = step
            print(json.dumps(evaluation, sort_keys=True), flush=True)
            if all(value is not None for value in threshold_steps.values()):
                break

    final_step = step
    checkpoints = {}
    for name in ("spatial", "fourier"):
        checkpoint_path = output_dir / f"final_{name}_model.pt"
        with checkpoint_path.open("xb") as handle:
            torch.save(
                {
                    "model": models[name].state_dict(),
                    "ema": emas[name].state_dict(),
                    "global_step": final_step,
                    "threshold_step": threshold_steps[name],
                },
                handle,
            )
        checkpoints[name] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        }
    summary = {
        "schema_version": 1,
        "status": "completed",
        "paired_seed": args.seed,
        "final_step": final_step,
        "threshold_steps": threshold_steps,
        "right_censored": {
            name: threshold_steps[name] is None for name in threshold_steps
        },
        "training_cuda_seconds": {
            name: training_ms[name] / 1000.0 for name in training_ms
        },
        "mean_cuda_ms_per_step": {
            name: training_ms[name] / final_step for name in training_ms
        },
        "wall_seconds_including_evaluation": time.perf_counter() - wall_start,
        "maximum_forward_commutation_abs_error": maximum_commutation_error,
        "initialization": initialization,
        "git": git,
        "run_commit": run_commit,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "subset_indices": [subset_indices[0], subset_indices[-1]],
        "system": system_identity(device),
        "checkpoints": checkpoints,
    }
    write_json_exclusive(output_dir / "metrics.json", summary)
    write_json_exclusive(
        output_dir / "run_manifest.json",
        {
            "run_commit": run_commit,
            "paired_seed": args.seed,
            "config_sha256": summary["config_sha256"],
            "summary_sha256": sha256_file(output_dir / "metrics.json"),
            "metrics_jsonl_sha256": sha256_file(metrics_path),
            "evaluations_jsonl_sha256": sha256_file(evaluations_path),
        },
    )
    completed.touch(exist_ok=False)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
