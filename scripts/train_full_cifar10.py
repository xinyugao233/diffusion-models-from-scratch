#!/usr/bin/env python3
"""Train the frozen full CIFAR-10 DDPM configuration under Slurm."""

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
from typing import Any

import torch
from _experiment_utils import (
    git_identity,
    load_json,
    md5_file,
    repository_path,
    save_loss_curve,
    sha256_file,
    write_json_exclusive,
)
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
    checkpoint_steps,
    ema_tensors_are_finite,
    fixed_initial_noise,
    model_tensors_are_finite,
    production_train_step,
    require_slurm_environment,
    resolve_study_configuration,
    sampling_steps,
)
from diffusion_models.models import CIFAR10UNet, primary_unet_config
from diffusion_models.spectral_boundary import (
    RadialPower,
    SpectralBoundaryConfig,
    StaticSpectralConfig,
    load_radial_power,
    load_static_spectral_control,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_configuration(config: dict[str, Any], stage: str) -> dict[str, Any]:
    if config["experiment"]["baseline_commit"] != (
        "706103861c7d12ff3cb7dee037b6a10514b46b5e"
    ):
        raise ValueError("Unexpected Milestone 6 baseline commit.")
    if config["model"]["configuration"] != "primary_unet":
        raise ValueError("Milestone 6 is frozen to the primary U-Net.")
    if config["training"]["precision"] != "float32":
        raise ValueError("Milestone 6 is frozen to float32.")
    if config["diffusion"]["schedule"] != "linear":
        raise ValueError("Milestone 6 is frozen to the linear schedule.")
    if stage not in config["stages"]:
        raise ValueError(f"Unknown stage {stage!r}.")
    return config["stages"][stage]


def load_spectral_boundary_configuration(
    config: dict[str, Any],
) -> tuple[RadialPower, SpectralBoundaryConfig | StaticSpectralConfig] | None:
    """Load the optional loss feature without touching the baseline path."""
    values = config.get("spectral_boundary_loss")
    if values is None or not values.get("enabled", False):
        return None
    mode = values.get("mode", "moving")
    if mode == "static":
        return load_static_spectral_control(
            repository_path(values["static_control_path"])
        )
    if mode != "moving":
        raise ValueError("spectral_boundary_loss mode must be 'moving' or 'static'.")
    radial_power = load_radial_power(repository_path(values["radial_power_path"]))
    boundary_config = SpectralBoundaryConfig(
        tau=float(values["tau"]),
        weight_floor=float(values["weight_floor"]),
        normalization=values["normalization"],
        fft_normalization=values["fft_normalization"],
        sigma_min=values.get("sigma_min"),
        sigma_max=values.get("sigma_max"),
    )
    return radial_power, boundary_config


def dataset_identity(root: Path) -> dict[str, Any]:
    archive = root / "cifar-10-python.tar.gz"
    batch_root = root / "cifar-10-batches-py"
    required = [
        batch_root / "batches.meta",
        *(batch_root / f"data_batch_{index}" for index in range(1, 6)),
    ]
    missing = [str(path) for path in [archive, *required] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-10 files: {missing}")
    return {
        "archive": str(archive),
        "archive_md5": md5_file(archive),
        "archive_sha256": file_sha256(archive),
        "training_files": {path.name: file_sha256(path) for path in required},
    }


def cuda_identity(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "gpu_index": index,
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def read_losses(path: Path) -> list[float]:
    return [
        float(json.loads(line)["loss"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_tensor_exclusive(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite tensor: {path}")
    with path.open("xb") as handle:
        torch.save(tensor.detach().cpu(), handle)


@torch.no_grad()
def generate_ema_samples(
    model: CIFAR10UNet,
    ema: ExponentialMovingAverage,
    schedule,
    config: dict[str, Any],
    *,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    evaluation_model = copy.deepcopy(model).eval()
    ema.copy_to(evaluation_model)
    seeds = config["evaluation"]["fixed_initial_noise_seeds"][:sample_count]
    initial_noise = fixed_initial_noise(seeds).to(device)
    reverse_generator = torch.Generator(device=device).manual_seed(
        config["evaluation"]["reverse_noise_seed"]
    )
    result = p_sample_loop(
        evaluation_model,
        schedule,
        initial_noise.shape,
        initial_noise=initial_noise,
        generator=reverse_generator,
        clip_x_start=config["evaluation"]["clip_x_start"],
    )
    del evaluation_model
    if not torch.isfinite(result.sample).all():
        raise FloatingPointError("EMA sample contains nonfinite values.")
    return result.sample.detach().cpu()


def main() -> int:
    job_id = require_slurm_environment(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-end", required=True, type=int)
    parser.add_argument("--resume")
    parser.add_argument("--condition")
    parser.add_argument("--pair", type=int)
    args = parser.parse_args()

    config_path = repository_path(args.config)
    source_config = load_json(config_path)
    config = resolve_study_configuration(
        source_config, condition=args.condition, pair=args.pair
    )
    stage_config = validate_configuration(config, args.stage)
    spectral_boundary = load_spectral_boundary_configuration(config)
    maximum_steps = int(stage_config["max_steps"])
    if not 0 < args.segment_end <= maximum_steps:
        raise ValueError("segment-end must lie in [1, stage max_steps].")

    git = git_identity()
    submitted_commit = os.environ.get("RUN_COMMIT", "")
    if not submitted_commit or git["commit"] != submitted_commit:
        raise RuntimeError(
            f"Execution commit {git['commit']} differs from RUN_COMMIT "
            f"{submitted_commit!r}."
        )
    if git["working_tree_dirty"]:
        raise RuntimeError("Refusing to train from a dirty execution checkout.")
    if not torch.cuda.is_available():
        raise RuntimeError("Milestone 6 requires a Slurm-assigned CUDA GPU.")

    device = torch.device("cuda", torch.cuda.current_device())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(
        bool(config["training"]["deterministic_algorithms"])
    )
    torch.backends.cudnn.benchmark = False
    random.seed(config["model"]["initialization_seed"])
    torch.manual_seed(config["model"]["initialization_seed"])
    torch.cuda.manual_seed_all(config["training"]["training_noise_seed"] + 1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"
    result_dir = output_dir / "results"

    dataset_root = Path(config["data"]["root"])
    dataset_manifest = dataset_identity(dataset_root)
    dataset = load_cifar10(dataset_root, train=True, download=False)
    if len(dataset) != config["data"]["expected_train_size"]:
        raise RuntimeError(
            f"Expected {config['data']['expected_train_size']} images, "
            f"found {len(dataset)}."
        )

    model = CIFAR10UNet(primary_unet_config()).to(device)
    initial_model_sha256 = state_dict_sha256(model.state_dict())
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != config["model"]["expected_parameter_count"]:
        raise RuntimeError(
            f"Expected {config['model']['expected_parameter_count']} parameters, "
            f"found {parameter_count}."
        )
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
    ema = ExponentialMovingAverage(model, decay=config["training"]["ema_decay"])
    schedule = make_linear_ddpm_schedule(
        num_steps=config["diffusion"]["num_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )
    training_generator = torch.Generator(device=device).manual_seed(
        config["training"]["training_noise_seed"]
    )
    initial_training_generator_sha256 = tensor_sha256(training_generator.get_state())
    checkpoint_configuration = {"config": config, "stage": args.stage}
    source_configuration_sha256 = sha256_file(config_path)
    resolved_configuration_sha256 = mapping_sha256(config)

    start_step = 0
    resume_metadata = None
    if args.resume:
        loaded = load_training_checkpoint(
            args.resume,
            model=model,
            ema=ema,
            optimizer=optimizer,
            expected_configuration=checkpoint_configuration,
            generators={"training": training_generator},
            map_location=device,
            restore_rng=True,
        )
        start_step = loaded.global_step
        resume_metadata = loaded.metadata
    if not start_step < args.segment_end:
        raise ValueError(
            f"Resume step {start_step} must be below segment end {args.segment_end}."
        )
    if args.stage == "gate_b" and start_step not in (0, 250):
        raise ValueError("Gate B may begin only at step 0 or its frozen step 250.")

    logger = AppendOnlyJsonlLogger(
        metrics_path,
        expected_last_step=start_step,
    )
    batch_sampler = DeterministicStepBatchSampler(
        dataset_size=len(dataset),
        batch_size=config["data"]["batch_size"],
        data_seed=config["data"]["data_order_seed"],
        start_step=start_step,
        end_step=args.segment_end,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["data"]["num_workers"] > 0,
    )

    first_parameter_before = next(model.parameters()).detach().clone()
    due_checkpoints = checkpoint_steps(stage_config)
    due_samples = sampling_steps(
        args.stage,
        stage_config,
        config["evaluation"],
    )
    checkpoint_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    optimizer_step_times: list[float] = []
    cumulative_optimizer_seconds = 0.0
    segment_examples = 0
    torch.cuda.reset_peak_memory_stats(device)

    def write_checkpoint(step: int) -> None:
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            global_step=step,
            configuration=checkpoint_configuration,
            metadata={
                "run_commit": git["commit"],
                "source_configuration_sha256": source_configuration_sha256,
                "resolved_configuration_sha256": resolved_configuration_sha256,
                "stage": args.stage,
                "condition": args.condition,
                "pair": args.pair,
                "slurm_job_id": job_id,
                "dataset_archive_md5": dataset_manifest["archive_md5"],
                "initial_model_sha256": initial_model_sha256,
                "initial_training_generator_sha256": initial_training_generator_sha256,
            },
            generators={"training": training_generator},
            cuda_devices=[device.index or 0],
        )
        checkpoint_records.append(
            {
                "step": step,
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
            }
        )

    if start_step == 0 and 0 in due_checkpoints:
        write_checkpoint(0)

    model.train()
    for batch_index, (images, _) in enumerate(loader, start=1):
        step = start_step + batch_index
        images = images.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        optimizer_start = time.perf_counter()
        metrics = production_train_step(
            model,
            optimizer,
            ema,
            images,
            schedule,
            generator=training_generator,
            max_gradient_norm=config["training"]["gradient_clip"],
            spectral_boundary=spectral_boundary,
        )
        torch.cuda.synchronize(device)
        optimizer_step_seconds = time.perf_counter() - optimizer_start
        optimizer_step_times.append(optimizer_step_seconds)
        cumulative_optimizer_seconds += optimizer_step_seconds
        segment_examples += images.shape[0]
        elapsed = time.perf_counter() - start_time
        metric_values: dict[str, float | str] = {
            "loss": metrics.loss,
            "optimization_loss": metrics.loss,
            "original_unweighted_epsilon_mse": metrics.unweighted_loss,
            "gradient_norm": metrics.gradient_norm,
            "clipped_gradient_norm": metrics.clipped_gradient_norm,
            "ema_num_updates": metrics.ema_num_updates,
            "elapsed_seconds": elapsed,
            "training_wall_clock_seconds": elapsed,
            "optimizer_step_seconds": optimizer_step_seconds,
            "cumulative_optimizer_seconds": cumulative_optimizer_seconds,
            "examples_seen": step * config["data"]["batch_size"],
            "steps_per_second": batch_index / elapsed,
            "examples_per_second": segment_examples / elapsed,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        if metrics.spectral_boundary is not None:
            diagnostics = metrics.spectral_boundary
            metric_values.update(
                weighted_loss=metrics.loss,
                unweighted_loss=metrics.unweighted_loss,
                sigma_mean=float(diagnostics.sigma_mean.detach().item()),
                sigma_min=float(diagnostics.sigma_min.detach().item()),
                sigma_max=float(diagnostics.sigma_max.detach().item()),
                spectral_boundary_active_fraction=float(
                    diagnostics.active_fraction.detach().item()
                ),
                peak_radial_shell_mean=float(
                    diagnostics.peak_shell_mean.detach().item()
                ),
                information_radius_mean=float(
                    diagnostics.information_radius_mean.detach().item()
                ),
                mean_spectral_weight=float(diagnostics.mean_weight.detach().item()),
                maximum_spectral_weight=float(
                    diagnostics.maximum_weight.detach().item()
                ),
                effective_weighted_shell_count=float(
                    diagnostics.effective_shell_count.detach().item()
                ),
                effective_weighted_coefficient_count=float(
                    diagnostics.effective_coefficient_count.detach().item()
                ),
            )
        logger.log(step, **metric_values)
        if step == 1 or step % 10 == 0 or step == args.segment_end:
            print(
                f"step={step} loss={metrics.loss:.6f} "
                f"grad={metrics.gradient_norm:.6f} "
                f"steps_per_second={batch_index / elapsed:.4f}",
                flush=True,
            )

        if step in due_checkpoints:
            write_checkpoint(step)

        if step in due_samples:
            sample_count = int(stage_config["sample_count"])
            samples = generate_ema_samples(
                model,
                ema,
                schedule,
                config,
                sample_count=sample_count,
                device=device,
            )
            reproducible = None
            repeated_hash = None
            if args.stage == "gate_b":
                repeated = generate_ema_samples(
                    model,
                    ema,
                    schedule,
                    config,
                    sample_count=sample_count,
                    device=device,
                )
                reproducible = torch.equal(samples, repeated)
                repeated_hash = tensor_sha256(repeated)
                if not reproducible:
                    raise RuntimeError("Fixed-seed Gate B EMA sampling is not exact.")
            tensor_path = result_dir / f"ema_samples_step_{step:06d}.pt"
            figure_path = figure_dir / f"ema_samples_step_{step:06d}.png"
            save_tensor_exclusive(tensor_path, samples)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            if figure_path.exists():
                raise FileExistsError(f"Refusing to overwrite figure: {figure_path}")
            save_image(
                inverse_normalize(samples, clamp=True),
                figure_path,
                nrow=4,
                padding=2,
            )
            losses = read_losses(metrics_path)
            loss_curve_path = figure_dir / f"loss_curve_step_{step:06d}.png"
            save_loss_curve(
                loss_curve_path,
                losses,
                title=f"Full CIFAR-10 DDPM loss through step {step}",
            )
            sample_records.append(
                {
                    "step": step,
                    "tensor_path": str(tensor_path),
                    "tensor_sha256": tensor_sha256(samples),
                    "tensor_file_sha256": file_sha256(tensor_path),
                    "figure_path": str(figure_path),
                    "figure_sha256": file_sha256(figure_path),
                    "loss_curve_path": str(loss_curve_path),
                    "loss_curve_sha256": file_sha256(loss_curve_path),
                    "reproducible": reproducible,
                    "repeated_tensor_sha256": repeated_hash,
                }
            )
            model.train()

    duration = time.perf_counter() - start_time
    parameters_changed = not torch.equal(
        first_parameter_before,
        next(model.parameters()).detach(),
    )
    if not parameters_changed:
        raise RuntimeError("Model parameters did not change during the segment.")
    if ema.num_updates != args.segment_end:
        raise RuntimeError(
            f"Expected EMA update count {args.segment_end}, got {ema.num_updates}."
        )
    if not model_tensors_are_finite(model) or not ema_tensors_are_finite(ema):
        raise FloatingPointError("Model or EMA state contains nonfinite values.")
    ema_differs_from_model = any(
        not torch.equal(ema.parameters[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    if not ema_differs_from_model:
        raise RuntimeError("EMA did not diverge from the optimized model.")

    losses = read_losses(metrics_path)
    if len(losses) != args.segment_end:
        raise RuntimeError(
            f"Expected {args.segment_end} total loss records, found {len(losses)}."
        )
    system = cuda_identity(device)
    timing_warmup_steps = int(config["training"].get("timing_warmup_steps", 100))
    post_warmup_times = optimizer_step_times[timing_warmup_steps:]
    if not post_warmup_times:
        post_warmup_times = optimizer_step_times
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"]["name"],
        "try": config["experiment"]["try"],
        "stage": args.stage,
        "segment": {"start_step": start_step, "end_step": args.segment_end},
        "passed": True,
        "git": git,
        "submitted_commit": submitted_commit,
        "configuration_path": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "resolved_configuration_sha256": resolved_configuration_sha256,
        "condition": args.condition,
        "pair": args.pair,
        "initial_model_sha256": initial_model_sha256,
        "initial_training_generator_sha256": initial_training_generator_sha256,
        "dataset": dataset_manifest,
        "dataset_size": len(dataset),
        "parameter_count": parameter_count,
        "batch_size": config["data"]["batch_size"],
        "duration_seconds": duration,
        "optimizer_compute_seconds": cumulative_optimizer_seconds,
        "timing_warmup_steps_excluded": timing_warmup_steps,
        "mean_optimizer_step_seconds_after_warmup": statistics.fmean(post_warmup_times),
        "median_optimizer_step_seconds_after_warmup": statistics.median(
            post_warmup_times
        ),
        "segment_examples": segment_examples,
        "steps_per_second": (args.segment_end - start_step) / duration,
        "examples_per_second": segment_examples / duration,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "final_loss": losses[-1],
        "all_losses_finite": all(math.isfinite(loss) for loss in losses),
        "metric_record_count": len(losses),
        "ema_num_updates": ema.num_updates,
        "parameters_changed": parameters_changed,
        "ema_differs_from_model": ema_differs_from_model,
        "resume_checkpoint": args.resume,
        "resume_metadata": resume_metadata,
        "checkpoints_written": checkpoint_records,
        "samples_written": sample_records,
        "system": system,
    }
    summary_path = result_dir / (
        f"segment_summary_{start_step:06d}_{args.segment_end:06d}.json"
    )
    manifest_path = result_dir / (
        f"run_manifest_{start_step:06d}_{args.segment_end:06d}.json"
    )
    write_json_exclusive(summary_path, summary)
    write_json_exclusive(
        manifest_path,
        {
            "schema_version": 1,
            "run_identity": {
                "git_commit": git["commit"],
                "configuration_sha256": summary["configuration_sha256"],
                "slurm_job_id": job_id,
                "stage": args.stage,
                "start_step": start_step,
                "end_step": args.segment_end,
            },
            "dataset": dataset_manifest,
            "system": system,
            "summary_path": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
