#!/usr/bin/env python3
"""Evaluate one final Study 2 checkpoint under common and native protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from _experiment_utils import git_identity, load_json, repository_path, sha256_file
from torch.utils.data import Dataset
from torchvision.utils import save_image

from diffusion_models.checkpointing import load_training_checkpoint
from diffusion_models.data import inverse_normalize, load_cifar10
from diffusion_models.diffusion import (
    make_linear_ddpm_schedule,
    p_sample_loop,
    q_sample,
)
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.full_training import require_slurm_environment
from diffusion_models.models import CIFAR10UNet, primary_unet_config
from diffusion_models.spectral_boundary import (
    effective_additive_sigma,
    load_radial_power,
)
from diffusion_models.spectral_noise import (
    SpectralNoiseConfig,
    build_spectral_noise_schedule,
    p_sample_loop_spectral,
    q_sample_spectral,
)
from diffusion_models.spectral_noise_study import (
    full_fft_shell_mse,
    resolve_spectral_noise_study_configuration,
)


class UInt8Images(Dataset[torch.Tensor]):
    def __init__(self, images: torch.Tensor) -> None:
        if images.dtype != torch.uint8 or images.ndim != 4:
            raise ValueError("KID inputs must be uint8 NCHW tensors.")
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


def uint8_images(samples: torch.Tensor) -> torch.Tensor:
    return (
        (samples.clamp(-1.0, 1.0) + 1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )


@torch.no_grad()
def evaluate_residuals(
    model: CIFAR10UNet,
    images: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon: torch.Tensor,
    baseline,
    spectral,
    *,
    protocol: str,
    radial_power,
    batch_size: int,
) -> tuple[float, list[dict[str, Any]]]:
    squared_sum = 0.0
    element_count = 0
    rows: list[dict[str, Any]] = []
    for offset in range(0, images.shape[0], batch_size):
        clean = images[offset : offset + batch_size]
        t = timesteps[offset : offset + batch_size]
        noise = epsilon[offset : offset + batch_size]
        if protocol == "native" and spectral is not None:
            noisy = q_sample_spectral(clean, t, spectral, noise)
        else:
            noisy = q_sample(clean, t, baseline, noise=noise)
        residual = model(noisy, t) - noise
        squared_sum += float(residual.square().sum())
        element_count += residual.numel()
        shell_values = full_fft_shell_mse(
            residual, num_shells=radial_power.power.numel()
        ).cpu()
        sigma = effective_additive_sigma(baseline, t, dtype=torch.float64).cpu()
        timestep_values = t.cpu()
        distance = (
            torch.log(radial_power.power)[None, :] - 2.0 * torch.log(sigma)[:, None]
        ).abs()
        for batch_index in range(shell_values.shape[0]):
            for shell in range(shell_values.shape[1]):
                rows.append(
                    {
                        "protocol": protocol,
                        "example_index": offset + batch_index,
                        "timestep": int(timestep_values[batch_index]),
                        "shell": shell,
                        "shell_epsilon_mse": float(shell_values[batch_index, shell]),
                        "boundary_log_distance": float(distance[batch_index, shell]),
                    }
                )
    return squared_sum / element_count, rows


@torch.no_grad()
def sample_model(
    model: CIFAR10UNet,
    baseline,
    spectral,
    *,
    count: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    grids = []
    generated = []
    for offset in range(0, count, batch_size):
        current = min(batch_size, count - offset)
        shape = (current, 3, 32, 32)
        if spectral is None:
            samples = p_sample_loop(
                model,
                baseline,
                shape,
                device=device,
                generator=generator,
                clip_x_start=True,
            ).sample
        else:
            samples = p_sample_loop_spectral(
                model,
                spectral,
                shape,
                device=device,
                generator=generator,
                clip_x_start=True,
            ).sample
        if not torch.isfinite(samples).all():
            raise FloatingPointError(
                "Matched reverse sampling produced nonfinite data."
            )
        if len(grids) * batch_size < 64:
            grids.append(samples[: min(current, 64 - len(grids) * batch_size)].cpu())
        generated.append(uint8_images(samples).cpu())
    return torch.cat(grids)[:64], torch.cat(generated)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    job_id = require_slurm_environment(os.environ)
    config_path = repository_path(args.config)
    source = load_json(config_path)
    config = resolve_spectral_noise_study_configuration(
        source, condition=args.condition, seed=args.seed
    )
    git = git_identity()
    if git["working_tree_dirty"] or git["commit"] != os.environ.get("RUN_COMMIT"):
        raise RuntimeError("Evaluation requires the exact clean RUN_COMMIT.")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a Slurm-assigned CUDA GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    evaluation = config["evaluation"]
    asset_path = Path(evaluation["asset_path"])
    if sha256_file(asset_path) != evaluation["asset_sha256"]:
        raise RuntimeError("Held-out evaluation asset SHA-256 differs.")
    weights_path = Path(evaluation["inception_weights_path"])
    if sha256_file(weights_path) != evaluation["inception_weights_sha256"]:
        raise RuntimeError("Inception weights SHA-256 differs.")
    asset = torch.load(asset_path, weights_only=True)

    test_set = load_cifar10(config["data"]["root"], train=False, download=False)
    heldout_images = torch.stack(
        [test_set[int(index)][0] for index in asset["example_indices"]]
    ).to(device)
    heldout_t = asset["timesteps"].to(device)
    heldout_epsilon = asset["epsilon"].to(device)
    real_uint8 = torch.from_numpy(test_set.data).permute(0, 3, 1, 2).contiguous()
    radial_power = load_radial_power(repository_path(config["spectrum"]))
    baseline = make_linear_ddpm_schedule(
        num_steps=config["diffusion"]["num_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )
    condition = config["spectral_noise_condition"]
    spectral = None
    if condition["enabled"]:
        weighted = config["weighted_noise"]
        spectral = build_spectral_noise_schedule(
            baseline,
            radial_power,
            SpectralNoiseConfig(
                tau=float(condition["tau"]),
                weight_floor=float(weighted["weight_floor"]),
                rho=float(weighted["rho"]),
                allocation=weighted["allocation"],
                fft_normalization=weighted["fft_normalization"],
            ),
        )

    model = CIFAR10UNet(primary_unet_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    ema = ExponentialMovingAverage(model, decay=config["training"]["ema_decay"])
    step = int(evaluation["checkpoint_step"])
    checkpoint = (
        Path(args.training_dir) / "checkpoints" / f"checkpoint_step_{step:06d}.pt"
    )
    checkpoint_configuration = {
        "config": config,
        "condition": args.condition,
        "seed": args.seed,
    }
    loaded = load_training_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        expected_configuration=checkpoint_configuration,
        map_location=device,
        restore_rng=False,
    )
    if loaded.global_step != step:
        raise RuntimeError("Checkpoint step differs from frozen evaluation step.")
    ema.copy_to(model)
    model.eval()
    started = time.perf_counter()
    protocol_metrics = {}
    raw_rows = []
    for protocol in evaluation["protocols"]:
        mse, rows = evaluate_residuals(
            model,
            heldout_images,
            heldout_t,
            heldout_epsilon,
            baseline,
            spectral,
            protocol=protocol,
            radial_power=radial_power,
            batch_size=int(evaluation["batch_size"]),
        )
        protocol_metrics[protocol] = {"ordinary_epsilon_mse": mse}
        raw_rows.extend(rows)
    raw_path = output / "frequency_resolved_residuals.csv"
    with raw_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    sampling = evaluation["sampling"]
    sample_started = time.perf_counter()
    grid_samples, generated_uint8 = sample_model(
        model,
        baseline,
        spectral,
        count=int(sampling["generated_image_count"]),
        batch_size=int(sampling["generated_batch_size"]),
        seed=int(sampling["seed"]),
        device=device,
    )
    torch.cuda.synchronize(device)
    sample_seconds = time.perf_counter() - sample_started
    grid_path = output / "samples_step_010000.png"
    save_image(inverse_normalize(grid_samples, clamp=True), grid_path, nrow=8)

    from torch_fidelity import calculate_metrics

    kid = evaluation["kid"]
    quality = calculate_metrics(
        input1=UInt8Images(generated_uint8),
        input2=UInt8Images(real_uint8),
        cuda=True,
        isc=False,
        fid=False,
        kid=True,
        feature_extractor=kid["feature_extractor"].split()[-1],
        feature_layer_kid=kid["feature_layer"],
        feature_extractor_weights_path=str(weights_path),
        batch_size=int(kid["batch_size"]),
        kid_subsets=int(kid["subsets"]),
        kid_subset_size=int(kid["subset_size"]),
        kid_kernel_poly_degree=int(kid["polynomial_degree"]),
        kid_kernel_poly_gamma=kid["polynomial_gamma"],
        kid_kernel_poly_coef0=float(kid["polynomial_coef0"]),
        rng_seed=int(kid["rng_seed"]),
        cache=True,
        cache_root=str(Path(os.environ["TMPDIR"]) / "torch-fidelity-cache"),
        input2_cache_name="cifar10-test-10k-raw-uint8",
        verbose=False,
    )
    summary = {
        "schema_version": 1,
        "passed": True,
        "condition": args.condition,
        "seed": args.seed,
        "git": git,
        "slurm_job_id": job_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluation_asset_sha256": evaluation["asset_sha256"],
        "protocol_metrics": protocol_metrics,
        "frequency_resolved_csv": str(raw_path),
        "frequency_resolved_csv_sha256": sha256_file(raw_path),
        "kid_mean": float(quality["kernel_inception_distance_mean"]),
        "kid_std": float(quality["kernel_inception_distance_std"]),
        "fid": None,
        "fid_status": evaluation["fid"]["reason"],
        "generated_image_count": int(sampling["generated_image_count"]),
        "generated_uint8_sha256": hashlib.sha256(
            generated_uint8.numpy().tobytes()
        ).hexdigest(),
        "sample_grid": str(grid_path),
        "sample_grid_sha256": sha256_file(grid_path),
        "sampling_seconds": sample_seconds,
        "duration_seconds": time.perf_counter() - started,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
