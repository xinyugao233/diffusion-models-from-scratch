#!/usr/bin/env python3
"""Evaluate every frozen checkpoint of one spectral-boundary 20K run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from _experiment_utils import git_identity, load_json, repository_path
from torch.utils.data import Dataset
from torchvision.utils import save_image

from diffusion_models.checkpointing import load_training_checkpoint
from diffusion_models.data import inverse_normalize, load_cifar10
from diffusion_models.diffusion import (
    ddim_sample_loop,
    make_linear_ddpm_schedule,
    q_sample,
)
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.full_training import (
    require_slurm_environment,
    resolve_study_configuration,
)
from diffusion_models.memorization import cifar_pixel_memorization_metrics
from diffusion_models.models import CIFAR10UNet, primary_unet_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UInt8Images(Dataset[torch.Tensor]):
    def __init__(self, images: torch.Tensor) -> None:
        if images.dtype != torch.uint8 or images.ndim != 4:
            raise ValueError("KID/FID inputs must be uint8 NCHW tensors.")
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


def uint8_images(model_samples: torch.Tensor) -> torch.Tensor:
    return (
        (model_samples.clamp(-1.0, 1.0) + 1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )


@torch.no_grad()
def heldout_loss(
    model: CIFAR10UNet,
    schedule,
    images: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon: torch.Tensor,
    batch_size: int,
) -> float:
    squared_sum = 0.0
    element_count = 0
    for offset in range(0, images.shape[0], batch_size):
        x = images[offset : offset + batch_size]
        t = timesteps[offset : offset + batch_size]
        noise = epsilon[offset : offset + batch_size]
        noisy = q_sample(x, t, schedule, noise=noise)
        prediction = model(noisy, t)
        squared_sum += float((prediction - noise).square().sum().item())
        element_count += noise.numel()
    return squared_sum / element_count


@torch.no_grad()
def sample_checkpoint(
    model: CIFAR10UNet,
    schedule,
    *,
    count: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    floats = []
    bytes_ = []
    for offset in range(0, count, batch_size):
        current = min(batch_size, count - offset)
        result = ddim_sample_loop(
            model,
            schedule,
            (current, 3, 32, 32),
            num_inference_steps=50,
            device=device,
            generator=generator,
            clip_x_start=True,
        ).sample
        floats.append(result.cpu())
        bytes_.append(uint8_images(result).cpu())
    return torch.cat(floats), torch.cat(bytes_)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--pair", required=True, type=int)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    job_id = require_slurm_environment(os.environ)
    source = load_json(repository_path(args.config))
    config = resolve_study_configuration(
        source, condition=args.condition, pair=args.pair
    )
    git = git_identity()
    if git["working_tree_dirty"] or git["commit"] != os.environ.get("RUN_COMMIT"):
        raise RuntimeError("Evaluation requires the exact clean RUN_COMMIT.")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    evaluation = config["evaluation"]
    asset_path = Path(evaluation["asset_path"])
    if file_sha256(asset_path) != evaluation["asset_sha256"]:
        raise RuntimeError("Held-out evaluation asset SHA-256 differs.")
    weights_path = Path(evaluation["inception_weights_path"])
    if file_sha256(weights_path) != evaluation["inception_weights_sha256"]:
        raise RuntimeError("Inception weights SHA-256 differs.")
    asset = torch.load(asset_path, weights_only=True)

    test_set = load_cifar10(config["data"]["root"], train=False, download=False)
    train_set = load_cifar10(config["data"]["root"], train=True, download=False)
    heldout_images = torch.stack(
        [test_set[int(i)][0] for i in asset["example_indices"]]
    ).to(device)
    heldout_t = asset["timesteps"].to(device)
    heldout_epsilon = asset["epsilon"].to(device)
    references = torch.stack([train_set[i][0] for i in range(1000)]).to(device)
    real_uint8 = torch.from_numpy(test_set.data).permute(0, 3, 1, 2).contiguous()

    schedule = make_linear_ddpm_schedule(
        num_steps=config["diffusion"]["num_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )
    model = CIFAR10UNet(primary_unet_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    ema = ExponentialMovingAverage(model, decay=config["training"]["ema_decay"])
    records: list[dict[str, Any]] = []
    training_dir = Path(args.training_dir)
    started = time.perf_counter()

    from torch_fidelity import calculate_metrics

    for step in evaluation["kid"]["checkpoint_steps"]:
        checkpoint = training_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pt"
        loaded = load_training_checkpoint(
            checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer,
            expected_configuration={"config": config, "stage": "stage_20k"},
            map_location=device,
            restore_rng=False,
        )
        if loaded.global_step != step:
            raise RuntimeError("Checkpoint step differs from filename.")
        ema.copy_to(model)
        model.eval()
        validation_loss = heldout_loss(
            model,
            schedule,
            heldout_images,
            heldout_t,
            heldout_epsilon,
            int(evaluation["sampling"]["generated_batch_size"]),
        )
        sample_started = time.perf_counter()
        samples, generated_uint8 = sample_checkpoint(
            model,
            schedule,
            count=int(evaluation["kid"]["generated_image_count"]),
            batch_size=int(evaluation["sampling"]["generated_batch_size"]),
            seed=int(evaluation["sampling"]["checkpoint_seed"]),
            device=device,
        )
        torch.cuda.synchronize(device)
        sample_seconds = time.perf_counter() - sample_started
        memorization = cifar_pixel_memorization_metrics(
            samples[: evaluation["memorization"]["generated_image_count"]].to(device),
            references,
            chunk_size=int(evaluation["memorization"]["chunk_size"]),
        )
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
        grid_path = output / f"samples_step_{step:06d}.png"
        save_image(inverse_normalize(samples[:64], clamp=True), grid_path, nrow=8)
        record = {
            "step": step,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "heldout_original_epsilon_mse": validation_loss,
            "kid_mean": float(quality["kernel_inception_distance_mean"]),
            "kid_std": float(quality["kernel_inception_distance_std"]),
            "fid": None,
            "fid_status": evaluation["fid"]["status_at_20k"],
            "sample_seconds": sample_seconds,
            "generated_uint8_sha256": hashlib.sha256(
                generated_uint8.numpy().tobytes()
            ).hexdigest(),
            "memorization": memorization,
            "sample_grid": str(grid_path),
            "sample_grid_sha256": file_sha256(grid_path),
        }
        records.append(record)
        (output / f"metrics_step_{step:06d}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        del samples, generated_uint8

    summary = {
        "schema_version": 1,
        "passed": True,
        "condition": args.condition,
        "pair": args.pair,
        "slurm_job_id": job_id,
        "git": git,
        "evaluation_asset_sha256": evaluation["asset_sha256"],
        "inception_weights_sha256": evaluation["inception_weights_sha256"],
        "duration_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
