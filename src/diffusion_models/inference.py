"""Versioned EMA-only inference bundles; no Slurm or optimizer at inference."""

from dataclasses import asdict
from pathlib import Path

import torch

from diffusion_models.diffusion import (
    ddim_sample_loop,
    make_linear_ddpm_schedule,
    p_sample_loop,
)
from diffusion_models.models import CIFAR10UNet, primary_unet_config
from diffusion_models.models.unet import UNetConfig
from diffusion_models.recovery import atomic_publish, read_checkpoint, sha256

SCHEMA_VERSION = 1
NORMALIZATION = {"model_range": [-1, 1], "display": "(x + 1) / 2, clamped to [0,1]"}


def load_bundle(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Inference bundle missing: {path}; export a trusted training checkpoint first."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["schema_version"] != SCHEMA_VERSION
            or payload["normalization"] != NORMALIZATION
        ):
            raise ValueError("unsupported schema or normalization")
        if payload["diffusion"]["schedule"] != "linear":
            raise ValueError("only the recorded linear schedule is supported")
        if type(payload["training_step"]) is not int or payload["training_step"] < 0:
            raise ValueError("invalid training step")
        config = UNetConfig(**payload["architecture"])
        model = CIFAR10UNet(config)
        if any(
            t.dtype != torch.float32 or not torch.isfinite(t).all()
            for t in payload["ema_state_dict"].values()
        ):
            raise ValueError("bundle weights must be finite float32")
        model.load_state_dict(payload["ema_state_dict"], strict=True)
        schedule = make_linear_ddpm_schedule(
            **{
                k: payload["diffusion"][k]
                for k in ("num_steps", "beta_start", "beta_end")
            },
            device=device,
        )
        return model.to(device).eval(), schedule, payload
    except Exception as error:
        raise ValueError(f"Incompatible inference bundle {path}: {error}") from error


def export_checkpoint(source, destination, *, expected_sha256, attribution):
    """Export only a trusted checksum-identified full-training checkpoint."""
    if sha256(source) != expected_sha256:
        raise ValueError(
            "Source checkpoint checksum differs from the supplied identity"
        )
    payload = read_checkpoint(source)
    config = payload["configuration"]["config"]
    if (
        config["model"]["configuration"] != "primary_unet"
        or config["training"]["precision"] != "float32"
    ):
        raise ValueError("Only the recorded primary float32 U-Net is exportable")
    model = CIFAR10UNet(primary_unet_config())
    model.load_state_dict(payload["model_state_dict"], strict=True)
    ema = payload["ema_state_dict"]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "ema_state_dict": {**ema["parameters"], **ema["buffers"]},
        "architecture": asdict(primary_unet_config()),
        "diffusion": config["diffusion"],
        "training_step": payload["global_step"],
        "source_sha256": expected_sha256,
        "source_metadata": payload["metadata"],
        "normalization": NORMALIZATION,
        "attribution": attribution,
    }
    return atomic_publish(
        destination, lambda handle: torch.save(bundle, handle), load_bundle
    )


def sample_bundle(path, *, device="cpu", sampler="ddim", steps=50, count=16, seed=0):
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if count < 1 or seed < 0:
        raise ValueError("count must be positive and seed nonnegative")
    model, schedule, metadata = load_bundle(path, device)
    config = model.config
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (count, config.in_channels, config.image_size, config.image_size)
    if sampler == "ddim":
        result = ddim_sample_loop(
            model, schedule, shape, generator=generator, num_inference_steps=steps
        )
    elif sampler == "ddpm":
        result = p_sample_loop(model, schedule, shape, generator=generator)
    else:
        raise ValueError("sampler must be ddim or ddpm")
    return result.sample.cpu(), metadata
