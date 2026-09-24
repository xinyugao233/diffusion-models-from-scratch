#!/usr/bin/env python3
"""Fail closed if the frozen Study 2 10K protocol drifts."""

from __future__ import annotations

import argparse
import json
import math
import random

import torch
from _experiment_utils import load_json, repository_path

from diffusion_models.diffusion import make_linear_ddpm_schedule
from diffusion_models.models import CIFAR10UNet, primary_unet_config
from diffusion_models.spectral_boundary import load_radial_power
from diffusion_models.spectral_noise import (
    SpectralNoiseConfig,
    build_spectral_noise_schedule,
)
from diffusion_models.spectral_noise_study import (
    resolve_spectral_noise_study_configuration,
    state_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    source = load_json(repository_path(args.config))
    if set(source["study_conditions"]) != {
        "baseline",
        "moving_narrow",
        "moving_broad",
    }:
        raise RuntimeError("Study 2 conditions differ from the frozen design.")
    if [entry["seed"] for entry in source["paired_seeds"]] != [0, 1, 2]:
        raise RuntimeError("Study 2 paired seeds must be exactly 0, 1, and 2.")
    if source["training"]["steps"] != 10000:
        raise RuntimeError("Study 2 must run exactly 10,000 steps.")
    if source["training"]["checkpoint_steps"] != [1000, 2000, 5000, 10000]:
        raise RuntimeError("Study 2 checkpoint protocol changed.")
    if source["weighted_noise"]["rho"] != 1.0:
        raise RuntimeError("Study 2 rho must remain one.")

    initial_hashes: dict[str, dict[str, str]] = {}
    for seed in (0, 1, 2):
        initial_hashes[str(seed)] = {}
        for condition in source["study_conditions"]:
            resolved = resolve_spectral_noise_study_configuration(
                source, condition=condition, seed=seed
            )
            random.seed(resolved["model"]["initialization_seed"])
            torch.manual_seed(resolved["model"]["initialization_seed"])
            model = CIFAR10UNet(primary_unet_config())
            initial_hashes[str(seed)][condition] = state_sha256(model)
        if len(set(initial_hashes[str(seed)].values())) != 1:
            raise RuntimeError(f"Initial model hashes differ within seed {seed}.")

    baseline = make_linear_ddpm_schedule(dtype=torch.float32)
    radial_power = load_radial_power(repository_path(source["spectrum"]))
    schedule_summaries = {}
    for condition in ("moving_narrow", "moving_broad"):
        values = source["study_conditions"][condition]
        schedule = build_spectral_noise_schedule(
            baseline,
            radial_power,
            SpectralNoiseConfig(
                tau=float(values["tau"]),
                weight_floor=float(source["weighted_noise"]["weight_floor"]),
                rho=float(source["weighted_noise"]["rho"]),
                allocation=source["weighted_noise"]["allocation"],
                fft_normalization=source["weighted_noise"]["fft_normalization"],
            ),
        )
        terminal_difference = float(
            (schedule.alpha_bars[-1] - baseline.alpha_bars[-1]).abs().max()
        )
        if terminal_difference > 1e-15:
            raise RuntimeError("Terminal attenuation exceeds frozen tolerance.")
        if not torch.all(schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1]):
            raise RuntimeError("Cumulative attenuation is not monotone.")
        if not math.isfinite(float(schedule.betas.min())) or not math.isfinite(
            float(schedule.betas.max())
        ):
            raise RuntimeError("Schedule beta range is nonfinite.")
        schedule_summaries[condition] = {
            "tau": schedule.config.tau,
            "beta_min": float(schedule.betas.min()),
            "beta_max": float(schedule.betas.max()),
            "terminal_max_abs_difference": terminal_difference,
        }

    result = {
        "passed": True,
        "conditions": list(source["study_conditions"]),
        "paired_seeds": [0, 1, 2],
        "initial_model_hashes": initial_hashes,
        "schedule_summaries": schedule_summaries,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
