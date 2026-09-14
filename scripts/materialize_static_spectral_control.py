#!/usr/bin/env python3
"""Materialize and validate the preregistered static narrow spectral control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch

from diffusion_models.diffusion import make_linear_ddpm_schedule
from diffusion_models.spectral_boundary import (
    SpectralBoundaryConfig,
    compute_spectral_weights,
    compute_static_matched_weights,
    load_radial_power,
    load_static_spectral_control,
    radial_shell_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radial-power", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--tau", type=float, default=math.log(2.0))
    parser.add_argument("--weight-floor", type=float, default=0.1)
    parser.add_argument("--allow-non-slurm", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if not args.allow_non_slurm and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Refusing static-control materialization outside Slurm")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    radial_power = load_radial_power(args.radial_power)
    schedule = make_linear_ddpm_schedule(
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        dtype=torch.float32,
    )
    moving = SpectralBoundaryConfig(
        tau=args.tau,
        weight_floor=args.weight_floor,
        normalization="coefficient_mean",
        fft_normalization="ortho",
    )
    static = compute_static_matched_weights(
        radial_power, schedule, moving, dtype=torch.float64
    )

    csv_path = args.output_dir / "static_narrow_matched_50k.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "radius",
                "shell_power",
                "full_fft_sites_per_channel",
                "static_weight",
            ],
        )
        writer.writeheader()
        for radius in range(radial_power.power.numel()):
            writer.writerow(
                {
                    "radius": radius,
                    "shell_power": float(radial_power.power[radius]),
                    "full_fft_sites_per_channel": int(
                        radial_power.coefficient_counts[radius]
                    ),
                    "static_weight": float(static[radius]),
                }
            )

    loaded_power, loaded_static = load_static_spectral_control(csv_path)
    serialized_error = torch.max(torch.abs(loaded_static.weights - static)).item()
    probe_sigma = torch.tensor([0.01, 1.0, 100.0], dtype=torch.float64)
    probe_weights = compute_spectral_weights(loaded_power, probe_sigma, loaded_static)
    timestep_invariant = bool(
        torch.equal(probe_weights[0], probe_weights[1])
        and torch.equal(probe_weights[1], probe_weights[2])
    )
    coefficient_mean = (
        loaded_static.weights * loaded_power.coefficient_counts
    ).sum() / loaded_power.coefficient_counts.sum()
    shell_grid = radial_shell_indices(32, 32, device=torch.device("cpu"))
    observed_counts = torch.bincount(
        shell_grid.flatten(), minlength=loaded_power.power.numel()
    ).to(torch.float64)
    multiplicity_exact = bool(
        torch.equal(observed_counts, loaded_power.coefficient_counts)
    )
    checks = {
        "timestep_invariance_exact": timestep_invariant,
        "coefficient_mean": float(coefficient_mean),
        "coefficient_mean_absolute_error": abs(float(coefficient_mean) - 1.0),
        "serialized_marginal_max_absolute_error": serialized_error,
        "serialized_marginal_exact": serialized_error == 0.0,
        "full_fft_multiplicity_exact": multiplicity_exact,
        "full_fft_coefficient_count": int(observed_counts.sum()),
    }
    if not all(
        (
            checks["timestep_invariance_exact"],
            checks["coefficient_mean_absolute_error"] < 1e-14,
            checks["serialized_marginal_exact"],
            checks["full_fft_multiplicity_exact"],
        )
    ):
        raise RuntimeError(f"Static-control validation failed: {checks}")

    manifest = {
        "status": "completed",
        "object": "static narrow matched-spectrum control",
        "definition": "uniform mean over t=0,...,999 of individually coefficient-mean-normalized moving narrow weights",
        "source_radial_power": str(args.radial_power.resolve()),
        "source_radial_power_sha256": sha256(args.radial_power),
        "static_control_csv": str(csv_path.resolve()),
        "static_control_csv_sha256": sha256(csv_path),
        "tau": args.tau,
        "weight_floor": args.weight_floor,
        "normalization": "coefficient_mean",
        "fft_normalization": "ortho",
        "timestep_distribution": "uniform discrete over every DDPM timestep",
        "schedule": {
            "implementation_dtype": "torch.float32",
            "num_steps": args.num_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "sigma_eff": "sqrt((1-alpha_bar)/alpha_bar)",
        },
        "static_weight_range": {
            "minimum": float(static.min()),
            "maximum": float(static.max()),
        },
        "checks": checks,
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
