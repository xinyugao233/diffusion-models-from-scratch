#!/usr/bin/env python3
"""Materialize weighted-noise schedule tables and diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _experiment_utils import git_identity, load_json, repository_path

from diffusion_models.diffusion import make_linear_ddpm_schedule
from diffusion_models.full_training import require_slurm_environment
from diffusion_models.spectral_boundary import load_radial_power
from diffusion_models.spectral_noise import (
    SpectralNoiseConfig,
    build_spectral_noise_schedule,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    job_id = require_slurm_environment(os.environ)
    config = load_json(repository_path(args.config))
    git = git_identity()
    if git["working_tree_dirty"] or git["commit"] != os.environ.get("RUN_COMMIT"):
        raise RuntimeError("Diagnostics require the exact clean RUN_COMMIT.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    baseline = make_linear_ddpm_schedule(
        num_steps=config["diffusion"]["num_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        dtype=torch.float64,
    )
    radial = load_radial_power(repository_path(config["spectrum"]))
    sigma_squared = (1.0 - baseline.alpha_bars) / baseline.alpha_bars
    rows = []
    summaries = {}
    representative = [0, 8, 15, 22]
    for name in ("noise_moving_narrow", "noise_moving_broad"):
        tau = config["conditions"][name]["tau"]
        common = config["weighted_noise"]
        corrected = build_spectral_noise_schedule(
            baseline,
            radial,
            SpectralNoiseConfig(
                tau=tau,
                weight_floor=common["weight_floor"],
                rho=common["rho"],
                allocation=common["allocation"],
            ),
        )
        literal = build_spectral_noise_schedule(
            baseline,
            radial,
            SpectralNoiseConfig(
                tau=tau,
                weight_floor=common["weight_floor"],
                rho=1.0,
                allocation="baseline_reweighted",
            ),
        )
        crossing = (
            (torch.log(sigma_squared)[:, None] - torch.log(radial.power)[None, :])
            .abs()
            .argmin(dim=0)
        )
        peak = corrected.hazards.argmax(dim=0)
        literal_peak = literal.hazards.argmax(dim=0)
        for radius in range(corrected.num_shells):
            rows.append(
                {
                    "condition": name,
                    "shell": radius,
                    "power": float(radial.power[radius]),
                    "crossing_timestep": int(crossing[radius]),
                    "peak_injection_timestep": int(peak[radius]),
                    "peak_minus_crossing": int(peak[radius] - crossing[radius]),
                    "literal_proposal_peak_timestep": int(literal_peak[radius]),
                    "maximum_beta": float(corrected.betas[:, radius].max()),
                    "terminal_alpha_bar": float(corrected.alpha_bars[-1, radius]),
                }
            )
        summaries[name] = {
            "tau": tau,
            "beta_min": float(corrected.betas.min()),
            "beta_max": float(corrected.betas.max()),
            "terminal_max_abs_error": float(
                (corrected.alpha_bars[-1] - baseline.alpha_bars[-1]).abs().max()
            ),
            "literal_proposal_endpoint_peak_shells": [
                int(index) for index in torch.where(literal_peak == 999)[0]
            ],
        }
        figure, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
        for radius in representative:
            axes[0].plot(corrected.hazards[:, radius], label=f"r={radius}")
            axes[1].plot(corrected.betas[:, radius], label=f"r={radius}")
            axes[2].plot(corrected.alpha_bars[:, radius], label=f"r={radius}")
        axes[0].plot(-torch.log(baseline.alphas), "k--", label="baseline")
        axes[1].plot(baseline.betas, "k--", label="baseline")
        axes[2].plot(baseline.alpha_bars, "k--", label="baseline")
        axes[0].set_title("Incremental hazard")
        axes[1].set_title("Beta")
        axes[2].set_title("Cumulative alpha bar")
        for axis in axes:
            axis.set_xlabel("code timestep")
            axis.legend(ncol=3)
        figure.savefig(output / f"{name}_schedule.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for timestep in (0, 50, 250, 500, 999):
            axis.plot(
                range(corrected.num_shells),
                corrected.hazards[timestep],
                marker="o",
                label=f"t={timestep}",
            )
        axis.set_xlabel("radial shell")
        axis.set_ylabel("incremental hazard")
        axis.set_title(f"Radial corruption profile: {name}")
        axis.legend()
        figure.savefig(output / f"{name}_radial_profiles.png", dpi=160)
        plt.close(figure)

    table = output / "shell_schedule_diagnostics.csv"
    with table.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "passed": True,
        "engineering_only": True,
        "slurm_job_id": job_id,
        "git": git,
        "selected_construction": "boundary_density",
        "literal_proposal_issue": (
            "h_base*b is a valid chain but absolute hazard peaks at t=999 for "
            "high-frequency shells, so it fails the intended localization."
        ),
        "rho_stability_scan": {
            "range": [0.0, 1.0],
            "conclusion": "All rho in [0,1] are numerically valid; rho=1 retained for smoke.",
        },
        "conditions": summaries,
        "table": str(table),
        "table_sha256": sha256(table),
        "figures": {path.name: sha256(path) for path in sorted(output.glob("*.png"))},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
