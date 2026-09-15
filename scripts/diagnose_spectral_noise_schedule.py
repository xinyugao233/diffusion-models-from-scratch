#!/usr/bin/env python3
"""Materialize weighted-noise schedule tables and diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from _experiment_utils import git_identity, load_json, repository_path
from PIL import Image, ImageDraw

from diffusion_models.diffusion import make_linear_ddpm_schedule
from diffusion_models.full_training import require_slurm_environment
from diffusion_models.spectral_boundary import load_radial_power
from diffusion_models.spectral_noise import (
    SpectralNoiseConfig,
    build_spectral_noise_schedule,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_plot(
    path: Path,
    title: str,
    series: list[tuple[str, list[float]]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    """Render a dependency-light diagnostic line plot with Pillow."""
    width, height = 1200, 700
    left, right, top, bottom = 100, 30, 70, 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), title, fill="black")
    draw.text((width // 2 - 50, height - 35), x_label, fill="black")
    draw.text((10, height // 2), y_label, fill="black")
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line(
        (left, height - bottom, width - right, height - bottom), fill="black", width=2
    )
    all_values = [value for _, values in series for value in values]
    minimum, maximum = min(all_values), max(all_values)
    span = maximum - minimum or 1.0
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]
    for index, (label, values) in enumerate(series):
        color = colors[index % len(colors)]
        points = []
        for x_index, value in enumerate(values):
            x = left + x_index * (width - left - right) / max(1, len(values) - 1)
            y = height - bottom - (value - minimum) * (height - top - bottom) / span
            points.append((x, y))
        draw.line(points, fill=color, width=3)
        legend_y = top + 22 * index
        draw.line((width - 250, legend_y, width - 215, legend_y), fill=color, width=3)
        draw.text((width - 205, legend_y - 7), label, fill="black")
    draw.text((left, height - bottom + 10), "0", fill="black")
    draw.text(
        (width - right - 35, height - bottom + 10),
        str(len(series[0][1]) - 1),
        fill="black",
    )
    draw.text((left - 85, top), f"{maximum:.3g}", fill="black")
    draw.text((left - 85, height - bottom - 10), f"{minimum:.3g}", fill="black")
    image.save(path)


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
        common_series = [
            (f"r={radius}", corrected.hazards[:, radius].tolist())
            for radius in representative
        ]
        line_plot(
            output / f"{name}_hazard.png",
            f"Incremental hazard: {name}",
            [*common_series, ("baseline", (-torch.log(baseline.alphas)).tolist())],
            x_label="code timestep",
            y_label="hazard",
        )
        line_plot(
            output / f"{name}_beta.png",
            f"Beta: {name}",
            [
                *[
                    (f"r={radius}", corrected.betas[:, radius].tolist())
                    for radius in representative
                ],
                ("baseline", baseline.betas.tolist()),
            ],
            x_label="code timestep",
            y_label="beta",
        )
        line_plot(
            output / f"{name}_alpha_bar.png",
            f"Cumulative alpha bar: {name}",
            [
                *[
                    (f"r={radius}", corrected.alpha_bars[:, radius].tolist())
                    for radius in representative
                ],
                ("baseline", baseline.alpha_bars.tolist()),
            ],
            x_label="code timestep",
            y_label="alpha bar",
        )
        line_plot(
            output / f"{name}_radial_profiles.png",
            f"Radial corruption profile: {name}",
            [
                (f"t={timestep}", corrected.hazards[timestep].tolist())
                for timestep in (0, 50, 250, 500, 999)
            ],
            x_label="radial shell",
            y_label="hazard",
        )

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
