#!/usr/bin/env python3
"""Compare the E006 first-1K CIFAR-10 radial spectrum with all 50K images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np


IMAGE_SIZE = 32
CHANNELS = 3
RMAX = 22
E006_SIGMAS = (
    0.02,
    0.05,
    0.1,
    0.14,
    0.2,
    0.3,
    0.4,
    0.6,
    0.8,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    8.0,
    12.0,
    20.0,
    40.0,
    80.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument(
        "--material-relative-threshold",
        type=float,
        default=0.05,
        help="Predeclared materiality threshold for max absolute shell-relative change.",
    )
    parser.add_argument(
        "--allow-non-slurm",
        action="store_true",
        help="Only for tiny tests; production dataset processing must use Slurm.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_batch(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    raw = np.asarray(payload[b"data"], dtype=np.float64)
    return raw.reshape(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE) / 127.5 - 1.0


def radius_grids_and_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(IMAGE_SIZE) * IMAGE_SIZE
    fx = np.fft.rfftfreq(IMAGE_SIZE) * IMAGE_SIZE
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.floor(np.sqrt(xx * xx + yy * yy) + 1e-12).astype(np.int64)
    weights = np.full_like(radius, 2.0, dtype=np.float64)
    weights[:, 0] = 1.0
    weights[:, -1] = 1.0

    full_f = np.fft.fftfreq(IMAGE_SIZE) * IMAGE_SIZE
    full_yy, full_xx = np.meshgrid(full_f, full_f, indexing="ij")
    full_radius = np.floor(
        np.sqrt(full_xx * full_xx + full_yy * full_yy) + 1e-12
    ).astype(np.int64)
    return radius, weights, full_radius


def radial_power(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce E006 radial_power for an in-memory NCHW float64 batch."""
    radius, weights, full_radius = radius_grids_and_weights()
    spectrum = np.fft.rfft2(images, axes=(-2, -1), norm="ortho")
    weighted_power = np.square(np.abs(spectrum)) * weights[None, None, :, :]
    powers = np.empty(RMAX + 1, dtype=np.float64)
    counts = np.empty(RMAX + 1, dtype=np.int64)
    for shell in range(RMAX + 1):
        mask = radius == shell
        counts[shell] = int(np.sum(full_radius == shell))
        shell_sum = weighted_power[:, :, mask].sum(axis=2)
        powers[shell] = float(np.mean(shell_sum / counts[shell]))
    return powers, counts


def aggregate_all_training_images(
    data_dir: Path, batch_size: int
) -> tuple[np.ndarray, np.ndarray, int, dict[str, str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths = [data_dir / f"data_batch_{index}" for index in range(1, 6)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-10 training batches: {missing}")

    weighted_sum = np.zeros(RMAX + 1, dtype=np.float64)
    expected_counts: np.ndarray | None = None
    total = 0
    hashes = {path.name: sha256(path) for path in paths}
    for path in paths:
        images = load_batch(path)
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            powers, counts = radial_power(chunk)
            if expected_counts is None:
                expected_counts = counts
            elif not np.array_equal(counts, expected_counts):
                raise RuntimeError("Radial multiplicities changed between batches")
            weighted_sum += len(chunk) * powers
            total += len(chunk)
    if total != 50_000 or expected_counts is None:
        raise RuntimeError(f"Expected 50,000 CIFAR-10 training images, found {total}")
    return weighted_sum / total, expected_counts, total, hashes


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    radii = [int(row["radius"]) for row in rows]
    if radii != list(range(RMAX + 1)):
        raise ValueError(f"Reference radii must be contiguous [0, {RMAX}]")
    powers = np.asarray([float(row["shell_power"]) for row in rows])
    counts = np.asarray(
        [int(row["full_fft_sites_per_channel"]) for row in rows], dtype=np.int64
    )
    return powers, counts


def effective_sigmas(num_steps: int, beta_start: float, beta_end: float) -> np.ndarray:
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    alpha_bars = np.cumprod(1.0 - betas)
    return np.sqrt((1.0 - alpha_bars) / alpha_bars)


def information_radius(powers: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    valid = powers[None, :] >= np.square(sigmas[:, None])
    radii = np.arange(len(powers), dtype=np.int64)[None, :]
    return np.where(valid, radii, -1).max(axis=1)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    output_path: Path,
    reference: np.ndarray,
    full: np.ndarray,
    relative: np.ndarray,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False
    radii = np.arange(RMAX + 1)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True)
    axes[0].semilogy(radii, reference, "o-", label="E006 first 1K")
    axes[0].semilogy(radii, full, "s--", label="CIFAR-10 all 50K")
    axes[0].set(xlabel="Radial shell r", ylabel="Mean power per coefficient")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].bar(radii, 100.0 * relative)
    axes[1].set(xlabel="Radial shell r", ylabel="Relative change (%)")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    args = parse_args()
    if not args.allow_non_slurm and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Refusing dataset processing outside a Slurm allocation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    reference, reference_counts = read_reference(args.reference_csv)
    full, counts, image_count, dataset_hashes = aggregate_all_training_images(
        args.data_dir, args.batch_size
    )
    if not np.array_equal(counts, reference_counts):
        raise RuntimeError("50K and E006 shell multiplicities disagree")
    relative = (full - reference) / reference

    radial_rows = [
        {
            "radius": radius,
            "shell_power": full[radius],
            "full_fft_sites_per_channel": int(counts[radius]),
        }
        for radius in range(RMAX + 1)
    ]
    write_csv(
        args.output_dir / "radial_power_50k.csv",
        ["radius", "shell_power", "full_fft_sites_per_channel"],
        radial_rows,
    )
    comparison_rows = [
        {
            "radius": radius,
            "shell_power_1k": reference[radius],
            "shell_power_50k": full[radius],
            "relative_difference": relative[radius],
            "absolute_relative_difference": abs(relative[radius]),
            "full_fft_sites_per_channel": int(counts[radius]),
        }
        for radius in range(RMAX + 1)
    ]
    write_csv(
        args.output_dir / "radial_power_comparison.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )

    ddpm_sigmas = effective_sigmas(args.num_steps, args.beta_start, args.beta_end)
    ddpm_1k = information_radius(reference, ddpm_sigmas)
    ddpm_50k = information_radius(full, ddpm_sigmas)
    timestep_rows = [
        {
            "timestep": timestep,
            "sigma_eff": ddpm_sigmas[timestep],
            "R_MI_1k": int(ddpm_1k[timestep]),
            "R_MI_50k": int(ddpm_50k[timestep]),
            "shell_difference": int(ddpm_50k[timestep] - ddpm_1k[timestep]),
        }
        for timestep in range(args.num_steps)
    ]
    write_csv(
        args.output_dir / "information_radius_ddpm.csv",
        list(timestep_rows[0]),
        timestep_rows,
    )
    e006_sigmas = np.asarray(E006_SIGMAS)
    e006_1k = information_radius(reference, e006_sigmas)
    e006_50k = information_radius(full, e006_sigmas)
    e006_rows = [
        {
            "sigma": sigma,
            "R_MI_1k": int(one),
            "R_MI_50k": int(fifty),
            "shell_difference": int(fifty - one),
        }
        for sigma, one, fifty in zip(e006_sigmas, e006_1k, e006_50k)
    ]
    write_csv(
        args.output_dir / "information_radius_e006_grid.csv",
        list(e006_rows[0]),
        e006_rows,
    )
    figure_created = plot_comparison(
        args.output_dir / "radial_power_1k_vs_50k.png", reference, full, relative
    )

    max_index = int(np.argmax(np.abs(relative)))
    changed = ddpm_1k != ddpm_50k
    shell_material = bool(
        np.max(np.abs(relative)) > args.material_relative_threshold
    )
    crossing_material = bool(
        changed.mean() > args.material_relative_threshold
        or np.max(np.abs(ddpm_50k - ddpm_1k)) > 1
    )
    material = shell_material or crossing_material
    summary = {
        "status": "completed",
        "decision_rule": {
            "metrics": [
                "maximum absolute per-shell relative difference",
                "fraction of DDPM timesteps with changed R_MI",
                "maximum absolute DDPM R_MI shell difference",
            ],
            "material_relative_threshold": args.material_relative_threshold,
            "maximum_allowed_R_MI_shell_difference": 1,
            "use_50k_if_material": True,
        },
        "spectrum_decision": "use_50k" if material else "retain_e006_1k",
        "material_difference": material,
        "shell_power_difference_material": shell_material,
        "R_MI_difference_material": crossing_material,
        "max_absolute_relative_difference": float(np.max(np.abs(relative))),
        "max_difference_radius": max_index,
        "mean_absolute_relative_difference": float(np.mean(np.abs(relative))),
        "ddpm_timesteps_with_changed_R_MI": int(changed.sum()),
        "ddpm_fraction_with_changed_R_MI": float(changed.mean()),
        "ddpm_max_absolute_R_MI_difference": int(np.max(np.abs(ddpm_50k - ddpm_1k))),
        "e006_grid_points_with_changed_R_MI": int(np.sum(e006_1k != e006_50k)),
        "train_count": image_count,
        "normalization": "x / 127.5 - 1.0",
        "fft_convention": "numpy rfft2 norm=ortho with exact Parseval half-plane weights",
        "radial_binning": "floor(sqrt(kx^2+ky^2)+1e-12)",
        "ddpm_schedule": {
            "num_steps": args.num_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "sigma_eff": "sqrt((1-alpha_bar)/alpha_bar)",
        },
        "reference_csv": str(args.reference_csv.resolve()),
        "reference_csv_sha256": sha256(args.reference_csv),
        "dataset_file_sha256": dataset_hashes,
        "script_sha256": sha256(Path(__file__)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "runtime_seconds": time.time() - started,
        "warnings": []
        if figure_created
        else ["matplotlib unavailable; comparison PNG was not generated"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
