"""Small provenance and plotting helpers for local experiment scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def repository_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_json(path: str | Path) -> dict[str, Any]:
    with repository_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_exclusive(path: str | Path, value: Any) -> Path:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "working_tree_dirty": bool(status)}


def environment_identity() -> dict[str, Any]:
    return {
        "device": "cpu",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


def write_loss_csv(path: str | Path, losses: list[float]) -> Path:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(
            {"step": step, "loss": loss} for step, loss in enumerate(losses)
        )
    return destination


def save_loss_curve(path: str | Path, losses: list[float]) -> Path:
    """Save a dependency-light, labeled PNG curve for experiment inspection."""
    if len(losses) < 2:
        raise ValueError("At least two losses are required for a curve.")
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {destination}")

    width, height = 1200, 700
    left, right, top, bottom = 105, 50, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    minimum = min(losses)
    maximum = max(losses)
    span = max(maximum - minimum, 1e-12)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((left, top, left, top + plot_height), fill="black", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="black",
        width=2,
    )

    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height - fraction * plot_height
        value = minimum + fraction * span
        draw.line((left - 6, y, left, y), fill="black", width=2)
        draw.line((left, y, left + plot_width, y), fill="#e5e7eb", width=1)
        draw.text((10, y - 7), f"{value:.3f}", fill="black")

    points = []
    for step, loss in enumerate(losses):
        x = left + step / (len(losses) - 1) * plot_width
        y = top + (maximum - loss) / span * plot_height
        points.append((x, y))
    draw.line(points, fill="#2563eb", width=3)
    draw.text((left, 24), "Fixed-16 CIFAR-10 epsilon-prediction loss", fill="black")
    draw.text((left + plot_width / 2 - 20, height - 38), "Step", fill="black")
    draw.text((14, top - 24), "MSE", fill="black")
    draw.text((left, top + plot_height + 18), "0", fill="black")
    draw.text(
        (left + plot_width - 30, top + plot_height + 18),
        str(len(losses) - 1),
        fill="black",
    )
    image.save(destination)
    return destination
