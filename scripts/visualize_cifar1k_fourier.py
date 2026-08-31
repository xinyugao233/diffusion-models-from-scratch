#!/usr/bin/env python3
"""Visual audit of the exact first-1,000 CIFAR-10 subset and its FFT channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_first_1000(root: Path) -> tuple[np.ndarray, np.ndarray]:
    with (root / "data_batch_1").open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    images = np.asarray(payload[b"data"][:1000], dtype=np.uint8).reshape(
        -1, 3, 32, 32
    )
    labels = np.asarray(payload[b"labels"][:1000], dtype=np.int64)
    return images, labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb(image: np.ndarray, scale: int = 4) -> Image.Image:
    array = np.moveaxis(image, 0, -1)
    return Image.fromarray(array, "RGB").resize((32 * scale, 32 * scale))


def normalize_rgb(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> Image.Image:
    lo, hi = np.percentile(values, [low, high], axis=(-2, -1), keepdims=True)
    scaled = np.clip((values - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
    array = np.moveaxis((scaled * 255).astype(np.uint8), 0, -1)
    return Image.fromarray(array, "RGB").resize((128, 128))


def signed_gray(values: np.ndarray) -> Image.Image:
    bound = max(float(np.percentile(np.abs(values), 99)), 1e-12)
    scaled = np.clip(0.5 + values / (2 * bound), 0.0, 1.0)
    return Image.fromarray((scaled * 255).astype(np.uint8), "L").resize((96, 96))


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    box = draw.textbbox((0, 0), text)
    draw.text((x - (box[2] - box[0]) / 2, y), text, fill="black")


def save_subset_contact(images: np.ndarray, labels: np.ndarray, path: Path) -> list[int]:
    indices = np.linspace(0, 999, 100, dtype=int).tolist()
    cell, label_h = 96, 22
    canvas = Image.new("RGB", (10 * cell, 10 * (cell + label_h) + 38), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "100 evenly spaced examples from the exact first 1,000 CIFAR-10 training images", fill="black")
    for position, index in enumerate(indices):
        row, column = divmod(position, 10)
        tile = rgb(images[index], scale=3)
        x, y = column * cell, 38 + row * (cell + label_h)
        canvas.paste(tile, (x, y))
        draw_centered(draw, (x + cell // 2, y + cell), f"{index}: {LABELS[labels[index]]}")
    canvas.save(path)
    return indices


def save_fourier_overview(images: np.ndarray, labels: np.ndarray, path: Path) -> list[int]:
    indices = np.linspace(0, 999, 8, dtype=int).tolist()
    columns = ("original", "log magnitude (RGB)", "phase (RGB)", "exact IFFT")
    cell, header, row_h, left = 128, 50, 154, 180
    canvas = Image.new("RGB", (left + len(columns) * cell, header + len(indices) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(columns):
        draw_centered(draw, (left + column * cell + cell // 2, 16), title)
    for row, index in enumerate(indices):
        normalized = images[index].astype(np.float64) / 127.5 - 1.0
        spectrum = np.fft.fftshift(np.fft.fft2(normalized, norm="ortho"), axes=(-2, -1))
        magnitude = np.log1p(np.abs(spectrum))
        phase = (np.angle(spectrum) + np.pi) / (2 * np.pi)
        reconstructed = np.fft.ifft2(np.fft.ifftshift(spectrum, axes=(-2, -1)), norm="ortho").real
        reconstructed_u8 = np.clip((reconstructed + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
        y = header + row * row_h
        draw.text((8, y + 50), f"index {index}\n{LABELS[labels[index]]}", fill="black")
        tiles = (
            rgb(images[index]),
            normalize_rgb(magnitude),
            Image.fromarray(np.moveaxis((phase * 255).astype(np.uint8), 0, -1), "RGB").resize((128, 128)),
            rgb(reconstructed_u8),
        )
        for column, tile in enumerate(tiles):
            canvas.paste(tile, (left + column * cell, y))
        error = np.max(np.abs(reconstructed - normalized))
        draw.text((left + 3 * cell + 4, y + 130), f"max error {error:.1e}", fill="black")
    canvas.save(path)
    return indices


def save_six_channels(images: np.ndarray, labels: np.ndarray, path: Path) -> list[int]:
    indices = [0, 333, 666, 999]
    titles = ("original", "Re R", "Im R", "Re G", "Im G", "Re B", "Im B")
    cell, header, row_h, left = 96, 48, 120, 150
    canvas = Image.new("RGB", (left + len(titles) * cell, header + len(indices) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        draw_centered(draw, (left + column * cell + cell // 2, 15), title)
    for row, index in enumerate(indices):
        normalized = images[index].astype(np.float64) / 127.5 - 1.0
        spectrum = np.fft.fftshift(np.fft.fft2(normalized, norm="ortho"), axes=(-2, -1))
        channels = np.stack((spectrum.real, spectrum.imag), axis=1).reshape(6, 32, 32)
        y = header + row * row_h
        draw.text((8, y + 35), f"index {index}\n{LABELS[labels[index]]}", fill="black")
        canvas.paste(rgb(images[index], scale=3), (left, y))
        for channel in range(6):
            tile = signed_gray(channels[channel]).convert("RGB")
            canvas.paste(tile, (left + (channel + 1) * cell, y))
        draw.text((left + cell, y + 98), "gray=0, white=+, black=- (per-panel robust scale)", fill="black")
    canvas.save(path)
    return indices


def save_histogram(labels: np.ndarray, path: Path) -> None:
    counts = np.bincount(labels, minlength=10)
    width, height, left, top, bottom = 900, 420, 80, 45, 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    maximum = int(counts.max())
    plot_h = height - top - bottom
    bar_w = 58
    gap = 24
    for index, count in enumerate(counts):
        x = left + index * (bar_w + gap)
        bar_h = int(plot_h * count / maximum)
        draw.rectangle((x, top + plot_h - bar_h, x + bar_w, top + plot_h), fill="#2563eb")
        draw_centered(draw, (x + bar_w // 2, top + plot_h - bar_h - 18), str(int(count)))
        draw_centered(draw, (x + bar_w // 2, top + plot_h + 10), LABELS[index])
    draw.text((left, 15), "Class counts in the first 1,000 canonical CIFAR-10 training examples", fill="black")
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels = load_first_1000(data_dir)
    contact_indices = save_subset_contact(images, labels, output_dir / "cifar1k_examples_100.png")
    overview_indices = save_fourier_overview(images, labels, output_dir / "fourier_examples_overview.png")
    channel_indices = save_six_channels(images, labels, output_dir / "fourier_six_channels.png")
    save_histogram(labels, output_dir / "cifar1k_class_counts.png")
    manifest = {
        "selection": "first 1000 examples of CIFAR-10 data_batch_1 in canonical order",
        "indices": [0, 999],
        "data_batch_1_sha256": sha256(data_dir / "data_batch_1"),
        "class_counts": {LABELS[i]: int(v) for i, v in enumerate(np.bincount(labels, minlength=10))},
        "contact_indices": contact_indices,
        "overview_indices": overview_indices,
        "six_channel_indices": channel_indices,
        "fft": "numpy fft2 norm=ortho; full complex spectrum; fftshift only for display",
    }
    (output_dir / "subset_visual_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
