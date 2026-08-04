"""Generate a deterministic visual check of the CIFAR-10 pipeline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from diffusion_models.data import inverse_normalize, load_cifar10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cifar10_train_grid_seed0.png"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images <= 0 or args.nrow <= 0:
        raise ValueError("num-images and nrow must be positive.")

    dataset = load_cifar10(
        args.data_root,
        train=True,
        download=not args.no_download,
    )
    if args.num_images > len(dataset):
        raise ValueError(
            f"Requested {args.num_images} images from a dataset of size {len(dataset)}."
        )

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_images]
    images = torch.stack([dataset[int(index)][0] for index in indices])
    display_images = inverse_normalize(images, clamp=True)

    # White padding makes image boundaries unambiguous without modifying pixels.
    grid = make_grid(
        display_images,
        nrow=args.nrow,
        padding=2,
        pad_value=1.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.output)

    rows = math.ceil(args.num_images / args.nrow)
    print(
        f"Saved {args.num_images} images as a {rows}x{args.nrow} grid to "
        f"{args.output} (seed={args.seed})."
    )


if __name__ == "__main__":
    main()
