"""Visualize closed-form forward noising of one CIFAR-10 image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from diffusion_models.data import inverse_normalize, load_cifar10
from diffusion_models.diffusion import make_linear_ddpm_schedule, q_sample

TIMESTEPS = (0, 100, 250, 500, 750, 999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cifar10_forward_process_seed0.png"),
    )
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_cifar10(
        args.data_root,
        train=True,
        download=not args.no_download,
    )
    if not 0 <= args.image_index < len(dataset):
        raise IndexError(
            f"image-index must lie in [0, {len(dataset) - 1}], "
            f"but received {args.image_index}."
        )

    x_start = dataset[args.image_index][0].unsqueeze(0)
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(x_start.shape, generator=generator, dtype=x_start.dtype)
    batch_size = len(TIMESTEPS)
    images = x_start.expand(batch_size, -1, -1, -1)
    shared_noise = noise.expand_as(images)
    timesteps = torch.tensor(TIMESTEPS, dtype=torch.long)
    schedule = make_linear_ddpm_schedule()
    x_t = q_sample(images, timesteps, schedule, shared_noise)

    display_images = inverse_normalize(x_t, clamp=True)
    grid = make_grid(
        display_images,
        nrow=batch_size,
        padding=2,
        pad_value=1.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.output)
    print(
        f"Saved forward-process grid to {args.output} using image "
        f"{args.image_index}, seed {args.seed}, and timesteps {list(TIMESTEPS)}."
    )


if __name__ == "__main__":
    main()
