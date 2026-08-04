"""CIFAR-10 input and visualization utilities."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# These values intentionally map image intensities from [0, 1] to [-1, 1].
# They are a model-space convention, not empirical CIFAR-10 channel statistics.
CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)


def _channel_values(values: Sequence[float], image: Tensor) -> Tensor:
    """Create a broadcastable channel tensor on the image device and dtype."""
    if image.ndim not in (3, 4):
        raise ValueError(
            f"Expected a CHW image or NCHW batch, but received shape {image.shape}."
        )
    if len(values) != image.shape[-3]:
        raise ValueError(
            f"Expected {image.shape[-3]} channel values, but received {len(values)}."
        )
    shape = (1, image.shape[-3], 1, 1) if image.ndim == 4 else (-1, 1, 1)
    return image.new_tensor(values).view(shape)


def normalize(
    image: Tensor,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> Tensor:
    """Normalize a CHW image or NCHW batch channel-wise."""
    mean_tensor = _channel_values(mean, image)
    std_tensor = _channel_values(std, image)
    if torch.any(std_tensor == 0):
        raise ValueError("Standard deviations must be nonzero.")
    return (image - mean_tensor) / std_tensor


def inverse_normalize(
    image: Tensor,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
    *,
    clamp: bool = False,
) -> Tensor:
    """Undo channel-wise normalization for a CHW image or NCHW batch.

    Clamping is opt-in so tests and callers can detect out-of-range model values
    rather than silently hiding them. Visualization code should set it to true.
    """
    restored = image * _channel_values(std, image) + _channel_values(mean, image)
    return restored.clamp(0.0, 1.0) if clamp else restored


def load_cifar10(
    root: str | Path = "data",
    *,
    train: bool = True,
    download: bool = False,
) -> datasets.CIFAR10:
    """Return CIFAR-10 with deterministic tensor conversion and [-1, 1] scaling."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return datasets.CIFAR10(
        root=Path(root).expanduser(),
        train=train,
        transform=transform,
        download=download,
    )


def make_cifar10_loader(
    root: str | Path = "data",
    *,
    train: bool = True,
    batch_size: int = 128,
    shuffle: bool | None = None,
    num_workers: int = 0,
    seed: int = 0,
    download: bool = False,
) -> DataLoader:
    """Build a reproducible CIFAR-10 data loader."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")

    dataset = load_cifar10(root, train=train, download=download)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
