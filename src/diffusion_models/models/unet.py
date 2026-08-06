"""A readable time-conditioned U-Net for 32x32 diffusion inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UNetConfig:
    """Architecture settings for :class:`CIFAR10UNet`."""

    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 4)
    residual_blocks_per_level: int = 2
    normalization: str = "group_norm"
    group_norm_groups: int = 32
    activation: str = "silu"
    dropout: float = 0.1
    attention_resolutions: tuple[int, ...] = (16,)
    time_embedding_dim: int = 256
    downsampling: str = "stride_2_convolution"
    upsampling: str = "nearest_neighbor_plus_convolution"
    parameter_budget_max: int | None = 25_000_000

    def __post_init__(self) -> None:
        positive_fields = {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "residual_blocks_per_level": self.residual_blocks_per_level,
            "group_norm_groups": self.group_norm_groups,
            "time_embedding_dim": self.time_embedding_dim,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, but received {value}.")
        if not self.channel_multipliers or any(
            multiplier <= 0 for multiplier in self.channel_multipliers
        ):
            raise ValueError("channel_multipliers must contain positive integers.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.normalization != "group_norm":
            raise ValueError("Only group_norm is supported.")
        if self.activation != "silu":
            raise ValueError("Only silu activation is supported.")
        if self.downsampling != "stride_2_convolution":
            raise ValueError("Only stride_2_convolution downsampling is supported.")
        if self.upsampling != "nearest_neighbor_plus_convolution":
            raise ValueError(
                "Only nearest_neighbor_plus_convolution upsampling is supported."
            )
        divisor = 2 ** (len(self.channel_multipliers) - 1)
        if self.image_size % divisor != 0:
            raise ValueError(
                f"image_size must be divisible by {divisor} for this architecture."
            )
        valid_resolutions = {
            self.image_size // (2**level)
            for level in range(len(self.channel_multipliers))
        }
        invalid_attention = set(self.attention_resolutions) - valid_resolutions
        if invalid_attention:
            raise ValueError(
                "attention_resolutions must be encoder resolutions; invalid values: "
                f"{sorted(invalid_attention)}."
            )
        if self.parameter_budget_max is not None and self.parameter_budget_max <= 0:
            raise ValueError("parameter_budget_max must be positive or None.")


def primary_unet_config() -> UNetConfig:
    """Return the frozen primary CIFAR-10 U-Net configuration."""
    return UNetConfig()


def smoke_unet_config() -> UNetConfig:
    """Return the frozen small configuration used by CPU smoke tests."""
    return UNetConfig(
        base_channels=32,
        channel_multipliers=(1, 2),
        residual_blocks_per_level=1,
        dropout=0.0,
        attention_resolutions=(),
        time_embedding_dim=128,
        parameter_budget_max=None,
    )


def _group_count(channels: int, maximum_groups: int) -> int:
    """Return the largest valid GroupNorm divisor no greater than the maximum."""
    for groups in range(min(channels, maximum_groups), 0, -1):
        if channels % groups == 0:
            return groups
    raise RuntimeError("Every positive channel count must have a GroupNorm divisor.")


def _group_norm(channels: int, maximum_groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels, maximum_groups), channels)


class SinusoidalTimeEmbedding(nn.Module):
    """Map a scalar timestep per batch item to fixed sinusoidal features."""

    def __init__(self, embedding_dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if embedding_dim < 2 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even integer of at least 2.")
        if max_period <= 1.0:
            raise ValueError("max_period must be greater than 1.")
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have shape [B], got {tuple(timesteps.shape)}."
            )
        if not timesteps.is_floating_point() and timesteps.dtype != torch.long:
            raise TypeError("timesteps must be floating point or torch.long.")
        if not torch.isfinite(timesteps).all():
            raise ValueError("timesteps must be finite.")

        half_dim = self.embedding_dim // 2
        exponent = torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(self.max_period) * exponent)
        angles = timesteps.to(dtype=torch.float32)[:, None] * frequencies[None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


class TimestepEmbedding(nn.Module):
    """Apply a learned MLP to fixed sinusoidal timestep features."""

    def __init__(self, sinusoidal_dim: int, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.sinusoidal = SinusoidalTimeEmbedding(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(timesteps))


class ResidualBlock(nn.Module):
    """A spatial residual block conditioned by one embedding per example."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        *,
        dropout: float = 0.0,
        maximum_groups: int = 32,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, time_embedding_dim) <= 0:
            raise ValueError("Channel and embedding dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim
        self.norm1 = _group_norm(in_channels, maximum_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, out_channels)
        self.norm2 = _group_norm(out_channels, maximum_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, image: Tensor, time_embedding: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [B, C, H, W].")
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} image channels, got {image.shape[1]}."
            )
        if time_embedding.shape != (image.shape[0], self.time_embedding_dim):
            raise ValueError(
                "time_embedding must have shape "
                f"[{image.shape[0]}, {self.time_embedding_dim}], got "
                f"{tuple(time_embedding.shape)}."
            )
        if image.device != time_embedding.device:
            raise ValueError("image and time_embedding must share a device.")

        hidden = self.conv1(F.silu(self.norm1(image)))
        time_bias = self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = hidden + time_bias
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return self.skip(image) + hidden


class Downsample(nn.Module):
    """Halve spatial resolution using a stride-two convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.convolution(image)


class Upsample(nn.Module):
    """Double spatial resolution using nearest-neighbor resize and convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, image: Tensor) -> Tensor:
        image = F.interpolate(image, scale_factor=2.0, mode="nearest")
        return self.convolution(image)


class SelfAttention2d(nn.Module):
    """Single-head spatial self-attention with a residual connection."""

    def __init__(self, channels: int, *, maximum_groups: int = 32) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.channels = channels
        self.normalization = _group_norm(channels, maximum_groups)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != self.channels:
            raise ValueError(
                f"Expected image shape [B, {self.channels}, H, W], got "
                f"{tuple(image.shape)}."
            )
        batch, channels, height, width = image.shape
        query, key, value = self.qkv(self.normalization(image)).chunk(3, dim=1)
        query = query.reshape(batch, channels, height * width).transpose(1, 2)
        key = key.reshape(batch, channels, height * width)
        value = value.reshape(batch, channels, height * width).transpose(1, 2)
        weights = torch.softmax(torch.bmm(query, key) / math.sqrt(channels), dim=-1)
        attended = torch.bmm(weights, value).transpose(1, 2)
        attended = attended.reshape(batch, channels, height, width)
        return image + self.output_projection(attended)


class CIFAR10UNet(nn.Module):
    """Time-conditioned U-Net that predicts noise with the input image shape."""

    def __init__(self, config: UNetConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else primary_unet_config()
        config = self.config
        self.resolutions = tuple(
            config.image_size // (2**level)
            for level in range(len(config.channel_multipliers))
        )
        self.level_channels = tuple(
            config.base_channels * multiplier
            for multiplier in config.channel_multipliers
        )

        self.time_embedding = TimestepEmbedding(
            config.base_channels, config.time_embedding_dim
        )
        self.input_convolution = nn.Conv2d(
            config.in_channels, config.base_channels, kernel_size=3, padding=1
        )

        self.encoder_levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        current_channels = config.base_channels
        for level, (resolution, level_channels) in enumerate(
            zip(self.resolutions, self.level_channels, strict=True)
        ):
            blocks = nn.ModuleList()
            for block_index in range(config.residual_blocks_per_level):
                block_in = current_channels if block_index == 0 else level_channels
                blocks.append(
                    ResidualBlock(
                        block_in,
                        level_channels,
                        config.time_embedding_dim,
                        dropout=config.dropout,
                        maximum_groups=config.group_norm_groups,
                    )
                )
            attention = (
                SelfAttention2d(level_channels, maximum_groups=config.group_norm_groups)
                if resolution in config.attention_resolutions
                else nn.Identity()
            )
            self.encoder_levels.append(
                nn.ModuleDict({"blocks": blocks, "attention": attention})
            )
            current_channels = level_channels
            if level < len(self.level_channels) - 1:
                next_channels = self.level_channels[level + 1]
                self.downsamplers.append(Downsample(current_channels, next_channels))
                current_channels = next_channels

        self.middle_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    current_channels,
                    current_channels,
                    config.time_embedding_dim,
                    dropout=config.dropout,
                    maximum_groups=config.group_norm_groups,
                )
                for _ in range(2)
            ]
        )

        self.decoder_levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        reversed_levels = list(
            zip(reversed(self.resolutions), reversed(self.level_channels), strict=True)
        )
        for level, (resolution, skip_channels) in enumerate(reversed_levels):
            blocks = nn.ModuleList()
            for block_index in range(config.residual_blocks_per_level):
                block_in = (
                    current_channels + skip_channels
                    if block_index == 0
                    else skip_channels
                )
                blocks.append(
                    ResidualBlock(
                        block_in,
                        skip_channels,
                        config.time_embedding_dim,
                        dropout=config.dropout,
                        maximum_groups=config.group_norm_groups,
                    )
                )
                current_channels = skip_channels
            attention = (
                SelfAttention2d(skip_channels, maximum_groups=config.group_norm_groups)
                if resolution in config.attention_resolutions
                else nn.Identity()
            )
            self.decoder_levels.append(
                nn.ModuleDict({"blocks": blocks, "attention": attention})
            )
            if level < len(reversed_levels) - 1:
                next_channels = reversed_levels[level + 1][1]
                self.upsamplers.append(Upsample(current_channels, next_channels))
                current_channels = next_channels

        self.output_normalization = _group_norm(
            current_channels, config.group_norm_groups
        )
        self.output_convolution = nn.Conv2d(
            current_channels, config.out_channels, kernel_size=3, padding=1
        )

        parameter_count = count_trainable_parameters(self)
        if (
            config.parameter_budget_max is not None
            and parameter_count >= config.parameter_budget_max
        ):
            raise ValueError(
                f"Model has {parameter_count:,} trainable parameters, reaching or "
                f"exceeding the budget of {config.parameter_budget_max:,}."
            )

    @property
    def trainable_parameter_count(self) -> int:
        """Return the number of parameters updated by optimization."""
        return count_trainable_parameters(self)

    def _validate_inputs(self, image: Tensor, timesteps: Tensor) -> None:
        config = self.config
        expected_shape = (config.in_channels, config.image_size, config.image_size)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected image shape [B, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}], got "
                f"{tuple(image.shape)}."
            )
        if not image.is_floating_point():
            raise TypeError("image must use a floating-point dtype.")
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have shape [B], got {tuple(timesteps.shape)}."
            )
        if timesteps.dtype != torch.long:
            raise TypeError("timesteps must use torch.long.")
        if timesteps.shape[0] != image.shape[0]:
            raise ValueError(
                "Image and timestep batch sizes differ: "
                f"{image.shape[0]} versus {timesteps.shape[0]}."
            )
        if image.device != timesteps.device:
            raise ValueError("image and timesteps must share a device.")
        if timesteps.numel() and int(timesteps.min().item()) < 0:
            raise ValueError("timesteps must be nonnegative.")

    def forward(self, image: Tensor, timesteps: Tensor) -> Tensor:
        self._validate_inputs(image, timesteps)
        time_embedding = self.time_embedding(timesteps)
        hidden = self.input_convolution(image)
        skips: list[Tensor] = []

        for level, modules in enumerate(self.encoder_levels):
            for block in modules["blocks"]:
                hidden = block(hidden, time_embedding)
            hidden = modules["attention"](hidden)
            skips.append(hidden)
            if level < len(self.downsamplers):
                hidden = self.downsamplers[level](hidden)

        for block in self.middle_blocks:
            hidden = block(hidden, time_embedding)

        for level, modules in enumerate(self.decoder_levels):
            skip = skips.pop()
            if hidden.shape[0] != skip.shape[0] or hidden.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    "Decoder and skip tensor shapes are incompatible: "
                    f"{tuple(hidden.shape)} versus {tuple(skip.shape)}."
                )
            hidden = torch.cat((hidden, skip), dim=1)
            for block in modules["blocks"]:
                hidden = block(hidden, time_embedding)
            hidden = modules["attention"](hidden)
            if level < len(self.upsamplers):
                hidden = self.upsamplers[level](hidden)

        if skips:
            raise RuntimeError(f"Decoder left {len(skips)} skip tensors unused.")
        return self.output_convolution(F.silu(self.output_normalization(hidden)))


def count_trainable_parameters(module: nn.Module) -> int:
    """Count parameters whose gradients are enabled."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
