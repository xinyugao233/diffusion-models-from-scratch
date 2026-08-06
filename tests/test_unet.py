import pytest
import torch
from torch import nn

from diffusion_models.models import (
    CIFAR10UNet,
    ResidualBlock,
    SelfAttention2d,
    SinusoidalTimeEmbedding,
    count_trainable_parameters,
    primary_unet_config,
    smoke_unet_config,
)


def _assert_present_and_finite_parameter_gradients(module: nn.Module) -> None:
    parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_sinusoidal_timestep_embedding_is_deterministic_and_distinguishes_time() -> (
    None
):
    embedding = SinusoidalTimeEmbedding(32)
    timesteps = torch.tensor([7, 7, 19], dtype=torch.long)

    output = embedding(timesteps)

    assert output.shape == (3, 32)
    torch.testing.assert_close(output[0], output[1])
    assert not torch.equal(output[0], output[2])
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("invalid_dim", [0, 1, 3])
def test_sinusoidal_timestep_embedding_rejects_invalid_dimensions(
    invalid_dim: int,
) -> None:
    with pytest.raises(ValueError, match="even integer"):
        SinusoidalTimeEmbedding(invalid_dim)


def test_sinusoidal_timestep_embedding_rejects_invalid_timestep_shape() -> None:
    embedding = SinusoidalTimeEmbedding(32)
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        embedding(torch.zeros((2, 1), dtype=torch.long))


@pytest.mark.parametrize("in_channels,out_channels", [(32, 32), (32, 64)])
def test_residual_block_shapes_conditioning_and_backward(
    in_channels: int,
    out_channels: int,
) -> None:
    torch.manual_seed(1)
    block = ResidualBlock(in_channels, out_channels, 128, dropout=0.0)
    image = torch.randn(2, in_channels, 8, 8, requires_grad=True)
    time_zeros = torch.zeros(2, 128)
    time_ones = torch.ones(2, 128)

    output_zeros = block(image, time_zeros)
    output_ones = block(image, time_ones)

    assert output_zeros.shape == (2, out_channels, 8, 8)
    assert not torch.equal(output_zeros, output_ones)
    assert torch.isfinite(output_zeros).all()
    output_zeros.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    _assert_present_and_finite_parameter_gradients(block)


def test_attention_preserves_shape_and_starts_as_residual_identity() -> None:
    torch.manual_seed(2)
    attention = SelfAttention2d(32)
    image = torch.randn(2, 32, 8, 8, requires_grad=True)

    output = attention(image)

    assert output.shape == image.shape
    torch.testing.assert_close(output, image)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    _assert_present_and_finite_parameter_gradients(attention)
    assert attention.qkv.weight.grad is not None
    assert torch.count_nonzero(attention.qkv.weight.grad) == 0
    assert attention.output_projection.weight.grad is not None
    assert torch.count_nonzero(attention.output_projection.weight.grad) > 0


@pytest.mark.parametrize("batch_size", [1, 3])
def test_smoke_unet_output_contract_for_multiple_batch_sizes(batch_size: int) -> None:
    torch.manual_seed(3)
    model = CIFAR10UNet(smoke_unet_config()).eval()
    image = torch.randn(batch_size, 3, 32, 32)
    timesteps = torch.arange(batch_size, dtype=torch.long)

    output = model(image, timesteps)

    assert output.shape == image.shape
    assert torch.isfinite(output).all()


def test_smoke_unet_backward_reaches_all_trainable_parameters() -> None:
    torch.manual_seed(4)
    model = CIFAR10UNet(smoke_unet_config())
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    timesteps = torch.tensor([0, 999], dtype=torch.long)

    output = model(image, timesteps)
    output.square().mean().backward()

    assert image.grad is not None and torch.isfinite(image.grad).all()
    _assert_present_and_finite_parameter_gradients(model)


def test_unet_rejects_timestep_batch_mismatch() -> None:
    model = CIFAR10UNet(smoke_unet_config())
    image = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="batch sizes differ"):
        model(image, timesteps)


def test_unet_rejects_incompatible_image_and_timestep_shapes() -> None:
    model = CIFAR10UNet(smoke_unet_config())
    with pytest.raises(ValueError, match="Expected image shape"):
        model(torch.randn(1, 3, 16, 16), torch.tensor([0], dtype=torch.long))
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        model(torch.randn(1, 3, 32, 32), torch.tensor([[0]], dtype=torch.long))


def test_primary_unet_respects_parameter_budget_and_attention_placement() -> None:
    config = primary_unet_config()
    model = CIFAR10UNet(config)

    assert count_trainable_parameters(model) == model.trainable_parameter_count
    assert model.trainable_parameter_count < config.parameter_budget_max
    assert model.resolutions == (32, 16, 8, 4)
    encoder_attention = [
        resolution
        for resolution, level in zip(
            model.resolutions, model.encoder_levels, strict=True
        )
        if isinstance(level["attention"], SelfAttention2d)
    ]
    decoder_attention = [
        resolution
        for resolution, level in zip(
            reversed(model.resolutions), model.decoder_levels, strict=True
        )
        if isinstance(level["attention"], SelfAttention2d)
    ]
    assert encoder_attention == [16]
    assert decoder_attention == [16]
