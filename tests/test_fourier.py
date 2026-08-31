import torch

from diffusion_models.fourier import (
    fourier_channels_to_image,
    image_to_fourier_channels,
    inverse_imaginary_residual,
    parseval_equivalent_mse,
    project_hermitian,
)


def test_fourier_round_trip_and_parseval() -> None:
    generator = torch.Generator().manual_seed(11)
    image = torch.randn((4, 3, 32, 32), generator=generator, dtype=torch.float64)
    encoded = image_to_fourier_channels(image)
    reconstructed = fourier_channels_to_image(encoded)
    torch.testing.assert_close(reconstructed, image, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        encoded.square().sum(), image.square().sum(), rtol=1e-12, atol=1e-12
    )
    assert inverse_imaginary_residual(encoded).item() < 1e-12


def test_hermitian_projection_is_idempotent() -> None:
    channels = torch.randn(
        (2, 6, 32, 32), generator=torch.Generator().manual_seed(12)
    )
    projected = project_hermitian(channels)
    torch.testing.assert_close(project_hermitian(projected), projected)
    assert inverse_imaginary_residual(projected).item() < 1e-5


def test_parseval_equivalent_mse_matches_pixel_mse() -> None:
    generator = torch.Generator().manual_seed(13)
    prediction = torch.randn((3, 3, 32, 32), generator=generator)
    target = torch.randn((3, 3, 32, 32), generator=generator)
    pixel_mse = torch.nn.functional.mse_loss(prediction, target)
    spectral_mse = parseval_equivalent_mse(
        image_to_fourier_channels(prediction), image_to_fourier_channels(target)
    )
    torch.testing.assert_close(spectral_mse, pixel_mse)


def test_paired_forward_noise_commutes_with_fft() -> None:
    generator = torch.Generator().manual_seed(14)
    image = torch.randn((2, 3, 32, 32), generator=generator)
    noise = torch.randn((2, 3, 32, 32), generator=generator)
    alpha = torch.tensor(0.37).sqrt()
    sigma = torch.tensor(0.63).sqrt()
    spatial = alpha * image + sigma * noise
    spectral = alpha * image_to_fourier_channels(image) + sigma * image_to_fourier_channels(
        noise
    )
    torch.testing.assert_close(image_to_fourier_channels(spatial), spectral)
