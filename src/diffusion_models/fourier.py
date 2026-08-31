"""Lossless real-image Fourier coordinates for paired diffusion experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def image_to_fourier_channels(image: Tensor) -> Tensor:
    """Encode NCHW real images as interleaved full-spectrum real/imag channels."""
    if image.ndim != 4 or not image.is_floating_point():
        raise ValueError("image must be a floating-point NCHW tensor.")
    spectrum = torch.fft.fft2(image, norm="ortho")
    return torch.stack((spectrum.real, spectrum.imag), dim=2).flatten(1, 2)


def fourier_channels_to_complex(channels: Tensor) -> Tensor:
    """Decode interleaved real/imag channels to an NCHW complex spectrum."""
    if channels.ndim != 4 or channels.shape[1] % 2:
        raise ValueError("channels must have shape [N, 2*C, H, W].")
    paired = channels.unflatten(1, (channels.shape[1] // 2, 2))
    return torch.complex(paired[:, :, 0], paired[:, :, 1])


def project_hermitian(channels: Tensor) -> Tensor:
    """Orthogonally project full complex coordinates onto the real-image subspace."""
    spectrum = fourier_channels_to_complex(channels)
    real_image = torch.fft.ifft2(spectrum, norm="ortho").real
    return image_to_fourier_channels(real_image)


def fourier_channels_to_image(channels: Tensor) -> Tensor:
    """Return the real image represented by possibly imperfect Fourier channels."""
    projected = project_hermitian(channels)
    return torch.fft.ifft2(fourier_channels_to_complex(projected), norm="ortho").real


def inverse_imaginary_residual(channels: Tensor) -> Tensor:
    """Measure the largest imaginary IFFT component before projection."""
    inverse = torch.fft.ifft2(fourier_channels_to_complex(channels), norm="ortho")
    return inverse.imag.abs().amax()


def parseval_equivalent_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """MSE per independent real image coordinate for redundant full spectra."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share a four-dimensional shape.")
    if prediction.shape[1] % 2:
        raise ValueError("Fourier tensors must have an even channel count.")
    return 2.0 * F.mse_loss(prediction, target)


class HermitianPrediction(nn.Module):
    """Project a six-channel denoiser prediction onto valid real-image spectra."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor, timesteps: Tensor) -> Tensor:
        return project_hermitian(self.model(image, timesteps))
