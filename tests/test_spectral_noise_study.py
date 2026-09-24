from __future__ import annotations

import torch

from diffusion_models.spectral_noise_study import (
    full_fft_shell_mse,
    resolve_spectral_noise_study_configuration,
)


def source_configuration():
    return {
        "experiment": {"name": "test"},
        "model": {},
        "data": {},
        "training": {},
        "study_conditions": {
            "baseline": {"enabled": False, "tau": None},
            "moving": {"enabled": True, "tau": 0.5},
        },
        "paired_seeds": [
            {
                "seed": 0,
                "initialization_seed": 11,
                "data_order_seed": 13,
                "training_noise_seed": 17,
            }
        ],
    }


def test_resolve_study_configuration_pairs_every_rng_stream() -> None:
    baseline = resolve_spectral_noise_study_configuration(
        source_configuration(), condition="baseline", seed=0
    )
    moving = resolve_spectral_noise_study_configuration(
        source_configuration(), condition="moving", seed=0
    )
    assert baseline["model"] == moving["model"] == {"initialization_seed": 11}
    assert baseline["data"] == moving["data"] == {"data_order_seed": 13}
    assert baseline["training"] == moving["training"] == {
        "training_noise_seed": 17
    }
    assert not baseline["spectral_noise_condition"]["enabled"]
    assert moving["spectral_noise_condition"]["enabled"]


def test_full_fft_shell_mse_preserves_global_parseval_average() -> None:
    residual = torch.randn(
        (4, 3, 32, 32), generator=torch.Generator().manual_seed(23)
    )
    shell_mse = full_fft_shell_mse(residual, num_shells=23)
    fy = torch.fft.fftfreq(32) * 32
    yy, xx = torch.meshgrid(fy, fy, indexing="ij")
    shells = torch.floor(torch.sqrt(xx.square() + yy.square()) + 1e-12).long()
    counts = torch.bincount(shells.flatten(), minlength=23).to(residual.dtype)
    reconstructed = (shell_mse * counts[None, :]).sum(dim=1) / counts.sum()
    torch.testing.assert_close(
        reconstructed, residual.square().mean(dim=(1, 2, 3)), rtol=1e-5, atol=1e-6
    )
