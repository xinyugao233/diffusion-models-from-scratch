import pytest
import torch

from diffusion_models.data import inverse_normalize, normalize


@pytest.mark.parametrize("shape", [(3, 32, 32), (4, 3, 32, 32)])
def test_normalization_round_trip(shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(shape, generator=generator)

    reconstructed = inverse_normalize(normalize(image))

    torch.testing.assert_close(reconstructed, image)


def test_normalization_maps_unit_interval_to_symmetric_interval() -> None:
    endpoints = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1).expand(2, 3, 1, 1)

    normalized = normalize(endpoints)

    torch.testing.assert_close(normalized[0], -torch.ones_like(normalized[0]))
    torch.testing.assert_close(normalized[1], torch.ones_like(normalized[1]))


def test_normalization_rejects_non_image_shapes() -> None:
    with pytest.raises(ValueError, match="CHW image or NCHW batch"):
        normalize(torch.zeros(3, 32))
