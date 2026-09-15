from __future__ import annotations

import pytest
import torch

from diffusion_models.memorization import cifar_pixel_memorization_metrics


def test_frozen_memorization_rule_and_bookkeeping() -> None:
    references = torch.tensor([0.0, 3.0, 8.0]).reshape(3, 1, 1, 1)
    samples = torch.tensor([0.1, 2.0, 7.0]).reshape(3, 1, 1, 1)

    result = cifar_pixel_memorization_metrics(samples, references, chunk_size=2)

    assert result["sample_count"] == 3
    assert result["memorized_count"] == 2
    assert result["memorization_rate"] == pytest.approx(2 / 3)
    assert result["unique_training_neighbors_hit"] == 2
    assert result["maximum_duplicate_count"] == 1
