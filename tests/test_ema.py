from __future__ import annotations

import torch
from torch import nn

from diffusion_models.ema import ExponentialMovingAverage


class ScalarModel(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value


def test_ema_initialization_equals_model_and_is_detached() -> None:
    model = ScalarModel(2.0)
    ema = ExponentialMovingAverage(model, decay=0.9)

    torch.testing.assert_close(ema.parameters["weight"], model.weight)
    torch.testing.assert_close(ema.buffers["counter"], model.counter)
    assert not ema.parameters["weight"].requires_grad
    assert ema.parameters["weight"].data_ptr() != model.weight.data_ptr()
    assert ema.num_updates == 0


def test_one_ema_update_matches_hand_calculation_and_copies_buffer() -> None:
    model = ScalarModel(1.0)
    ema = ExponentialMovingAverage(model, decay=0.75)
    with torch.no_grad():
        model.weight.fill_(5.0)
        model.counter.fill_(7)

    ema.update(model)

    torch.testing.assert_close(ema.parameters["weight"], torch.tensor([2.0]))
    torch.testing.assert_close(ema.buffers["counter"], torch.tensor(7))
    assert ema.num_updates == 1


def test_multiple_ema_updates_match_hand_calculation() -> None:
    model = ScalarModel(0.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    with torch.no_grad():
        model.weight.fill_(6.0)
    ema.update(model)

    torch.testing.assert_close(ema.parameters["weight"], torch.tensor([3.5]))
    assert ema.num_updates == 2


def test_ema_and_training_parameter_diverge_after_optimization() -> None:
    model = ScalarModel(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    model(torch.tensor([2.0])).square().mean().backward()
    optimizer.step()
    ema.update(model)

    assert not torch.equal(model.weight, ema.parameters["weight"])


def test_ema_state_round_trip_and_copy_to_are_exact() -> None:
    source_model = ScalarModel(1.0)
    source_ema = ExponentialMovingAverage(source_model, decay=0.8)
    with torch.no_grad():
        source_model.weight.fill_(4.0)
        source_model.counter.fill_(3)
    source_ema.update(source_model)

    restored_ema = ExponentialMovingAverage(ScalarModel(-2.0), decay=0.1)
    restored_ema.load_state_dict(source_ema.state_dict())
    target_model = ScalarModel(-5.0)
    restored_ema.copy_to(target_model)

    assert restored_ema.decay == 0.8
    assert restored_ema.num_updates == 1
    assert torch.equal(
        restored_ema.parameters["weight"], source_ema.parameters["weight"]
    )
    assert torch.equal(target_model.weight, source_ema.parameters["weight"])
    assert torch.equal(target_model.counter, source_ema.buffers["counter"])
    assert target_model.state_dict().keys() == restored_ema.model_state_dict().keys()
