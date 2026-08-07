"""Explicit exponential moving averages for diffusion-model parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class ExponentialMovingAverage:
    """Maintain detached EMA parameters and exact copies of model buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must lie in [0, 1), got {decay}.")
        self.decay = float(decay)
        self.num_updates = 0
        self._parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self._buffers = {
            name: buffer.detach().clone() for name, buffer in model.named_buffers()
        }

    @property
    def parameters(self) -> Mapping[str, Tensor]:
        """Return read-only-by-convention detached EMA parameter tensors."""
        return self._parameters

    @property
    def buffers(self) -> Mapping[str, Tensor]:
        """Return read-only-by-convention copied model buffers."""
        return self._buffers

    @staticmethod
    def _validate_tensor_mapping(
        expected: Mapping[str, Tensor],
        actual: Mapping[str, Tensor],
        *,
        kind: str,
    ) -> None:
        if expected.keys() != actual.keys():
            missing = sorted(expected.keys() - actual.keys())
            unexpected = sorted(actual.keys() - expected.keys())
            raise ValueError(
                f"EMA {kind} names differ; missing={missing}, unexpected={unexpected}."
            )
        for name, expected_tensor in expected.items():
            actual_tensor = actual[name]
            if expected_tensor.shape != actual_tensor.shape:
                raise ValueError(
                    f"EMA {kind} shape mismatch for {name}: "
                    f"{tuple(expected_tensor.shape)} versus {tuple(actual_tensor.shape)}."
                )
            if expected_tensor.dtype != actual_tensor.dtype:
                raise TypeError(
                    f"EMA {kind} dtype mismatch for {name}: "
                    f"{expected_tensor.dtype} versus {actual_tensor.dtype}."
                )
            if expected_tensor.device != actual_tensor.device:
                raise ValueError(
                    f"EMA {kind} device mismatch for {name}: "
                    f"{expected_tensor.device} versus {actual_tensor.device}."
                )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update parameters after an optimizer step and copy model buffers."""
        model_parameters = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        self._validate_tensor_mapping(
            self._parameters, model_parameters, kind="parameter"
        )
        self._validate_tensor_mapping(self._buffers, model_buffers, kind="buffer")

        for name, ema_parameter in self._parameters.items():
            model_parameter = model_parameters[name].detach()
            ema_parameter.mul_(self.decay).add_(
                model_parameter,
                alpha=1.0 - self.decay,
            )
        for name, ema_buffer in self._buffers.items():
            ema_buffer.copy_(model_buffers[name].detach())
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        """Return a structured, independently owned EMA state dictionary."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "parameters": {
                name: tensor.detach().clone()
                for name, tensor in self._parameters.items()
            },
            "buffers": {
                name: tensor.detach().clone() for name, tensor in self._buffers.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore EMA state after strict schema, tensor, and scalar validation."""
        required = {"decay", "num_updates", "parameters", "buffers"}
        if set(state) != required:
            raise ValueError(
                "EMA state keys differ; "
                f"expected={sorted(required)}, actual={sorted(state)}."
            )
        decay = state["decay"]
        num_updates = state["num_updates"]
        if not isinstance(decay, (float, int)) or not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA state decay must lie in [0, 1).")
        if not isinstance(num_updates, int) or num_updates < 0:
            raise ValueError("EMA state num_updates must be a nonnegative integer.")
        parameters = state["parameters"]
        buffers = state["buffers"]
        if not isinstance(parameters, Mapping) or not isinstance(buffers, Mapping):
            raise TypeError("EMA parameters and buffers must be mappings.")
        self._validate_tensor_mapping(self._parameters, parameters, kind="parameter")
        self._validate_tensor_mapping(self._buffers, buffers, kind="buffer")

        for name, tensor in self._parameters.items():
            tensor.copy_(parameters[name])
        for name, tensor in self._buffers.items():
            tensor.copy_(buffers[name])
        self.decay = float(decay)
        self.num_updates = num_updates

    def model_state_dict(self) -> dict[str, Tensor]:
        """Return a flat model-compatible state for future EMA evaluation."""
        state = {
            name: tensor.detach().clone() for name, tensor in self._parameters.items()
        }
        state.update(
            {name: tensor.detach().clone() for name, tensor in self._buffers.items()}
        )
        return state

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters and buffers into a compatible model explicitly."""
        model_parameters = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        self._validate_tensor_mapping(
            self._parameters, model_parameters, kind="parameter"
        )
        self._validate_tensor_mapping(self._buffers, model_buffers, kind="buffer")
        for name, parameter in model_parameters.items():
            parameter.copy_(self._parameters[name])
        for name, buffer in model_buffers.items():
            buffer.copy_(self._buffers[name])
