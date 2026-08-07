"""Structured, explicit training checkpoints with reproducible RNG state."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from diffusion_models.ema import ExponentialMovingAverage

CHECKPOINT_VERSION = 1
CHECKPOINT_KEYS = {
    "checkpoint_version",
    "model_state_dict",
    "ema_state_dict",
    "optimizer_state_dict",
    "global_step",
    "configuration",
    "rng_state",
    "metadata",
}


class CheckpointError(RuntimeError):
    """Raised when a training checkpoint is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated non-tensor metadata returned after checkpoint restoration."""

    global_step: int
    configuration: dict[str, Any]
    metadata: dict[str, Any]


def states_exactly_equal(first: Any, second: Any) -> bool:
    """Compare nested state dictionaries without requiring mapping subclasses."""
    if isinstance(first, Tensor) or isinstance(second, Tensor):
        return (
            isinstance(first, Tensor)
            and isinstance(second, Tensor)
            and torch.equal(first, second)
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return False
        return first.keys() == second.keys() and all(
            states_exactly_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        if type(first) is not type(second) or len(first) != len(second):
            return False
        return all(
            states_exactly_equal(first_item, second_item)
            for first_item, second_item in zip(first, second, strict=True)
        )
    return type(first) is type(second) and first == second


def _validate_json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    copied = dict(value)
    try:
        json.dumps(copied, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible.") from error
    return copied


def capture_rng_state(
    *,
    generators: Mapping[str, torch.Generator] | None = None,
    cuda_devices: Sequence[int] = (),
    include_numpy: bool = False,
) -> dict[str, Any]:
    """Capture only the explicitly used Python, torch, generator, and device RNGs."""
    named_generators = generators or {}
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "generators": {
            name: generator.get_state().clone()
            for name, generator in named_generators.items()
        },
        "cuda": {},
        "numpy": None,
    }
    for device_index in cuda_devices:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state requested but CUDA is unavailable.")
        state["cuda"][int(device_index)] = torch.cuda.get_rng_state(
            int(device_index)
        ).clone()
    if include_numpy:
        import numpy as np

        algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        state["numpy"] = {
            "algorithm": algorithm,
            "keys": torch.from_numpy(keys.copy()),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    """Restore every RNG stream present in a validated checkpoint state."""
    required = {"python", "torch_cpu", "generators", "cuda", "numpy"}
    if set(state) != required:
        raise CheckpointError(
            "RNG state keys differ; "
            f"expected={sorted(required)}, actual={sorted(state)}."
        )
    torch_cpu_state = state["torch_cpu"]
    if not isinstance(torch_cpu_state, Tensor) or torch_cpu_state.dtype != torch.uint8:
        raise CheckpointError("torch_cpu RNG state must be a uint8 tensor.")
    random.setstate(state["python"])
    torch.set_rng_state(torch_cpu_state.cpu())

    named_generators = generators or {}
    saved_generators = state["generators"]
    if set(named_generators) != set(saved_generators):
        raise CheckpointError(
            "Named generator sets differ; "
            f"expected={sorted(saved_generators)}, actual={sorted(named_generators)}."
        )
    for name, generator in named_generators.items():
        generator.set_state(saved_generators[name].cpu())

    for device_index, cuda_state in state["cuda"].items():
        if not torch.cuda.is_available():
            raise CheckpointError("Checkpoint contains CUDA RNG state without CUDA.")
        torch.cuda.set_rng_state(cuda_state.cpu(), int(device_index))

    numpy_state = state["numpy"]
    if numpy_state is not None:
        import numpy as np

        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"].cpu().numpy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    configuration: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
    cuda_devices: Sequence[int] = (),
    include_numpy: bool = False,
) -> Path:
    """Save one complete structured training state without overwriting evidence."""
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a nonnegative integer.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "configuration": _validate_json_mapping(configuration, name="configuration"),
        "rng_state": capture_rng_state(
            generators=generators,
            cuda_devices=cuda_devices,
            include_numpy=include_numpy,
        ),
        "metadata": _validate_json_mapping(metadata or {}, name="metadata"),
    }
    try:
        with destination.open("xb") as handle:
            torch.save(payload, handle)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    expected_configuration: Mapping[str, Any] | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> LoadedCheckpoint:
    """Load and explicitly restore a complete structured training checkpoint."""
    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointError(
            f"Could not read checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint root must be a mapping.")
    if set(payload) != CHECKPOINT_KEYS:
        raise CheckpointError(
            "Checkpoint keys differ; "
            f"expected={sorted(CHECKPOINT_KEYS)}, actual={sorted(payload)}."
        )
    if payload["checkpoint_version"] != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint version {payload['checkpoint_version']}."
        )
    global_step = payload["global_step"]
    if not isinstance(global_step, int) or global_step < 0:
        raise CheckpointError("Checkpoint global_step must be nonnegative integer.")
    configuration = payload["configuration"]
    metadata = payload["metadata"]
    if not isinstance(configuration, Mapping) or not isinstance(metadata, Mapping):
        raise CheckpointError("Checkpoint configuration and metadata must be mappings.")
    configuration = dict(configuration)
    metadata = dict(metadata)
    if expected_configuration is not None and configuration != dict(
        expected_configuration
    ):
        raise CheckpointError("Checkpoint configuration does not match expectation.")

    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        ema.load_state_dict(payload["ema_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if restore_rng:
            restore_rng_state(payload["rng_state"], generators=generators)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointError(f"Checkpoint state is incompatible: {error}") from error

    return LoadedCheckpoint(
        global_step=global_step,
        configuration=configuration,
        metadata=metadata,
    )
