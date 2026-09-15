"""Production orchestration helpers for full CIFAR-10 DDPM training."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Sampler

from diffusion_models.diffusion import DDPMSchedule
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.spectral_boundary import (
    RadialPower,
    SpectralBoundaryConfig,
    SpectralBoundaryDiagnostics,
    StaticSpectralConfig,
    weighted_spectral_loss,
)
from diffusion_models.training import (
    epsilon_prediction_loss,
    gradient_norm,
    sample_epsilon_training_batch,
)


@dataclass(frozen=True)
class ProductionStepMetrics:
    """Measurements from one complete production optimizer/EMA update."""

    loss: float
    unweighted_loss: float
    gradient_norm: float
    clipped_gradient_norm: float
    ema_num_updates: int
    spectral_boundary: SpectralBoundaryDiagnostics | None = None


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """Map every global step to a reproducible shuffled dataset batch."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        data_seed: int,
        start_step: int,
        end_step: int,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive.")
        if batch_size > dataset_size:
            raise ValueError("batch_size cannot exceed dataset_size.")
        if data_seed < 0:
            raise ValueError("data_seed must be nonnegative.")
        if start_step < 0 or end_step <= start_step:
            raise ValueError("Expected 0 <= start_step < end_step.")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.start_step = start_step
        self.end_step = end_step
        self.batches_per_epoch = dataset_size // batch_size

    def __len__(self) -> int:
        return self.end_step - self.start_step

    def __iter__(self) -> Iterator[list[int]]:
        current_epoch = -1
        permutation: Tensor | None = None
        for global_step in range(self.start_step, self.end_step):
            epoch, batch_index = divmod(global_step, self.batches_per_epoch)
            if epoch != current_epoch:
                generator = torch.Generator().manual_seed(self.data_seed + epoch)
                permutation = torch.randperm(
                    self.dataset_size,
                    generator=generator,
                )
                current_epoch = epoch
            if permutation is None:
                raise RuntimeError("Dataset permutation was not initialized.")
            offset = batch_index * self.batch_size
            yield permutation[offset : offset + self.batch_size].tolist()


class AppendOnlyJsonlLogger:
    """Append scalar records while validating the prior resume boundary."""

    def __init__(self, path: str | Path, *, expected_last_step: int) -> None:
        if expected_last_step < 0:
            raise ValueError("expected_last_step must be nonnegative.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_step = 0
        if self.path.exists():
            records = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not records:
                raise ValueError(f"Existing log is empty: {self.path}")
            steps = [record.get("step") for record in records]
            if steps != list(range(1, len(records) + 1)):
                raise ValueError("Existing metric steps are not contiguous from one.")
            self.last_step = int(steps[-1])
        if self.last_step != expected_last_step:
            raise ValueError(
                "Metric log resume boundary differs: "
                f"expected {expected_last_step}, found {self.last_step}."
            )

    def log(self, step: int, **values: float | str) -> None:
        if step != self.last_step + 1:
            raise ValueError(
                f"Expected metric step {self.last_step + 1}, received {step}."
            )
        for name, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"Cannot log nonfinite {name}={value}.")
        record: dict[str, Any] = {"step": step, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.last_step = step


def production_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    x_start: Tensor,
    schedule: DDPMSchedule,
    *,
    generator: torch.Generator,
    max_gradient_norm: float,
    spectral_boundary: tuple[RadialPower, SpectralBoundaryConfig | StaticSpectralConfig]
    | None = None,
    event_callback: Callable[[str], None] | None = None,
) -> ProductionStepMetrics:
    """Run one explicit checked optimizer step followed by one EMA update."""
    if max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive.")
    notify = event_callback or (lambda event: None)

    batch = sample_epsilon_training_batch(
        x_start,
        schedule,
        generator=generator,
    )
    optimizer.zero_grad(set_to_none=True)
    notify("zero_grad")
    unweighted_loss, prediction = epsilon_prediction_loss(model, batch)
    diagnostics = None
    if spectral_boundary is None:
        # Preserve the original scalar and computation graph exactly.
        loss = unweighted_loss
    else:
        radial_power, boundary_config = spectral_boundary
        loss, diagnostics = weighted_spectral_loss(
            prediction - batch.target_noise,
            batch.timesteps,
            schedule,
            radial_power,
            boundary_config,
        )
    loss.backward()
    notify("backward")

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    before_clipping = gradient_norm(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    after_clipping = gradient_norm(parameters)
    optimizer.step()
    notify("optimizer_step")
    ema.update(model)
    notify("ema_update")

    return ProductionStepMetrics(
        loss=float(loss.detach().item()),
        unweighted_loss=float(unweighted_loss.detach().item()),
        gradient_norm=float(before_clipping.item()),
        clipped_gradient_norm=float(after_clipping.item()),
        ema_num_updates=ema.num_updates,
        spectral_boundary=diagnostics,
    )


def checkpoint_steps(
    stage_configuration: Mapping[str, Any],
) -> set[int]:
    """Return all frozen checkpoint steps for one training stage."""
    maximum = int(stage_configuration["max_steps"])
    if "checkpoint_steps" in stage_configuration:
        steps = {int(step) for step in stage_configuration["checkpoint_steps"]}
    else:
        cadence = int(stage_configuration["checkpoint_every"])
        if cadence <= 0:
            raise ValueError("checkpoint_every must be positive.")
        steps = set(range(cadence, maximum + 1, cadence))
    if not steps or any(step < 0 or step > maximum for step in steps):
        raise ValueError("Checkpoint steps must lie in [0, max_steps].")
    return steps


def sampling_steps(
    stage_name: str,
    stage_configuration: Mapping[str, Any],
    evaluation_configuration: Mapping[str, Any],
) -> set[int]:
    """Return frozen EMA evaluation steps for one stage."""
    if int(stage_configuration["sample_count"]) == 0:
        return set()
    source = (
        stage_configuration["sample_steps"]
        if "sample_steps" in stage_configuration
        else evaluation_configuration["sample_steps"]
    )
    steps = {int(step) for step in source}
    maximum = int(stage_configuration["max_steps"])
    if any(step <= 0 or step > maximum for step in steps):
        raise ValueError(f"Invalid sampling step for stage {stage_name}.")
    return steps


def fixed_initial_noise(
    seeds: Sequence[int],
    *,
    image_shape: Sequence[int] = (3, 32, 32),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create one independently seeded CPU Gaussian image per seed."""
    if not seeds:
        raise ValueError("At least one fixed initial-noise seed is required.")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Fixed seeds must be unique nonnegative integers.")
    if len(tuple(image_shape)) != 3 or any(size <= 0 for size in image_shape):
        raise ValueError("image_shape must contain positive CHW dimensions.")
    return torch.stack(
        [
            torch.randn(
                tuple(image_shape),
                generator=torch.Generator().manual_seed(seed),
                dtype=dtype,
            )
            for seed in seeds
        ]
    )


def tensors_are_finite(tensors: Mapping[str, Tensor]) -> bool:
    """Return whether every named tensor is finite."""
    return all(torch.isfinite(tensor).all().item() for tensor in tensors.values())


def model_tensors_are_finite(model: nn.Module) -> bool:
    """Check every model parameter and buffer for finiteness."""
    return tensors_are_finite(dict(model.state_dict()))


def ema_tensors_are_finite(ema: ExponentialMovingAverage) -> bool:
    """Check every EMA parameter and buffer for finiteness."""
    return tensors_are_finite({**dict(ema.parameters), **dict(ema.buffers)})


def require_slurm_environment(environment: Mapping[str, str]) -> str:
    """Return the Slurm job ID or refuse production execution."""
    job_id = environment.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError("Refusing to run training outside Slurm.")
    return job_id


def resolve_study_configuration(
    config: Mapping[str, Any],
    *,
    condition: str | None,
    pair: int | None,
) -> dict[str, Any]:
    """Resolve one frozen condition/pair without changing shared config input."""
    resolved = copy.deepcopy(dict(config))
    conditions = resolved.pop("study_conditions", None)
    pairs = resolved.pop("paired_runs", None)
    if conditions is None and pairs is None:
        if condition is not None or pair is not None:
            raise ValueError("Condition/pair overrides require a study configuration.")
        return resolved
    if not isinstance(conditions, Mapping) or not isinstance(pairs, list):
        raise TypeError("Study configuration requires conditions and paired runs.")
    if condition not in conditions or pair is None:
        raise ValueError("A known condition and pair index are required.")
    matches = [entry for entry in pairs if int(entry["pair"]) == pair]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one seed record for pair {pair}.")
    seed_record = matches[0]
    treatment = conditions[condition]
    resolved["model"]["initialization_seed"] = int(seed_record["initialization_seed"])
    resolved["data"]["data_order_seed"] = int(seed_record["data_order_seed"])
    resolved["training"]["training_noise_seed"] = int(
        seed_record["training_noise_seed"]
    )
    spectral = treatment.get("spectral_boundary_loss")
    if spectral is None:
        resolved.pop("spectral_boundary_loss", None)
    else:
        resolved["spectral_boundary_loss"] = copy.deepcopy(spectral)
    resolved["experiment"]["condition"] = condition
    resolved["experiment"]["pair"] = pair
    resolved["resolved_study"] = {
        "condition": condition,
        "pair": pair,
        "seeds": copy.deepcopy(seed_record),
    }
    return resolved
