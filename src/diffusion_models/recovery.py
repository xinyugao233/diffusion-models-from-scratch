"""Non-destructive log reconciliation and atomic checkpoint publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch

from diffusion_models.checkpointing import (
    CHECKPOINT_KEYS,
    CHECKPOINT_VERSION,
    CheckpointError,
)


def sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def read_checkpoint(path, expected_configuration=None):
    """Check portable structural invariants without executing pickle globals."""
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if (
            set(payload) != CHECKPOINT_KEYS
            or payload["checkpoint_version"] != CHECKPOINT_VERSION
        ):
            raise ValueError("invalid checkpoint schema")
        step = payload["global_step"]
        if type(step) is not int or step < 0:
            raise ValueError("invalid checkpoint step")
        if (
            expected_configuration is not None
            and payload["configuration"] != expected_configuration
        ):
            raise ValueError("incompatible configuration")
        if not isinstance(payload["metadata"], dict) or not isinstance(
            payload["configuration"], dict
        ):
            raise TypeError("invalid checkpoint metadata/configuration")
        ema = payload["ema_state_dict"]
        if (
            set(ema) != {"decay", "num_updates", "parameters", "buffers"}
            or ema["num_updates"] != step
        ):
            raise ValueError("EMA step/schema mismatch")
        if not 0 <= ema["decay"] < 1:
            raise ValueError("invalid EMA decay")
        state = payload["model_state_dict"]
        ema_state = {**ema["parameters"], **ema["buffers"]}
        if not state or state.keys() != ema_state.keys():
            raise ValueError("EMA/model keys mismatch")
        for key, tensor in state.items():
            other = ema_state[key]
            if tensor.shape != other.shape or tensor.dtype != other.dtype:
                raise ValueError("EMA/model tensor mismatch")
            if not torch.isfinite(tensor).all() or not torch.isfinite(other).all():
                raise ValueError("nonfinite state")
        optimizer = payload["optimizer_state_dict"]
        if set(optimizer) != {"state", "param_groups"} or not optimizer["param_groups"]:
            raise ValueError("invalid optimizer state")
        rng = payload["rng_state"]
        if set(rng) != {"python", "torch_cpu", "generators", "cuda", "numpy"}:
            raise ValueError("invalid RNG schema")
        random.Random().setstate(rng["python"])
        torch.Generator().set_state(rng["torch_cpu"])
        for value in rng["generators"].values():
            if value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("invalid generator state")
        return payload
    except Exception as error:
        raise CheckpointError(f"Invalid checkpoint {path}: {error}") from error


def atomic_publish(destination, writer, validator):
    """Fsync and validate a sibling temporary file, then link without clobbering.

    Hard-link publication is atomic and exclusive on supported local/POSIX filesystems.
    Unsupported filesystems fail explicitly. This is not a cluster durability claim.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        validator(temporary)
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def stage_checkpoint(source, destination):
    """Copy a complete validated snapshot; never expose a partial destination."""
    source = Path(source)
    read_checkpoint(source)
    digest = sha256(source)

    def copy(handle):
        with source.open("rb") as original:
            shutil.copyfileobj(original, handle)

    def validate(path):
        if sha256(path) != digest:
            raise CheckpointError("Staged checksum differs")
        read_checkpoint(path)

    return atomic_publish(destination, copy, validate)


def recover_run(
    checkpoint, log, destination, *, expected_configuration, checkpoint_validator=None
):
    """Copy complete records 1..k; retain evidence of all later interrupted bytes.

    A caller-specific validator can additionally restore model/optimizer shapes.
    The production CLI supplies this validator; reusable toy callers may supply theirs.
    """
    checkpoint, log, destination = Path(checkpoint), Path(log), Path(destination)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(
            f"Recovery destination must be new or empty: {destination}"
        )
    payload = read_checkpoint(checkpoint, expected_configuration)
    if checkpoint_validator is not None:
        checkpoint_validator(checkpoint)
    step = payload["global_step"]
    data = log.read_bytes()
    lines = data.splitlines(keepends=True)
    prefix = []
    for index in range(step):
        try:
            line = lines[index]
            record = json.loads(line)
            if (
                not line.endswith(b"\n")
                or type(record.get("step")) is not int
                or record["step"] != index + 1
            ):
                raise ValueError("missing/noncontiguous/incomplete record")
            if "loss" not in record or not math.isfinite(float(record["loss"])):
                raise ValueError("missing/nonfinite loss")
            if any(
                isinstance(v, float) and not math.isfinite(v) for v in record.values()
            ):
                raise ValueError("nonfinite record")
            prefix.append(line)
        except (IndexError, ValueError, TypeError, AttributeError) as error:
            raise ValueError(
                f"Invalid committed log prefix at step {index + 1}: {error}"
            ) from error
    tail = data[len(b"".join(prefix)) :]
    malformed_tail = False
    for line in tail.splitlines(keepends=True):
        try:
            json.loads(line)
            malformed_tail |= not line.endswith(b"\n")
        except (ValueError, UnicodeDecodeError):
            malformed_tail = True
    destination.mkdir(parents=True, exist_ok=True)
    copied = stage_checkpoint(checkpoint, destination / "checkpoints" / checkpoint.name)
    if step > 0:
        (destination / "metrics.jsonl").write_bytes(b"".join(prefix))
    if tail:
        (destination / "interrupted_tail.bin").write_bytes(tail)
    manifest = {
        "schema_version": 1,
        "checkpoint_step": step,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_log": str(log.resolve()),
        "source_log_sha256": sha256(log),
        "prefix_sha256": hashlib.sha256(b"".join(prefix)).hexdigest(),
        "checkpoint": str(copied),
        "configuration": payload["configuration"],
        "metadata": payload["metadata"],
        "tail_bytes": len(tail),
        "tail_sha256": hashlib.sha256(tail).hexdigest(),
        "malformed_uncommitted_tail": malformed_tail,
        "diagnostic": "Only records through the checkpoint are committed; original and discarded tail preserved.",
    }
    (destination / "recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest
