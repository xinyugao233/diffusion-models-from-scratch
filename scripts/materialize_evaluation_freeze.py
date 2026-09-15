#!/usr/bin/env python3
"""Materialize the frozen held-out identities without consuming training RNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from _experiment_utils import git_identity, load_json, repository_path

from diffusion_models.full_training import require_slurm_environment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    job_id = require_slurm_environment(os.environ)
    config_path = repository_path(args.config)
    config = load_json(config_path)
    git = git_identity()
    if git["working_tree_dirty"] or git["commit"] != os.environ.get("RUN_COMMIT"):
        raise RuntimeError("Evaluation freeze requires the exact clean RUN_COMMIT.")

    evaluation = config["evaluation"]
    heldout = evaluation["heldout_denoising"]
    count = int(heldout["example_count"])
    num_steps = int(config["diffusion"]["num_steps"])
    indices = torch.arange(count, dtype=torch.int64)
    timesteps = torch.div(indices * num_steps, count, rounding_mode="floor")
    generator = torch.Generator(device="cpu").manual_seed(int(heldout["noise_seed"]))
    noise = torch.randn((count, 3, 32, 32), generator=generator, dtype=torch.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "heldout_denoising.pt"
    with artifact.open("xb") as handle:
        torch.save(
            {
                "schema_version": 1,
                "example_indices": indices,
                "timesteps": timesteps,
                "epsilon": noise,
                "noise_seed": int(heldout["noise_seed"]),
                "timestep_definition": heldout["timestep_assignment"],
                "git_commit": git["commit"],
            },
            handle,
        )
    reloaded = torch.load(artifact, weights_only=True)
    if not (
        torch.equal(reloaded["example_indices"], indices)
        and torch.equal(reloaded["timesteps"], timesteps)
        and torch.equal(reloaded["epsilon"], noise)
    ):
        raise RuntimeError("Serialized evaluation identity failed exact round trip.")
    manifest = {
        "schema_version": 1,
        "slurm_job_id": job_id,
        "git": git,
        "config_path": str(config_path),
        "artifact_path": str(artifact),
        "artifact_sha256": sha256(artifact),
        "example_count": count,
        "timestep_min": int(timesteps.min()),
        "timestep_max": int(timesteps.max()),
        "noise_tensor_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
        "sampling_seed": evaluation["sampling"]["checkpoint_seed"],
        "rng_isolation": evaluation["rng_isolation"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
