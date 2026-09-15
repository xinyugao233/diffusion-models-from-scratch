#!/usr/bin/env python3
"""Exercise frozen spectral-study identities and the pinned KID implementation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from _experiment_utils import git_identity, load_json, repository_path

from diffusion_models.full_training import (
    require_slurm_environment,
    resolve_study_configuration,
)
from diffusion_models.models import CIFAR10UNet, primary_unet_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(seed: int) -> str:
    torch.manual_seed(seed)
    model = CIFAR10UNet(primary_unet_config())
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    job_id = require_slurm_environment(os.environ)
    config = load_json(repository_path("configs/spectral_boundary_20k.json"))
    git = git_identity()
    if git["working_tree_dirty"] or git["commit"] != os.environ.get("RUN_COMMIT"):
        raise RuntimeError("Protocol preflight requires the exact clean RUN_COMMIT.")
    evaluation = config["evaluation"]
    asset_path = Path(evaluation["asset_path"])
    weights_path = Path(evaluation["inception_weights_path"])
    if file_sha256(asset_path) != evaluation["asset_sha256"]:
        raise RuntimeError("Held-out asset identity mismatch.")
    if file_sha256(weights_path) != evaluation["inception_weights_sha256"]:
        raise RuntimeError("Inception weight identity mismatch.")
    asset = torch.load(asset_path, weights_only=True)
    count = evaluation["heldout_denoising"]["example_count"]
    expected_indices = torch.arange(count)
    expected_timesteps = torch.div(
        expected_indices * 1000, count, rounding_mode="floor"
    )
    expected_epsilon = torch.randn(
        (count, 3, 32, 32),
        generator=torch.Generator().manual_seed(
            evaluation["heldout_denoising"]["noise_seed"]
        ),
    )
    if not (
        torch.equal(asset["example_indices"], expected_indices)
        and torch.equal(asset["timesteps"], expected_timesteps)
        and torch.equal(asset["epsilon"], expected_epsilon)
    ):
        raise RuntimeError("Held-out tensors do not reproduce exactly.")

    initial_hashes: dict[str, dict[str, str]] = {}
    for pair in range(3):
        pair_hashes = {}
        for condition in config["study_conditions"]:
            resolved = resolve_study_configuration(
                config, condition=condition, pair=pair
            )
            pair_hashes[condition] = model_sha256(
                resolved["model"]["initialization_seed"]
            )
        if len(set(pair_hashes.values())) != 1:
            raise RuntimeError(f"Pair {pair} initial model identities differ.")
        initial_hashes[str(pair)] = pair_hashes

    from torch_fidelity import calculate_metrics

    class Images(torch.utils.data.Dataset):
        def __init__(self, values: torch.Tensor) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> torch.Tensor:
            return self.values[index]

    generator = torch.Generator().manual_seed(701)
    first = torch.randint(
        0, 256, (4, 3, 32, 32), dtype=torch.uint8, generator=generator
    )
    second = torch.randint(
        0, 256, (4, 3, 32, 32), dtype=torch.uint8, generator=generator
    )
    smoke = calculate_metrics(
        input1=Images(first),
        input2=Images(second),
        cuda=False,
        isc=False,
        fid=False,
        kid=True,
        feature_extractor="inception-v3-compat",
        feature_layer_kid="2048",
        feature_extractor_weights_path=str(weights_path),
        batch_size=4,
        kid_subsets=1,
        kid_subset_size=4,
        kid_kernel_poly_degree=3,
        kid_kernel_poly_gamma=None,
        kid_kernel_poly_coef0=1.0,
        rng_seed=2020,
        cache=False,
        verbose=False,
    )
    result = {
        "passed": True,
        "slurm_job_id": job_id,
        "git": git,
        "evaluation_asset_sha256": evaluation["asset_sha256"],
        "inception_weights_sha256": evaluation["inception_weights_sha256"],
        "initial_model_sha256_by_pair_and_condition": initial_hashes,
        "kid_smoke": smoke,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
