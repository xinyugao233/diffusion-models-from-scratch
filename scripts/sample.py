"""Portable local inference from a validated EMA-only bundle."""

import argparse
import json
import platform
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion_models.inference import sample_bundle
from diffusion_models.recovery import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    samples, metadata = sample_bundle(
        args.checkpoint,
        device=args.device,
        sampler=args.sampler,
        steps=args.inference_steps,
        count=args.sample_count,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(samples, args.output_dir / "samples.pt")
    save_image(((samples + 1) / 2).clamp(0, 1), args.output_dir / "samples.png", nrow=4)
    manifest = {
        **vars(args),
        "checkpoint_sha256": sha256(args.checkpoint),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "training_step": metadata["training_step"],
        "attribution": metadata["attribution"],
        "nfe": args.inference_steps
        if args.sampler == "ddim"
        else metadata["diffusion"]["num_steps"],
        "scope": "Local inference; not a reproduction of the historical H100 timing protocol",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, default=str, indent=2) + "\n"
    )
    print(f"Saved samples and manifest to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
