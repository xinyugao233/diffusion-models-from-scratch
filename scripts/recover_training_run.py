"""Reconcile a trusted full-training snapshot into a new continuation directory."""

import argparse
import json
from pathlib import Path

import torch

from diffusion_models.checkpointing import load_training_checkpoint
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.models import CIFAR10UNet, primary_unet_config
from diffusion_models.recovery import read_checkpoint, recover_run, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["gate_a", "gate_b", "full"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    expected = {"config": config, "stage": args.stage}
    payload = read_checkpoint(args.checkpoint, expected)
    if payload["metadata"].get("configuration_sha256") != sha256(args.config):
        raise ValueError("Checkpoint metadata/config file checksum mismatch")
    model = CIFAR10UNet(primary_unet_config())
    ema = ExponentialMovingAverage(model, config["training"]["ema_decay"])
    optimizer = torch.optim.AdamW(model.parameters())

    def validate(path):
        load_training_checkpoint(
            path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            expected_configuration=expected,
            restore_rng=False,
        )

    print(
        json.dumps(
            recover_run(
                args.checkpoint,
                args.log,
                args.output_dir,
                expected_configuration=expected,
                checkpoint_validator=validate,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
