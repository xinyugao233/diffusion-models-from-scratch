"""Export a trusted training checkpoint to an EMA-only inference bundle."""

import argparse
from pathlib import Path

from diffusion_models.inference import export_checkpoint
from diffusion_models.recovery import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attribution", required=True)
    args = parser.parse_args()
    result = export_checkpoint(
        args.checkpoint,
        args.output,
        expected_sha256=args.expected_sha256,
        attribution=args.attribution,
    )
    print(f"{sha256(result)}  {result}")


if __name__ == "__main__":
    main()
