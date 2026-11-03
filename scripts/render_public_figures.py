"""Derived layout only: preserve all historical sample selections and timing values."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    historical = (
        root / "Experiments/exp005-ddim-sampling-speed-quality-comparison/try01"
    )
    records = json.loads((historical / "results/comparison.json").read_text())[
        "records"
    ]
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), layout="constrained")
    for axis, record in zip(axes, records, strict=True):
        axis.imshow(
            Image.open(historical / "figures" / f"{record['name']}_grid.png"),
            interpolation="nearest",
        )
        axis.axis("off")
        axis.set_title(
            f"{record['name'].upper().replace('_', ' · ')} NFE\n{record['median_seconds']:.3f} s · {record['speedup_vs_ddpm_1000']:.1f}×",
            fontsize=13,
        )
    fig.suptitle(
        "Same 50K-step EMA checkpoint · same 16 initial-noise seeds · recorded H100 NVL run",
        fontsize=15,
    )
    fig.savefig(args.output_dir / "sampling_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    names = [record["name"].replace("_", "-").upper() for record in records]
    values = [record["median_seconds"] for record in records]
    bars = ax.barh(names, values, color=["#263c5a", "#3d76b6", "#5794ce", "#83b6df"])
    ax.invert_yaxis()
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Median seconds per 16-image batch (3 timed repeats)")
    ax.set_title("Historical H100 NVL sampling latency · 50K EMA checkpoint", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, record in zip(bars, records, strict=True):
        ax.text(
            bar.get_width() + 0.06,
            bar.get_y() + bar.get_height() / 2,
            f"{record['median_seconds']:.4f} s  ({record['speedup_vs_ddpm_1000']:.2f}×)",
            va="center",
            fontsize=11,
        )
    fig.savefig(args.output_dir / "sampling_latency.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
