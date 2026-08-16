from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_ddim_comparison import make_runtime_figure


def test_runtime_figure_keeps_annotations_inside_right_margin(tmp_path: Path) -> None:
    output = tmp_path / "runtime.png"
    records = [
        {"name": "ddpm_1000", "median_seconds": 4.611, "model_evaluations": 1000},
        {"name": "ddim_100", "median_seconds": 0.450, "model_evaluations": 100},
        {"name": "ddim_50", "median_seconds": 0.225, "model_evaluations": 50},
        {"name": "ddim_25", "median_seconds": 0.112, "model_evaluations": 25},
    ]

    make_runtime_figure(output, records)

    with Image.open(output).convert("RGB") as image:
        assert image.size == (1000, 420)
        right_margin = image.crop((951, 0, 1000, 420))
        assert right_margin.getextrema() == ((255, 255), (255, 255), (255, 255))
