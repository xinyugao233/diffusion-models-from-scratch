from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_cifar10_radial_power.py"
SPEC = importlib.util.spec_from_file_location("compare_cifar10_radial_power", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_radial_power_matches_spatial_parseval() -> None:
    rng = np.random.default_rng(7)
    images = rng.standard_normal((5, 3, 32, 32))

    powers, counts = MODULE.radial_power(images)

    spectral_mean = float(np.sum(powers * counts))
    spatial_mean = float(np.mean(np.square(images).sum(axis=(-2, -1))))
    np.testing.assert_allclose(spectral_mean, spatial_mean, rtol=1e-13, atol=1e-13)
    assert int(counts.sum()) == 32 * 32


def test_information_radius_uses_largest_crossing_shell() -> None:
    powers = np.asarray([9.0, 4.0, 1.0, 0.25])
    sigmas = np.asarray([0.4, 1.0, 2.1, 4.0])

    actual = MODULE.information_radius(powers, sigmas)

    np.testing.assert_array_equal(actual, [3, 2, 0, -1])


def test_effective_sigmas_match_ddpm_definition() -> None:
    actual = MODULE.effective_sigmas(2, 0.1, 0.2)
    alpha_bars = np.asarray([0.9, 0.9 * 0.8])
    expected = np.sqrt((1.0 - alpha_bars) / alpha_bars)
    np.testing.assert_allclose(actual, expected)
