"""Unit tests for :func:`wdfkit.cosmic_ray.remove_cosmic_rays_1d`."""

from __future__ import annotations

import numpy as np
import pytest

from wdfkit.cosmic_ray import remove_cosmic_rays_1d


def test_validate_y_must_be_1d():
    with pytest.raises(ValueError, match="1D"):
        remove_cosmic_rays_1d(np.zeros((2, 3)), "median", kernel_size=5)


def test_validate_kernel_must_be_odd():
    with pytest.raises(ValueError, match="kernel_size"):
        remove_cosmic_rays_1d(np.zeros(10), "median", kernel_size=4)


def test_validate_method():
    with pytest.raises(ValueError, match="method"):
        remove_cosmic_rays_1d(
            np.arange(20, dtype=float),
            "pca",
            kernel_size=5,
        )


def test_validate_threshold_positive():
    with pytest.raises(ValueError, match="threshold"):
        remove_cosmic_rays_1d(
            np.arange(20, dtype=float), "median", threshold=0.0
        )


def test_median_constant_spectrum_no_change():
    y = np.full(50, 100.0)
    out, mask = remove_cosmic_rays_1d(y, "median", kernel_size=5)
    np.testing.assert_array_equal(out, y)
    assert not mask.any()


def test_interpolate_positive_spike_reduced():
    y = np.linspace(0, 1, 100, dtype=np.float64) + 10.0
    y[50] = 500.0
    out, mask = remove_cosmic_rays_1d(
        y, "interpolate", kernel_size=5, threshold=3.0
    )
    assert out[50] < y[50] / 5
    assert mask[50]


def test_remove_cosmic_rays_1d_casts_to_float64():
    y = np.arange(25, dtype=np.int32)
    out, _ = remove_cosmic_rays_1d(y, "median", kernel_size=5)
    assert out.dtype == np.float64
