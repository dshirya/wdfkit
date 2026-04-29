"""Tests for spectral preprocessing (normalize, cosmic-ray removal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wdfkit import CosmicRayRemover, WDFReader, normalize

TEST_DATA = Path(__file__).resolve().parent / "test_data"


@pytest.fixture(scope="module")
def single_da():
    da, _ = WDFReader(TEST_DATA / "test.wdf")
    return da


@pytest.fixture(scope="module")
def map_da():
    da, _ = WDFReader(TEST_DATA / "test_2.wdf")
    return da


def test_normalize_default_spectral_dim_matches_reader(single_da):
    out = normalize(single_da, method="max")
    assert out.dims == single_da.dims
    assert "normalization" in out.attrs["treatments"]
    assert out.attrs["treatments"]["normalization"]["method"] == "max"
    # Max-abs row norm, then per-row min subtraction (legacy): rows touch zero.
    row_span = out.values.max(axis=-1) - out.values.min(axis=-1)
    assert np.all((out.values >= -1e-9) & (row_span > 0.95))


def test_normalize_explicit_spectral_dim_when_not_last(single_da):
    da = single_da.transpose("nm", "Time")
    assert da.dims[0] == "nm"
    out = normalize(da, method="min_max", spectral_dim="nm")
    assert out.dims == da.dims
    # min_max to [0, 1] per spectrum; final min-subtract leaves [0, 1].
    assert float(np.nanmax(out.values)) <= 1.0 + 1e-6
    assert float(np.nanmin(out.values)) >= -1e-6


def test_normalize_invalid_spectral_dim_raises(single_da):
    with pytest.raises(ValueError, match="spectral_dim"):
        normalize(single_da, spectral_dim="not_a_dim")


def test_cosmic_ray_single_no_spike_unchanged():
    n = 300
    spec = np.full(n, 100.0, dtype=np.float64)
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("idx", "wavenumber"),
        coords={"idx": [0], "wavenumber": np.arange(n)},
        attrs={"treatments": {}},
    )
    cr = CosmicRayRemover(threshold=5.0, kernel_size=5)
    out = cr.transform(da)
    np.testing.assert_array_equal(out.values[0], spec)
    meta = out.attrs["treatments"]["Cosmic Ray Correction"]
    assert "CRs found (spectral indices)" not in meta


def test_cosmic_ray_single_removes_spike():
    n = 200
    spec = np.linspace(0, 1, n, dtype=np.float64) + 100.0
    spec[50] = 5000.0
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("Time", "nm"),
        coords={"Time": [0.0], "nm": np.arange(n)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(threshold=3.0, kernel_size=5).transform(da)
    assert out.values[0, 50] < da.values[0, 50] / 10
    assert (
        "CRs found (spectral indices)"
        in out.attrs["treatments"]["Cosmic Ray Correction"]
    )


def test_cosmic_ray_single_interpolate_removes_spike():
    n = 200
    spec = np.linspace(0, 1, n, dtype=np.float64) + 100.0
    spec[50] = 5000.0
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("Time", "nm"),
        coords={"Time": [0.0], "nm": np.arange(n)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(
        threshold=3.0,
        kernel_size=5,
        single_spectrum_method="interpolate",
    ).transform(da)
    assert out.values[0, 50] < da.values[0, 50] / 10
    assert (
        out.attrs["treatments"]["Cosmic Ray Correction"][
            "single_spectrum_method"
        ]
        == "interpolate"
    )


def test_cosmic_ray_single_derivative_removes_spike():
    n = 256
    spec = np.linspace(0, 1, n, dtype=np.float64) * 20.0 + 50.0
    spec[100] = 800.0
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("t", "eV"),
        coords={"t": [0], "eV": np.linspace(1.5, 3.0, n)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(
        kernel_size=5,
        single_spectrum_method="derivative",
        threshold=2.5,
    ).transform(da)
    assert out.values[0, 100] < spec[100] / 8
    assert (
        out.attrs["treatments"]["Cosmic Ray Correction"][
            "single_spectrum_method"
        ]
        == "derivative"
    )


def test_cosmic_ray_single_derivative_smooth_high_threshold():
    n = 300
    spec = np.asarray(
        np.sin(np.linspace(0, 2 * np.pi, n)) * 2.0 + 50.0,
        dtype=np.float64,
    )
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("i", "cm"),
        coords={"i": [0], "cm": np.arange(n, dtype=float)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(
        kernel_size=5,
        single_spectrum_method="derivative",
        threshold=12.0,
    ).transform(da)
    np.testing.assert_allclose(out.values[0], spec, rtol=0, atol=0.08)


def test_cosmic_ray_single_invalid_method_raises():
    with pytest.raises(ValueError, match="single_spectrum_method"):
        CosmicRayRemover(**{"single_spectrum_method": "pca"})


def test_cosmic_ray_map_removes_spike():
    ny, nx, n = 5, 5, 64
    cube = np.random.default_rng(0).random((ny, nx, n)).astype(np.float64) * 10
    cube[2, 2, 20] = 500.0
    da = xr.DataArray(
        cube,
        dims=("Y", "X", "nm"),
        coords={
            "Y": np.arange(ny),
            "X": np.arange(nx),
            "nm": np.arange(n),
        },
        attrs={"treatments": {}},
    )
    cr = CosmicRayRemover(sensitivity=0.5, width=0.08, disk_radius=2)
    out = cr.transform(da)
    assert out.values[2, 2, 20] < cube[2, 2, 20] / 5


def test_cosmic_ray_degenerate_map_uses_single_path():
    n = 100
    spec = np.ones(n, dtype=np.float64) * 50
    spec[30] = 800.0
    da = xr.DataArray(
        spec.reshape(1, 1, n),
        dims=("Y", "X", "nm"),
        coords={"Y": [0], "X": [0], "nm": np.arange(n)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(threshold=2.5, kernel_size=5).transform(da)
    assert out.values[0, 0, 30] < 100.0


def test_cosmic_ray_rejects_series_like_multispectrum(map_da):
    path = TEST_DATA / "test.wdf"
    da2, _ = WDFReader(path)
    fake_series = xr.concat([da2, da2], dim="Time")
    with pytest.raises(ValueError, match="CosmicRayRemover"):
        CosmicRayRemover().transform(fake_series)
