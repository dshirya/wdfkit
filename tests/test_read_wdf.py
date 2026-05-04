"""Parity tests for :class:`wdfkit.WDFReader` against golden spectral
arrays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from wdfkit import WDFReader

TEST_DATA = Path(__file__).resolve().parent / "test_data"

# SHA256 of contiguous spectral ``values`` bytes — pins full arrays.
_VALUES_SHA256 = {
    "test.wdf": (
        "1121d8054198265a1c476ebf662816edb7714a8c163f774b0e3457ba3e46ec65"
    ),
    "test_2.wdf": (
        "3b389ea645ecc6f712147b264d607800598bc76354cff6a28b558c08b06c25a9"
    ),
}


def _sha256_values(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_read_wdf_single_scan_matches_golden():
    path = TEST_DATA / "test.wdf"
    da, img = WDFReader(path)

    assert dict(da.sizes) == {"Time": 1, "nm": 9341}
    assert da.attrs["WdfFlag"] == "WdfXYXY"
    assert da.attrs["MeasurementType"] == "Single"
    assert da.attrs["PointsPerSpectrum"] == 9341
    assert da.attrs["Capacity"] == 1
    assert da.attrs["Count"] == 1
    assert da.attrs["ScanShape"] == (1, 1)
    assert da.attrs["ColCoord"] is None
    assert da.attrs["RowCoord"] == "Time"
    assert da.attrs["Filename"] == "test.wdf"
    assert da.attrs["Folder name"] == str(path.parent)
    assert da.attrs["FileSize"] == "357.5kB"
    assert da.attrs["SpectralUnits"] == "Counts"
    assert np.isclose(da.attrs["LaserWaveLength"], 354.74)
    assert img is None

    np.testing.assert_allclose(
        da["Time"].values,
        np.array([137.53774643]),
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_array_equal(da["nm"].shape, (9341,))

    row = da.values[0]
    np.testing.assert_allclose(
        row[:3], [959.03619385, 961.67810059, 973.14605713]
    )
    np.testing.assert_allclose(
        row[-3:], [17035.36328125, 17592.48242188, 17531.3671875]
    )

    assert _sha256_values(da.values) == _VALUES_SHA256["test.wdf"]


def test_read_wdf_map_matches_golden():
    path = TEST_DATA / "test_2.wdf"
    da, img = WDFReader(path)

    assert dict(da.sizes) == {"Y": 17, "X": 25, "nm": 9341}
    assert da.attrs["WdfFlag"] == "16: UnknownFlag (LiveTrack?)"
    assert da.attrs["MeasurementType"] == "Map"
    assert da.attrs["MapAreaType"] == "RandomPoints"
    assert da.attrs["Capacity"] == 425
    assert da.attrs["Count"] == 425
    assert da.attrs["PointsPerSpectrum"] == 9341
    assert da.attrs["ScanShape"] == (17, 25)
    assert da.attrs["ColCoord"] == "X"
    assert da.attrs["RowCoord"] == "Y"
    assert da.attrs["Filename"] == "test_2.wdf"
    assert da.attrs["Folder name"] == str(path.parent)
    assert da.attrs["FileSize"] == "16.2MB"
    assert img is not None
    assert getattr(img, "mode") == "RGB"

    corner = da.values[0, 0, :3]
    np.testing.assert_allclose(
        corner, [668.75933838, 673.73608398, 695.56280518]
    )
    np.testing.assert_allclose(
        da.values[0, 0, -3:],
        [12142.55175781, 12009.16113281, 12445.69824219],
    )

    assert _sha256_values(da.values) == _VALUES_SHA256["test_2.wdf"]


def test_read_wdf_missing_file_raises():
    with pytest.raises(IOError, match="does not exist"):
        WDFReader(TEST_DATA / "nonexistent_file.wdf")


def test_wdf_reader_idempotent_same_file():
    """Two reads of the same path yield identical spectral cubes."""
    path = TEST_DATA / "test.wdf"
    da_a, _ = WDFReader(path)
    da_b, _ = WDFReader(path)
    np.testing.assert_array_equal(da_a.values, da_b.values)


def test_spectral_dim_override_restores_legacy_name():
    """Force spectral coordinate dimension name (e.g. ``shifts``)."""
    path = TEST_DATA / "test.wdf"
    da, _ = WDFReader(path, spectral_dim="shifts")
    assert dict(da.sizes) == {"Time": 1, "shifts": 9341}
    assert da["shifts"].attrs.get("units") == "nm"


def test_exposure_time_and_laser_power_single_scan():
    """ExposureTime and LaserPower are read from WXDM/WXIS blocks."""
    path = TEST_DATA / "test.wdf"
    da, _ = WDFReader(path)

    # Exposure Time: stored as int ms in WXDM, exposed as float seconds.
    assert "ExposureTime" in da.attrs
    assert np.isclose(da.attrs["ExposureTime"], 10.0)

    # Laser Power: ND Transmission % from WXIS, stored as float percent.
    assert "LaserPower" in da.attrs
    assert np.isclose(da.attrs["LaserPower"], 10.0)


def test_exposure_time_and_laser_power_map():
    """ExposureTime and LaserPower are read correctly for 2-D map data."""
    path = TEST_DATA / "test_2.wdf"
    da, _ = WDFReader(path)

    assert "ExposureTime" in da.attrs
    assert np.isclose(da.attrs["ExposureTime"], 10.0)

    assert "LaserPower" in da.attrs
    assert np.isclose(da.attrs["LaserPower"], 5.0)
