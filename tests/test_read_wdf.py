"""Parity tests for `wdfkit.read_WDF` against legacy golden outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from wdfkit import WDFReader, read_WDF

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
    da, img = read_WDF(str(path))

    assert dict(da.sizes) == {"Time": 1, "shifts": 9341}
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
    np.testing.assert_array_equal(da["shifts"].shape, (9341,))

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
    da, img = read_WDF(str(path))

    assert dict(da.sizes) == {"Y": 17, "X": 25, "shifts": 9341}
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
        read_WDF(str(TEST_DATA / "nonexistent_file.wdf"))


def test_wdf_reader_matches_read_wdf_function():
    """Class API should match the legacy-style function wrapper."""
    path = TEST_DATA / "test.wdf"
    da_fn, img_fn = read_WDF(str(path))
    da_cls, img_cls = WDFReader(path).read()
    np.testing.assert_array_equal(da_fn.values, da_cls.values)
    assert img_fn is img_cls is None
