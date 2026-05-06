# -*- coding: utf-8 -*-
"""Public :class:`WDFReader` API plus module-level :func:`read` and
:func:`classify`."""

from __future__ import annotations

import os
from typing import Union

import xarray as xr

from .wdf.dispatch import classify_kind, dispatch
from .wdf.io import parse_wdf_header, parse_wdf_to_parsed

StrPath = Union[str, os.PathLike[str]]


class WDFReader:
    """Load spectra and metadata from a Renishaw WiRE ``.wdf`` binary
    file.

    Typical usage::

        data_array, white_light_image = WDFReader(path)

    After construction, ``.data`` and ``.image`` hold the same objects as the
    unpacked tuple.

    Parameters
    ----------
    spectral_dim
        Name for the spectral axis coordinate (default ``None`` /
        ``"auto"``).  WiRE ``XLST`` ``XlistDataUnits`` selects the
        default (e.g. ``RamanShift`` → dimension ``"raman_shift"``).
        Set to ``"shifts"`` for legacy notebooks.
    chunks
        Enable lazy Dask-backed reading.  ``False`` (default) = eager;
        ``True`` = auto-chunk at ~128 MB per chunk; ``int`` = target MB.
    """

    def __init__(
        self,
        path: StrPath,
        *,
        verbose: bool = False,
        time_coord: str = "seconds_elapsed",
        spectral_dim: str | None = None,
        chunks: bool | int = False,
    ) -> None:
        self._path = os.fspath(path)
        self._verbose = verbose
        self._time_coord = (
            None if time_coord != "seconds_elapsed" else time_coord
        )
        self._spectral_dim = spectral_dim
        self._chunks = chunks

        parsed = parse_wdf_to_parsed(
            self._path,
            verbose=self._verbose,
            time_coord=self._time_coord,
            spectral_dim=self._spectral_dim,
            chunks=self._chunks,
        )
        # Override spectral dim name when explicitly requested
        if spectral_dim and spectral_dim != "auto":
            parsed.xlst.dim_name = spectral_dim

        self.data: xr.DataArray = dispatch(parsed)
        self.image = parsed.img

    def __iter__(self):
        yield self.data
        yield self.image


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------


def read(
    path: StrPath,
    *,
    verbose: bool = False,
    spectral_dim: str | None = None,
    chunks: bool | int = False,
) -> xr.DataArray:
    """Read a WiRE ``.wdf`` file and return a :class:`xarray.DataArray`.

    Parameters
    ----------
    path
        Path to the ``.wdf`` file.
    spectral_dim
        Override for the spectral-axis dimension name.
    chunks
        Dask chunking: ``False`` (eager), ``True`` (auto), or int (target MB).

    Returns
    -------
    xarray.DataArray
        Shape and dims depend on scan kind; spectral axis is always last.
    """
    parsed = parse_wdf_to_parsed(
        path,
        verbose=verbose,
        spectral_dim=spectral_dim,
        chunks=chunks,
    )
    if spectral_dim and spectral_dim != "auto":
        parsed.xlst.dim_name = spectral_dim
    return dispatch(parsed)


def classify(path: StrPath) -> dict:
    """Return scan classification for a WiRE ``.wdf`` file *without*
    loading the spectral data.

    Returns
    -------
    dict
        Keys: ``kind``, ``measurement_type``, ``scan_type``,
        ``wmap_flag``, ``nspectra``, ``npoints``, ``nsteps``.
    """
    parsed = parse_wdf_header(path)
    kind = classify_kind(parsed)
    info: dict = {
        "kind": kind,
        "measurement_type": parsed.params.get("MeasurementType", ""),
        "scan_type": parsed.params.get("ScanType", ""),
        "wmap_flag": parsed.wmap.flag if parsed.wmap else None,
        "nspectra": parsed.nspectra,
        "npoints": parsed.npoints,
        "nsteps": (
            parsed.wmap.nsteps.tolist() if parsed.wmap is not None else None
        ),
    }
    return info
