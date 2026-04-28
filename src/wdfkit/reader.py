# -*- coding: utf-8 -*-
"""Public :class:`WDFReader` API."""

from __future__ import annotations

import os
from typing import Union

from .wdf.io import read_wdf_file

StrPath = Union[str, os.PathLike[str]]


class WDFReader:
    """Load spectra and metadata from a Renishaw WiRE ``.wdf`` binary file.

    Typical usage::

        data_array, white_light_image = WDFReader(path)

    After construction, ``.data`` and ``.image`` hold the same objects as the
    unpacked tuple.

    Parameters
    ----------
    spectral_dim
        Name for the spectral axis coordinate (default ``None`` /
        ``\"auto\"``). WiRE ``XLST`` ``XlistDataUnits`` selects the default
        (e.g. ``Nanometre`` → dimension ``\"nm\"``). Set to ``\"shifts\"`` for
        legacy notebooks.
    """

    def __init__(
        self,
        path: StrPath,
        *,
        verbose: bool = False,
        time_coord: str = "seconds_elapsed",
        spectral_dim: str | None = None,
    ) -> None:
        self._path = os.fspath(path)
        self._verbose = verbose
        self._time_coord = (
            None if time_coord != "seconds_elapsed" else time_coord
        )
        self._spectral_dim = spectral_dim
        self.data, self.image = read_wdf_file(
            self._path,
            self._verbose,
            self._time_coord,
            self._spectral_dim,
        )

    def __iter__(self):
        yield self.data
        yield self.image
