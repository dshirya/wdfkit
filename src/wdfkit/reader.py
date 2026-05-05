# -*- coding: utf-8 -*-
"""Public :class:`WDFReader` API."""

from __future__ import annotations

import os
from typing import Union

from .wdf.io import read_wdf_file

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
        ``\"auto\"``). WiRE ``XLST`` ``XlistDataUnits`` selects the default
        (e.g. ``Nanometre`` → dimension ``\"nm\"``). Set to ``\"shifts\"`` for
        legacy notebooks.
    chunks
        Enable lazy Dask-backed reading to handle large maps without loading
        the full data into RAM.

        - ``False`` (default): eager — entire DATA block is read into a
          NumPy array immediately, matching the previous behaviour.
        - ``True``: lazy — auto-compute chunk size targeting ~128 MB per
          chunk along the Y (row) axis, capped at 20 chunks.
        - ``int``: lazy — use that value as the target chunk size in MB
          (e.g. ``chunks=256`` targets 256 MB per chunk).

        Before reading, ``WDFReader`` checks available RAM. If the full
        array would exceed free RAM it raises ``MemoryError`` and asks you
        to re-open with ``chunks=True``. A ``UserWarning`` is emitted if
        the array would use more than 75 % of free RAM.
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
        self.data, self.image = read_wdf_file(
            self._path,
            self._verbose,
            self._time_coord,
            self._spectral_dim,
            chunks=self._chunks,
        )

    def __iter__(self):
        yield self.data
        yield self.image
