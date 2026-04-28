# -*- coding: utf-8 -*-
"""Public :class:`WDFReader` API and legacy :func:`read_WDF` wrapper."""

from __future__ import annotations

import os
from typing import Union

from .wdf_io import read_wdf_file

StrPath = Union[str, os.PathLike[str]]


class WDFReader:
    """Load spectra and metadata from a Renishaw WiRE ``.wdf`` binary file.

    Typical usage::

        reader = WDFReader(path)
        data_array, white_light_image = reader.read()

    The function :func:`read_WDF` is a thin wrapper around this class for
    backwards compatibility with legacy scripts.
    """

    def __init__(
        self,
        path: StrPath,
        *,
        verbose: bool = False,
        time_coord: str = "seconds_elapsed",
    ) -> None:
        self._path = os.fspath(path)
        self._verbose = verbose
        self._time_coord = (
            None if time_coord != "seconds_elapsed" else time_coord
        )

    def read(self):
        """Parse the file and return ``(data_array, white_light_image)``."""
        return read_wdf_file(self._path, self._verbose, self._time_coord)


def read_WDF(filename, verbose=False, time_coord="seconds_elapsed", **kwargs):
    """Read the data (and metadata) from the binary .wdf file.

    Prefer :class:`WDFReader` for library-style use; this function matches the
    legacy ``read_WDF(filename)`` calling convention.
    """
    return WDFReader(filename, verbose=verbose, time_coord=time_coord).read()
