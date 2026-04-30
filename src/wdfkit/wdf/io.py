# -*- coding: latin-1 -*-
"""Orchestrate WiRE ``.wdf`` parsing: block index → parsers → xarray
assembly."""

from __future__ import annotations

import os

from PIL import ImageFile

from .assemble import assemble_data_array
from .block_index import scan_blocks
from .blocks.data_block import parse_data
from .blocks.origin import parse_orgn, print_coord_lengths_if_verbose
from .blocks.wdf1 import parse_wdf1
from .blocks.whtl import parse_whtl
from .blocks.wmap import parse_wmap
from .blocks.xlst import parse_xlst
from .blocks.ylst import parse_ylst
from .parse_context import ParseContext

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_wdf_file(filename, verbose, time_coord, spectral_dim=None):
    """Parse a WiRE WDF file (invoked by
    :class:`~wdfkit.reader.WDFReader`).

    Parameters
    ----------
    spectral_dim
        Passed to :func:`~wdfkit.spectral.resolve_spectral_axis` during
        ``XLST`` handling.
    """
    try:
        file_obj = open(filename, "rb")
        print(f'Reading the file: "{filename.split("/")[-1]}"\n')
    except IOError:
        raise IOError(f"File {filename} does not exist!") from None

    filesize = os.path.getsize(filename)
    ctx = ParseContext(
        filename=filename,
        verbose=verbose,
        time_coord=time_coord,
        spectral_dim=spectral_dim,
        filesize=filesize,
        f=file_obj,
    )

    try:
        ctx.blocks = scan_blocks(ctx.f, filesize)
        parse_wdf1(ctx)
        if verbose:
            for key, val in ctx.params.items():
                print(f"{key:-<40s} : \t{val}")

        parse_wmap(ctx)
        if verbose:
            for key, val in ctx.map_params.items():
                print(f"{key:-<40s} : \t{val}")

        parse_data(ctx)
        parse_xlst(ctx)
        parse_ylst(ctx)
        parse_whtl(ctx)
        parse_orgn(ctx)
        print_coord_lengths_if_verbose(ctx)

        return assemble_data_array(ctx)
    finally:
        file_obj.close()
