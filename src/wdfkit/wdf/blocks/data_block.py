# -*- coding: utf-8 -*-
"""Parse ``DATA`` spectral intensity block."""

from __future__ import annotations

import numpy as np

from ..binary_io import read_from_file
from ..block_index import indices_named
from ..parse_context import ParseContext


def parse_data(ctx: ParseContext) -> None:
    name = "DATA"
    for i in indices_named(ctx.blocks, name):
        data_points_count = ctx.ncollected * ctx.npoints
        ctx.spectra = np.zeros((ctx.nspectra, ctx.npoints))
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        ctx.spectra[: ctx.ncollected] = read_from_file(
            ctx.f, "<f", count=data_points_count
        ).reshape(ctx.ncollected, ctx.npoints)
        if ctx.verbose:
            print(f'{"The number of spectra":-<40s} : \t{ctx.ncollected}')
            print(
                f'{"The number of points in each spectra":-<40s} : \t'
                f"{ctx.npoints}"
            )
