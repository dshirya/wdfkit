# -*- coding: utf-8 -*-
"""Parse ``WMAP`` map-area metadata block."""

from __future__ import annotations

import numpy as np

from ...internal import constants as const
from ..binary_io import read_from_file
from ..block_index import indices_named
from ..parse_context import ParseContext


def parse_wmap(ctx: ParseContext) -> None:
    name = "WMAP"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        m_flag = read_from_file(ctx.f)
        ctx.map_params["MapAreaType"] = const.MAP_TYPES[m_flag]
        read_from_file(ctx.f)
        ctx.map_params["InitialCoordinates"] = np.round(
            read_from_file(ctx.f, "<f", count=3), 2
        )
        ctx.map_params["StepSizes"] = np.round(
            read_from_file(ctx.f, "<f", count=3), 2
        )
        ctx.map_params["NbSteps"] = read_from_file(ctx.f, np.uint32, count=3)
        ctx.map_params["LineFocusSize"] = read_from_file(ctx.f)
