# -*- coding: utf-8 -*-
"""Parse ``WHTL`` embedded white-light image block."""

from __future__ import annotations

import io

from PIL import Image, ImageFile

from ..binary_io import read_from_file
from ..block_index import indices_named
from ..parse_context import ParseContext


def parse_whtl(ctx: ParseContext) -> None:
    name = "WHTL"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        nbytes = int((ctx.blocks["BlockSizes"][i] - 16) / 4)
        img_bytes = read_from_file(ctx.f, count=nbytes)
        # Allow truncated JPEG/PNG thumbnails embedded in some WDF files.
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        ctx.img = Image.open(io.BytesIO(img_bytes))
