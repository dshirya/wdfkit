# -*- coding: utf-8 -*-
"""Parse ``WXIS`` (WiRE Instrument State) block.

The ``WXIS`` block stores a snapshot of all hardware motor positions and
instrument configuration at the time of acquisition.

Currently extracted
-------------------
``LaserPower``
    ND-filter transmission in **percent** (e.g. ``10.0`` for 10 %).
    Stored in WiRE as ``"ND Transmission %"`` which may be a bare numeric
    string (``"10"``) or a string with a ``" Percent"`` suffix
    (``"0 Percent"``).  If neither form is parseable the attribute is omitted.
"""

from __future__ import annotations

from ..block_index import indices_named
from ..parse_context import ParseContext
from ..pset import decode_pset


def _parse_nd_percent(raw: object) -> float | None:
    """Convert a raw ``"ND Transmission %"`` value to a plain float.

    Accepts:
    * ``"10"`` → ``10.0``
    * ``"0 Percent"`` → ``0.0``
    * ``10`` (int) → ``10.0``

    Returns ``None`` when the value cannot be interpreted.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    # Bare number
    try:
        return float(s)
    except ValueError:
        pass
    # "N Percent" format
    s_lower = s.lower()
    for suffix in (" percent", "%"):
        if s_lower.endswith(suffix):
            try:
                return float(s[: len(s) - len(suffix)].strip())
            except ValueError:
                pass
    return None


def parse_wxis(ctx: ParseContext) -> None:
    """Extract instrument-state parameters from the ``WXIS`` block.

    Sets ``ctx.params["LaserPower"]`` (float, %) when the ND-transmission
    value is present and parseable.  Missing or malformed blocks are silently
    ignored.
    """
    name = "WXIS"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        block_offset = ctx.blocks["BlockOffsets"][i]
        block_size = ctx.blocks["BlockSizes"][i]
        ctx.f.seek(block_offset + 16)
        payload = ctx.f.read(block_size - 16)

        props = decode_pset(payload)

        nd = _parse_nd_percent(props.get("ND Transmission %"))
        if nd is not None:
            ctx.params["LaserPower"] = nd
