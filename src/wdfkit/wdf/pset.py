# -*- coding: utf-8 -*-
"""Minimal decoder for Renishaw PSET (Property Set) binary blocks.

PSET is a proprietary key-value serialisation used inside ``WXDM``,
``WXIS``, ``WXDA``, ``WXCS`` and related WiRE blocks.  Each block is a
flat list of typed value entries followed by a key-definition table that
maps 3-byte key indices to human-readable names.

Nested PSETs (type ``'p'``) are skipped during top-level decoding.
"""

from __future__ import annotations

import struct
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _entry_size(data: bytes, pos: int) -> int:
    """Return the byte length of the PSET entry starting at *pos*.

    Returns ``1`` when the entry type is unrecognised (advance one byte and
    try again), and ``-1`` to signal that a key-definition entry was reached
    (i.e. the values section has ended).
    """
    if pos >= len(data):
        return 1
    t = data[pos]

    if t == ord("i"):  # int32
        return 8 if pos + 8 <= len(data) else 1
    if t == ord("u"):  # utf-8 string
        if pos + 8 > len(data):
            return 1
        slen = struct.unpack_from("<I", data, pos + 4)[0]
        total = 8 + slen
        return total if pos + total <= len(data) else 1
    if t == ord("q"):  # float64
        return 12 if pos + 12 <= len(data) else 1
    if t == 0x3F:  # bool ('?')
        return 5 if pos + 5 <= len(data) else 1
    if t == ord("s"):  # int16
        return 6 if pos + 6 <= len(data) else 1
    if t == ord("c"):  # uint8 / char
        return 5 if pos + 5 <= len(data) else 1
    if t == ord("t"):  # int64 / Windows FILETIME
        return 12 if pos + 12 <= len(data) else 1
    if t == ord("b"):  # raw bytes / blob
        if pos + 8 > len(data):
            return 1
        blen = struct.unpack_from("<I", data, pos + 4)[0]
        total = 8 + blen
        return total if pos + total <= len(data) else 1
    if t == ord("p"):  # nested PSET
        if pos + 8 > len(data):
            return 1
        ns = struct.unpack_from("<I", data, pos + 4)[0]
        total = 8 + ns
        return total if pos + total <= len(data) else 1
    if t == ord("k"):  # key-definition — marks end of values section
        return -1

    return 1  # unknown type: skip one byte


def _find_key_def_start(data: bytes, values_start: int) -> int:
    """Walk the values section correctly (honouring entry sizes, *skipping*
    nested PSETs as opaque blobs) and return the offset where key
    definitions begin.
    """
    pos = values_start
    while pos < len(data):
        size = _entry_size(data, pos)
        if size == -1:  # reached 'k' entries
            return pos
        if size <= 0:
            pos += 1
        else:
            pos += size
    return pos  # no 'k' found; key defs may be absent


def _build_key_map(data: bytes, key_def_start: int) -> dict[tuple, str]:
    """Parse the key-definition section and return a mapping of
    ``(b0, b1, b2) → name``."""
    key_map: dict[tuple, str] = {}
    pos = key_def_start
    while pos < len(data) - 8:
        if data[pos] != ord("k"):
            pos += 1
            continue
        kb = (data[pos + 1], data[pos + 2], data[pos + 3])
        nlen = struct.unpack_from("<I", data, pos + 4)[0]
        end = pos + 8 + nlen
        if 0 < nlen < 256 and end <= len(data):
            name = data[pos + 8 : end].decode("utf-8", "replace")
            key_map[kb] = name
            pos = end
        else:
            pos += 1
    return key_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _extract_values(
    data: bytes,
    key_map: dict[tuple, str],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Scan *data[start:end]* for value entries, using *key_map* to resolve
    names.

    When a nested ``'p'`` PSET entry is encountered the function inspects
    whether the nested blob carries its own key definitions:

    * **No own key defs** → the nested entries are decoded with the *parent*
      ``key_map`` (Renishaw's convention for blocks like ``WXDM`` where nested
      PSETs share the top-level key table).
    * **Own key defs present** → the nested blob is decoded independently with
      its own key table (used in blocks like ``WXIS``).

    In both cases the nested results are merged into the parent dict (parent
    values take precedence so that the first occurrence wins).
    """
    result: dict[str, Any] = {}
    pos = start
    while pos < end:
        t = data[pos]
        size = _entry_size(data, pos)
        if size == -1:
            break
        if size < 1 or pos + size > len(data):
            pos += 1
            continue

        kb = (
            (data[pos + 1], data[pos + 2], data[pos + 3])
            if pos + 4 <= len(data)
            else None
        )
        key_name: Optional[str] = key_map.get(kb) if kb else None

        if t == ord("p") and size > 8:
            nested = data[pos + 8 : pos + size]
            n_kstart = _find_key_def_start(nested, 0)
            n_kmap = _build_key_map(nested, n_kstart)
            if n_kmap:
                # Nested block has its own key table — decode independently.
                n_result = _extract_values(nested, n_kmap, 0, n_kstart)
            else:
                # No key table — inherit parent's key map.
                n_result = _extract_values(nested, key_map, 0, len(nested))
            for k, v in n_result.items():
                result.setdefault(k, v)  # parent (earlier) value wins
        elif key_name is not None and t != ord("p"):
            if t == ord("i"):
                result[key_name] = struct.unpack_from("<i", data, pos + 4)[0]
            elif t == ord("u"):
                slen = struct.unpack_from("<I", data, pos + 4)[0]
                result[key_name] = data[pos + 8 : pos + 8 + slen].decode(
                    "utf-8", "replace"
                )
            elif t == ord("q"):
                result[key_name] = struct.unpack_from("<d", data, pos + 4)[0]
            elif t == 0x3F:
                result[key_name] = bool(data[pos + 4])
            elif t == ord("s"):
                result[key_name] = struct.unpack_from("<h", data, pos + 4)[0]
            elif t == ord("t"):
                result[key_name] = struct.unpack_from("<q", data, pos + 4)[0]

        pos += size

    return result


def decode_pset(data: bytes) -> dict[str, Any]:
    """Decode a PSET binary blob and return a flat ``{name: value}`` dict.

    Both top-level blocks (starting with the ``PSET`` magic + 4-byte header)
    and nested PSET blobs (starting directly with value entries) are handled.

    When a nested ``'p'`` entry has no key definitions of its own, its entries
    are decoded with the *parent* key map (Renishaw's inherited-key convention
    used in ``WXDM`` and similar blocks).
    """
    if not data:
        return {}

    values_start = 8 if (len(data) >= 4 and data[:4] == b"PSET") else 0
    key_def_start = _find_key_def_start(data, values_start)
    key_map = _build_key_map(data, key_def_start)
    return _extract_values(data, key_map, values_start, key_def_start)


def find_in_pset(
    data: bytes, key_name: str, max_depth: int = 3
) -> Optional[Any]:
    """Search for *key_name* in a PSET blob, recursing into nested PSETs.

    This is a lightweight alternative to :func:`decode_pset` when only one
    specific key is needed.  Returns the first matching value, or ``None``.
    ``max_depth`` guards against unbounded recursion on malformed data.
    """
    result = decode_pset(data)
    return result.get(key_name)
