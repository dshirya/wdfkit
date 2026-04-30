# -*- coding: utf-8 -*-
"""Parse ``ORGN`` per-spectrum origin coordinates (time, stage axes,
...)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ...internal import constants as const
from ...internal.utils import convert_time, pad_if_unfinished
from ..binary_io import read_from_file
from ..block_index import indices_named
from ..parse_context import ParseContext

_EPOCH = datetime(year=1601, month=1, day=1, tzinfo=timezone.utc)


def parse_orgn(ctx: ParseContext) -> None:
    name = "ORGN"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        nb_origin_sets = read_from_file(ctx.f)
        assert (
            nb_origin_sets == ctx.params["DataOriginCount"]
        ), "Not the same!?"
        for _set_n in range(nb_origin_sets):
            data_type_flag = read_from_file(ctx.f).astype(np.uint16)
            data_type = const.DATA_TYPES.get(
                data_type_flag, f"{data_type_flag}_unknown"
            )
            ctx.origin_set_dtypes.append(data_type)
            coord_units_flag = read_from_file(ctx.f)
            coord_units = const.DATA_UNITS.get(
                coord_units_flag, f"{coord_units_flag}_unknown"
            ).lower()
            ctx.origin_set_units.append(coord_units)
            label_raw = read_from_file(ctx.f, "<S16") + b"\0"
            ndx = label_raw.index(b"\0")
            label = label_raw[:ndx].decode("utf-8")
            ctx.origin_labels.append(label)

            if data_type_flag == 11:
                microseconds_from_epoch = 0.1 * read_from_file(
                    ctx.f, np.uint64, count=ctx.nspectra
                )
                if ctx.time_coord == "seconds_elapsed":
                    st = ctx.params["StartTime"] - _EPOCH
                    recording_time = (
                        1e-6 * microseconds_from_epoch - st.total_seconds()
                    )
                else:
                    recording_time = convert_time(microseconds_from_epoch)

                if recording_time.ndim == 0:
                    recording_time = np.expand_dims(recording_time, 0)

                if ctx.params["Count"] < ctx.params["Capacity"]:
                    recording_time = pad_if_unfinished(
                        recording_time,
                        count=ctx.params["Count"],
                        capacity=ctx.params["Capacity"],
                        extend=True,
                    )
                ctx.coord_dict = {
                    **ctx.coord_dict,
                    label: (
                        "points",
                        recording_time,
                        {
                            "units": coord_units,
                            "long_name": data_type,
                        },
                    ),
                }
            elif data_type_flag in (16, 17):
                np.array(
                    np.round(
                        read_from_file(ctx.f, "<Q", count=ctx.nspectra),
                        2,
                    )
                )
            elif data_type_flag not in (0, 11, 16, 17):
                coord_values = np.array(
                    np.round(
                        read_from_file(ctx.f, "<d", count=ctx.nspectra),
                        2,
                    )
                )
                ctx.coord_dict = {
                    **ctx.coord_dict,
                    label: (
                        "points",
                        coord_values,
                        {
                            "units": coord_units,
                            "long_name": data_type,
                        },
                    ),
                }


def print_coord_lengths_if_verbose(ctx: ParseContext) -> None:
    if not ctx.verbose:
        return
    try:
        print(
            [
                f"{c} : {len(ctx.coord_dict[c][1])}"
                for c in ctx.coord_dict.keys()
            ]
        )
    except Exception:
        pass
