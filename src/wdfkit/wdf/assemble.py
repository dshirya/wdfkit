# -*- coding: utf-8 -*-
"""Build :class:`xarray.DataArray` and final attrs from parsed
``ParseContext``."""

from __future__ import annotations

import os
import warnings

import numpy as np
import xarray as xr

from ..internal.utils import hr_filesize
from .parse_context import ParseContext


def warn_if_incomplete_recording(ctx: ParseContext) -> None:
    if ctx.params["Count"] == ctx.params["Capacity"]:
        return
    warnings.warn(
        f"Not all spectra was recorded. \nExpected {ctx.nspectra}, "
        f"but only {ctx.ncollected} spectra were recorded.\n"
        f"The {ctx.nspectra - ctx.ncollected} missing spectra will be filled "
        "with zeros."
        "\n\nPlease bear in mind that working with such incomplete"
        " recordings might (and probably will) lead to odd results"
        " further down the pipeline.",
        UserWarning,
        stacklevel=3,
    )


def assemble_data_array(ctx: ParseContext) -> tuple[xr.DataArray, object]:
    """Turn filled ``ParseContext`` into ``(data_array,
    white_light_image)``."""
    if ctx.spectral_dim_name is None:
        raise ValueError("WDF file has no XLST block; cannot build spectra.")

    warn_if_incomplete_recording(ctx)

    sdim = ctx.spectral_dim_name
    da = xr.DataArray(
        ctx.spectra,
        dims=("points", sdim),
        coords=ctx.coord_dict,
    )
    da = da.sortby([sdim, "Time"])

    new_coord_dict = {sdim: da[sdim]}
    ctx.coord_dict.pop(sdim)

    rowdim = None
    coldim = None

    if ctx.params["MeasurementType"].lower().startswith("map"):
        da, new_coord_dict, rowdim, coldim, nrows, ncols = _assemble_map_scan(
            ctx, da, new_coord_dict
        )
    else:
        da, new_coord_dict, rowdim, coldim, nrows, ncols = (
            _assemble_series_scan(ctx, da, new_coord_dict)
        )

    da.attrs["ScanShape"] = (nrows, ncols)
    da.attrs["ColCoord"] = coldim
    da.attrs["RowCoord"] = rowdim
    da.attrs["Folder name"], da.attrs["Filename"] = os.path.split(ctx.filename)
    da.attrs["FileSize"] = hr_filesize(ctx.filesize)
    da.attrs["treatments"] = dict()

    return da, ctx.img


def _assemble_map_scan(ctx, da, new_coord_dict):
    coord_dict = ctx.coord_dict
    map_params = ctx.map_params
    params = ctx.params
    sdim = ctx.spectral_dim_name
    npoints = ctx.npoints

    if "R" not in coord_dict.keys():
        coldim, rowdim = "X", "Y"
    else:
        coldim, rowdim = "R", "Z"
        if map_params["StepSizes"][0] == 0:
            params["Angle R"] = 90
        else:
            params["Angle R"] = np.rad2deg(
                np.arcsin(
                    -map_params["StepSizes"][1] / map_params["StepSizes"][0]
                )
            )

    assert coldim in ctx.origin_labels, f"No {coldim} in coords?"
    da = da.sortby(coldim)
    x_coord_vals, x_attrs = np.unique(da[coldim].data), da[coldim].attrs
    ncols = len(x_coord_vals)
    new_coord_dict = {
        **new_coord_dict,
        **{coldim: (coldim, x_coord_vals, x_attrs)},
    }
    coord_dict.pop(coldim)

    assert rowdim in ctx.origin_labels, f"No {rowdim} in coords?"
    da = da.sortby(rowdim)
    y_coord_vals, y_attrs = np.unique(da[rowdim].data), da[rowdim].attrs
    nrows = len(y_coord_vals)
    new_coord_dict = {
        **new_coord_dict,
        **{rowdim: (rowdim, y_coord_vals, y_attrs)},
    }
    coord_dict.pop(rowdim)

    for k in [k for k in da.coords.keys() if k not in [sdim, rowdim, coldim]]:
        new_coord_dict = {
            **new_coord_dict,
            **{
                k: (
                    (rowdim, coldim),
                    da[k].data.reshape(nrows, ncols),
                    da[k].attrs,
                )
            },
        }

    da = xr.DataArray(
        da.values.reshape(nrows, ncols, npoints),
        dims=(rowdim, coldim, sdim),
        coords=new_coord_dict,
        attrs={
            **params,
            **map_params,
            "FileSize": hr_filesize(ctx.filesize),
        },
    )
    return da, new_coord_dict, rowdim, coldim, nrows, ncols


def _assemble_series_scan(ctx, da, new_coord_dict):
    params = ctx.params
    map_params = ctx.map_params
    sdim = ctx.spectral_dim_name
    npoints = ctx.npoints

    rowdim = "Time"
    coldim = None
    assert rowdim in ctx.origin_labels, f"No {rowdim} in coords?"
    y_coord_vals, y_attrs = np.unique(da[rowdim].data), da[rowdim].attrs
    nrows = len(y_coord_vals)
    ncols = 1
    new_coord_dict = {
        **new_coord_dict,
        **{rowdim: (rowdim, y_coord_vals, y_attrs)},
    }

    for k in [k for k in da.coords.keys() if k not in [sdim, rowdim]]:
        new_coord_dict = {
            **new_coord_dict,
            **{k: (rowdim, da[k].data, da[k].attrs)},
        }

    da = xr.DataArray(
        da.values.reshape(nrows, npoints),
        dims=(rowdim, sdim),
        coords=new_coord_dict,
        attrs={**params, **map_params},
    )
    return da, new_coord_dict, rowdim, coldim, nrows, ncols
