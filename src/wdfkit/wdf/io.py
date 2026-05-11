# -*- coding: latin-1 -*-
"""Orchestrate WiRE ``.wdf`` parsing: block index → parsers → xarray
assembly."""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

from ..spectral import resolve_spectral_axis
from .block_index import scan_blocks
from .blocks.data_block import parse_data
from .blocks.origin import parse_orgn, print_coord_lengths_if_verbose
from .blocks.wdf1 import parse_wdf1
from .blocks.whtl import parse_whtl
from .blocks.wmap import parse_wmap
from .blocks.wxdm import parse_wxdm
from .blocks.wxis import parse_wxis
from .blocks.xlst import parse_xlst
from .blocks.ylst import parse_ylst
from .memory_check import check_memory
from .parse_context import ParseContext
from .parsed import OrgnEntry, ParsedWDF, WMAPInfo, XLSTInfo, YLSTInfo
from .types import get_spectral_dim_name


def _ctx_to_parsed(ctx: ParseContext) -> ParsedWDF:
    """Build an immutable :class:`~wdfkit._parsed.ParsedWDF` from a filled
    :class:`ParseContext`."""
    # XLST — re-derive dim name using new mapping (overrides xlst.py)
    xlst_units = ctx.params.get("XlistDataUnits", "Arbitrary")
    xlst_dtype = ctx.params.get("XlistDataType", "Spectral")
    new_sdim = get_spectral_dim_name(
        xlst_units, xlst_dtype, override=ctx.spectral_dim
    )
    # Retrieve values from coord_dict (keyed by old dim name set by xlst.py)
    old_sdim = ctx.spectral_dim_name or "spectral"
    xlst_entry = ctx.coord_dict.get(old_sdim) or ctx.coord_dict.get(new_sdim)
    xlst_values = xlst_entry[1] if xlst_entry else np.array([])
    spectral_spec = resolve_spectral_axis(xlst_units, ctx.spectral_dim)
    xlst = XLSTInfo(
        values=np.asarray(xlst_values, dtype="float32"),
        data_type=xlst_dtype,
        units=xlst_units,
        dim_name=new_sdim,
        coord_units=spectral_spec.units,
    )

    # YLST
    ylst: YLSTInfo | None = None
    if ctx.ylst_values is not None:
        ylst = YLSTInfo(
            values=np.asarray(ctx.ylst_values, dtype="float32"),
            data_type=ctx.ylst_data_type,
            units=ctx.ylst_units,
        )

    # ORGN entries — pull from coord_dict (stores ("points", values, attrs))
    orgn_entries: list[OrgnEntry] = []
    for idx, label in enumerate(ctx.origin_labels):
        if label in ctx.coord_dict:
            _dim, values, attrs = ctx.coord_dict[label]
            is_primary = (
                ctx.origin_is_primary[idx]
                if idx < len(ctx.origin_is_primary)
                else False
            )
            orgn_entries.append(
                OrgnEntry(
                    label=label,
                    data_type=(
                        ctx.origin_set_dtypes[idx]
                        if idx < len(ctx.origin_set_dtypes)
                        else ""
                    ),
                    units=(
                        ctx.origin_set_units[idx]
                        if idx < len(ctx.origin_set_units)
                        else ""
                    ),
                    values=np.asarray(values),
                    is_primary=is_primary,
                )
            )

    # WMAP
    wmap: WMAPInfo | None = None
    mp = ctx.map_params
    if "NbSteps" in mp:
        wmap = WMAPInfo(
            flag=ctx.wmap_flag_raw,
            start_xyz=np.asarray(
                mp.get("InitialCoordinates", [0, 0, 0]), dtype="float32"
            ),
            step_xyz=np.asarray(
                mp.get("StepSizes", [0, 0, 0]), dtype="float32"
            ),
            nsteps=np.asarray(mp["NbSteps"], dtype="uint32"),
            linefocus_size=int(mp.get("LineFocusSize", 0)),
        )

    return ParsedWDF(
        filename=str(ctx.filename),
        filesize=ctx.filesize,
        nspectra=ctx.nspectra,
        ncollected=ctx.ncollected,
        npoints=ctx.npoints,
        ylist_length=int(ctx.params.get("YlistLength", 1)),
        measurement_type=ctx.measurement_type_raw,
        scan_type=ctx.scan_type_raw,
        app_name=ctx.params.get("ApplicationName", ""),
        app_version=ctx.params.get("ApplicationVersion", ""),
        naccum=int(ctx.params.get("AccumulationCount", 1)),
        laser_wavelength=ctx.params.get("LaserWaveLength", "Unspecified"),
        title=ctx.params.get("Title", ""),
        spectral_units_str=ctx.params.get("SpectralUnits", ""),
        params=dict(ctx.params),
        map_params=dict(ctx.map_params),
        data=ctx.spectra if ctx.spectra is not None else None,
        xlst=xlst,
        ylst=ylst,
        orgn=orgn_entries,
        wmap=wmap,
        img=ctx.img,
        exposure_time=ctx.params.get("ExposureTime"),
        laser_power=ctx.params.get("LaserPower"),
        stage_xyz=ctx.stage_xyz,
    )


def _run_parsers(
    ctx: ParseContext,
    *,
    load_data: bool = True,
) -> None:
    """Run block parsers against *ctx*.

    When *load_data* is ``False`` the DATA, WHTL, ORGN, WXDM, and WXIS
    blocks are skipped (used by :func:`parse_wdf_header` for fast
    classification).
    """
    ctx.blocks = scan_blocks(ctx.f, ctx.filesize)
    parse_wdf1(ctx)
    parse_wmap(ctx)
    if load_data:
        check_memory(ctx)
        parse_data(ctx)
        parse_xlst(ctx)
        parse_ylst(ctx)
        parse_whtl(ctx)
        parse_orgn(ctx)
        parse_wxdm(ctx)
        parse_wxis(ctx)
    else:
        parse_xlst(ctx)


def parse_wdf_to_parsed(
    filename: str | os.PathLike[str],
    verbose: bool = False,
    time_coord: str | None = None,
    spectral_dim: str | None = None,
    chunks: bool | int = False,
) -> ParsedWDF:
    """Parse a WDF file and return a :class:`~wdfkit._parsed.ParsedWDF`.

    This is the low-level entry point used by handlers and
    :func:`classify_wdf`.
    """
    try:
        file_obj = open(filename, "rb")
        if verbose:
            print(f'Reading the file: "{str(filename).split("/")[-1]}"\n')
    except IOError as e:
        raise IOError(f"File {filename} does not exist!") from e

    filesize = os.path.getsize(filename)
    ctx = ParseContext(
        filename=filename,
        verbose=verbose,
        time_coord=time_coord,
        spectral_dim=spectral_dim,
        filesize=filesize,
        f=file_obj,
        chunks=chunks,
    )
    try:
        _run_parsers(ctx, load_data=True)
        if verbose:
            print_coord_lengths_if_verbose(ctx)
        return _ctx_to_parsed(ctx)
    finally:
        file_obj.close()


def parse_wdf_header(
    filename: str | os.PathLike[str],
    verbose: bool = False,
    spectral_dim: str | None = None,
) -> ParsedWDF:
    """Parse only the header blocks of a WDF file (no DATA).

    Used by :func:`wdfkit.classify` for fast scan-type triage.
    """
    try:
        file_obj = open(filename, "rb")
    except IOError as e:
        raise IOError(f"File {filename} does not exist!") from e

    filesize = os.path.getsize(filename)
    ctx = ParseContext(
        filename=filename,
        verbose=verbose,
        time_coord=None,
        spectral_dim=spectral_dim,
        filesize=filesize,
        f=file_obj,
        chunks=False,
    )
    try:
        _run_parsers(ctx, load_data=False)
        return _ctx_to_parsed(ctx)
    finally:
        file_obj.close()


def read_wdf_file(
    filename: str | os.PathLike[str],
    verbose: bool,
    time_coord: str | None,
    spectral_dim: str | None = None,
    chunks: bool | int = False,
) -> tuple[xr.DataArray, object]:
    """Parse a WiRE WDF file (invoked by
    :class:`~wdfkit.reader.WDFReader`).

    Parameters
    ----------
    spectral_dim
        Passed to :func:`~wdfkit.spectral.resolve_spectral_axis` during
        ``XLST`` handling.
    chunks
        ``False`` for eager NumPy reading (default); ``True`` or an ``int``
        MB value for lazy Dask-backed reading.
    """
    from .dispatch import dispatch

    parsed = parse_wdf_to_parsed(
        filename,
        verbose=verbose,
        time_coord=time_coord,
        spectral_dim=spectral_dim,
        chunks=chunks,
    )
    return dispatch(parsed), parsed.img
