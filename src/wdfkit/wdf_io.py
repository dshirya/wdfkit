# -*- coding: latin-1 -*-
"""Low-level binary reader for Renishaw WiRE ``.wdf`` files."""

from __future__ import annotations

import io
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from PIL import Image, ImageFile

from . import constants as const
from .utils import convert_time, hr_filesize, pad_if_unfinished

# Embedded WHTL JPEG chunks may be truncated in some recordings.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_wdf_file(filename, verbose, time_coord):
    """Parse a WiRE WDF file (invoked by
    :meth:`~wdfkit.reader.WDFReader.read`)."""

    def _read(f, dtype=np.uint32, count=1):
        """Reads bytes from binary file, with the most common values given as
        default.

        Returns the value itself if one value, or list if count > 1 Note
        that you should do ".decode()" on strings to avoid getting
        strings like "b'string'" For further information, refer to
        numpy.fromfile() function
        """

        if count == 1:
            return np.fromfile(f, dtype=dtype, count=count)[0]
        else:
            return np.fromfile(f, dtype=dtype, count=count)[0:count]

    def print_block_header(name, i, verbose=verbose):
        if verbose:
            print(
                f"\n{' Block : ' + name + ' ':=^80s}\n"
                f"size: {blocks['BlockSizes'][i]},"
                f"offset: {blocks['BlockOffsets'][i]}"
            )

    try:
        f = open(filename, "rb")
        if True:  # verbose:
            print(f'Reading the file: "{filename.split("/")[-1]}"\n')
    except IOError:
        raise IOError(f"File {filename} does not exist!")

    filesize = os.path.getsize(filename)
    header_dt = np.dtype(
        [
            ("block_name", "|S4"),
            ("block_id", np.int32),
            ("block_size", np.int64),
        ]
    )
    offset = 0
    _EPOCH = datetime(year=1601, month=1, day=1, tzinfo=timezone.utc)
    rowdim = None
    coldim = None
    params = dict()
    map_params = dict()
    blocks = dict()
    coord_dict = dict()
    blocks["BlockNames"] = []
    blocks["BlockSizes"] = []
    blocks["BlockOffsets"] = []
    # Reading all block names, offsets, and sizes (second pass reads payloads).
    while offset < filesize - 1:
        f.seek(offset)
        block_header = np.fromfile(f, dtype=header_dt, count=1)
        block_size = block_header["block_size"][0]
        # Zero-sized block avoids infinite scan on malformed files.
        if block_size == 0:
            break
        blocks["BlockOffsets"].append(offset)
        blocks["BlockNames"].append(block_header["block_name"][0].decode())
        blocks["BlockSizes"].append(block_size)
        offset += block_size
    if verbose:
        pass
        # print(blocks)
    # WDF1 (main header)
    name = "WDF1"
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        # TEST_WDF_FLAG = _read(f, np.uint64)
        params["WdfFlag"] = const.WDF_FLAGS[_read(f, np.uint64)]
        f.seek(60)
        params["PointsPerSpectrum"] = npoints = _read(f)
        # Number of spectra measured (nspectra):
        params["Capacity"] = nspectra = int(_read(f, np.uint64))
        # Number of spectra recorded (ncollected):
        params["Count"] = ncollected = int(_read(f, np.uint64))
        # Number of accumulations per spectrum:
        params["AccumulationCount"] = _read(f)
        # Number of elements in the y-list (>1 for image):
        params["YlistLength"] = _read(f)
        params["XlistLength"] = _read(f)  # number of elements in the x-list
        params["DataOriginCount"] = _read(f)  # number of data origin lists
        params["ApplicationName"] = _read(f, "|S24").decode()
        version = _read(f, np.uint16, count=4)
        params["ApplicationVersion"] = (
            ".".join([str(x) for x in version[0:-1]])
            + " build "
            + str(version[-1])
        )
        params["ScanType"] = const.SCAN_TYPES[_read(f)]
        params["MeasurementType"] = const.MEASUREMENT_TYPES[_read(f)]
        params["StartTime"] = convert_time(0.1 * _read(f, np.uint64))
        params["EndTime"] = convert_time(0.1 * _read(f, np.uint64))
        params["SpectralUnits"] = const.DATA_UNITS[_read(f)]
        laser_wavenumber = _read(f, "<f")
        params["LaserWaveLength"] = (
            np.round(10e6 / laser_wavenumber, 2)
            if laser_wavenumber
            else "Unspecified"
        )
        f.seek(240)
        params["Title"] = _read(f, "|S160").decode()
    # Printing params if verbose:
    if verbose:
        for key, val in params.items():
            print(f"{key:-<40s} : \t{val}")
    # WMAP (map metadata)
    name = "WMAP"
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        m_flag = _read(f)
        map_params["MapAreaType"] = const.MAP_TYPES[m_flag]  # _read(f)]
        _read(f)
        map_params["InitialCoordinates"] = np.round(_read(f, "<f", count=3), 2)
        map_params["StepSizes"] = np.round(_read(f, "<f", count=3), 2)
        map_params["NbSteps"] = n_x, n_y, n_z = _read(f, np.uint32, count=3)
        map_params["LineFocusSize"] = _read(f)
    if verbose:
        for key, val in map_params.items():
            print(f"{key:-<40s} : \t{val}")
    # DATA (spectra)
    name = "DATA"
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        data_points_count = ncollected * npoints
        # Empty container shape (capacity, n spectral points).
        spectra = np.zeros((nspectra, npoints))
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        # Fill recorded spectra; remainder of spectra stays zero-filled.
        spectra[:ncollected] = _read(f, "<f", count=data_points_count).reshape(
            ncollected, npoints
        )
        if verbose:
            print(f'{"The number of spectra":-<40s} : \t{ncollected}')
            print(
                f'{"The number of points in each spectra":-<40s} : \t'
                f"{npoints}"
            )
    # XLST (x-axis list / spectral axis)
    name = "XLST"
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        xldt = _read(f)
        params["XlistDataType"] = const.DATA_TYPES.get(xldt, f"{xldt}_unknown")
        xldu = _read(f)
        params["XlistDataUnits"] = const.DATA_UNITS.get(
            xldu, f"{xldu}_unknown"
        )
        x_values = _read(f, "<f", count=npoints)
        if params["XlistDataUnits"] == "RamanShift":
            shift_units = "1/cm"
        elif params["XlistDataUnits"] == "ElectronVolt":
            shift_units = "eV"
        else:
            shift_units = "unknown"
        coord_dict = {
            **coord_dict,
            "shifts": (
                "shifts",
                x_values,
                {"long_name": params["XlistDataUnits"], "units": shift_units},
            ),
        }
    if verbose:
        print(f"{'The shape of the x_values is':-<40s} : \t{x_values.shape} ")
        print(
            f"*These are the \"{params['XlistDataType']}"
            f"\" recordings in \"{params['XlistDataUnits']}\" units"
        )
    # YLST (y-axis list)
    name = "YLST"  # Not sure what's this about
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        yldt = _read(f)
        params["YlistDataType"] = const.DATA_TYPES.get(yldt, f"{yldt}_unknown")
        yldu = _read(f)
        params["YlistDataUnits"] = const.DATA_UNITS.get(
            yldu, f"{yldu}_unknown"
        )
        y_values_count = int((blocks["BlockSizes"][i] - 24) / 4)
        if y_values_count > 1:
            y_values = _read(f, "<f", count=y_values_count)
            print(y_values)
    # WHTL (embedded white-light image)
    name = "WHTL"  # This is where the image is
    img = None
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        img_bytes = _read(f, count=int((blocks["BlockSizes"][i] - 16) / 4))
        img = Image.open(io.BytesIO(img_bytes))

    # ORGN (origin / auxiliary coordinates per spectrum)
    name = "ORGN"
    origin_labels = []
    origin_set_dtypes = []
    origin_set_units = []
    # origin_values = np.empty(
    #     (params["DataOriginCount"], nspectra), dtype="<d"
    # )
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        # coord_names = {3:"X", 4:"Y", 5:"Z", 6:"R"}
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        nb_origin_sets = _read(f)
        # nb_origin_sets should be the same as params['DataOriginCount']
        assert nb_origin_sets == params["DataOriginCount"], "Not the same!?"
        for set_n in range(nb_origin_sets):
            data_type_flag = _read(f).astype(np.uint16)
            """Not sure why I had to convert the above to uint16, but if I just
            read it as uint32, I got rubbish sometimes."""
            data_type = const.DATA_TYPES.get(
                data_type_flag, f"{data_type_flag}_unknown"
            )
            origin_set_dtypes.append(data_type)
            coord_units_flag = _read(f)
            coord_units = const.DATA_UNITS.get(
                coord_units_flag, f"{coord_units_flag}_unknown"
            ).lower()
            origin_set_units.append(coord_units)
            label = _read(f, "<S16") + b"\0"
            ndx = label.index(b"\0")
            label = label[:ndx].decode("utf-8")
            origin_labels.append(label)
            # Reading Time coordinate:
            if data_type_flag == 11:  # special case for reading timestamps
                microseconds_from_epoch = 0.1 * _read(
                    f, np.uint64, count=nspectra
                )

                if time_coord == "seconds_elapsed":
                    st = params["StartTime"] - _EPOCH
                    recording_time = (
                        1e-6 * microseconds_from_epoch - st.total_seconds()
                    )
                else:
                    recording_time = convert_time(microseconds_from_epoch)

                if recording_time.ndim == 0:  # for single scan measurement
                    recording_time = np.expand_dims(recording_time, 0)

                if params["Count"] < params["Capacity"]:
                    recording_time = pad_if_unfinished(
                        recording_time,
                        count=params["Count"],
                        capacity=params["Capacity"],
                        extend=True,
                    )
                # return microseconds_from_epoch, recording_time
                coord_dict = {
                    **coord_dict,
                    label: (
                        "points",
                        recording_time,
                        {"units": coord_units, "long_name": data_type},
                    ),
                }
            # Other coordinates
            elif data_type_flag in [16, 17]:
                coord_values = np.array(
                    np.round(_read(f, "<Q", count=nspectra), 2)
                )
                # coord_dict = {**coord_dict,
                #                 label: ("points", coord_values,
                #                         {"units": coord_units,
                #                         "long_name": data_type}
                #                         )
                #             }
            elif data_type_flag not in [0, 11, 16, 17]:
                # [3, 4, 5, 6]:  # X, Y, Z, R
                # 0:?
                # 11:Time - a special case already dealt with above
                # 16:Checksum - never saw anything useful recorded here
                # 17: Flags — rarely used (Renishaw may omit).

                coord_values = np.array(
                    np.round(_read(f, "<d", count=nspectra), 2)
                )

                coord_dict = {
                    **coord_dict,
                    label: (
                        "points",
                        coord_values,
                        {"units": coord_units, "long_name": data_type},
                    ),
                }
    # Printing out some info
    if verbose:
        try:
            print(
                [f"{c} : {len(coord_dict[c][1])}" for c in coord_dict.keys()]
            )
        except Exception:
            pass
    if params["Count"] != params["Capacity"]:
        warnings.warn(
            f"Not all spectra was recorded. \nExpected {nspectra}, "
            f"but only {ncollected} spectra were recorded.\n"
            f"The {nspectra-ncollected} missing spectra will be filled with "
            "zeros."
            "\n\nPlease bear in mind that working with such incomplete"
            " recordings might (and probably will) lead to odd results"
            " further down the pipeline.",
            UserWarning,
            stacklevel=3,
        )
    # Preliminary DataArray (points × shifts); reorder below by scan type.
    da = xr.DataArray(spectra, dims=("points", "shifts"), coords=coord_dict)
    da = da.sortby(["shifts", "Time"])

    new_coord_dict = {"shifts": da.shifts}
    _ = coord_dict.pop("shifts")
    # For Maps (Slices included)
    if params["MeasurementType"].lower().startswith("map"):
        if "R" not in coord_dict.keys():
            coldim, rowdim = "X", "Y"
        else:  # Slice
            coldim, rowdim = "R", "Z"
            if map_params["StepSizes"][0] == 0:
                params["Angle R"] = 90
            else:
                params["Angle R"] = np.rad2deg(
                    np.arcsin(
                        -map_params["StepSizes"][1]
                        / map_params["StepSizes"][0]
                    )
                )
        # Create Column coordinate dimension:
        assert coldim in origin_labels, f"No {coldim} in coords?"
        da = da.sortby(coldim)
        x_coord_vals, x_attrs = np.unique(da[coldim].data), da[coldim].attrs
        ncols = len(x_coord_vals)
        new_coord_dict = {
            **new_coord_dict,
            **{coldim: (coldim, x_coord_vals, x_attrs)},
        }
        _ = coord_dict.pop(coldim)
        # Create Row coordinate dimension:
        assert rowdim in origin_labels, f"No {rowdim} in coords?"
        da = da.sortby(rowdim)
        y_coord_vals, y_attrs = np.unique(da[rowdim].data), da[rowdim].attrs
        nrows = len(y_coord_vals)
        new_coord_dict = {
            **new_coord_dict,
            **{rowdim: (rowdim, y_coord_vals, y_attrs)},
        }
        _ = coord_dict.pop(rowdim)
        # For the remaining da coordinates:
        for k in [
            k for k in da.coords.keys() if k not in ["shifts", rowdim, coldim]
        ]:
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
        # Create the new DataArray:
        da = xr.DataArray(
            da.values.reshape(nrows, ncols, npoints),
            dims=(rowdim, coldim, "shifts"),
            coords=new_coord_dict,
            attrs={
                **params,
                **map_params,
                "FileSize": hr_filesize(filesize),
            },
        )
    # For Series:
    else:  # Non-map (series / single / etc.)
        # Principal dimension/coordinate is Time:
        rowdim = "Time"
        coldim = None
        assert rowdim in origin_labels, f"No {rowdim} in coords?"
        y_coord_vals, y_attrs = np.unique(da[rowdim].data), da[rowdim].attrs
        nrows = len(y_coord_vals)
        ncols = 1
        new_coord_dict = {
            **new_coord_dict,
            **{rowdim: (rowdim, y_coord_vals, y_attrs)},
        }
        # For the remaining da coordinates:
        for k in [k for k in da.coords.keys() if k not in ["shifts", rowdim]]:
            new_coord_dict = {
                **new_coord_dict,
                **{k: (rowdim, da[k].data, da[k].attrs)},
            }
        # Create the new DataArray:
        da = xr.DataArray(
            da.values.reshape(nrows, npoints),
            dims=(rowdim, "shifts"),
            coords=new_coord_dict,
            attrs={**params, **map_params},
        )

    # Add additional attributes:
    da.attrs["ScanShape"] = (nrows, ncols)
    da.attrs["ColCoord"] = coldim
    da.attrs["RowCoord"] = rowdim
    da.attrs["Folder name"], da.attrs["Filename"] = os.path.split(filename)
    da.attrs["FileSize"] = hr_filesize(filesize)
    da.attrs["treatments"] = dict()

    # # Close the file and return the da and img
    f.close()
    return da, img
