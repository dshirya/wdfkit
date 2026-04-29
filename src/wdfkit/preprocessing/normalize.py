# -*- coding: utf-8 -*-
"""Per-spectrum normalization (dynamic spectral coordinate)."""

from __future__ import annotations

from warnings import warn

import numpy as np
import xarray as xr
from sklearn import preprocessing

from ._common import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)


def _trapz_y(
    y: np.ndarray,
    x: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Trapezoidal integration along ``axis``.

    Prefer ``numpy.trapezoid`` (NumPy 2+); fall back to ``numpy.trapz``.
    Avoid ``getattr(..., np.trapz)``: the default is evaluated eagerly and
    ``np.trapz`` is absent on some NumPy 2 builds.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def normalize(
    input_spectra: xr.DataArray | np.ndarray,
    method: str = "robust_scale",
    *,
    spectral_dim: str | None = None,
    **kwargs,
) -> xr.DataArray | np.ndarray:
    """Scale spectra along the spectral axis.

    For :class:`xarray.DataArray` input, the spectral axis defaults to the
    **last** dimension (e.g. ``nm``, ``raman_shift``, ``shifts``, …). Pass
    ``spectral_dim`` to select another dimension when spectra are not last.

    Parameters
    ----------
    input_spectra
        DataArray or 2D ndarray of shape ``(n_spectra, n_points)``.
    method
        One of ``\"l1\"``, ``\"l2\"``, ``\"max\"``, ``\"min_max\"``,
        ``\"wave_number\"``, ``\"robust_scale\"``, ``\"area\"``.
    spectral_dim
        Spectral dimension name when ``input_spectra`` is a DataArray.
    x_values
        Spectral abscissa for ndarray input (default ``arange(n_points)``).

    Returns
    -------
    Same type as ``input_spectra`` with updated ``attrs[\"treatments\"]`` for
    DataArray output.
    """
    if isinstance(input_spectra, xr.DataArray):
        sdim = resolve_spectral_dim(input_spectra, spectral_dim)
        da_w, orig_order = transpose_spectral_last(input_spectra, sdim)
        x_values = da_w[sdim].values
        spectra = da_w.values.reshape(-1, da_w.shape[-1])
        target_shape_w = da_w.shape
        as_xarray = True
    else:
        spectra = np.asarray(input_spectra)
        if spectra.ndim != 2:
            raise ValueError(
                "ndarray input must be 2D with shape (n_spectra, n_points)"
            )
        x_values = kwargs.get("x_values")
        if x_values is None:
            x_values = np.arange(spectra.shape[-1])
        else:
            x_values = np.asarray(x_values)
        target_shape_w = spectra.shape
        da_w = None
        orig_order = None
        as_xarray = False

    meta_keys = (
        "quantile",
        "centering",
        "wave_number",
        "x_values",
        "spectral_dim",
    )
    meta = {k: kwargs[k] for k in meta_keys if k in kwargs}

    if method in ("l1", "l2", "max"):
        normalized_spectra = preprocessing.normalize(
            spectra, axis=1, norm=method, copy=False
        )
    elif method == "min_max":
        normalized_spectra = preprocessing.minmax_scale(
            spectra, axis=1, copy=False
        )
    elif method == "area":
        denom = _trapz_y(spectra, x_values, axis=-1)[:, np.newaxis]
        denom = np.where(np.abs(denom) < np.finfo(float).eps, 1.0, denom)
        normalized_spectra = spectra / denom
    elif method == "wave_number":
        wave_number = kwargs.get("wave_number", float(np.min(x_values)))
        idx = int(np.nanargmin(np.abs(x_values - wave_number)))
        divider = spectra[:, [idx]]
        mean_divider = float(np.mean(divider))
        divider = np.where(divider == 0, mean_divider, divider)
        normalized_spectra = spectra / divider
    elif method == "robust_scale":
        quantile = kwargs.get("quantile", (5.0, 95.0))
        centering = kwargs.get("centering", False)
        normalized_spectra = preprocessing.robust_scale(
            spectra,
            axis=1,
            with_centering=centering,
            quantile_range=quantile,
        )
    else:
        warn(
            '"method" must be one of '
            '["l1", "l2", "max", "min_max", "wave_number", '
            '"robust_scale", "area"]'
        )
        normalized_spectra = spectra.copy()

    normalized_spectra = normalized_spectra - np.min(
        normalized_spectra, axis=-1, keepdims=True
    )

    treatment_payload = {"method": method, **meta}

    if as_xarray:
        assert da_w is not None and orig_order is not None
        packed_w = reshape_row_stack_to(normalized_spectra, target_shape_w)
        out_w = da_w.copy(data=packed_w)
        if tuple(out_w.dims) != orig_order:
            out = out_w.transpose(*orig_order)
        else:
            out = out_w
        return with_new_values(
            input_spectra,
            out.values,
            "normalization",
            treatment_payload,
        )

    return normalized_spectra.reshape(target_shape_w)
