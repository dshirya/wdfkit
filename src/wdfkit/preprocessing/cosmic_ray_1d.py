# -*- coding: utf-8 -*-
"""1D spectrum cosmic-ray (positive spike) removal."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import medfilt

from .cosmic_ray_mad import (
    noise_estimate_too_small,
    robust_mad_noise_with_floor,
)

SingleSpectrumMethod = Literal["median", "interpolate", "derivative"]


def _coerce_float_1d_spectrum(
    y: np.ndarray,
    method: str,
    kernel_size: int,
) -> np.ndarray:
    """Cast ``y`` to float 1D; validate inputs.

    Ensures ``method`` is allowed and ``kernel_size`` is odd and ≥ 3.
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {arr.shape}")
    allowed = ("median", "interpolate", "derivative")
    if method not in allowed:
        raise ValueError(f"method must be one of {allowed!r}, got {method!r}")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd and >= 3, got {kernel_size}"
        )
    return arr


def positive_spike_mask_vs_median_smooth(
    y: np.ndarray,
    median_smoothed_y: np.ndarray,
    threshold_multiplier: float,
) -> tuple[np.ndarray, float]:
    """Mask where positive residual exceeds ``threshold_multiplier * noise``.

    Residual is ``y - median_smoothed_y``; ``noise`` is scaled MAD of residual.
    """
    residual = y - median_smoothed_y
    if not np.any(residual):
        return np.zeros(y.shape, dtype=bool), 0.0
    amplitude_reference = max(
        float(np.nanmax(np.abs(y))),
        float(np.nanmax(np.abs(median_smoothed_y))),
    )
    noise = robust_mad_noise_with_floor(
        residual,
        amplitude_reference,
    )
    mask = residual > threshold_multiplier * noise
    return mask.astype(bool), noise


def positive_spike_mask_from_derivative_peaks(
    y: np.ndarray,
    threshold_multiplier: float,
) -> np.ndarray:
    """Interior ``i`` where ``y[i]`` is above both neighbors by
    ``threshold_multiplier * noise``.

    ``noise`` is scaled MAD of ``diff(y)``.
    """
    dy = np.diff(y)
    n = y.size
    mask = np.zeros(n, dtype=bool)
    if dy.size == 0:
        return mask
    amplitude_reference = max(
        float(np.nanmax(np.abs(y))),
        float(np.nanmax(np.abs(dy))),
    )
    noise = robust_mad_noise_with_floor(dy, amplitude_reference)
    max_abs_dy = float(np.nanmax(np.abs(dy))) + np.finfo(float).tiny
    if noise_estimate_too_small(noise, max_abs_dy):
        return mask
    threshold = threshold_multiplier * noise
    for i in range(1, n - 1):
        if (y[i] - y[i - 1] > threshold) and (y[i] - y[i + 1] > threshold):
            mask[i] = True
    return mask


def linear_interpolate_masked_channels_1d(
    y: np.ndarray,
    bad_channel_mask: np.ndarray,
) -> np.ndarray:
    """Fill masked channels by linear interpolation from good ones."""
    if not np.any(bad_channel_mask):
        return y.copy()
    good = ~bad_channel_mask
    if not np.any(good):
        return y.copy()
    n = y.size
    x = np.arange(n, dtype=float)
    out = y.copy()
    bad_idx = np.flatnonzero(bad_channel_mask)
    good_idx = np.flatnonzero(good)
    if good_idx.size == 1:
        out[bad_idx] = out[good_idx[0]]
        return out
    out[bad_idx] = np.interp(x[bad_idx], x[good_idx], y[good_idx])
    return out


def remove_cosmic_rays_1d(
    y: np.ndarray,
    method: SingleSpectrumMethod,
    *,
    kernel_size: int = 5,
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove sharp positive spikes from one 1D spectrum (PL-style).

    Operates on the raw counts / intensity array (only masked indices change).
    Cosmic rays: **positive** excursions vs a robust noise model.

    Parameters
    ----------
    y
        One spectral trace (any numeric dtype; cast to float).
    method
        ``\"median\"`` — ``scipy.signal.medfilt`` reference, MAD on residual,
        replace spikes with median-filtered values (default, conservative).
        ``\"interpolate\"`` — same detection, replace spikes via ``np.interp``.
        ``\"derivative\"`` — neighbour-difference test on ``diff(y)`` MAD;
        interior points only; interpolate repairs.
    kernel_size
        Odd length ``>= 3`` for ``medfilt`` (median / interpolate only).
    threshold
        Multiplier on MAD-derived noise (larger → fewer detections).

    Returns
    -------
    corrected_y
        Same shape as ``y``; unchanged if noise is degenerate or (for
        ``interpolate``) if every point would be flagged.
    cosmic_mask
        Boolean mask, same shape as ``y``; ``True`` were spikes **would** be
        corrected. If noise is too small, all ``False``. If all channels are
        flagged for ``interpolate``, returns original ``y`` and all-``False``.
    """
    y1 = _coerce_float_1d_spectrum(y, method, kernel_size)
    n = y1.size
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")

    if method in ("median", "interpolate"):
        median_filtered = medfilt(y1, kernel_size=kernel_size)
        spike_mask, _noise = positive_spike_mask_vs_median_smooth(
            y1,
            median_filtered,
            threshold,
        )
        if not np.any(spike_mask):
            return y1.copy(), spike_mask
        if method == "median":
            out = y1.copy()
            out[spike_mask] = median_filtered[spike_mask]
            return out, spike_mask
        if np.all(spike_mask):
            return y1.copy(), np.zeros(n, dtype=bool)
        out = linear_interpolate_masked_channels_1d(y1, spike_mask)
        return out, spike_mask

    spike_mask = positive_spike_mask_from_derivative_peaks(y1, threshold)
    if not np.any(spike_mask):
        return y1.copy(), spike_mask
    out = linear_interpolate_masked_channels_1d(y1, spike_mask)
    return out, spike_mask
