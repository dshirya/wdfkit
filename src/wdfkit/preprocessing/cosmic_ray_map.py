# -*- coding: utf-8 -*-
"""Spatial (3D map) cosmic-ray detection and replacement."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import grey_dilation
from skimage import filters, morphology

from .cosmic_ray_1d import linear_interpolate_masked_channels_1d
from .cosmic_ray_mad import robust_mad_noise_with_floor

_LEGACY_SENSITIVITY_REFERENCE = 0.01


def min_subtract_median_normalize_map_cube(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per spectrum: subtract min along λ, divide by median intensity."""
    spe = np.asarray(values, dtype=float)
    spe = spe - np.min(spe, axis=-1, keepdims=True)
    per_spectrum_median = np.median(spe, axis=-1, keepdims=True)
    per_spectrum_median = np.where(
        np.abs(per_spectrum_median) < np.finfo(float).eps,
        1.0,
        per_spectrum_median,
    )
    return spe / per_spectrum_median, per_spectrum_median


def unique_spatial_indices_from_nonzero(
    nonzero_axes: tuple[np.ndarray, ...],
    spatial_ndim: int,
) -> list[tuple[int, ...]]:
    """Unique ``(y, x, …)`` from ``np.nonzero``-style sparse index
    arrays."""
    if spatial_ndim == 0:
        return []
    per_ax = [nonzero_axes[i] for i in range(spatial_ndim)]
    return list({*zip(*per_ax)})


def _spatial_robust_noise_per_wavelength(
    residual_to_spatial_median: np.ndarray,
    preprocessed: np.ndarray,
) -> np.ndarray:
    """Scaled MAD over ``(y, x)`` at each spectral channel."""
    _, _, nlam = residual_to_spatial_median.shape
    noises = np.empty(nlam, dtype=float)
    for k in range(nlam):
        sl = residual_to_spatial_median[:, :, k].ravel()
        amp = (
            float(
                np.nanmax(np.abs(preprocessed[:, :, k])),
            )
            + np.finfo(float).tiny
        )
        noises[k] = robust_mad_noise_with_floor(sl, amp)
    return noises


def _per_wavelength_cutoff_relax_factors(
    noise_per_channel: np.ndarray,
    relax_floor: float,
) -> np.ndarray:
    """Factors in ``[relax_floor, 1]`` that **lower** the cutoff in
    noisy bands.

    Large per-channel noise → smaller factor → **higher** sensitivity
    there.
    """
    med = float(np.median(noise_per_channel))
    if (not np.isfinite(med)) or med <= 0:
        return np.ones_like(noise_per_channel, dtype=float)
    ratio = med / np.maximum(noise_per_channel, np.finfo(float).tiny)
    return np.clip(ratio, relax_floor, 1.0)


def _spectral_dilation_footprint_length(
    n_channels: int,
    spectral_width_fraction: float,
    spectral_dilate_cap: int,
) -> int:
    """1D window length along the spectral axis for binary dilation of
    the mask.

    Capped so dilation does not cover most of each spectrum (keeps
    repair local).
    """
    req = max(int(round(spectral_width_fraction * n_channels)), 1)
    cap = max(int(spectral_dilate_cap), 1)
    return min(req, cap, n_channels)


def _limit_mask_runs_along_spectral_axis(
    mask: np.ndarray,
    residual: np.ndarray,
    max_channels: int,
) -> np.ndarray:
    """For each ``(y, x)``, shorten any contiguous True run along λ to
    at most ``max_channels``, centered on the largest ``residual`` in
    that run."""
    if max_channels < 1:
        return mask
    ny, nx, _nlam = mask.shape
    out = mask.copy()
    for y in range(ny):
        for x in range(nx):
            m = out[y, x, :]
            r = residual[y, x, :]
            n = m.size
            idx = 0
            while idx < n:
                if not m[idx]:
                    idx += 1
                    continue
                start = idx
                while idx < n and m[idx]:
                    idx += 1
                run_len = idx - start
                if run_len <= max_channels:
                    continue
                seg = r[start:idx]
                peak = start + int(np.nanargmax(seg))
                half_lo = (max_channels - 1) // 2
                half_hi = max_channels - 1 - half_lo
                a = max(start, peak - half_lo)
                b = min(idx, peak + half_hi + 1)
                if b - a < max_channels:
                    a = max(start, b - max_channels)
                    b = min(idx, a + max_channels)
                m[start:idx] = False
                m[a:b] = True
    return out


def _dilate_mask_along_spectral_axis(
    mask: np.ndarray,
    footprint_length: int,
) -> np.ndarray:
    win = max(int(footprint_length), 1)
    if mask.ndim == 3:
        footprint = np.ones((1, 1, win), dtype=bool)
    else:
        footprint = np.ones(win, dtype=bool)
    return morphology.binary_dilation(mask.astype(bool), footprint=footprint)


def _strict_spatial_local_max_mask(
    field: np.ndarray,
) -> np.ndarray:
    """Where ``field`` exceeds all eight in-plane neighbours (strict).

    Same spectral slice; suppresses extended bright patches that are not
    spike-like spatially.
    """
    ny, nx, nlam = field.shape
    neighbour_footprint = np.ones((3, 3), dtype=bool)
    neighbour_footprint[1, 1] = False
    out = np.zeros_like(field, dtype=bool)
    for k in range(nlam):
        sl = field[:, :, k]
        neigh_max = grey_dilation(
            sl, footprint=neighbour_footprint, mode="nearest"
        )
        out[:, :, k] = (sl > neigh_max) & (sl > 0)
    return out


def interpolate_cosmic_ray_regions_spectrally(
    preprocessed: np.ndarray,
    spatial_median_reference: np.ndarray,
    repair_mask: np.ndarray,
) -> np.ndarray:
    """Inpaint ``repair_mask`` points by interp along λ.

    Reference curve is ``spatial_median_reference[y, x, :]``; other channels
    keep **original** ``preprocessed`` values.
    """
    ny, nx, nlam = preprocessed.shape
    flat_out = preprocessed.reshape(-1, nlam).copy()
    flat_med = spatial_median_reference.reshape(-1, nlam)
    flat_m = repair_mask.reshape(-1, nlam).astype(bool)
    for r in np.flatnonzero(np.any(flat_m, axis=1)):
        m = flat_m[r]
        ref = np.asarray(flat_med[r], dtype=float)
        filled = linear_interpolate_masked_channels_1d(ref, m)
        flat_out[r, :] = np.where(m, filled, flat_out[r, :])
    return flat_out.reshape(ny, nx, nlam)


def correct_cosmic_rays_on_map_cube(
    values: np.ndarray,
    *,
    sensitivity: float,
    spectral_width_fraction: float,
    disk_radius: int,
    map_mad_multiplier: float = 7.0,
    map_noisy_channel_relax_min: float = 0.82,
    map_spectral_dilate_cap: int = 5,
    map_max_spectral_repair_extent: int | None = 12,
    map_min_residual_over_cutoff: float = 1.05,
    map_require_spatial_local_max: bool = True,
    return_diagnostic_masks: bool = False,
) -> (
    tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, dict[str, Any], dict[str, Any]]
):
    """Spatial disk median on a per-spectrum normalized cube; robust
    positive residual test per wavelength.

    Per channel λ, the cutoff is
    ``map_mad_multiplier * (0.01/sensitivity) * relax_λ * noise_λ``,
    where ``noise_λ`` is scaled MAD of
    ``(preprocessed - spatial_median_reference)`` in the ``(y, x)`` plane, and
    ``relax_λ`` comes from ``map_noisy_channel_relax_min`` (noisy bands more
    sensitive).

    Spectral dilation length is ``min(width×N, map_spectral_dilate_cap)``.
    After dilation, each contiguous ``True`` segment along λ at fixed
    ``(y, x)`` is clipped to at most ``map_max_spectral_repair_extent``
    channels (``None`` disables) so repair stays localized.

    Detection uses ``residual > map_min_residual_over_cutoff * cutoff``.

    If ``map_require_spatial_local_max``, a voxel must be a strict spatial
    maximum in its λ slice among 8 neighbours (reduces false positives).

    Repair: dilate core hits along λ, then for each ``(y, x)`` interpolate
    masked samples along λ from ``spatial_median_reference[y, x, :]``;
    unmasked λ keep ``preprocessed``.

    If ``return_diagnostic_masks`` is True, returns a third dict (large numpy
    arrays — do not put them in ``DataArray.attrs``).
    """
    preprocessed, per_spectrum_median = min_subtract_median_normalize_map_cube(
        values,
    )
    disk = morphology.disk(disk_radius)[:, :, np.newaxis]
    spatial_median_reference = filters.median(preprocessed, footprint=disk)
    residual = preprocessed - spatial_median_reference
    noise_ch = _spatial_robust_noise_per_wavelength(residual, preprocessed)
    relax_ch = _per_wavelength_cutoff_relax_factors(
        noise_ch,
        map_noisy_channel_relax_min,
    )
    sens_scale = _LEGACY_SENSITIVITY_REFERENCE / float(sensitivity)
    cutoff = (
        map_mad_multiplier
        * sens_scale
        * relax_ch[np.newaxis, np.newaxis, :]
        * noise_ch[np.newaxis, np.newaxis, :]
    )
    rel = float(map_min_residual_over_cutoff)
    if rel <= 0 or not np.isfinite(rel):
        rel = 1.0
    core_mask = residual > (cutoff * rel)
    if map_require_spatial_local_max:
        core_mask &= _strict_spatial_local_max_mask(residual)
    bad = np.nonzero(core_mask)
    spatial_pairs = unique_spatial_indices_from_nonzero(bad, spatial_ndim=2)
    dil_len = _spectral_dilation_footprint_length(
        preprocessed.shape[-1],
        spectral_width_fraction,
        map_spectral_dilate_cap,
    )
    dilated = _dilate_mask_along_spectral_axis(core_mask, dil_len)
    if map_max_spectral_repair_extent is not None:
        dilated = _limit_mask_runs_along_spectral_axis(
            dilated,
            residual,
            int(map_max_spectral_repair_extent),
        )
    corrected_norm = interpolate_cosmic_ray_regions_spectrally(
        preprocessed,
        spatial_median_reference,
        dilated,
    )
    corrected_physical_units = corrected_norm * per_spectrum_median
    meta: dict[str, Any] = {
        "map_detection": "per_channel_spatial_mad",
        "map_mad_multiplier": map_mad_multiplier,
        "map_noisy_channel_relax_min": map_noisy_channel_relax_min,
        "map_spectral_dilate_cap": map_spectral_dilate_cap,
        "map_max_spectral_repair_extent": map_max_spectral_repair_extent,
        "map_min_residual_over_cutoff": map_min_residual_over_cutoff,
        "map_spectral_dilate_used": dil_len,
        "map_require_spatial_local_max": map_require_spatial_local_max,
    }
    if spatial_pairs:
        meta["CRs found"] = [list(p) for p in spatial_pairs]
    if not return_diagnostic_masks:
        return corrected_physical_units, meta

    diag: dict[str, Any] = {
        "core_mask": core_mask.copy(),
        "repair_mask": dilated.copy(),
        "residual": residual.copy(),
        "preprocessed": preprocessed.copy(),
        "spatial_median_reference": spatial_median_reference.copy(),
        "noise_per_channel": noise_ch.copy(),
        "relax_per_channel": relax_ch.copy(),
        "cutoff": cutoff.copy(),
        "per_spectrum_median": per_spectrum_median.copy(),
    }
    return corrected_physical_units, meta, diag


def correct_cosmic_rays_collection(
    values: np.ndarray,
    *,
    method: str = "median",
    threshold: float = 5.0,
    spectral_width_fraction: float = 0.02,
    spectral_dilate_cap: int = 5,
    max_repair_extent: int | None = 12,
    n_components: int = 3,
    return_diagnostics: bool = False,
) -> (
    tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, dict[str, Any], dict[str, Any]]
):
    """Global-median or PCA-reference cosmic-ray removal for collections.

    Works on any shape ``(..., n_channels)``: spatial dims are flattened
    internally; detection and repair run on the flat
    ``(n_spectra, n_channels)`` view, then the result is reshaped back.

    Unlike :func:`correct_cosmic_rays_on_map_cube`, this function requires no
    spatial neighbourhood — it is suitable for 2-D line/series/point arrays
    and for 3-D maps when ``map_method="pca"`` is selected.

    Parameters
    ----------
    values
        Input array of shape ``(..., n_channels)``.
    method
        ``"median"``: global median spectrum across all positions used as
        reference — same reference for every spectrum.
        ``"pca"``: per-spectrum PCA reconstruction used as reference (captures
        smooth spatial/temporal variation; cosmic rays appear in the residual).
    threshold
        Positive residual cutoff in units of per-channel MAD noise.
    n_components
        PCA only: number of principal components for reconstruction.
        Typically 3–5; increase for samples with many distinct spectral shapes.
    """
    orig_shape = np.asarray(values).shape
    n_channels = orig_shape[-1]
    flat = np.asarray(values, dtype=float).reshape(-1, n_channels)
    n_spectra = flat.shape[0]

    # Build reference array (n_spectra, n_channels)
    if method == "median":
        ref = np.tile(np.median(flat, axis=0), (n_spectra, 1))
    elif method == "pca":
        mean = flat.mean(axis=0)
        centered = flat - mean
        k = min(n_components, n_spectra - 1, n_channels - 1)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        ref = (U[:, :k] * S[np.newaxis, :k]) @ Vt[:k, :] + mean
    else:
        raise ValueError(f"method must be 'median' or 'pca', got {method!r}")

    residual = flat - ref  # (n_spectra, n_channels)

    # Per-channel MAD noise estimate across all spectra
    noise_ch = np.empty(n_channels)
    for ch in range(n_channels):
        col = residual[:, ch]
        amp = float(np.nanmax(np.abs(flat[:, ch]))) + np.finfo(float).tiny
        noise_ch[ch] = robust_mad_noise_with_floor(col, amp)

    cutoff = threshold * noise_ch  # (n_channels,)
    core_mask_flat = (
        residual > cutoff[np.newaxis, :]
    )  # (n_spectra, n_channels)

    # Reshape to pseudo-3D (n_spectra, 1, n_channels) to reuse existing
    # spectral dilation and extent-limiting helpers
    core_mask_3d = core_mask_flat.reshape(n_spectra, 1, n_channels)
    residual_3d = residual.reshape(n_spectra, 1, n_channels)

    dil_len = _spectral_dilation_footprint_length(
        n_channels, spectral_width_fraction, spectral_dilate_cap
    )
    dilated_3d = _dilate_mask_along_spectral_axis(core_mask_3d, dil_len)
    if max_repair_extent is not None:
        dilated_3d = _limit_mask_runs_along_spectral_axis(
            dilated_3d, residual_3d, int(max_repair_extent)
        )
    dilated_flat = dilated_3d.reshape(n_spectra, n_channels)

    # Repair masked channels by spectral interpolation from the reference
    corrected = flat.copy()
    for r in range(n_spectra):
        m = dilated_flat[r]
        if np.any(m):
            filled = linear_interpolate_masked_channels_1d(ref[r], m)
            corrected[r] = np.where(m, filled, flat[r])

    bad_rows = list(map(int, np.unique(np.argwhere(core_mask_flat)[:, 0])))

    meta: dict[str, Any] = {
        "collection_detection": method,
        "spike_threshold": threshold,
        "collection_dilate_used": dil_len,
    }
    if method == "pca":
        meta["map_n_components"] = k  # type: ignore[possibly-undefined]
    if bad_rows:
        meta["CRs found (spectrum indices)"] = bad_rows

    corrected_out = corrected.reshape(orig_shape)

    if not return_diagnostics:
        return corrected_out, meta

    spatial_shape = orig_shape[:-1]
    diag: dict[str, Any] = {
        "core_mask": core_mask_flat.reshape(spatial_shape + (n_channels,)),
        "repair_mask": dilated_flat.reshape(spatial_shape + (n_channels,)),
        "residual": residual.reshape(orig_shape),
        "reference": ref.reshape(orig_shape),
        "noise_per_channel": noise_ch,
        "cutoff": cutoff,
    }
    return corrected_out, meta, diag


__all__ = [
    "min_subtract_median_normalize_map_cube",
    "correct_cosmic_rays_on_map_cube",
    "correct_cosmic_rays_collection",
    "interpolate_cosmic_ray_regions_spectrally",
    "unique_spatial_indices_from_nonzero",
]
