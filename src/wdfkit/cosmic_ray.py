# -*- coding: utf-8 -*-
"""
High-level cosmic-ray removal: :class:`CosmicRayRemover` for maps and singles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .preprocessing._common import resolve_spectral_dim, with_new_values
from .preprocessing.cosmic_ray_1d import (
    SingleSpectrumMethod,
    remove_cosmic_rays_1d,
)
from .preprocessing.cosmic_ray_map import correct_cosmic_rays_on_map_cube
from .preprocessing.spectral_harmonic_removal import harmonic_correct_dataarray


@dataclass
class CosmicRayRemover:
    """Cosmic-ray removal: spatial median for maps; robust 1D for singles.

    Optionally removes broad Nd:YAG laser harmonics on ~355 nm excitation
    before narrow spike removal (:meth:`harmonic_check`, :meth:`remove`).

    **Maps** (3D): spatial disk median on a min/median-normalized cube;
    **per-λ scaled MAD** cutoffs and noisy-band ``relax_λ``; repair by
    **spectral interpolation** (not copying the full median surface).

    **Single spectrum** (2D ``(1, n_spectral)`` or 1×1 map): see
    :func:`remove_cosmic_rays_1d` — ``scipy.signal.medfilt``, MAD-based
    positive spike detection, optional interpolation repair.

    Parameters
    ----------
    sensitivity
        Map path: scales aggressiveness. The cutoff includes
        ``(0.01 / sensitivity)`` times the per-channel MAD level (``0.01`` is
        the legacy default reference). Larger ``sensitivity`` → **more** hits.
    width
        Map path: spectral dilation of the CR mask (fraction of length).
    disk_radius
        Map path: spatial disk radius for the reference median filter.
    map_mad_multiplier
        Map path: multiplier on ``noise_λ × relax_λ`` (like 1D ``threshold``;
        larger → fewer false positives).
    map_noisy_channel_relax_min
        Map path: floor on ``relax_λ`` in noisy channels. **Higher** → weaker
        boost in noisy bands → fewer false positives.
    map_spectral_dilate_cap
        Map path: max footprint length (in spectral channels) when dilating
        hits along λ. Caps ``width × N`` so repair stays **narrow** and interp
        stays accurate.
    map_require_spatial_local_max
        Map path: if True (default), keep only voxels that are strict maxima in
        their ``(y, x)`` slice at fixed λ (8-neighbour), reducing extended
        bright features being treated as CRs.
    map_max_spectral_repair_extent
        Map path: after spectral dilation, each contiguous repair segment along
        λ is clipped to at most this many channels (centered on max residual in
        the segment). ``None`` disables (not recommended for noisy maps).
    map_min_residual_over_cutoff
        Map path: require ``residual > cutoff *`` this factor (> 1 stricter,
        fewer false positives). Use ``1.0`` for the legacy strict inequality.
    single_spectrum_method
        ``\"median\"``, ``\"interpolate\"``, or ``\"derivative\"``.
    kernel_size
        Odd, ``>= 3``. Passed to ``medfilt`` for single-spectrum median-based
        methods.
    threshold
        Single-spectrum only: spike cutoff is ``threshold * MAD_noise``.
    spectral_dim
        Name of the spectral axis (default: last dimension). Used for harmonic
        cleanup and when the spectral dimension is not last.
    """

    sensitivity: float = 0.01
    width: float = 0.02
    disk_radius: int = 3
    single_spectrum_method: SingleSpectrumMethod = "median"
    kernel_size: int = 5
    threshold: float = 5.0
    spectral_dim: str | None = None
    map_mad_multiplier: float = 7.0
    map_noisy_channel_relax_min: float = 0.82
    map_spectral_dilate_cap: int = 5
    map_max_spectral_repair_extent: int | None = 12
    map_min_residual_over_cutoff: float = 1.05
    map_require_spatial_local_max: bool = True

    def __post_init__(self) -> None:
        allowed: tuple[str, ...] = ("median", "interpolate", "derivative")
        if self.single_spectrum_method not in allowed:
            raise ValueError(
                f"single_spectrum_method must be one of {allowed!r}, got "
                f"{self.single_spectrum_method!r}"
            )
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd and >= 3, got {self.kernel_size}"
            )
        if self.threshold <= 0 or not np.isfinite(self.threshold):
            raise ValueError("threshold must be positive and finite")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        if not 0 < self.width <= 1:
            raise ValueError("width must be in (0, 1]")
        if self.map_mad_multiplier <= 0 or not np.isfinite(
            self.map_mad_multiplier
        ):
            raise ValueError("map_mad_multiplier must be positive and finite")
        if not 0 < self.map_noisy_channel_relax_min <= 1:
            raise ValueError("map_noisy_channel_relax_min must be in (0, 1]")
        if self.map_spectral_dilate_cap < 1:
            raise ValueError("map_spectral_dilate_cap must be >= 1")
        if self.map_max_spectral_repair_extent is not None and (
            self.map_max_spectral_repair_extent < 1
        ):
            raise ValueError(
                "map_max_spectral_repair_extent must be >= 1 or None"
            )
        if self.map_min_residual_over_cutoff <= 0 or not np.isfinite(
            self.map_min_residual_over_cutoff
        ):
            raise ValueError(
                "map_min_residual_over_cutoff must be positive and finite"
            )

    def harmonic_check(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Notch broad harmonics when ``LaserWaveLength`` is ~355 nm (Nd:YAG).

        If ``spectrum.attrs['LaserWaveLength']`` is outside 354–356 nm, returns
        ``spectrum`` unchanged.

        Searches 1064 / 532 / 355 / 266 nm (±2.5 nm); replaces ~1 nm around
        each found peak with linear interpolation. Prints one line per removal.
        """
        return harmonic_correct_dataarray(
            spectrum,
            spectral_dim=self.spectral_dim,
        )

    def remove_cosmic_rays(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Spike removal only (no harmonic notch)."""
        if spectrum.ndim == 3:
            ny, nx = spectrum.shape[0], spectrum.shape[1]
            if ny * nx <= 1:
                return self._single_spectrum_output(
                    spectrum,
                    spectrum.values.reshape(-1),
                )
            corrected, meta = correct_cosmic_rays_on_map_cube(
                spectrum.values,
                sensitivity=self.sensitivity,
                spectral_width_fraction=self.width,
                disk_radius=self.disk_radius,
                map_mad_multiplier=self.map_mad_multiplier,
                map_noisy_channel_relax_min=self.map_noisy_channel_relax_min,
                map_spectral_dilate_cap=self.map_spectral_dilate_cap,
                map_max_spectral_repair_extent=(
                    self.map_max_spectral_repair_extent
                ),
                map_min_residual_over_cutoff=(
                    self.map_min_residual_over_cutoff
                ),
                map_require_spatial_local_max=(
                    self.map_require_spatial_local_max
                ),
            )
            return with_new_values(
                spectrum,
                corrected,
                "Cosmic Ray Correction",
                meta,
            )
        if spectrum.ndim == 2 and spectrum.shape[0] == 1:
            return self._single_spectrum_output(spectrum, spectrum.values[0])
        raise ValueError(
            "CosmicRayRemover supports 3D maps or a single spectrum as 2D "
            "(1, n_spectral); got "
            f"ndim={spectrum.ndim}, shape={spectrum.shape}"
        )

    def remove_cosmic_rays_with_diagnostics(
        self,
        spectrum: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Like :meth:`remove_cosmic_rays`, but returns a **diagnostics** dict
        for visualization / QC (not written to ``DataArray.attrs``).

        For 3D maps, ``diagnostics`` includes boolean ``core_mask``,
        ``repair_mask``, and float arrays ``residual``, ``preprocessed``,
        ``spatial_median_reference``, ``cutoff``, ``per_spectrum_median``, etc.
        Use matplotlib to overlay masks or compare spectra at selected
        ``(y, x)``.

        For 2D single-spectrum input, diagnostics contain ``cosmic_mask`` and
        ``corrected_1d`` (the 1D corrected intensity).
        """
        if spectrum.ndim == 3:
            ny, nx = spectrum.shape[0], spectrum.shape[1]
            if ny * nx <= 1:
                resolve_spectral_dim(spectrum, self.spectral_dim)
                sp = spectrum.values.reshape(-1)
                corrected, mask = remove_cosmic_rays_1d(
                    sp,
                    self.single_spectrum_method,
                    kernel_size=self.kernel_size,
                    threshold=self.threshold,
                )
                meta_1d: dict[str, Any] = {
                    "single_spectrum_method": self.single_spectrum_method,
                    "kernel_size": self.kernel_size,
                    "threshold": self.threshold,
                }
                if np.any(mask):
                    meta_1d["CRs found (spectral indices)"] = list(
                        np.flatnonzero(mask)
                    )
                out = with_new_values(
                    spectrum,
                    corrected.reshape(spectrum.shape),
                    "Cosmic Ray Correction",
                    meta_1d,
                )
                return out, {"cosmic_mask": mask}
            corrected, meta, diag = correct_cosmic_rays_on_map_cube(
                spectrum.values,
                sensitivity=self.sensitivity,
                spectral_width_fraction=self.width,
                disk_radius=self.disk_radius,
                map_mad_multiplier=self.map_mad_multiplier,
                map_noisy_channel_relax_min=self.map_noisy_channel_relax_min,
                map_spectral_dilate_cap=self.map_spectral_dilate_cap,
                map_max_spectral_repair_extent=(
                    self.map_max_spectral_repair_extent
                ),
                map_min_residual_over_cutoff=(
                    self.map_min_residual_over_cutoff
                ),
                map_require_spatial_local_max=(
                    self.map_require_spatial_local_max
                ),
                return_diagnostic_masks=True,
            )
            out = with_new_values(
                spectrum,
                corrected,
                "Cosmic Ray Correction",
                meta,
            )
            return out, diag
        if spectrum.ndim == 2 and spectrum.shape[0] == 1:
            resolve_spectral_dim(spectrum, self.spectral_dim)
            sp = spectrum.values[0]
            corrected, mask = remove_cosmic_rays_1d(
                sp,
                self.single_spectrum_method,
                kernel_size=self.kernel_size,
                threshold=self.threshold,
            )
            meta_1d = {
                "single_spectrum_method": self.single_spectrum_method,
                "kernel_size": self.kernel_size,
                "threshold": self.threshold,
            }
            if np.any(mask):
                meta_1d["CRs found (spectral indices)"] = list(
                    np.flatnonzero(mask)
                )
            out = with_new_values(
                spectrum,
                corrected.reshape(spectrum.shape),
                "Cosmic Ray Correction",
                meta_1d,
            )
            return out, {"cosmic_mask": mask, "corrected_1d": corrected}
        raise ValueError(
            "CosmicRayRemover supports 3D maps or a single spectrum as 2D "
            "(1, n_spectral); got "
            f"ndim={spectrum.ndim}, "
            f"shape={spectrum.shape}"
        )

    def remove(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Harmonic cleanup first, then cosmic-ray removal."""
        spectrum = self.harmonic_check(spectrum)
        return self.remove_cosmic_rays(spectrum)

    def remove_with_diagnostics(
        self,
        spectrum: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Harmonics, then :meth:`remove_cosmic_rays_with_diagnostics`."""
        after_h = self.harmonic_check(spectrum)
        return self.remove_cosmic_rays_with_diagnostics(after_h)

    def transform(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Alias of :meth:`remove` (harmonics then cosmic rays)."""
        return self.remove(spectrum)

    def _single_spectrum_output(
        self,
        da_template: xr.DataArray,
        spectrum_1d: np.ndarray,
    ) -> xr.DataArray:
        """1D robust spike removal without global intensity rescaling."""
        resolve_spectral_dim(da_template, self.spectral_dim)
        corrected, mask = remove_cosmic_rays_1d(
            spectrum_1d,
            self.single_spectrum_method,
            kernel_size=self.kernel_size,
            threshold=self.threshold,
        )
        meta: dict[str, Any] = {
            "single_spectrum_method": self.single_spectrum_method,
            "kernel_size": self.kernel_size,
            "threshold": self.threshold,
        }
        if np.any(mask):
            meta["CRs found (spectral indices)"] = list(np.flatnonzero(mask))
        return with_new_values(
            da_template,
            corrected.reshape(da_template.shape),
            "Cosmic Ray Correction",
            meta,
        )


__all__ = ["CosmicRayRemover", "SingleSpectrumMethod", "remove_cosmic_rays_1d"]
