# -*- coding: utf-8 -*-
"""
High-level spectral denoising: :class:`SpectraCleaner`.

Currently implements PCA-based reconstruction (legacy ``pca_clean``); the
``method`` switch is kept so other denoisers can be added (e.g. Savitzky-
Golay or wavelet) without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr

from .preprocessing._common import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
from .preprocessing.pca_clean import NComponents, denoise_spectra_pca

CleanMethod = Literal["pca"]

_TREATMENT_KEY = "spectra_cleaning"


@dataclass
class SpectraCleaner:
    """Denoise a population of spectra by low-rank reconstruction.

    Designed for **3D map cubes** ``(ny, nx, n_spectral)`` and **2D stacks**
    ``(n_spectra, n_spectral)``. PCA reconstruction needs more than one
    spectrum to separate shared signal from per-channel noise — a single
    spectrum is rejected with a clear error (use a 1D smoother instead).

    Parameters
    ----------
    method
        Denoising method. Currently only ``\"pca\"`` is implemented; the
        switch is kept for forward compatibility.
    n_components
        Forwarded to :class:`sklearn.decomposition.PCA`. ``\"mle\"``
        (default), a ``float`` in ``(0, 1)`` for variance-explained,
        an ``int`` count, or ``None`` for ``min(n_spectra, n_spectral)``.
    subtract_min
        Subtract per-spectrum min before the fit (legacy default ``True``).
        PCA also mean-centers internally, so this only changes the baseline
        offset fed to the fit.
    restore_min
        Add the saved per-spectrum min back after reconstruction. Off by
        default (legacy behavior); enable to preserve absolute intensities.
    spectral_dim
        Name of the spectral axis in DataArray inputs. Defaults to the last
        dimension; pass when spectra are not last (e.g. ``\"raman_shift\"``
        with leading spectral axis).
    pca_kwargs
        Extra kwargs forwarded to :class:`sklearn.decomposition.PCA`
        (e.g. ``{\"svd_solver\": \"full\"}``).
    """

    method: CleanMethod = "pca"
    n_components: NComponents = "mle"
    subtract_min: bool = True
    restore_min: bool = False
    spectral_dim: str | None = None
    pca_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed: tuple[str, ...] = ("pca",)
        if self.method not in allowed:
            raise ValueError(
                f"method must be one of {allowed!r}, got {self.method!r}"
            )
        if isinstance(self.n_components, float) and not (
            0.0 < self.n_components < 1.0
        ):
            raise ValueError(
                "float n_components must be in (0, 1) (variance ratio); "
                f"got {self.n_components}"
            )
        if isinstance(self.n_components, int) and self.n_components < 1:
            raise ValueError(
                f"int n_components must be >= 1, got {self.n_components}"
            )

    def clean(self, spectra: xr.DataArray) -> xr.DataArray:
        """Return a denoised copy of ``spectra`` (no decomposition
        payload)."""
        cleaned, meta, _ = self._clean_core(
            spectra, return_decomposition=False
        )
        return with_new_values(spectra, cleaned, _TREATMENT_KEY, meta)

    def clean_with_decomposition(
        self,
        spectra: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Like :meth:`clean`, but also returns the PCA decomposition.

        The returned ``decomposition`` dict has keys ``components`` (shape
        ``(n_components, n_spectral)``), ``coeffs`` (per-spectrum scores
        reshaped to the input's spatial layout + components axis), ``mean``,
        ``explained_variance``, ``explained_variance_ratio``, and
        ``noise_variance``. These arrays can be large — they're returned
        separately rather than written to ``DataArray.attrs``.
        """
        cleaned, meta, payload = self._clean_core(
            spectra, return_decomposition=True
        )
        out = with_new_values(spectra, cleaned, _TREATMENT_KEY, meta)
        return out, payload

    def transform(self, spectra: xr.DataArray) -> xr.DataArray:
        """Alias of :meth:`clean`."""
        return self.clean(spectra)

    def _clean_core(
        self,
        spectra: xr.DataArray,
        *,
        return_decomposition: bool,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None]:
        """Validate input, transpose spectral last, run PCA, restore
        order."""
        if not isinstance(spectra, xr.DataArray):
            raise TypeError(
                "SpectraCleaner.clean expects an xarray.DataArray; got "
                f"{type(spectra).__name__}"
            )

        sdim = resolve_spectral_dim(spectra, self.spectral_dim)
        da_w, orig_order = transpose_spectral_last(spectra, sdim)
        spatial_shape = da_w.shape[:-1]
        n_spectra = int(np.prod(spatial_shape)) if spatial_shape else 1
        if n_spectra < 2:
            raise ValueError(
                "SpectraCleaner needs more than one spectrum (PCA on a "
                "single spectrum is degenerate). Got input with shape "
                f"{tuple(spectra.shape)} → n_spectra={n_spectra} along "
                f"non-spectral dims. For a single spectrum use a 1D "
                "smoother (e.g. Savitzky-Golay) instead."
            )

        result = denoise_spectra_pca(
            da_w.values,
            n_components=self.n_components,
            subtract_min=self.subtract_min,
            restore_min=self.restore_min,
            pca_kwargs=self.pca_kwargs or None,
            return_decomposition=return_decomposition,
        )
        if return_decomposition:
            cleaned_w, meta, payload = result
        else:
            cleaned_w, meta = result
            payload = None
        meta = {**meta, "spectral_dim": sdim}

        cleaned_w_array = reshape_row_stack_to(
            cleaned_w.reshape(-1, cleaned_w.shape[-1]),
            da_w.shape,
        )
        if tuple(da_w.dims) != orig_order:
            cleaned_da = da_w.copy(data=cleaned_w_array).transpose(*orig_order)
            cleaned = cleaned_da.values
        else:
            cleaned = cleaned_w_array
        return cleaned, meta, payload


__all__ = ["SpectraCleaner", "CleanMethod"]
