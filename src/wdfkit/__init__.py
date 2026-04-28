"""Read Renishaw WiRE .wdf files (spectra and metadata)."""

from .reader import WDFReader
from .spectral_axis import SpectralAxisSpec, resolve_spectral_axis

__all__ = [
    "WDFReader",
    "SpectralAxisSpec",
    "resolve_spectral_axis",
]
