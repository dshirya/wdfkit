#!/usr/bin/env python
##############################################################################
#
# (c) 2026 Danila Shiryaev.
# All rights reserved.
#
# File coded by: Danila Shiryaev
#
##############################################################################
"""Read Renishaw WiRE .wdf files (spectra and metadata)."""

from .reader import WDFReader
from .spectral import SpectralAxisSpec, resolve_spectral_axis

# silence the pyflakes syntax checker
assert WDFReader or True
assert SpectralAxisSpec or True
assert resolve_spectral_axis or True

__all__ = [
    "__version__",
    "WDFReader",
    "SpectralAxisSpec",
    "resolve_spectral_axis",
]

# End of file
