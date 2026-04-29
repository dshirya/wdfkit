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

from .cosmic_ray import CosmicRayRemover, remove_cosmic_rays_1d
from .preprocessing import normalize
from .reader import WDFReader
from .spectral import SpectralAxisSpec, resolve_spectral_axis

# silence the pyflakes syntax checker
assert WDFReader or True
assert SpectralAxisSpec or True
assert resolve_spectral_axis or True
assert normalize or True
assert CosmicRayRemover or True
assert remove_cosmic_rays_1d or True

__all__ = [
    "WDFReader",
    "SpectralAxisSpec",
    "resolve_spectral_axis",
    "normalize",
    "CosmicRayRemover",
    "remove_cosmic_rays_1d",
]

# End of file
