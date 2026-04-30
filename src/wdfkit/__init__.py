#!/usr/bin/env python
##############################################################################
#
# (c) 2026 Ecole Polytechnique Palaiseau.
# All rights reserved.
#
# File coded by: Danila Shiryaev.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/dshirya/wdfkit/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Python package for WDF data treatment."""

# package version
from wdfkit.version import __version__  # noqa

from .cosmic_ray import CosmicRayRemover, remove_cosmic_rays_1d
from .preprocessing import normalize
from .reader import WDFReader
from .spectra_cleaner import SpectraCleaner
from .spectral import SpectralAxisSpec, resolve_spectral_axis

# silence the pyflakes syntax checker
assert WDFReader or True
assert SpectralAxisSpec or True
assert resolve_spectral_axis or True
assert normalize or True
assert CosmicRayRemover or True
assert remove_cosmic_rays_1d or True
assert SpectraCleaner or True
# silence the pyflakes syntax checker
assert __version__ or True

__all__ = [
    "WDFReader",
    "SpectralAxisSpec",
    "resolve_spectral_axis",
    "normalize",
    "CosmicRayRemover",
    "remove_cosmic_rays_1d",
    "SpectraCleaner",
]

# End of file
