# -*- coding: utf-8 -*-
"""Spectral preprocessing (normalization).

Cosmic-ray removal: see :mod:`wdfkit.cosmic_ray`.
PCA-based denoising: see :mod:`wdfkit.spectra_cleaner`.
"""

from .normalize import normalize
from .pca_clean import denoise_spectra_pca

__all__ = ["normalize", "denoise_spectra_pca"]
