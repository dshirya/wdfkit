:tocdepth: -1

.. index:: getting-started

.. _getting-started:

================
Getting started
================

.. image:: ./img/logo.png
    :alt: wdfkit logo
    :width: 200px
    :align: center

``wdfkit`` reads Renishaw WiRE ``.wdf`` spectra into ``xarray.DataArray``
objects and provides preprocessing tools: cosmic-ray removal, Nd:YAG
laser-harmonic notches, normalization, and PCA-based denoising.

Installation
------------

The quickest way to install is via pip::

    pip install wdfkit

For a conda-based development setup, see the
`README <https://github.com/dshirya/wdfkit#installation>`_.

Reading a ``.wdf`` file
-----------------------

:class:`~wdfkit.WDFReader` loads the file and returns an
``(xr.DataArray, image_or_None)`` pair. It can be unpacked directly:

.. code-block:: python

    from wdfkit import WDFReader

    data, image = WDFReader("measurement.wdf")

    # or keep as an object
    reader = WDFReader("measurement.wdf")
    data = reader.data    # xr.DataArray
    image = reader.image  # white-light image, or None

The ``DataArray`` dimensions depend on the measurement type:

- **Map** scan → shape ``(ny, nx, n_spectral)``
- **Single spectrum** or line → shape ``(1, n_spectral)``

The spectral coordinate name is inferred automatically from the file's
``XlistDataUnits`` (e.g. ``"nm"`` for wavelength data, ``"raman_shift"`` for
Raman shift). Override with the ``spectral_dim`` argument::

    data, _ = WDFReader("measurement.wdf", spectral_dim="shifts")

Cosmic-ray removal
------------------

Use :class:`~wdfkit.CosmicRayRemover` for both single spectra and maps.
The default pipeline runs **laser-harmonic notch first**, then **spike
removal**:

.. code-block:: python

    from wdfkit import WDFReader, CosmicRayRemover

    data, _ = WDFReader("map.wdf")

    remover = CosmicRayRemover()       # all defaults
    data_clean = remover.remove(data)  # harmonics + cosmic rays

For fine-grained control, call the steps separately:

.. code-block:: python

    remover = CosmicRayRemover(
        sensitivity=0.02,    # more aggressive detection
        threshold=4.0,       # lower threshold for single-spectrum path
        single_spectrum_method="interpolate",
    )

    data_no_harmonics = remover.harmonic_check(data)
    data_clean = remover.remove_cosmic_rays(data_no_harmonics)

To inspect what was detected (maps only), use the diagnostics method:

.. code-block:: python

    data_clean, diagnostics = remover.remove_with_diagnostics(data)
    # diagnostics["core_mask"] — boolean array of detected spikes
    # diagnostics["repair_mask"] — dilated mask that was interpolated

Key parameters for **single-spectrum** removal:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``single_spectrum_method``
     - ``"median"`` (default), ``"interpolate"``, or ``"derivative"``
   * - ``kernel_size``
     - Odd integer ≥ 3; median filter window
   * - ``threshold``
     - Multiplier on robust noise (higher → fewer detections)

Key parameters for **map** removal:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``sensitivity``
     - Scales detection aggressiveness (higher → more hits)
   * - ``disk_radius``
     - Spatial disk radius for the median reference filter
   * - ``map_mad_multiplier``
     - Multiplier on per-channel MAD noise (larger → fewer false positives)

Normalization
-------------

:func:`~wdfkit.normalize` scales spectra along the spectral axis:

.. code-block:: python

    from wdfkit import normalize

    data_norm = normalize(data, method="area")

Available methods: ``"l1"``, ``"l2"``, ``"max"``, ``"min_max"``,
``"robust_scale"`` (default), ``"area"``, ``"wave_number"``.

PCA denoising
-------------

:class:`~wdfkit.SpectraCleaner` removes noise from a **population** of spectra
(maps or stacks) using PCA reconstruction. It requires more than one spectrum —
for a single spectrum use a 1D smoother instead.

.. code-block:: python

    from wdfkit import WDFReader, SpectraCleaner

    data, _ = WDFReader("map.wdf")

    cleaner = SpectraCleaner(n_components="mle")  # Minka's MLE picks component count
    data_clean = cleaner.clean(data)

To also retrieve the PCA decomposition (components, per-spectrum scores,
explained variance):

.. code-block:: python

    cleaner = SpectraCleaner(n_components=0.95)   # keep 95 % of variance
    data_clean, decomp = cleaner.clean_with_decomposition(data)

    components = decomp["components"]              # shape (n_components, n_spectral)
    coeffs = decomp["coeffs"]                      # shape (ny, nx, n_components)
    print(decomp["explained_variance_ratio_total"])

The cleaned ``DataArray`` carries a ``treatments["spectra_cleaning"]`` entry
with the number of components used and variance-explained summary.

Typical workflow
----------------

A common end-to-end pipeline for a Raman map:

.. code-block:: python

    from wdfkit import WDFReader, CosmicRayRemover, normalize, SpectraCleaner

    # 1. Load
    data, image = WDFReader("raman_map.wdf")

    # 2. Remove cosmic rays (and Nd:YAG harmonics if applicable)
    data = CosmicRayRemover(sensitivity=0.015).remove(data)

    # 3. Normalize
    data = normalize(data, method="area")

    # 4. PCA denoise (maps only)
    data = SpectraCleaner(n_components="mle").clean(data)
