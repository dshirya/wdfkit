|Icon| |title|_
===============

.. |title| replace:: wdfkit
.. _title: https://dshirya.github.io/wdfkit

.. |Icon| image:: https://avatars.githubusercontent.com/dshirya
        :target: https://dshirya.github.io/wdfkit
        :height: 100px

|PyPI| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/dshirya/wdfkit/actions/workflows/matrix-and-codecov-on-merge-to-main.yml/badge.svg
        :target: https://github.com/dshirya/wdfkit/actions/workflows/matrix-and-codecov-on-merge-to-main.yml

.. |Codecov| image:: https://codecov.io/gh/dshirya/wdfkit/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/dshirya/wdfkit

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/wdfkit
        :target: https://anaconda.org/conda-forge/wdfkit

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff
        :target: https://github.com/dshirya/wdfkit/pulls

.. |PyPI| image:: https://img.shields.io/pypi/v/wdfkit
        :target: https://pypi.org/project/wdfkit/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/wdfkit
        :target: https://pypi.org/project/wdfkit/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/dshirya/wdfkit/issues

Read Renishaw WiRE ``.wdf`` spectra in Python. **Cosmic-ray removal,
Nd:YAG harmonic notches, and tuning notes** are documented in
`Cosmic-ray and laser-harmonic correction`_ below.

For more information about the wdfkit library, please consult our
`online documentation <https://dshirya.github.io/wdfkit>`_.

Citation
--------

If you use wdfkit in a scientific publication, we would like you to cite this package as

        wdfkit Package, https://github.com/dshirya/wdfkit

Installation
------------

``cd`` into the project directory::

        cd wdfkit

Create and activate a new conda environment::

        conda create -n wdfkit_env python=<max_python_version>
        conda activate wdfkit_env

**Method 1: Install with dependencies from pip**

The only command required is::

        pip install -e .

This automatically installs the dependencies listed in ``requirements/pip.txt``.

**Method 2: Install with dependencies from conda**

If you haven't already, add the conda-forge channel::

        conda config --add channels conda-forge

Install the dependencies::

        conda install --file requirements/conda.txt

Then install the package without re-downloading dependencies::

        pip install -e . --no-deps

Verify the installation::

        pip list

If you prefer to install from PyPI::

        pip install wdfkit

Getting Started
---------------

You may consult our `online documentation <https://dshirya.github.io/wdfkit>`_ for tutorials and API references.

Cosmic-ray and laser-harmonic correction
-----------------------------------------

``wdfkit`` can remove **sharp positive spikes** (cosmic rays / outliers on the
detector) from spectra, and optionally **broad laser-harmonic artifacts** before
that step. The design is general enough for **photoluminescence (PL)** and
**Raman**, but parameters must respect the trade-off between catching more spikes
and protecting narrow real peaks (especially Raman lines).

API overview
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry point
     - Role
   * - ``CosmicRayRemover``
     - High-level API on ``xarray.DataArray`` (single spectra and maps).
   * - ``remove_cosmic_rays_1d``
     - Low-level 1D array API: ``(y, method) → (corrected_y, mask)``.

Import examples:

.. code-block:: python

    from wdfkit import CosmicRayRemover, remove_cosmic_rays_1d
    from wdfkit.preprocessing import normalize  # separate from cosmic-ray module

``CosmicRayRemover`` **methods** (single spectrum: 2D ``(1, nλ)`` or degenerate
3D map; maps: 3D hypercubes):

- **``remove(da)``** (and **``transform(da)``**, same thing): ``harmonic_check``
  first, then cosmic-ray correction. Preferred default pipeline.
- **``harmonic_check(da)``**: laser-harmonic notch only (see below).
- **``remove_cosmic_rays(da)``**: cosmic-ray correction **only** (no harmonics).

Dataclass fields that matter most for **single-spectrum** spike removal:

- ``single_spectrum_method``: ``"median"`` | ``"interpolate"`` | ``"derivative"``
- ``kernel_size``: odd, ≥ 3 (median / interpolate)
- ``threshold``: multiplier on robust noise (higher → fewer detections)
- ``spectral_dim``: spectral axis name if it is not the last dimension (``None`` = last dim)

Laser-harmonic step (before cosmic rays)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When **``LaserWaveLength``** in ``da.attrs`` is in the **354–356 nm** band
(Nd:YAG third harmonic), ``harmonic_check`` / ``remove`` look for **broad**
positive features near catalogue wavelengths **1064, 532, 355, and 266 nm**
(±2.5 nm search), take the local maximum in each window, and replace a
**~1 nm** slice (in wavelength space) with **linear interpolation**. This is
**not** cosmic-ray logic (no MAD spike test).

Supported axes:

- **Wavelength** (e.g. dimension ``nm``, or coord units containing ``nm``).
- **Absolute wavenumber** (cm⁻¹): harmonics are converted with
  :math:`10^7/\lambda_{\mathrm{nm}}`; the notch width is still defined in
  nanometres and mapped to cm⁻¹.

Unsupported / skipped: missing laser metadata, wrong laser band, **Raman shift**
axes, or unknown spectral units (no guessing).

Messages use **``Filename``** from attrs, or ``"unknown file"``. Successful runs
merge ``treatments["Laser harmonic removal"]`` on the output ``DataArray``.

Single-spectrum cosmic-ray logic (``remove_cosmic_rays_1d``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All methods work on a **1D float** intensity vector ``y`` (same order as your
spectral axis). Only **positive** outliers are targeted; **negative** dips are
left as-is.

**``method="median"``** (default): ``scipy.signal.medfilt(y, kernel_size)`` gives
a smooth reference; residual ``y - median_filtered``; robust scale is
**scaled MAD** of the residual (with a small amplitude floor when MAD collapses).
Pixels with ``residual > threshold × noise`` are **replaced** by the
median-filtered value at those pixels.

**``method="interpolate"``**: Same **detection** as median; spikes are **filled**
with ``numpy.interp`` over **channel index** from unmasked neighbours. If every
channel were flagged, the function returns the **original** spectrum and an
**all-false** mask.

**``method="derivative"``**: Robust noise from **global** scaled MAD of
``numpy.diff(y)``. Interior index ``i`` is flagged if it exceeds **both**
neighbours by ``threshold × noise``. **Edges** are never flagged. Repair uses the
same linear interpolation as interpolate.

Why some spikes remain (and how to tune)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detector uses a **single global** noise scale on the whole residual (or whole
``diff``). Real structure (slopes, PL bands, many Raman edges) **inflates** that
scale, so weaker spikes may sit **below** ``threshold × noise``. The defaults are
**conservative** so **narrow material peaks** are not carved up.

Practical levers:

1. **``threshold``**: Lower (e.g. **3–4**) removes more spikes; raises risk on the
   **sharpest real lines**. Increase if you see good features being clipped.
2. **``kernel_size``**: Smaller **(3)** can help **very** narrow spikes; larger
   **(7)** isolates narrow spikes better but can make **real narrow peaks** look
   like positive residuals unless ``threshold`` is high enough.
3. **``method``**: ``derivative`` can help **isolated one-pixel** spikes in
   **flat** regions; on **busy** Raman spectra, global ``MAD(diff(y))`` is often
   large, so this path may **under-detect**.
4. **Two passes**: Applying ``remove_cosmic_rays_1d`` twice on the same trace
   sometimes catches leftovers (no built-in loop; call from your pipeline if
   needed).
5. **Input domain**: Pass **count-like** or linear intensity comparable to noise;
   heavily transformed data change residual statistics.

**Photon noise / PL curvature** can resemble structured residuals; **Raman**
stacks many steep **legitimate** gradients — both push the global MAD up. Tuning
is intentionally left to **``threshold``** and **``kernel_size``**.

Spatial maps (3D ``DataArray``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **multi-pixel** maps, ``CosmicRayRemover.remove_cosmic_rays`` / ``remove`` use
``preprocessing/cosmic_ray_map.py``:

- **Reference:** spatial median filter with a configurable disk radius
  (``disk_radius``); result is ``spatial_median_reference``.
- **Detection:** per-λ cutoff
  ``map_mad_multiplier × (0.01 / sensitivity) × relax_λ × noise_λ``,
  where ``noise_λ`` is **scaled MAD** of the spatial residual at that channel and
  ``relax_λ ∈ [map_noisy_channel_relax_min, 1]`` lowers the threshold in noisy
  bands (more hits there).
- **Repair:** for each ``(y, x)``, masked channels are replaced by **linear
  interpolation along λ** on the local ``spatial_median_reference[y, x, :]``
  curve; unmasked channels keep the original values.
- **Metadata:** treatment dict includes ``map_detection`` and MAD tuning keys;
  unique spatial pixels with core hits are stored under ``CRs found``.

Degenerate maps (single spatial pixel) still use the **1D** MAD pipeline.

Source layout
~~~~~~~~~~~~~

- `src/wdfkit/cosmic_ray.py <src/wdfkit/cosmic_ray.py>`_: ``CosmicRayRemover``
- `src/wdfkit/preprocessing/cosmic_ray_1d.py <src/wdfkit/preprocessing/cosmic_ray_1d.py>`_: ``remove_cosmic_rays_1d``
- `src/wdfkit/preprocessing/cosmic_ray_mad.py <src/wdfkit/preprocessing/cosmic_ray_mad.py>`_: MAD / noise floor
- `src/wdfkit/preprocessing/cosmic_ray_map.py <src/wdfkit/preprocessing/cosmic_ray_map.py>`_: 3D map correction
- `src/wdfkit/preprocessing/spectral_harmonic_removal.py <src/wdfkit/preprocessing/spectral_harmonic_removal.py>`_: Nd:YAG harmonic notches

Unit tests: ``tests/test_cosmic_ray_1d.py``, ``tests/test_harmonic_removal.py``,
cosmic-ray cases in ``tests/test_preprocessing.py``.

Support and Contribute
----------------------

If you see a bug or want to request a feature, please
`report it as an issue <https://github.com/dshirya/wdfkit/issues>`_ and/or
`submit a fix as a PR <https://github.com/dshirya/wdfkit/pulls>`_.

Feel free to fork the project and contribute. To install wdfkit in a development
mode, with its sources being directly used by Python rather than copied to a
package directory, use the following in the root directory::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch,
please set up the use of our pre-commit hooks.

1. Install pre-commit in your working environment by running
   ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8
before you can commit. If it fails by black or isort, just rerun and it should
pass (black and isort will modify the files so should pass after they are
modified). If the flake8 test fails please see the error messages and fix them
manually before trying to commit again.

Improvements and fixes are always appreciated.

Before contributing, please read our
`Code of Conduct <https://github.com/dshirya/wdfkit/blob/main/CODE-OF-CONDUCT.rst>`_.

Contact
-------

For more information on wdfkit please visit the project
`web-page <https://dshirya.github.io/>`_ or email the maintainers
``Danila Shiryaev(danila.shiryaev@polytechnique.edu)``.

Acknowledgements
----------------

``wdfkit`` is built and maintained with
`scikit-package <https://scikit-package.github.io/scikit-package/>`_.
