=============
Release notes
=============

.. current developments

0.1.0
=====

**Added:**

* Add optional ``chunks`` on ``WDFReader`` for lazy, Dask-backed map reads with Y-row-aligned targets and a RAM guard before the ``DATA`` block.
* Add ``wdfkit.read`` and header-only ``wdfkit.classify`` on the public API.
* Add a ``wdf/`` layout with per-scan-kind handlers, ``ParsedWDF``, and typed enums/constants for parsing and dispatch.
* Add ``CosmicRayRemover`` support for 2-D inputs and an iterative ``max_passes`` (default ``3``).
* Expose ``ExposureTime`` and ``LaserPower`` from ``WXDM`` / ``WXIS`` as ``DataArray`` attributes where applicable.
* Ship ``py.typed`` and extend type annotations on the read/assembly surface.

**Changed:**

* Make ``normalize``, ``CosmicRayRemover``, and ``SpectraCleaner`` Dask-aware with lazy paths or warnings when full materialisation is needed.
* Unify ``WDFReader``, ``wdfkit.read``, and ``read_wdf_file`` on one handler-based code path; fold the former top-level ``internal/``, ``spectral/`` package tree, and related modules into ``wdf/`` (public ``from wdfkit.spectral import SpectralAxisSpec`` unchanged).
* Return map dimensions as ``x`` / ``y`` (was ``X`` / ``Y``), single scans as 1-D spectral arrays, and sort the spectral coordinate ascending for all kinds.
* Tidy ``attrs`` (CamelCase only), use linear interpolation and slight mask dilation for 1-D cosmic-ray repair, and refresh docs and branding.

**Fixed:**

* Fix chunked-map reads that hit Dask ``nd`` fancy-index failures during ``sortby`` assembly.
* Fix the installed ``wdfkit`` CLI entry point (``ModuleNotFoundError`` under the previous target).
* Fix a file-pointer bug in ``origin.py`` for certain dtypes, preserve exception chaining on missing files in ``io.py``, and confine truncated-image PIL settings to parsing so import has no global side effect.
* Gate noisy YLST debug output behind verbose mode.
* Raise ``ValueError`` for unknown ``normalize`` methods instead of returning silently; replace coordinate ``assert`` checks with explicit errors.

**Removed:**

* Remove the old ``wdf/assemble.py``-centric layout, duplicate package roots absorbed into ``wdf/``, and a few obsolete test helpers.


0.0.1
=====

**Changed:**

* README.rst file updated.
* Logo added to README.rst file.
