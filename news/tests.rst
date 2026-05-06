**Added:**

* Add handler-based dispatch architecture: each WDF scan kind (``single``,
  ``series``, ``points``, ``line_xy``, ``raster_rowmajor``,
  ``raster_columnmajor``, ``raster_snake``, ``linefocus``, ``volume``) has
  its own module under ``wdf/handlers/``.
* Add ``wdfkit.read(path)`` module-level function returning an
  ``xr.DataArray`` directly.
* Add ``wdfkit.classify(path)`` for fast header-only scan-type triage without
  loading spectrum data.
* Add ``wdf/parsed.py`` with the ``ParsedWDF`` immutable dataclass as the
  intermediate representation between block parsing and handler assembly.
* Add ``wdf/types.py`` with ``MeasurementType``, ``ScanType``, ``DataType``,
  ``UnitType`` enums and ``MapFlag`` bitmask constants.
* Add ``wdf/dispatch.py`` with ``classify_kind`` and ``dispatch`` for routing
  a ``ParsedWDF`` to the correct handler.
* Add 7 new test modules covering each WDF scan-kind handler.

**Changed:**

* Consolidate all WDF I/O into a single ``wdf/`` subpackage: ``internal/``,
  ``spectral/``, top-level ``handlers/``, and the root ``_types.py``,
  ``_parsed.py``, ``_dispatch.py`` files are all absorbed into ``wdf/``.
* Collapse ``spectral/axis.py`` to a flat ``spectral.py`` module; the public
  import path ``from wdfkit.spectral import SpectralAxisSpec`` is unchanged.
* Replace ``wdf/assemble.py`` with handler dispatch in ``read_wdf_file``;
  ``WDFReader``, ``wdfkit.read``, and the legacy ``read_wdf_file`` now all
  share the same code path.
* Rename map ``xr.DataArray`` dimensions to lowercase ``x``/``y``
  (were ``X``/``Y``).
* Return single-scan ``xr.DataArray`` as 1-D (spectral axis only) instead of
  2-D ``(Time, spectral)``.
* Sort spectral coordinate ascending (low → high) for all scan kinds.
* Remove duplicate lowercase keys from ``DataArray.attrs``; only the existing
  CamelCase parameter keys are kept.

**Removed:**

* Remove ``wdf/assemble.py`` and its ``assemble_data_array`` helper.
* Remove ``internal/`` package (contents moved to ``wdf/``).
* Remove ``spectral/`` package (replaced by ``spectral.py``).
* Remove top-level ``_types.py``, ``_parsed.py``, ``_dispatch.py``, and
  ``handlers/`` (all moved into ``wdf/``).
