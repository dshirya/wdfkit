=============
Release notes
=============

.. current developments

0.1.0
=====

**Added:**

* Add ``chunks`` parameter (``False`` / ``True`` / int MB) to ``WDFReader``
for lazy, memory-efficient reading of large map files backed by
:mod:`dask`; chunks align to whole Y-rows and target ~128 MB each, capped
at 20 chunks to avoid scheduler overhead.
* Add pre-read RAM guard before the ``DATA`` block: raise ``MemoryError``
when the full array would exceed free RAM, or emit a ``UserWarning`` when
it would exceed 75 % of free RAM, with instructions to re-open with
``chunks=True``.
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
* Support 2-D data arrays in ``CosmicRayRemover`` (treated as line scans).
* Add ``max_passes`` parameter (default ``3``) to ``CosmicRayRemover`` for
iterative 1-D cosmic-ray detection so that large spikes no longer hide
smaller ones.
* Extract ``ExposureTime`` (CCD exposure in seconds) and ``LaserPower``
(ND-filter transmission in percent) from the ``WXDM`` and ``WXIS`` blocks
and store them as ``DataArray`` attributes for single spectra, maps, and
volume scans.
* Added test data for the new API.
* Add ``ensure_in_memory`` shared helper in ``wdf/utils.py`` to centralise
the Dask warn-and-compute pattern across ``CosmicRayRemover``,
``SpectraCleaner``, and ``normalize``.
* Add ``_build_cr_meta_1d`` method to ``CosmicRayRemover`` to deduplicate
repeated construction of the 1-D cosmic-ray correction metadata dict.
* Add ``py.typed`` marker (PEP 561) so that type checkers treat ``wdfkit``
as a typed package.
* Add type annotations to ``read_wdf_file`` and the private assembly helpers.
* Add ``tests/test_spectra_cleaner.py`` with 18 tests covering
``SpectraCleaner`` and the underlying ``denoise_spectra_pca``.

**Changed:**

* Make :func:`~wdfkit.preprocessing.normalize` Dask-aware: per-spectrum
methods (``"l1"``, ``"l2"``, ``"max"``, ``"min_max"``, ``"area"``)
process each chunk independently via :func:`xarray.apply_ufunc` and keep
the result lazy; global methods (``"robust_scale"``, ``"wave_number"``)
load the full array and emit a ``UserWarning`` explaining the memory
impact.
* Make :class:`~wdfkit.cosmic_ray.CosmicRayRemover` and
:class:`~wdfkit.spectra_cleaner.SpectraCleaner` detect Dask-backed input,
call ``.compute()`` before their NumPy/sklearn kernels, and emit a
``UserWarning`` with the estimated RAM cost.
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
* Use linear interpolation from the original signal for 1-D spike repair in
all cases (was: ``"median"`` method replaced spikes with the biased
median-filter value).
* Dilate the 1-D spike mask by 1 channel on each side before repair to cover
sub-threshold spike edges.
* Include the ``DataArray`` name and file identifier in ``CosmicRayRemover``
error messages for easier debugging.
* Update documentation to reflect the new API.
* Change the logo to a more modern one.
* Fix ``WDFReader(chunks=True)`` map assembly by splitting ``sortby`` into
two 1-D calls (``da.sortby(sdim).sortby("Time")``) to avoid an
unsupported nd fancy-index in Dask.
* Raise ``ValueError`` in ``normalize`` immediately when an unrecognised
``method`` is passed (was: silently returned an unchanged copy).
* Replace ``assert`` statements for coordinate validation with explicit
``raise ValueError`` (asserts are stripped in optimised mode).
* Replace ``except Exception: pass`` in ``print_coord_lengths_if_verbose``
with a ``UserWarning`` that surfaces the failure message.
* Replace ``assert X or True`` pyflakes workaround in ``__init__.py`` with
``# noqa: F401`` on each import line.
* Correct ``memory_check.py`` docstring to accurately describe that the
chunked path skips all RAM checks.
* Remove unused ``max_depth`` parameter from ``find_in_pset``.
* Fix grammar in incomplete-recording warning: "was recorded" → "were
recorded".
* Rename ``test_normalize_invalid_method_warns`` to
``test_normalize_invalid_method_raises`` to match the new ``ValueError``
behaviour.

**Fixed:**

* Fix ``WDFReader(chunks=True)`` on map files raising
``NotImplementedError: Don't yet support nd fancy indexing``.
* Fix ``wdfkit`` CLI entry point in ``pyproject.toml``:
``wdfkit.app:main`` → ``wdfkit.wdfkit_app:main`` (caused
``ModuleNotFoundError`` when the installed command was invoked).
* Fix ``origin.py`` dtype 16/17 branch to advance the file pointer without
discarding the result of an unassigned ``np.array`` call.
* Chain the original exception in ``io.py`` file-not-found error
(``raise IOError(...) from e`` instead of ``from None``).
* Move ``ImageFile.LOAD_TRUNCATED_IMAGES = True`` from module-level in
``io.py`` into ``parse_whtl`` to remove a hidden process-global PIL side
effect on import.
* Gate ``ylst.py`` debug ``print(y_values)`` behind ``ctx.verbose``
(was: printed unconditionally for every file with a YLST block).

**Removed:**

* Remove ``wdf/assemble.py`` and its ``assemble_data_array`` helper.
* Remove ``internal/`` package (contents moved to ``wdf/``).
* Remove ``spectral/`` package (replaced by ``spectral.py``).
* Remove top-level ``_types.py``, ``_parsed.py``, ``_dispatch.py``, and
``handlers/`` (all moved into ``wdf/``).
* Remove dead ``user_filesystem`` fixture from ``tests/conftest.py``.


0.0.1
=====

**Changed:**

* README.rst file updated.
* Logo added to README.rst file.
