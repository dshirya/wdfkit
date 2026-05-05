**Added:**

* Added ``ensure_in_memory`` shared helper in ``wdfkit.internal.utils`` to
  centralise the Dask warn-and-compute pattern used across
  ``CosmicRayRemover``, ``SpectraCleaner``, and ``normalize``.
* Added ``_build_cr_meta_1d`` method to ``CosmicRayRemover`` to deduplicate
  repeated construction of the 1D cosmic-ray correction metadata dict.
* Added ``_reorder_attrs`` in ``assemble.py`` so that ``ExposureTime`` and
  ``LaserPower`` always appear immediately after ``WdfFlag`` in
  ``DataArray.attrs``.
* Added ``py.typed`` marker (PEP 561) so that type checkers treat wdfkit as
  a typed package.
* Added type annotations to ``read_wdf_file`` and the private helpers
  ``_assemble_map_scan`` / ``_assemble_series_scan`` in ``assemble.py``.
* Added ``tests/test_spectra_cleaner.py`` with 18 tests covering
  ``SpectraCleaner`` and the underlying ``denoise_spectra_pca``.

**Changed:**

* ``WDFReader(chunks=True)`` now returns a fully **Dask-backed**
  ``xr.DataArray`` after map assembly. Previously, ``sortby`` with multiple
  keys triggered an unsupported nd fancy-index in Dask; the fix splits the
  call into ``da.sortby(sdim).sortby("Time")`` so each sort applies a 1-D
  index to its own dimension.
* ``normalize`` now raises ``ValueError`` immediately when an unrecognised
  ``method`` is passed (previously emitted a warning and silently returned
  an unchanged copy).
* ``assert`` statements used for runtime coordinate validation in
  ``assemble.py`` replaced with explicit ``raise ValueError`` (asserts are
  stripped in optimised mode).
* ``except Exception: pass`` in ``origin.print_coord_lengths_if_verbose``
  replaced with a ``UserWarning`` that surfaces the failure message.
* ``assert X or True`` pyflakes workaround in ``__init__.py`` replaced with
  ``# noqa: F401`` on each import line.
* ``memory_check.py`` docstring corrected to accurately describe that the
  chunked path skips all RAM checks (previously described a per-chunk check
  that was never implemented).
* ``find_in_pset`` signature simplified: the unused ``max_depth`` parameter
  has been removed.
* Grammar fix in incomplete-recording warning: "Not all spectra was
  recorded" → "were recorded".
* ``CosmicRayRemover._maybe_compute_for_map`` and ``SpectraCleaner``
  inline Dask compute both delegate to the new ``ensure_in_memory`` helper.
* ``test_normalize_invalid_method_warns`` renamed to
  ``test_normalize_invalid_method_raises`` to match the new ``ValueError``
  behaviour.

**Fixed:**

* ``WDFReader(chunks=True)`` on map files no longer raises
  ``NotImplementedError: Don't yet support nd fancy indexing``.
* ``wdfkit`` CLI entry point corrected in ``pyproject.toml``:
  ``wdfkit.app:main`` → ``wdfkit.wdfkit_app:main`` (previous value caused
  ``ModuleNotFoundError`` when the installed command was invoked).
* ``origin.py`` dtype 16/17 branch now correctly advances the file pointer
  without silently discarding a computed-but-unassigned ``np.array`` result.
* ``io.py`` file-not-found error now chains the original exception
  (``raise IOError(...) from e`` instead of ``from None``).
* ``ImageFile.LOAD_TRUNCATED_IMAGES = True`` moved from module-level in
  ``io.py`` into ``parse_whtl`` where it belongs, removing the hidden
  process-global PIL side effect on import.
* ``ylst.py`` debug ``print(y_values)`` gated behind ``ctx.verbose``
  (previously printed unconditionally for every file with a YLST block).

**Removed:**

* Dead ``user_filesystem`` fixture (``diffpyconfig.json`` setup unrelated to
  wdfkit) removed from ``tests/conftest.py``.
