**Added:**

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

* Remove dead ``user_filesystem`` fixture from ``tests/conftest.py``.
