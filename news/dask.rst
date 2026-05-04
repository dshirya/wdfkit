**Added:**

* ``WDFReader`` gains a ``chunks`` parameter (``False`` / ``True`` / int MB) for
  lazy, memory-efficient reading of large 3D map files backed by
  :mod:`dask`.  When enabled, the ``DATA`` block is never fully loaded into
  RAM; instead a :class:`dask.array.Array` is built from per-chunk
  :func:`dask.delayed` reads that are executed only when needed.
  Chunks align to whole Y-rows so that the assembled ``(Y, X, spectral)``
  :class:`xarray.DataArray` has clean ``(chunk_y, nx, npoints)`` chunk
  boundaries.  Chunk size is auto-computed to target ~128 MB per chunk
  (or a user-supplied MB value) while capping the total number of chunks
  at 20 to avoid scheduler overhead.

* Pre-read RAM guard: before touching the ``DATA`` block, ``WDFReader``
  checks available system memory via :mod:`psutil`.  If the full array
  would exceed free RAM a ``MemoryError`` is raised with instructions to
  re-open with ``chunks=True``; if it would use more than 75 % of free
  RAM a ``UserWarning`` is emitted.

**Changed:**

* :func:`~wdfkit.preprocessing.normalize` is now Dask-aware.
  Per-spectrum methods (``"l1"``, ``"l2"``, ``"max"``, ``"min_max"``,
  ``"area"``) process each chunk independently via
  :func:`xarray.apply_ufunc` and keep the result lazy.  Global methods
  (``"robust_scale"``, ``"wave_number"``) load the full array and emit a
  ``UserWarning`` explaining why.

* :class:`~wdfkit.cosmic_ray.CosmicRayRemover` and
  :class:`~wdfkit.spectra_cleaner.SpectraCleaner` now detect Dask-backed
  input, call ``.compute()`` before their NumPy/sklearn kernels, and emit
  a ``UserWarning`` with the estimated RAM cost so users are aware of the
  memory spike.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
