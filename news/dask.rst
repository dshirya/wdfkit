**Added:**

* Add ``chunks`` parameter (``False`` / ``True`` / int MB) to ``WDFReader``
  for lazy, memory-efficient reading of large map files backed by
  :mod:`dask`; chunks align to whole Y-rows and target ~128 MB each, capped
  at 20 chunks to avoid scheduler overhead.
* Add pre-read RAM guard before the ``DATA`` block: raise ``MemoryError``
  when the full array would exceed free RAM, or emit a ``UserWarning`` when
  it would exceed 75 % of free RAM, with instructions to re-open with
  ``chunks=True``.

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
