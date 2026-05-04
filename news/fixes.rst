**Added:**

* 2D data arrays are now supported in ``CosmicRayRemover`` (treated as line scans).
* ``CosmicRayRemover`` gains a ``max_passes`` parameter (default 3) for iterative
  1D cosmic-ray detection: each pass runs on the already-repaired signal so that
  large spikes no longer hide smaller ones.

**Changed:**

* 1D spike repair now always uses linear interpolation from the original signal
  (was: ``"median"`` method replaced spikes with the biased median-filter value).
* The 1D spike mask is dilated by 1 channel on each side before repair to cover
  sub-threshold spike edges.
* Error messages from ``CosmicRayRemover`` now include the ``DataArray`` name and
  file identifier for easier debugging.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
