**Added:**

* Support 2-D data arrays in ``CosmicRayRemover`` (treated as line scans).
* Add ``max_passes`` parameter (default ``3``) to ``CosmicRayRemover`` for
  iterative 1-D cosmic-ray detection so that large spikes no longer hide
  smaller ones.

**Changed:**

* Use linear interpolation from the original signal for 1-D spike repair in
  all cases (was: ``"median"`` method replaced spikes with the biased
  median-filter value).
* Dilate the 1-D spike mask by 1 channel on each side before repair to cover
  sub-threshold spike edges.
* Include the ``DataArray`` name and file identifier in ``CosmicRayRemover``
  error messages for easier debugging.
