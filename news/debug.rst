**Added:**

* Add ``CleanData`` class to detect oversaturated spectra (10+ consecutive zero channels) and automatically remove them from 2D/3D arrays; integrated as the first step in ``CosmicRayRemover`` and ``SpectraCleaner``.
* Add zero-saturation detection to ``CosmicRayRemover`` via ``_zero_saturation_mask``, flagging ADC-clipped channels before positive-spike removal.
* Add reading of InitialCoordinates for 2D files.

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Fix ``CosmicRayRemover`` collection-engine repair to interpolate from the original spectrum's clean channels instead of the PCA reference, eliminating residual negative spikes.
* Fix ``CosmicRayRemover`` collection engine to run a second detection pass on a reference rebuilt from clean data, improving sensitivity on heterogeneous maps.
* Lower default ``spike_threshold`` in ``CosmicRayRemover`` from ``5.0`` to ``3.5`` to improve cosmic-ray detection on typical spectra.

**Security:**

* <news item>
