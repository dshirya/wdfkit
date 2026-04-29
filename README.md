# wdfkit

Read Renishaw WiRE `.wdf` spectra in Python. **Cosmic-ray removal, Nd:YAG harmonic notches, and tuning notes** are documented in [Cosmic-ray and laser-harmonic correction](#cosmic-ray-and-laser-harmonic-correction) below.

Here is a quick tutorial on how to locally install the package.

## How to install `wdfkit` locally

`cd` into the project directory:

```bash
cd wdfkit
```

Create and activate a new conda environment:

```bash
conda create -n wdfkit_env python=<max_python_version>
conda activate wdfkit_env
```

### Method 1: Install your package with dependencies sourced from pip

It's simple. The only command required is the following:

```bash
pip install -e .
```

> The above command will automatically install the dependencies listed in `requirements/pip.txt`.

### Method 2: Install your package with dependencies sourced from conda

If you haven't already, ensure you have the conda-forge channel added as the highest priority channel.

```bash
conda config --add channels conda-forge
```

Install the dependencies listed under `conda.txt`:

```bash
conda install --file requirements/conda.txt
```

Then install your Python package locally:

```bash
pip install -e . --no-deps
```

> `--no-deps` is used to avoid installing the dependencies in `requirements/pip.txt` since they are already installed in the previous step.

## Verify your package has been installed

Verify the installation:

```bash
pip list
```

## Cosmic-ray and laser-harmonic correction

`wdfkit` can remove **sharp positive spikes** (cosmic rays / outliers on the detector) from spectra, and optionally **broad laser-harmonic artefacts** before that step. The design is general enough for **photoluminescence (PL)** and **Raman**, but parameters must respect the trade-off between _catching more spikes_ and _protecting narrow real peaks_ (especially Raman lines).

### API overview

| Entry point             | Role                                                            |
| ----------------------- | --------------------------------------------------------------- |
| `CosmicRayRemover`      | High-level API on `xarray.DataArray` (single spectra and maps). |
| `remove_cosmic_rays_1d` | Low-level 1D array API: `(y, method) → (corrected_y, mask)`.    |

Import examples:

```python
from wdfkit import CosmicRayRemover, remove_cosmic_rays_1d
from wdfkit.preprocessing import normalize  # separate from cosmic-ray module
```

**`CosmicRayRemover` methods** (single spectrum: 2D `(1, nλ)` or degenerate 3D map; maps: 3D hypercubes):

- **`remove(da)`** (and **`transform(da)`**, same thing): **`harmonic_check` first**, then cosmic-ray correction. Preferred default pipeline.
- **`harmonic_check(da)`**: laser-harmonic notch only (see below).
- **`remove_cosmic_rays(da)`**: cosmic-ray correction **only** (no harmonics).

Dataclass fields that matter most for **single-spectrum** spike removal:

- `single_spectrum_method`: `"median"` \| `"interpolate"` \| `"derivative"`
- `kernel_size`: odd, ≥ 3 (median / interpolate)
- `threshold`: multiplier on robust noise (higher → fewer detections)
- `spectral_dim`: spectral axis name if it is not the last dimension (`None` = last dim)

### Laser-harmonic step (before cosmic rays)

When **`LaserWaveLength`** in `da.attrs` is in the **354–356 nm** band (Nd:YAG third harmonic), `harmonic_check` / `remove` look for **broad** positive features near catalogue wavelengths **1064, 532, 355, and 266 nm** (±2.5 nm search), take the local maximum in each window, and replace a **~1 nm** slice (in wavelength space) with **linear interpolation**. This is **not** cosmic-ray logic (no MAD spike test).

Supported axes:

- **Wavelength** (e.g. dimension `nm`, or coord units containing `nm`).
- **Absolute wavenumber** (cm⁻¹): harmonics are converted with \(10^7/\lambda\_{\mathrm{nm}}\); the notch width is still defined in nanometres and mapped to cm⁻¹.

Unsupported / skipped: missing laser metadata, wrong laser band, **Raman shift** axes, or unknown spectral units (no guessing).

Messages use **`Filename`** from attrs, or `"unknown file"`. Successful runs merge `treatments["Laser harmonic removal"]` on the output `DataArray`.

### Single-spectrum cosmic-ray logic (`remove_cosmic_rays_1d`)

All methods work on a **1D float** intensity vector `y` (same order as your spectral axis). Only **positive** outliers are targeted; **negative** dips are left as-is.

**`method="median"`** (default): `scipy.signal.medfilt(y, kernel_size)` gives a smooth reference; residual `y - median_filtered`; robust scale is **scaled MAD** of the residual (with a small amplitude floor when MAD collapses). Pixels with `residual > threshold × noise` are **replaced** by the median-filtered value at those pixels (same channel grid; no extra wavelength interpolation).

**`method="interpolate"`**: Same **detection** as median; spikes are **filled** with `numpy.interp` over **channel index** from unmasked neighbours (not re-sampled in physical \(x\)). If every channel were flagged, the function returns the **original** spectrum and an **all-false** mask.

**`method="derivative"`**: Robust noise from **global** scaled MAD of `numpy.diff(y)`. Interior index `i` is flagged if it exceeds **both** neighbours by `threshold × noise`. **Edges** are never flagged. Repair uses the same linear interpolation as interpolate.

### Why some spikes remain (and how to tune)

The detector uses a **single global** noise scale on the whole residual (or whole `diff`). Real structure (slopes, PL bands, many Raman edges) **inflates** that scale, so weaker spikes may sit **below** `threshold × noise`. The defaults are **conservative** so **narrow material peaks** are not carved up.

Practical levers:

1. **`threshold`**: Lower (e.g. **3–4**) removes more spikes; raises risk on the **sharpest real lines**. Increase if you see good features being clipped.
2. **`kernel_size`**: Smaller **(3)** can help **very** narrow spikes but may understabilise the median reference; larger **(7)** isolates narrow spikes better but can make **real narrow peaks** look like positive residuals unless `threshold` is high enough.
3. **`method`**: `derivative` can help **isolated one-pixel** spikes in **flat** regions; on **busy** Raman spectra, global `MAD(diff(y))` is often large, so this path may **under-detect**.
4. **Two passes**: Applying `remove_cosmic_rays_1d` twice on the same trace sometimes catches leftovers after the first fill (no built-in loop; call from your pipeline if needed).
5. **Input domain**: Pass **count-like** or linear intensity comparable to noise; heavily transformed data change residual statistics.

**Photon noise / PL curvature** can resemble structured residuals; **Raman** stacks many steep **legitimate** gradients—both push the global MAD up. There is no separate “map vs Raman” mode today: tuning is intentionally left to **`threshold`** and **`kernel_size`**.

### Spatial maps (3D `DataArray`)

For **multi-pixel** maps, `CosmicRayRemover.remove_cosmic_rays` / `remove` use
`preprocessing/cosmic_ray_map.py`:

- **Same preprocessing as legacy** `remove_CRs`: per-spectrum min subtraction,
  then divide by the spectrum median → **preprocessed** cube; multiply back by
  **per_spectrum_median** for physical units.
- **Reference:** spatial median filter with a disk (footprint
  `disk[:, :, np.newaxis]`); we call this **spatial_median_reference**.
- **Detection (new vs legacy):** legacy used one global cutoff
  `preprocessed - spatial_median_reference > 10 / sensitivity` on the same
  normalized cube. The current code uses a **per-λ** cutoff:
  `map_mad_multiplier × (0.01 / sensitivity) × relax_λ × noise_λ`,
  where `noise_λ` is **scaled MAD** of the spatial residual slice at that
  channel and `relax_λ ∈ [map_noisy_channel_relax_min, 1]` is smaller when
  `noise_λ` is **above** the median across channels (noisy bands → **lower**
  effective threshold → more hits). Defaults: `map_mad_multiplier=4`,
  `map_noisy_channel_relax_min=0.55`. `sensitivity` still increases
  aggressiveness like legacy (higher → more detection), with `0.01` the
  reference matching the old default.
- **Repair (new vs legacy):** legacy dilated the hit mask along λ and set
  `preprocessed[mask] = spatial_median_reference[mask]` (every affected voxel
  **equal** to the spatial median surface). That can **erase spectral shape**
  inside the dilated tube. The current code dilates the mask the same way in λ,
  but for each `(y, x)` it replaces **only** masked channels: values come
  from **linear interpolation along λ** on the local curve
  `spatial_median_reference[y, x, :]`, while **unmasked** channels keep the
  original **preprocessed** values — so traces are not replaced by a flat copy
  of the median image across all λ.
- **Metadata:** treatment dict includes `map_detection` and the MAD tuning
  keys; unique spatial pixels with core hits remain under `CRs found`.

**Degenerate** maps (single spatial pixel) still use the **1D** MAD pipeline,
not this module.

#### Compared to `legacy/preprocessing.remove_CRs` (≈ lines 630–712)

| Aspect            | Legacy                                                                  | Current `cosmic_ray_map`                                                                        |
| ----------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Normalization     | Min along λ, ÷ median → preprocessed cube; output × per-spectrum median | Same                                                                                            |
| Spatial reference | `median_filter` with disk `radius=3` hard-coded                         | Configurable `disk_radius`                                                                      |
| Threshold         | Single scalar `10 / sensitivity` for all λ                              | Per-λ `noise_λ` (spatial MAD × floor) × `relax_λ` × `map_mad_multiplier` × `(0.01/sensitivity)` |
| Noisy bands       | No special rule                                                         | `relax_λ` clips `median(noise)/noise_λ` to `[relax_min, 1]`                                     |
| Spectral dilation | `width × n_channels` box along λ                                        | Capped; optional max repair extent along λ                                                      |
| Inpainting        | Paste spatial median at mask                                            | **Interp** along λ from `spatial_median_reference[y,x,:]` on bad channels only                  |
| Bug               | `if crs_found == 0` compares list to int                                | N/A                                                                                             |

### Source layout (for contributors)

- [`src/wdfkit/cosmic_ray.py`](src/wdfkit/cosmic_ray.py): `CosmicRayRemover`
- [`src/wdfkit/preprocessing/cosmic_ray_1d.py`](src/wdfkit/preprocessing/cosmic_ray_1d.py): `remove_cosmic_rays_1d`
- [`src/wdfkit/preprocessing/cosmic_ray_mad.py`](src/wdfkit/preprocessing/cosmic_ray_mad.py): MAD / noise floor
- [`src/wdfkit/preprocessing/cosmic_ray_map.py`](src/wdfkit/preprocessing/cosmic_ray_map.py): 3D map correction
- [`src/wdfkit/preprocessing/spectral_harmonic_removal.py`](src/wdfkit/preprocessing/spectral_harmonic_removal.py): Nd:YAG harmonic notches

Unit tests: `tests/test_cosmic_ray_1d.py`, `tests/test_harmonic_removal.py`, cosmic-ray cases in `tests/test_preprocessing.py`.
