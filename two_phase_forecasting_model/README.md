# Two-Phase Solar Flare Forecasting Model

Notebooks for a two-phase approach to forecasting solar flares from the
**SWAN-SF** (Space Weather ANalytics for Solar Flares) benchmark dataset.

* **Phase 1 – Classification.** Given a 3-hour (15 × 12-minute) window of
  SHARP magnetic parameters and GOES X-ray flux for an active region,
  classify the upcoming activity as *No Flare*, *M-class flare* or
  *X-class flare*.
* **Phase 2 – Flux forecasting.** For flaring regions, forecast the next
  1 hour (5 steps) of GOES soft X-ray flux (`XR_MAX`) with a sequence-to-sequence
  LSTM. The Fourier notebook explores the frequency content of the flux
  signal to inform the Phase 2 design.

## Folder contents

| Notebook | Purpose |
| --- | --- |
| `phase-1-data-collection-swan-sf.ipynb` | Scans SWAN-SF partitions 1–5, builds 15-step windows for flaring / non-flaring regions, cleans them and writes the Phase 1 training CSV. |
| `swan-sf-data-processing.ipynb` | Same content as the collection notebook (duplicate upload). |
| `phase-1-model.ipynb` | Phase 1 classifier: window statistics + XGBoost, evaluation, feature importance (gain + SHAP) and feature-ablation study. |
| `fourier-transform-for-phase-2.ipynb` | Frequency analysis of the X-ray flux of a single active region (FFT, low-pass filtering, spectrogram / STFT). |
| `phase-2-model.ipynb` | Phase 2 seq2seq LSTM (teacher forcing + attention) that forecasts `XR_MAX` after a flare onset; includes cross-validation, grid search, autoregressive inference and trajectory plots. |

## Data

### Source
SWAN-SF partitions 1–5 (Kaggle datasets `partition{1..5}-swanrafal-et-al`).
Each active region is a tab-separated CSV (e.g. `partition1/753.csv`,
1716 rows × 55 columns) sampled every 12 minutes. Columns:

* `Timestamp`
* 24 SHARP magnetic parameters: `TOTUSJH`, `TOTBSQ`, `TOTPOT`, `TOTUSJZ`,
  `ABSNJZH`, `SAVNCPP`, `USFLUX`, `TOTFZ`, `MEANPOT`, `EPSZ`, `MEANSHR`,
  `SHRGT45`, `MEANGAM`, `MEANGBT`, `MEANGBZ`, `MEANGBH`, `MEANJZH`, `TOTFY`,
  `MEANJZD`, `MEANALP`, `TOTFX`, `EPSY`, `EPSX`, `R_VALUE`
* Position / geometry: `CRVAL1`, `CRLN_OBS`, `CRLT_OBS`, `CRVAL2`, `HC_ANGLE`,
  `SPEI`, `LAT_MIN`, `LON_MIN`, `LAT_MAX`, `LON_MAX`, `QUALITY`
* Flare labels & locations for B / C / M / X classes
  (`MFLARE`, `MFLARE_LABEL`, `MFLARE_LOC`, …)
* GOES flux: `XR_MAX` (1–8 Å flux), `XR_QUAL` (quality flag), `IS_TMFI`

### Phase 1 dataset construction (`phase-1-data-collection-swan-sf.ipynb`)

Window sizes: `L = H = 3 h / 12 min = 15` steps.

1. **Find flaring regions** – files with any `MFLARE > 0` or `XFLARE > 0`
   across partitions 1–5: **191 files** (avg. 307.7 h, min 62.8 h,
   max 425.2 h of data each). Saved as `m_x_files_p1to5.csv`.
2. **Flare windows** – for every flare row, take the 15 steps immediately
   before it as `input`; label `2` for X flares (47 windows) and `1` for
   M flares (703 windows).
3. **Extra flare windows** – additional windows starting from 30 to 15 steps
   before each flare (stride 2): 376 X and 5539 M windows
   → 6665 flaring samples in total.
4. **Non-flaring regions** – files with no M/X flares: **3907 files**
   (`nonflares_files_p1to5.csv`). Three random, non-overlapping 15-step
   windows per file, label `0` → 11 602 windows.
5. **Cleaning** (`flareinstancesprocessing`) – drop any window that
   contains a NaN, any step with `IS_TMFI != 1`, or any step with
   `XR_QUAL < 6`.
6. **Result** – 7976 windows: 4913 non-flaring, 2882 M, 181 X.
   Saved as `cleanedfulldata_withrand_25COL.csv` with columns
   * `input` – JSON array of shape `(15, 27)`
     (24 SHARP parameters + `XR_MAX`, `XR_QUAL`, `IS_TMFI`),
   * `output` – `0` / `1` / `2`.

Nine features can be negative (`TOTFZ`, `EPSZ`, `MEANJZH`, `TOTFY`,
`MEANJZD`, `MEANALP`, `TOTFX`, `EPSY`, `EPSX`); see *Known caveats*.

The notebook also contains a `datvisualization` helper (mean ± std of a
feature per time step, flare vs. non-flare) and literature notes on useful
predictors (e.g. `SAVNCPP`, `TOTUSJH`, unsigned flux, current helicity,
twist parameters, PIL-based measures, historical flare rate).

An earlier 8-feature variant of the dataset
(`USFLUX, TOTUSJH, TOTBSQ, MEANPOT, R_VALUE, XR_MAX, XR_QUAL, IS_TMFI`,
8030 rows; `cleanedfullNONNANdata_withrand_FULLCOL.csv`) is also produced and
is the column layout used by Phase 2.

### Phase 2 dataset
`mflareinstances_FULLCOL.csv` (`/kaggle/input/mflareinstances-6col-cleaned/`):
311 M-flare instances with `input` of shape `(15, 8)` (8-feature layout
above) and `output` of shape `(5, 1)` – the five `XR_MAX` values following the
window. Only `XR_MAX` (feature index 5) is used as model input. The code that
generates this file is **not** in this folder.

## Phase 1 – flare classification (`phase-1-model.ipynb`)

### Pre-processing
1. Drop the last two features (`XR_QUAL`, `IS_TMFI`) → `(N, 15, 25)`.
2. Shuffle, then split 60 % / 40 % into train / validation
   (4785 / 3191 samples).
3. Transform features with `log1p(clip(x, 0, None))`, fit a `StandardScaler`
   on the training rows only and apply it to both splits.
4. Labels stay as integers.

| Split | No flare | M flare | X flare |
| --- | --- | --- | --- |
| Train | 2930 | 1740 | 115 |
| Validation | 1983 | 1142 | 66 |

### Model
Each `(15, 25)` window is collapsed into 125 features
(`mean`, `std`, `last`, `min`, `max` over time for each feature) and fed to

```python
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
              objective="multi:softprob", num_class=3,
              eval_metric="mlogloss", random_state=42)
