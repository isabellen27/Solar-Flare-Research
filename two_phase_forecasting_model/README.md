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
| --- | ---: | ---: | ---: |
| Train | 2930 | 1740 | 115 |
| Validation | 1983 | 1142 | 66 |

### Model

Each `(15, 25)` window is collapsed into 125 features
(`mean`, `std`, `last`, `min`, `max` over time for each feature) and fed to

```python
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
              objective="multi:softprob", num_class=3,
              eval_metric="mlogloss", random_state=42)
```

### Results (validation)

* Accuracy **0.979**; macro precision 0.950, macro recall 0.918, macro F1 0.93

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| No flare | 0.99 | 0.98 | 0.99 | 1983 |
| M flare | 0.96 | 0.98 | 0.97 | 1142 |
| X flare | 0.90 | 0.79 | 0.84 | 66 |

```text
Confusion matrix (rows = true, cols = predicted)
[[1950   33    0]
 [  13 1123    6]
 [   0   14   52]]
```

### Interpretability & ablation

* **Gain importance** – top features: `mean_TOTFX`, `mean_MEANSHR`,
  `mean_TOTUSJH`, `mean_MEANGBH`, `last_TOTFX`, `min_MEANALP`, `min_MEANSHR`,
  `last_TOTUSJH`, `mean_TOTBSQ`, `last_MEANALP`.
* **SHAP** – `TreeExplainer` summary plots per class and a global ranking by
  mean |SHAP| averaged over classes.
* **Ablation** – retrain on the top-k features
  (k ∈ {5, 10, 15, 20, 30, 50, 75, 100, 125}); one-vs-rest ROC AUC / PR AUC per
  class, shown as ROC curves (top 5/10/20) and a 3-D surface over (k, class).
* Artifacts: `xgb_model.pkl`, sample windows `X_cls.npy`, `y_cls.npy`.

## Phase 2 – frequency analysis (`fourier-transform-for-phase-2.ipynb`)

Worked example on AR 753 (partition 1):

1. Extract `XR_MAX`, apply `log1p` + `StandardScaler`.
2. FFT with `dt = 720 s`; power spectrum `|X(f)|²`.
3. **Low-pass filter**: zero all bins with period shorter than `T = 7200 s`,
   inverse FFT, and overlay the smoothed series on the raw one.
4. Stem plot of the FFT magnitude spectrum.
5. Time–frequency views: `scipy.signal.spectrogram` (Hann, `nperseg=128`,
   50 % overlap, per-column normalised dB) and `scipy.signal.stft`
   (`nperseg=512`), `fs = 1/720 Hz`.

## Phase 2 – flux forecasting (`phase-2-model.ipynb`)

### Pre-processing

1. Keep only `XR_MAX` → `X: (311, 15, 1)`, `Y: (311, 5, 1)`.
2. Remove the 4 samples whose targets contain the `-99999` fill value → 307.
3. Shuffle; split 60 % train (184), then the remainder 75 % / 25 % into
   test (92) and validation (31).
4. `log1p` + `StandardScaler` for both inputs and targets (fit on train only).
   Target range in the training set ≈ 9.3 × 10⁻⁵ W m⁻².

### Model (`build_lstm_teacher_forcing`)

Encoder–decoder LSTM trained with teacher forcing:

```text
encoder_inputs (15, 1)
  → GaussianNoise(σ = 0.02)                      # training only
  → LSTM(units, return_sequences, return_state)
  → Dropout
decoder_inputs (5, 1)  (previous ground-truth step, 0 at t = 0)
  → LSTM(units) initialised with encoder states
  → AdditiveAttention(query = decoder outputs, value = encoder outputs)
  → Concatenate → Dropout → TimeDistributed(Dense(1))
loss = Huber, optimizer = Adam, metric = MAE
```

A simpler baseline (`build_lstm_model`: LSTM → RepeatVector → LSTM → Dense)
is also defined. Two helper plots visualise the effect of the Gaussian input
noise (σ ∈ {0, 0.005, 0.01, 0.02}) on one encoder window.

### Inference

`make_inference_models` splits the trained network into an encoder and a
one-step decoder; `predict_autoregressive` feeds each prediction back as the
next decoder input. The decoder is seeded with the **last raw `XR_MAX`
value** of the input window, converted to the target's log/scaled space
(`start_token="last_x_raw"`).

### Model selection

* `cross_val_score_lstm`: `TimeSeriesSplit(n_splits=5)`, per-fold scalers,
  `EarlyStopping(patience=8, restore_best_weights=True)`,
  `ReduceLROnPlateau(factor=0.5, patience=4)`, RMSE computed in original units
  after autoregressive prediction.
* Grid: `lstm_units ∈ {32, 64, 128}`, `dropout ∈ {0.2, 0.3, 0.4}`,
  `lr ∈ {1e-3, 5e-4, 1e-4}`, `batch_size ∈ {8, 16, 32, 64}`
  (grid-search cell is kept as markdown; run it to reproduce).
* Final configuration: **units = 32, dropout = 0.3, lr = 1e-3, batch = 16**,
  up to 100 epochs. Training stopped at epoch 68 with best
  `val_loss ≈ 0.259` / `val_mae ≈ 0.48` (Huber / MAE in scaled log space).

### Evaluation

`plot_flare_trajectory` plots, for each test instance, the input `XR_MAX`
(t = −15 … −1), the true and predicted flux (t = 0 … 4) in original units on a
log axis, with a vertical line at flare onset. Predictions follow the decay of
the flux after onset but tend to under-predict the peak for the strongest
events (e.g. true 5.5 × 10⁻⁵ vs. predicted 1.4 × 10⁻⁵ W m⁻²).

## Environment

Developed on Kaggle (`/kaggle/input/...` paths). Dependencies: `numpy`,
`pandas`, `scikit-learn`, `xgboost`, `shap`, `tensorflow` / `keras`, `scipy`,
`matplotlib`, `joblib`.

## Known caveats

* **Leakage risk in Phase 1.** `X`/`Y` are shuffled with
  `np.random.permutation` before the split, even though the next cell is
  commented as a "chronological split". Overlapping windows from the same
  active region (and the stride-2 augmented windows) can land in both train
  and validation, which likely inflates the reported scores.
* **Negative features are clipped.** The `log1p(clip(x, 0, None))` transform
  zeroes the nine signed SHARP parameters (`TOTFZ`, `EPSZ`, `MEANJZH`,
  `TOTFY`, `MEANJZD`, `MEANALP`, `TOTFX`, `EPSY`, `EPSX`) whenever they are
  negative, discarding sign information.
* **Class imbalance.** X flares are rare (115 train / 66 validation);
  X-flare recall (0.79) is the weakest metric.
* **Phase 2 label.** The section header reads "X CLASS FLARE TRAINING" but the
  data file is `mflareinstances_FULLCOL.csv` (M-flare instances).
* **Phase 2 data provenance.** The generation of the `(15 → 5)` regression
  dataset is not included in this folder.
* The feature-dropping cells (`X = X[:, :, :-2]`, `X = X[:, :, 5:6]`) must be
  run exactly once per kernel session.
