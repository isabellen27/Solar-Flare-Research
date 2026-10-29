
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
SWAN-SF partit
