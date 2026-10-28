# Solar Flare Forecasting with EfficientNet

This repository contains the development and training pipeline for a deep learning model designed to forecast solar flares using magnetogram data. The model leverages the **EfficientNetV2B3** architecture and is optimized for training on TPUs (Tensor Processing Units) in environments like Kaggle.

## Project Overview

The core of this project is a Jupyter Notebook (`solarflareefficientnetmodel.ipynb`) that defines a custom data pipeline for astrophysical `.fits` files and trains an image classification model to predict solar flare occurrences.

### Key Features
* **TPU Acceleration:** Utilizes `tf.distribute.TPUStrategy` for distributed, high-performance training.
* **Custom FITS Data Generator:** Implements a custom Keras Sequence (`FitsDataGen`, inspired by Boucheron 2023) to read FITS files, scale intensities properly relative to the global dataset extremums, and apply real-time data augmentation.
* **Advanced Data Augmentation:** Includes random horizontal/vertical flips and random rotations (up to 10 degrees with nearest-neighbor fill mode) to improve model generalization on magnetograms.
* **Domain-Specific Metrics:** Because solar flares are rare events, standard accuracy is insufficient. The model tracks custom metrics tailored for space weather forecasting, including:
  * **TSS** (True Skill Statistic)
  * **HSS** (Heidke Skill Score)
  * **TPR** (True Positive Rate) & **TNR** (True Negative Rate)
  * **F1 Score**
* **Class Weighting:** Applies custom class weights during training to handle dataset imbalances between flare and non-flare events.

## Model Architecture
* **Base Model:** `EfficientNetV2B3` (pre-trained on ImageNet).
* **Input Shape:** $300 \times 300 \times 3$
* **Custom Top Layers:** 
  * `GlobalAveragePooling2D`
  * `Dropout` (rate: 0.5) to prevent overfitting.
  * `Dense` output layer with 2 units (Softmax activation) for binary classification.

## Dataset
The model trains on FITS magnetogram data representing Solar Active Regions (ARs: 2284-2731). The paths in the notebook are currently configured for Kaggle directories:
* `/kaggle/input/datasetsmagnetogram/` (Train/Test/Val splits)
* `/kaggle/input/solar-flare-medium-dataset/Lat60_Lon60_Nans0/` (Raw FITS images)

## Requirements
To run the notebook, you will need a TPU-enabled environment and the following libraries:
* `tensorflow` / `keras`
* `astropy` (for reading `.fits` files)
* `scikit-image` (`skimage` for image resizing)
* `pandas` & `numpy`

You can install the required specific dependencies via:
```bash
pip install astropy pandas numpy scikit-image
