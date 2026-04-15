# TinyML HAR Milestone 2 Package

Project: Energy-Efficient Human Activity Recognition on the Edge

Team:

- Lee Ting Sen, B00103724
- Khalifa Alshamsi, B00078654
- Vineetha Addanki, G00111196

This repository now contains the Milestone 2 implementation package for a TinyML human activity recognition project targeting Arduino Nano 33 BLE Sense in M3. The current M2 benchmark is Track B: public/open datasets.

Current status: Milestone 2 implementation package is runnable with TensorFlow/Keras training code, generated UCI HAR held-out metrics, confusion matrices, dataset inspection outputs, and a PDF-ready report draft. The paper reproduction run included here is a bounded CPU reproduction artifact; the closer 64-epoch reproduction command is documented below.

## What Is Implemented

Two baselines are intentionally kept separate:

- Paper Reproduction Baseline: offline reference baseline under `outputs/reproduction/`. It implements the Sensors Journal stacking design in TensorFlow/Keras with five level-0 hybrid learners: ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, CNN-LSTM, plus XGBoost as the level-1 meta-learner. This is not intended for direct Arduino deployment.
- Lightweight TinyML-Oriented Baseline: compact depthwise-separable 1D CNN under `outputs/lightweight/`. This is the M3 deployment path for later quantization and Arduino testing.

Primary dataset:

- UCI HAR v1.0, using prewindowed total-acceleration plus gyroscope inertial signal files with input shape `[128, 6]`. Source: `https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`.

Secondary dataset:

- Fordham WISDM Activity Prediction v1.1, used for inspection, timestamp regularity analysis, and domain-gap planning. Its label taxonomy differs from UCI HAR, so labels are not force-merged. Source: `https://www.cis.fordham.edu/wisdm/dataset.php`.

## Repository Layout

```text
data/
  raw/                 # downloaded archives/extracted public datasets, ignored by git
  interim/
  processed/
docs/
  milestone2_report.md
  reproduction_notes.md
  figures/
  tables/
notebooks/
src/
  config.py
  data/
  models/
    reproduction/
    lightweight/
  training/
  utils/
outputs/
  reproduction/
  lightweight/
  datacards/
```

## Setup

Use Python 3.12 in the current environment, then install pinned dependencies:

```powershell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name tinyml-m2 --display-name "TinyML M2 TensorFlow"
```

TensorFlow 2.18.0 and Keras 3.8.0 are pinned for model training and TensorFlow Lite conversion. TensorFlow runs on CPU in this Windows environment.

## Dataset Inspection

Run:

```powershell
python -m src.data.inspect_datasets
```

This downloads or reuses:

- UCI HAR from the official UCI URL when available, with a public mirror fallback if UCI returns HTTP 502.
- WISDM classic v1.1 from Fordham WISDM Lab.

Outputs:

- `outputs/datacards/dataset_inspection_summary.json`
- `outputs/datacards/uci_har_inspection.json`
- `outputs/datacards/wisdm_classic_inspection.json`
- `docs/tables/uci_har_class_counts.csv`
- `docs/tables/wisdm_classic_raw_counts.csv`

## Train Lightweight Baseline

Reportable run used for the current M2 artifact:

```powershell
python -m src.training.train_lightweight --epochs 40 --patience 8 --device cpu
```

Outputs:

- `outputs/lightweight/models/lightweight_tiny_cnn.keras`
- `outputs/lightweight/models/lightweight_tiny_cnn.tflite`
- `outputs/lightweight/models/lightweight_tiny_cnn_dynamic_range.tflite`
- `outputs/lightweight/metrics/lightweight_tiny_cnn_metrics.json`
- `outputs/lightweight/metrics/lightweight_tiny_cnn_per_class_metrics.csv`
- `outputs/lightweight/metrics/lightweight_tiny_cnn_confusion_matrix.csv`
- `outputs/lightweight/figures/lightweight_tiny_cnn_confusion_matrix.png`
- `outputs/lightweight/logs/lightweight_training_history.csv`

Current UCI HAR held-out test result:

- Accuracy: 0.9169
- Macro F1: 0.9173
- Test windows: 2,947
- Keras parameters: 1,922
- TFLite size: 13,460 bytes
- Host CPU Keras predict latency proxy: mean 62.65 ms over 20 runs. This is not an Arduino latency claim.

## Train Paper Reproduction Baseline

Bounded run used for the current M2 artifact:

```powershell
python -m src.training.train_reproduction --epochs 20 --patience 5 --device cpu --normalization standard --fast-xgb-grid
```

Closer-to-paper run:

```powershell
python -m src.training.train_reproduction --epochs 64 --patience 12 --device cpu
```

Outputs:

- base Keras model checkpoints in `outputs/reproduction/models/`
- base model metrics in `outputs/reproduction/metrics/`
- stacked XGBoost model in `outputs/reproduction/models/xgboost_meta_learner.joblib`
- final stacking metrics in `outputs/reproduction/metrics/paper_reproduction_stacking_metrics.json`
- confusion matrices in `outputs/reproduction/figures/`

Current bounded UCI HAR held-out test result:

- Accuracy: 0.9135
- Macro F1: 0.9128
- Test windows: 2,947

This bounded run is a working reproduction artifact. For the closest paper-style run, use the 64-epoch command and the full XGBoost grid.

## Notebook Model Lab

Use the notebook when multiple collaborators want to compare architecture ideas without touching preprocessing code:

```text
notebooks/m2_model_lab.ipynb
```

Launch it with:

```powershell
jupyter lab notebooks/m2_model_lab.ipynb
```

The notebook imports `src.training.experiment_lab.run_keras_architecture`, so model designers only provide a Keras model-builder function. The helper keeps the same UCI HAR loading, subject-aware validation split, train-only normalization, held-out test metrics, confusion matrix, TFLite size, and host latency measurement.

## Report Files

- `docs/milestone2_report.md`: PDF-ready M2 report draft.
- `docs/reproduction_notes.md`: exact vs approximate reproduction choices and data-access notes.

## Arduino / PlatformIO Starter

The original Wokwi/PlatformIO firmware remains available for embedded bring-up while waiting for Nano 33 BLE Sense hardware.

```powershell
pio run
```

Main firmware file:

- `src/main.cpp`

The M3 plan is to collect real Arduino Nano 33 BLE Sense accelerometer and gyroscope data at 50 Hz, segment into 2.56 s windows with 50% overlap, and evaluate the lightweight model path under quantization and on-device latency constraints.
