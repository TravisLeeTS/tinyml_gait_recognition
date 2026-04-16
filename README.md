# TinyML HAR Package

Project: Energy-Efficient Human Activity Recognition on the Edge

Team:

- Lee Ting Sen, B00103724
- Khalifa Alshamsi, B00078654
- Vineetha Addanki, G00111196

This repository contains a TinyML human activity recognition implementation package targeting Arduino Nano 33 BLE Sense for edge deployment. The current benchmark track uses public/open datasets.

Current status: The package is runnable with TensorFlow/Keras training code, generated UCI HAR held-out metrics, confusion matrices, dataset inspection outputs, and a PDF-ready report draft. The paper reproduction run included here is a bounded CPU reproduction artifact; the closer 64-epoch reproduction command is documented below.

## Phase Framework

| Phase | Title | Objective | Main Change | Evaluation Focus |
|---|---|---|---|---|
| 1 | Rebuilding the Paper Baseline | Reproduce the reference HGAR pipeline as closely as feasible to establish a strong upper-bound baseline. | No intentional simplification. | Accuracy |
| 2 | Lightweight Model Screening | Compare a small set of TinyML-friendly models under the same data pipeline and sensor setup, then select one winner. | Architecture only. | Accuracy + Latency |
| 3 (Optional) | Eco-Mode Sensor Ablation | Test whether accelerometer-only sensing is worthwhile after a lightweight model is chosen. | Sensor configuration only. | Energy + Accuracy |
| 4 | Quantized Edge Deployment | Deploy the selected final model on Arduino using TFLM and evaluate real edge performance. | INT8 quantization and on-device implementation. | Energy + Accuracy + Latency |

Current round scope: Phase 1 and Phase 2 are implemented in this repository. Phase 3 and Phase 4 are planned for the next round.

Phase 1, Rebuilding the Paper Baseline, is implemented under `outputs/reproduction/` with five level-0 hybrid learners (ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, CNN-LSTM) and XGBoost as the level-1 meta-learner.

Phase 2, Lightweight Model Screening, is intentionally narrow in scope to avoid over-expanding the study. The shortlist is depthwise separable CNN, compact CNN-GRU, and small TCN-style architectures under the same pipeline. 

Phase 4 is planned to use post-training quantization (INT8) as the main compression strategy for deployment.

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
  phase1_phase2_report.md
  phase1_phase2_report.docx
  phase1_phase2_report.pdf
  figures/
  tables/
notebooks/
  phase2_model_screening_lab.ipynb
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
python -m ipykernel install --user --name tinyml-har --display-name "TinyML HAR TensorFlow"
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

## Phase 2: Lightweight Model Screening

Reportable run used for the current artifact:

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

Current selected Phase 2 winner result on UCI HAR held-out test set:

- Accuracy: 0.9169
- Macro F1: 0.9173
- Test windows: 2,947
- Keras parameters: 1,922
- TFLite size: 13,460 bytes
- Host CPU Keras predict latency proxy: mean 62.65 ms over 20 runs. This is not an Arduino latency claim.

## Phase 1: Rebuilding the Paper Baseline

Bounded run used for the current artifact:

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

## Phase 2 Notebook Lab

Use the notebook when multiple collaborators want to compare architecture ideas without touching preprocessing code:

```text
notebooks/phase2_model_screening_lab.ipynb
```

Launch it with:

```powershell
jupyter lab notebooks/phase2_model_screening_lab.ipynb
```

The notebook imports `src.training.experiment_lab.run_keras_architecture`, so model designers only provide a Keras model-builder function. The helper keeps the same UCI HAR loading, subject-aware validation split, train-only normalization, held-out test metrics, confusion matrix, TFLite size, and host latency measurement.

## Project Documentation

- `docs/phase1_phase2_report.md`: editable Phase 1-2 project report draft in Markdown.
- `docs/phase1_phase2_report.docx`: editable Word version of the Phase 1-2 report.
- `docs/phase1_phase2_report.pdf`: exported PDF version of the Phase 1-2 report.
- `docs/tables/`: dataset inspection tables and class-count summaries.
- `outputs/lightweight/` and `outputs/reproduction/`: generated models, metrics, logs, and figures.

## Arduino / PlatformIO Starter

The original Wokwi/PlatformIO firmware remains available for embedded bring-up while waiting for Nano 33 BLE Sense hardware.

```powershell
pio run
```

Main firmware file:

- `src/main.cpp`

Next-round plan: run Phase 3 optional eco-mode sensor ablation (accelerometer-only check), then execute Phase 4 quantized edge deployment with INT8 post-training quantization and on-device TFLM evaluation on Arduino Nano 33 BLE Sense.
