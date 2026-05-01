# TinyML HAR Package

Project: Energy-Efficient Human Activity Recognition on the Edge

Team:

- Lee Ting Sen, B00103724
- Khalifa Alshamsi, B00078654
- Vineetha Addanki, G00111196

> **Important for Milestone 2 grading:**  
> This file is the general project README. Please also read [`README_MILESTONE2.md`](README_MILESTONE2.md), which is organized around the Milestone 2 handout and explains D2, D3, D4, preprocessing, dataset download, baseline scripts, held-out test results, and why `x_train/y_train/x_val/x_test` are generated in memory. Please also refer to [`tinyml_milestone2.pdf`](tinyml_milestone2.pdf), the Milestone 2 handout PDF, as the main grading reference.

> **Milestone 3 update:**  
> M3 laptop-side deployment prep is summarized in [`README_MILESTONE3.md`](README_MILESTONE3.md), with the report draft under [`docs/milestone3_report_draft.md`](docs/milestone3_report_draft.md) and the Arduino sketch under [`arduino/tinyml_har_m3/`](arduino/tinyml_har_m3/). Hardware-only metrics, video evidence, and live confusion matrices are still pending because they must be measured on the Arduino.

This repository contains a TinyML human activity recognition implementation package targeting Arduino Nano 33 BLE Sense for edge deployment. The current benchmark track uses public/open datasets.

Current status: The package is runnable with TensorFlow/Keras training code, generated UCI HAR held-out metrics, confusion matrices, dataset inspection outputs, INT8 TFLite conversion, and an Arduino Nano 33 BLE Sense sketch scaffold. The paper reproduction run included here is a bounded CPU reproduction artifact; the closer 64-epoch reproduction command is documented below.

## Phase Framework

| Phase | Title | Objective | Main Change | Evaluation Focus |
|---|---|---|---|---|
| 1 | Rebuilding the Paper Baseline | Reproduce the reference HGAR pipeline as closely as feasible to establish a strong upper-bound baseline. | No intentional simplification. | Accuracy |
| 2 | Lightweight Model Screening | Compare a small set of TinyML-friendly models under the same data pipeline and sensor setup, then select one winner. | Architecture only. | Accuracy + Latency |
| 3 | Quantization Strategy Selection | Take the selected Phase 2 model and compare deployable quantization strategies before Arduino flashing. | Quantization only. | Accuracy + Latency + Size |

Current round scope: Phase 1, Phase 2, and the laptop-side Phase 3 quantization experiment are implemented in this repository. Hardware-only Arduino evidence is still pending because it must be measured on the physical board.

Phase 1, Rebuilding the Paper Baseline, is implemented under `outputs/reproduction/` with five level-0 hybrid learners (ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, CNN-LSTM) and XGBoost as the level-1 meta-learner.

Phase 2, Lightweight Model Screening, is intentionally narrow in scope to avoid over-expanding the study. The shortlist is depthwise separable CNN, compact CNN-GRU, and small TCN-style architectures under the same pipeline.

Phase 3 replaces the earlier accelerometer-only sensor-ablation idea. That branch is removed from the current M3 plan so the work stays aligned with the rubric: choose the best Phase 2 model, quantize it properly, then measure real Arduino performance.

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
arduino/
  tinyml_har_m3/       # M3 live IMU inference sketch and generated model header
docs/
  tables/
notebooks/
  phase2_model_screening_lab.ipynb
src/
  config.py
  data/
  models/
    reproduction/
    lightweight/
  deployment/
  training/
  utils/
outputs/
  reproduction/
  lightweight/
  deployment/
    quantization_experiments/
  arduino_collectdata/
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

All model-architecture or feature experiments belong in Phase 2. That includes focal loss, posture/gravity features, compact recurrent models, or TCN-style candidates. Phase 3 should not add new sensor-ablation branches; it should quantize the selected Phase 2 winner.

The M3 posture/gravity feature probe is recorded as a Phase 2 error-analysis experiment:

```powershell
python -m src.training.evaluate_posture_gravity_feature --device cpu
```

Result: a mean-acceleration gravity feature did not improve the M2 `SITTING` <-> `STANDING` confusion. The DS-CNN baseline remains better (`0.9173` macro F1, 184 static confusions) than the gravity-rescue variant (`0.9117` macro F1, 201 static confusions), so the feature is documented but not adopted for M3 deployment.

## Phase 3: Quantization Strategy Selection

Run the quantization comparison after the Phase 2 winner has been selected:

```powershell
python -m src.deployment.quantization_experiment --representative-samples 512 --qat-trigger-drop 0.01
```

The experiment compares:

- full integer INT8 PTQ with random representative train windows
- class-balanced full integer INT8 PTQ
- INT8 QAT only if all PTQ variants drop macro F1 by more than `0.01`

Current Phase 3 result:

| Variant | Accuracy | Macro F1 | Size | Host TFLite Mean Latency | Selected |
|---|---:|---:|---:|---:|---|
| Full integer INT8 PTQ | 0.9138 | 0.9141 | 10,288 bytes | 0.0391 ms | No |
| Class-balanced full integer INT8 PTQ | 0.9145 | 0.9147 | 10,288 bytes | 0.0430 ms | Yes |

QAT decision: not run in this pass because the selected PTQ variant is within `0.0026` macro F1 of the Phase 2 FP32 baseline, which is below the `0.01` trigger threshold.

Outputs:

- `src/deployment/quantization_experiment.py`
- `outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.csv`
- `outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.json`
- `outputs/deployment/quantization_experiments/models/lightweight_tiny_cnn_full_integer_int8_ptq.tflite`
- `outputs/deployment/quantization_experiments/models/lightweight_tiny_cnn_class_balanced_full_integer_int8_ptq.tflite`
- selected deployment artifact copied to `outputs/deployment/models/lightweight_tiny_cnn_int8.tflite`
- selected Arduino header regenerated at `arduino/tinyml_har_m3/model_data.h`

## Milestone 3 Deployment Prep

Audit the teammate's Arduino CSV collection:

```powershell
python -m src.data.arduino_collectdata
```

Replay the collected Arduino CSV test data through the INT8 model on the laptop:

```powershell
python -m src.deployment.evaluate_arduino_replay
```

Outputs:

- `docs/tables/arduino_collectdata_class_counts.csv`
- `docs/tables/arduino_collectdata_file_quality.csv`
- `outputs/arduino_collectdata/arduino_collectdata_summary.json`
- `outputs/arduino_collectdata/metrics/arduino_replay_int8_metrics.json`
- `data/processed/arduino_collectdata_v1_windows_50hz.npz`

The audit found the CSV data is labeled and usable as the Track B real Arduino test dataset, but the observed cadence is about 37-38 Hz rather than 50 Hz and the current data volume is below 100 windows per class after 50 Hz resampling. Offline replay of the selected INT8 model on this real Arduino dataset gives `0.1789` accuracy and `0.0582` macro F1, which indicates a severe domain gap and likely axis/orientation/timing mismatch. This replay result is not live on-device accuracy. The next self-collected dataset round should supersede v1 before final Arduino-live accuracy is claimed.

Regenerate the selected full-int8 TFLite model and Arduino C header:

```powershell
python -m src.deployment.quantization_experiment --representative-samples 512 --qat-trigger-drop 0.01
```

Outputs:

- `outputs/deployment/models/lightweight_tiny_cnn_int8.tflite`
- `outputs/deployment/metrics/lightweight_tiny_cnn_int8_metrics.json`
- `outputs/deployment/metrics/m2_vs_m3_int8_comparison.csv`
- `arduino/tinyml_har_m3/model_data.h`

Current selected INT8 offline result on the UCI HAR held-out test set:

- Quantization: class-balanced full integer INT8 PTQ
- Accuracy: 0.9145
- Macro F1: 0.9147
- INT8 TFLite size: 10,288 bytes
- Planned tensor arena: 61,440 bytes

Arduino sketch:

```text
arduino/tinyml_har_m3/tinyml_har_m3.ino
```

Open the sketch in Arduino IDE, select Arduino Nano 33 BLE Sense, install `Arduino_LSM9DS1` and Arduino TensorFlow Lite / TensorFlow Lite Micro, upload, and monitor Serial at 115200 baud. The teammate with the board must still provide the M3 hardware evidence: 2-minute stable run, video/serial log, average latency over at least 50 invokes, compile flash/RAM summary, live 20-trials/class confusion matrix, and 10-trials/class robustness test.

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

- `README_MILESTONE2.md`: handout-aligned Milestone 2 explanation and D2/D3/D4 mapping.
- `README_MILESTONE3.md`: M3 deployment prep, commands, current artifacts, and hardware tasks.
- `docs/milestone3_report_draft.md`: M3 report draft with hardware-only fields marked pending.
- `docs/milestone3_teammate_message.md`: message/checklist for the teammate holding the Arduino.
- `docs/tables/`: dataset inspection tables and class-count summaries.
- `outputs/lightweight/` and `outputs/reproduction/`: generated models, metrics, logs, and figures.

## Arduino / PlatformIO Starter

The original Wokwi/PlatformIO firmware remains available for embedded bring-up while waiting for Nano 33 BLE Sense hardware.

```powershell
pio run
```

Main firmware file:

- `src/main.cpp`

Next-round plan: collect a clean Arduino v2 dataset at verified 50 Hz, then execute the live on-device TFLM evaluation on Arduino Nano 33 BLE Sense using the selected class-balanced INT8 PTQ model.
