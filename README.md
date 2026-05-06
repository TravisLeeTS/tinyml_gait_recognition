# TinyML HAR Package

Project: Energy-Efficient Human Activity Recognition on the Edge

Team:

- Lee Ting Sen, B00103724
- Khalifa Alshamsi, B00078654
- Vineetha Addanki, G00111196

> **Important for Milestone 3 grading:**  
> Use [`tinyml_milestone3.pdf`](tinyml_milestone3.pdf) as the main M3 report. M3 deployment prep and returned Arduino evidence are summarized in [`README_MILESTONE3.md`](README_MILESTONE3.md), with the Arduino sketch under [`arduino/tinyml_har_m3/`](arduino/tinyml_har_m3/). Hardware metrics, proof videos, and live Serial confusion matrices have been integrated; only the returned videos/logs still need to be attached with the submission.

> **Milestone 2 background:**  
> [`README_MILESTONE2.md`](README_MILESTONE2.md) remains available for the previous milestone's D2/D3/D4 details, preprocessing, dataset download, baseline scripts, held-out test results, and explanation of in-memory `x_train/y_train/x_val/x_test` arrays.

This repository contains a TinyML human activity recognition implementation package targeting Arduino Nano 33 BLE Sense for edge deployment. The current benchmark track uses public/open datasets.

Current status: Milestone 3 submitted. On-device inference working. The package is runnable with TensorFlow/Keras training code, generated UCI HAR held-out metrics, confusion matrices, dataset inspection outputs, target-domain adaptation experiments, and an Arduino Nano 33 BLE Sense M3 candidate deployment build. The candidate INT8 model has returned Arduino hardware evidence: compile flash/RAM, boot metadata, 34.113 ms average `Invoke()` latency over 50 inferences, 2-minute stability video, and live Serial trial scores.

## Phase Framework

| Phase | Title | Objective | Main Change | Evaluation Focus |
|---|---|---|---|---|
| 1 | Rebuilding the Paper Baseline | Reproduce the reference HGAR pipeline as closely as feasible to establish a strong upper-bound baseline. | No intentional simplification. | Accuracy |
| 2 | Lightweight Model Screening | Compare a small set of TinyML-friendly models under the same data pipeline and sensor setup, then select one winner. | Architecture only. | Accuracy + Latency |
| 3 | Target-Domain Adaptation Then Quantization | Use UCI-HAR as source knowledge, adapt to Arduino V2 target data, validate on held-out right/left pocket data, then quantize the selected model. | Preprocessing/model transfer first; quantization second. | Arduino-domain accuracy + UCI retention + latency + size |

Current round scope: Phase 1, Phase 2, V2 domain-gap auditing, target-domain adaptation scripts, candidate quantization, Arduino evidence templates, and returned teammate hardware/live Serial evidence are implemented in this repository.

Phase 1, Rebuilding the Paper Baseline, is implemented under `outputs/reproduction/` with five level-0 hybrid learners (ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, CNN-LSTM) and XGBoost as the level-1 meta-learner.

Phase 2, Lightweight Model Screening, is intentionally narrow in scope to avoid over-expanding the study. The main deployable family is the depthwise-separable CNN, with focal-loss and gravity/posture-feature probes added for M3 error analysis.

Phase 3 replaces the earlier accelerometer-only sensor-ablation idea. The current M3 direction is: align Arduino preprocessing and target-domain model behavior first, then quantize the selected candidate for live evidence collection and measure real Arduino performance separately.

Primary dataset:

- UCI HAR v1.0, using prewindowed total-acceleration plus gyroscope inertial signal files with input shape `[128, 6]`. Source: `https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`.

Secondary dataset:

- Fordham WISDM Activity Prediction v1.1, used for inspection, timestamp regularity analysis, and domain-gap planning. Its label taxonomy differs from UCI HAR, so labels are not force-merged. Source: `https://www.cis.fordham.edu/wisdm/dataset.php`.

Arduino M3 datasets:

- `data/raw/arduino_collectdata_v2/`: corrected 50 Hz Arduino target-domain data used for adaptation and grouped validation.
- `data/raw/arduino_collectdata_v2_1_long_adaptation/`: four 5-minute adaptation captures for `WALKING`, `SITTING`, `STANDING`, and `LAYING`.
- `data/raw/arduino_live_validation/right_60s/`: held-out controlled raw sensor replay validation, not training.
- `data/raw/arduino_live_validation/left_30s/`: held-out robustness raw sensor replay validation, not training.
- Returned live Serial logs: final M3 hardware evidence only; do not train on them before reporting.

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

All model-architecture or feature experiments belong in Phase 2 or the M3 target-domain comparison. For M3, the V2 replay result shows that the UCI-only winner is not enough for the demo environment, so the selected V2.1 mixed 6-channel candidate is quantized as the evidence-collection build and scored from returned live Arduino Serial logs.

### M3 Offline Improvement Candidates

For the Milestone 3 Table 4 row `Offline test set (M3 improved)`, two DS-CNN variants were trained and evaluated on the same UCI-HAR held-out test split as the Phase 2 baseline. Phase 1 and the original Phase 2 baseline were not rerun because the main preprocessing path did not change.

```powershell
python -m src.training.train_m3_second_improvements --experiment focal_loss --epochs 40 --patience 8 --device cpu
python -m src.training.train_m3_second_improvements --experiment gravity_feature --epochs 40 --patience 8 --device cpu
python -m src.training.train_m3_second_improvements --experiment all --evaluate-existing --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

| Offline UCI-HAR Test Run | Input | Change | Accuracy | Macro F1 | Weighted F1 | FP32 TFLite Size | Host Keras Mean Latency | SITTING<->STANDING Confusions | Current Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Phase 2 DS-CNN baseline | 128 x 6 | Cross entropy | 0.9169 | 0.9173 | 0.9168 | 13,460 bytes | 62.65 ms | 184 | UCI reference only; fails V2 replay |
| M3 focal-loss DS-CNN | 128 x 6 | Sparse focal loss, gamma 2.0 | 0.9074 | 0.9071 | 0.9073 | 13,460 bytes | 46.88 ms | 182 | Reject |
| M3 gravity-feature DS-CNN | 128 x 10 | Add mean acceleration unit vector XYZ + magnitude | 0.9179 | 0.9173 | 0.9167 | 14,036 bytes | 49.55 ms | 173 | Useful probe; needs V2 validation |

Strict decision after V2: do not use the UCI-only Phase 2 DS-CNN as the deployment candidate. Focal loss is still rejected because it loses about 0.010 macro F1. The gravity-feature DS-CNN remains useful because it improves the static-posture failure mode, but any 10-channel model must be validated in the V2-like Arduino environment before it becomes the deployment candidate.

Latency note: the host Keras latency numbers above are laptop proxy measurements, not Arduino latency. They were gathered in different runs, so they are useful for sanity checking only; final deployable latency must come from the selected quantized model on Arduino.

Saved outputs:

- `src/training/train_m3_second_improvements.py`
- `outputs/lightweight/experiments/m3_second_improvements_summary.csv`
- `outputs/lightweight/experiments/m3_focal_loss_tiny_cnn/metrics/m3_focal_loss_tiny_cnn_metrics.json`
- `outputs/lightweight/experiments/m3_focal_loss_tiny_cnn/models/m3_focal_loss_tiny_cnn.tflite`
- `outputs/lightweight/experiments/m3_gravity_feature_tiny_cnn/metrics/m3_gravity_feature_tiny_cnn_metrics.json`
- `outputs/lightweight/experiments/m3_gravity_feature_tiny_cnn/models/m3_gravity_feature_tiny_cnn.tflite`

## Phase 3: M3 Arduino Deployment

Milestone 3 is organized as a deployment story rather than a single
quantization experiment:

1. Professor-advised improvements were tested first. Focal loss did not improve
   the overall DS-CNN enough to replace the baseline. The 10-feature
   gravity/posture probe reduced static-posture confusion on UCI-HAR, so it is
   useful evidence for error analysis and future work.
2. The UCI-only DS-CNN then failed on real Arduino V2 pocket data, producing
   `0.1864` accuracy and `0.0524` macro F1 on offline replay. This exposed a
   public-dataset-to-Arduino domain gap rather than a quantization bug.
3. More self-collected Arduino data was added: V2, V2.1 long adaptation,
   held-out right-pocket raw validation, held-out left-pocket raw validation,
   and final live Serial logs.
4. The final M3 deployment candidate is `m3_v2_1_mixed_6ch`, a mixed UCI +
   Arduino 6-channel DS-CNN quantized to INT8 and exported to the Nano 33 BLE
   Sense sketch.

Current selected deployment artifact:

```text
outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite
arduino/tinyml_har_m3/model_data.h
arduino/tinyml_har_m3/tinyml_har_m3.ino
```

Final M3 hardware and live evidence:

| Evidence | Result |
|---|---:|
| INT8 TFLite model size | 10,288 bytes / 10.05 KiB |
| Generated `model_data.h` source size | 65,933 bytes / 64.39 KiB |
| Tensor arena | 61,440 bytes / 60.00 KiB |
| Arduino flash usage | 177,504 bytes / 18% |
| Arduino dynamic memory | 111,144 bytes / 108.54 KiB / 42% |
| Average `Invoke()` latency | 34.113 ms over 50 windows |
| Right-pocket live Serial accuracy / macro F1 | 0.9040 / 0.9089 |
| Left-pocket robustness live Serial accuracy / macro F1 | 0.8901 / 0.8826 |

Quantization details for the handout: the selected candidate uses full-integer
INT8 post-training quantization. Representative calibration windows come only
from training/adaptation data, not UCI-HAR test, right/left validation, or live
Serial evidence. The selected FP32 candidate is 13,460 bytes / 13.14 KiB as
TFLite; the deployment INT8 `.tflite` is 10,288 bytes / 10.05 KiB. QAT was not
used because PTQ did not create a material macro-F1 drop on the selected
candidate.

Main unresolved limitation: `WALKING_UPSTAIRS` is still often predicted as
`WALKING_DOWNSTAIRS`. Static postures are strong in the returned live Serial
logs, but stair-direction robustness remains the main M4 improvement target.

Reproduce the final M3 retraining, INT8 packaging, and Arduino header
regeneration:

```powershell
python -m src.training.train_m3_retrained_with_v2_1 --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Score the returned live Serial logs:

```powershell
python -m src.reporting.score_live_serial_trials --input outputs\live_evidence\raw_teammate_return\7_control_trials --condition right_pocket_controlled --output-dir outputs\live_evidence
python -m src.reporting.score_live_serial_trials --input outputs\live_evidence\raw_teammate_return\8_robust_trials --condition left_pocket_robustness --output-dir outputs\live_evidence
```

See `README_MILESTONE3.md` for the full dataset register, commands, challenge
summary, and artifact index.

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
- `README_MILESTONE3.md`: M3 deployment prep, commands, current artifacts, and returned hardware/live evidence.
- `tinyml_milestone3.pdf`: final M3 PDF report with R1-R7 evidence integrated.
- `docs/m3_submission_checklist.md`: final M3 submission checklist.
- `docs/m3_handout_compliance_check.md`: handout-aligned D1-D4 and R1-R7 compliance table.
- `docs/milestone3_live_trial_protocol.md`: live trial protocol used for returned Arduino evidence.
- `docs/tables/m3_live_serial_results.md`: scored right-pocket controlled and left-pocket robustness live Serial results.
- `docs/tables/`: dataset inspection tables and class-count summaries.
- `outputs/lightweight/` and `outputs/reproduction/`: generated models, metrics, logs, and figures.

## Arduino / PlatformIO Starter

The original Wokwi/PlatformIO firmware remains available for embedded bring-up while waiting for Nano 33 BLE Sense hardware.

```powershell
pio run
```

Main firmware file:

- `src/main.cpp`

Next-round plan: submit `tinyml_milestone3.pdf` with the returned videos/logs for M3, then improve stair-direction robustness for M4 with more balanced upstairs/downstairs data and calibration/orientation-aware features.
