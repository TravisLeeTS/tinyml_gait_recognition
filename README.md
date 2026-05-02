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

Current status: The package is runnable with TensorFlow/Keras training code, generated UCI HAR held-out metrics, confusion matrices, dataset inspection outputs, target-domain adaptation experiments, and an Arduino Nano 33 BLE Sense M3 candidate deployment build. The candidate INT8 model has returned Arduino hardware evidence: compile flash/RAM, boot metadata, 34.113 ms average `Invoke()` latency over 50 inferences, 2-minute stability video, and live Serial trial scores.

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

## Phase 3: Prototype Quantization Strategy

This section records the existing quantization proof-of-concept. It should be rerun after the V2-robust model is selected:

```powershell
python -m src.deployment.quantization_experiment --representative-samples 512 --qat-trigger-drop 0.01
```

The experiment compares:

- full integer INT8 PTQ with random representative train windows
- class-balanced full integer INT8 PTQ
- INT8 QAT only if all PTQ variants drop macro F1 by more than `0.01`

Prototype Phase 3 result on the UCI-trained DS-CNN:

| Variant | Accuracy | Macro F1 | Size | Host TFLite Mean Latency | Prototype Selection |
|---|---:|---:|---:|---:|---|
| Full integer INT8 PTQ | 0.9138 | 0.9141 | 10,288 bytes | 0.0168 ms | No |
| Class-balanced full integer INT8 PTQ | 0.9145 | 0.9147 | 10,288 bytes | 0.0192 ms | Yes |

QAT decision for this prototype: not run because the selected PTQ variant is within `0.0026` macro F1 of the Phase 2 FP32 baseline, which is below the `0.01` trigger threshold. This decision must be revisited if the final V2-robust model has a larger PTQ drop.

Outputs:

- `src/deployment/quantization_experiment.py`
- `outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.csv`
- `outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.json`
- `outputs/deployment/quantization_experiments/models/lightweight_tiny_cnn_full_integer_int8_ptq.tflite`
- `outputs/deployment/quantization_experiments/models/lightweight_tiny_cnn_class_balanced_full_integer_int8_ptq.tflite`
- prototype deployment artifact copied to `outputs/deployment/models/lightweight_tiny_cnn_int8.tflite`
- prototype Arduino header regenerated at `arduino/tinyml_har_m3/model_data.h`

## Milestone 3 Deployment Prep

Audit the teammate's Arduino CSV collection:

```powershell
python -m src.data.arduino_collectdata
python -m src.data.audit_arduino_domain
```

Replay the collected Arduino V2 data through the current UCI-trained prototype on the laptop:

```powershell
python -m src.deployment.evaluate_arduino_replay --root data\raw\arduino_collectdata_v2 --output-dir outputs\arduino_collectdata_v2_baseline_replay --windowed-output data\processed\arduino_collectdata_v2_baseline_replay_windows_50hz.npz
```

Outputs:

- `docs/tables/arduino_collectdata_class_counts.csv`
- `docs/tables/arduino_collectdata_file_quality.csv`
- `outputs/arduino_collectdata/arduino_collectdata_summary.json`
- `outputs/arduino_collectdata_v2_baseline_replay/metrics/arduino_replay_int8_metrics.json`
- `data/processed/arduino_collectdata_v2_windows_50hz.npz`

The V2 audit found corrected 50 Hz timing and 236 usable windows, but each class is still below the earlier 100-window target. Offline replay of the current UCI-trained baseline on V2 gives `0.1864` accuracy and `0.0524` macro F1; all replay windows collapse to `LAYING`. This is a domain-gap diagnostic, not live on-device accuracy. A deeper replay check found the same collapse in FP32 and INT8, so this is not a quantization bug. The stronger cause is axis/orientation and placement shift: V2 upright/motion windows are centered near negative x-axis gravity, while UCI-HAR upright/motion classes are mostly positive x-axis after preprocessing.

Because the demo will use the same Arduino/pocket/clothing environment family as V2, the current project strategy is model-first: use UCI-HAR as public source knowledge, use V2 and V2.1 as target-domain adaptation data, keep held-out right/left validation and future live Serial evidence separate, and quantize only the selected candidate for Arduino evidence collection.

The domain audit saves `docs/tables/m3_domain_orientation_comparison.csv` and `outputs/arduino_domain_audit/m3_domain_standardized_shift.json`. Current finding: Arduino V2 `acc_x` is about `-3.40` standard deviations from the UCI-HAR train-only standardizer mean.

Run the V2 mixed-training experiment for the deployable DS-CNN family:

```powershell
python -m src.training.train_m3_mixed_arduino --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Run the earlier V2-only target-domain adaptation experiment suite:

```powershell
python -m src.training.train_m3_target_domain --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Run the latest V2.1 long-adaptation retraining, held-out validation, INT8 packaging, and header regeneration workflow:

```powershell
python -m src.training.train_m3_retrained_with_v2_1 --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Latest retrained result selects `m3_v2_1_mixed_6ch` as the current candidate:

- Held-out right_60s raw replay accuracy: 0.8583
- Held-out right_60s raw replay macro F1: 0.7647
- Held-out left_30s raw replay accuracy: 0.4261
- Held-out left_30s raw replay macro F1: 0.3721
- Arduino V2 grouped validation accuracy: 0.7379
- Arduino V2 grouped validation macro F1: 0.7097
- UCI-HAR official test accuracy: 0.7957
- UCI-HAR official test macro F1: 0.7932
- INT8 TFLite size: 10,288 bytes

The raw replay rows are not live Arduino accuracy. The returned true live Serial evidence for the same candidate gives `0.9040` accuracy / `0.9089` macro F1 for right-pocket controlled trials and `0.8901` accuracy / `0.8826` macro F1 for left-pocket robustness trials. Static postures are strong in the live logs; the main remaining live failure is `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS`.

Deployment finding: the Arduino model is placement- and orientation-sensitive. Use right pocket for the controlled demo, keep the board orientation fixed, and align left-pocket placement to resemble right-pocket orientation if robustness testing is repeated. Static classes depend strongly on board angle, while stair direction remains the main dynamic-class weakness.

Detailed results are saved in `outputs/deployment/m3_retrained_with_v2_1/`, `docs/tables/m3_retrained_model_comparison.md`, and `docs/tables/m3_live_validation_results.md`.

The 10-channel feature computation is shared in `src/data/target_domain_features.py`. The Arduino sketch is also forward-compatible with a future `kChannelCount = 10` `model_data.h`: it computes gravity direction XYZ and acceleration magnitude from the 128-sample window before applying exported normalization constants.

Prototype quantization command for the older UCI-trained DS-CNN, retained only as historical reference:

```powershell
python -m src.deployment.quantization_experiment --representative-samples 512 --qat-trigger-drop 0.01
```

Outputs:

- `outputs/deployment/models/lightweight_tiny_cnn_int8.tflite`
- `outputs/deployment/metrics/lightweight_tiny_cnn_int8_metrics.json`
- `outputs/deployment/metrics/m2_vs_m3_int8_comparison.csv`
- `arduino/tinyml_har_m3/model_data.h`

Current prototype INT8 offline result on the UCI HAR held-out test set:

- Quantization: class-balanced full integer INT8 PTQ
- Accuracy: 0.9145
- Macro F1: 0.9147
- INT8 TFLite size: 10,288 bytes
- Planned tensor arena: 61,440 bytes

Arduino sketch:

```text
arduino/tinyml_har_m3/tinyml_har_m3.ino
```

Open the sketch in Arduino IDE with the current candidate `model_data.h`. Returned hardware evidence shows the sketch compiles on Arduino Nano 33 BLE Sense, uses 177,504 bytes flash and 111,144 bytes RAM, reports a 61,440-byte tensor arena, and averages 34.113 ms per `Invoke()` over 50 windows. See `tinyml_milestone3.pdf` and `README_MILESTONE3.md` for the full M3 evidence table and final video/log submission notes.

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
