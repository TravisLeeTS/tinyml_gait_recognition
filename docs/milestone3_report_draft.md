# Milestone 3 Report Draft

Project: Energy-Efficient Human Activity Recognition on the Edge  
Team: Lee Ting Sen, Khalifa Alshamsi, Vineetha Addanki

Status as of repo update: laptop-side M3 conversion, Arduino sketch, Arduino CSV dataset integration, dataset audit, offline replay evaluation, and report scaffolding are complete. Hardware-only results are pending because the Arduino is currently with a teammate. Do not submit the pending fields as final numbers until they are measured on the board.

## R1 Deployment Pipeline

Pipeline:

```text
LSM9DS1 IMU -> 50 Hz sampler -> 128-sample ring buffer -> train-stat standardization -> INT8 input quantization -> TFLite Micro Invoke -> Serial prediction + LED state
```

Conversion path:

```text
outputs/lightweight/models/lightweight_tiny_cnn.keras
  -> quantization strategy experiment with 512 representative UCI-HAR train windows
  -> selected class-balanced full-integer INT8 post-training quantization
  -> outputs/deployment/models/lightweight_tiny_cnn_int8.tflite
  -> arduino/tinyml_har_m3/model_data.h
```

Conversion command:

```powershell
python -m src.deployment.quantization_experiment --representative-samples 512 --qat-trigger-drop 0.01
```

Quantization: class-balanced full integer INT8 post-training quantization. Input and output tensors are INT8. The saved input quantization constants are `scale=0.0921273530` and `zero_point=2`.

Arduino sketch: `arduino/tinyml_har_m3/tinyml_har_m3.ino`

- Reads accelerometer and gyroscope from `Arduino_LSM9DS1`.
- Targets 50 Hz using a `micros()` interval of 20,000 us.
- Uses 128 samples per inference and shifts by 64 samples after each prediction.
- Converts gyroscope readings from deg/s to rad/s before applying the saved UCI-HAR training standardizer.
- Quantizes the standardized input with the saved INT8 input scale and zero-point generated during the class-balanced PTQ experiment.
- Invokes the INT8 TFLite Micro model and prints prediction, confidence, instantaneous latency, average latency, and tensor arena size.
- Uses the built-in LED as a movement/static indicator; Serial is the primary class output.

Arduino libraries and versions: pending teammate compile. Record the exact Arduino Mbed OS Nano Boards, `Arduino_LSM9DS1`, and TensorFlow Lite Micro / Arduino TensorFlow Lite versions from the machine used to flash the board.

## R2 Deployment Metrics

| Metric | Current Value | Source | Status |
|---|---:|---|---|
| INT8 `.tflite` model size | 10,288 bytes / 10.05 KB | `outputs/deployment/models/lightweight_tiny_cnn_int8.tflite` | Done |
| Planned tensor arena | 61,440 bytes / 60.00 KB | `model_data.h` | Must confirm on board |
| Total sketch flash/RAM | TBD | Arduino compile summary | Hardware teammate needed |
| Arduino invoke latency | TBD average over >=50 calls | Serial `latency_us` field | Hardware teammate needed |
| Host TFLite latency proxy | 0.0430 ms mean over 50 runs | `outputs/deployment/metrics/lightweight_tiny_cnn_int8_metrics.json` | Not a hardware metric |

The host latency proxy is not an M3 deployment claim. The final report must use the Arduino `micros()` measurements printed by the sketch.

## R3 Optimization: M2 Baseline vs. M3 INT8

| Metric | Phase 2 Lightweight Keras / FP32 TFLite | Phase 3 Selected INT8 TFLite | Change |
|---|---:|---:|---:|
| Offline UCI-HAR accuracy | 0.9169 | 0.9145 | -0.0024 |
| Offline macro F1 | 0.9173 | 0.9147 | -0.0026 |
| TFLite model size | 13,460 bytes | 10,288 bytes | -3,172 bytes |

Optimization used: class-balanced full-integer post-training quantization. The accuracy tradeoff is small, about 0.24 percentage points accuracy and 0.26 macro-F1 points. The model-size reduction is 23.6%; it is less dramatic than the usual 4x headline reduction because the original DS-CNN is already tiny and TFLite flatbuffer metadata is a meaningful fraction of total size.

Quantization variants tested:

| Variant | Accuracy | Macro F1 | Host TFLite Mean Latency | Selected |
|---|---:|---:|---:|---|
| Full integer INT8 PTQ | 0.9138 | 0.9141 | 0.0391 ms | No |
| Class-balanced full integer INT8 PTQ | 0.9145 | 0.9147 | 0.0430 ms | Yes |

INT8 QAT decision: QAT was not run in this pass because the selected PTQ variant stayed within the 0.01 macro-F1 trigger threshold. If a future Phase 2 model or v2 Arduino-calibrated model loses more than 0.01 macro F1 after PTQ, QAT becomes the next experiment.

M2 error-analysis motivation: the model is already small enough for deployment, so M3 prioritizes deployability and measured on-device cost. Static posture confusion remains the main accuracy issue and belongs in Phase 2 model/feature screening or M4 data adaptation, not in the Phase 3 quantization selection step.

Second-improvement experiment: gravity-direction posture feature.

| Method | Accuracy | Macro F1 | SITTING<->STANDING Confusions | Decision |
|---|---:|---:|---:|---|
| Phase 2 DS-CNN baseline | 0.9169 | 0.9173 | 184 | Keep |
| Gravity static classifier, static subset only | 0.8712 | 0.8684 | 201 | Not enough |
| DS-CNN + gravity static rescue | 0.9111 | 0.9117 | 201 | Do not adopt |
| DS-CNN + thresholded gravity rescue | 0.9111 | 0.9117 | 201 | Do not adopt |

The simple mean-acceleration gravity feature was tested and did not reduce SITTING/STANDING confusion. Therefore it is documented as a completed negative experiment, not adopted into the M3 deployable model. The focal-loss path was also corrected for runtime: the previous CPU run produced only a training log and an incomplete `.keras` file, so it is not reported as a valid test result. The fixed focal-loss runner now defaults to one bounded experiment, writes live epoch logs, and skips TFLite/latency work unless requested.

## R4 On-Device Accuracy

Required final evidence: 20 live Arduino trials per class under controlled conditions, producing a live confusion matrix.

Current status: pending hardware run.

Track B Arduino-collected test dataset:

- Stored in `data/raw/arduino_collectdata_v1/`.
- Source archive: `TinyML_arduino_collectdata_v1.zip`.
- CSV columns: `time_ms`, `ax`, `ay`, `az`, `gx`, `gy`, `gz`, `label`.
- Sensor configuration: accelerometer XYZ plus gyroscope XYZ.
- Pocket variation: left and right pocket files are present.
- Exact collection date and environment are not embedded in the files; teammate confirmation is needed before final submission.

Current Arduino CSV audit:

| Class | Valid Rows | Duration (s) | Resampled 50 Hz Windows |
|---|---:|---:|---:|
| LAYING | 2,245 | 59.927 | 44 |
| SITTING | 2,249 | 59.925 | 44 |
| STANDING | 3,397 | 89.898 | 66 |
| WALKING | 2,228 | 59.922 | 44 |
| WALKING_DOWNSTAIRS | 2,271 | 59.604 | 24 |
| WALKING_UPSTAIRS | 2,272 | 59.619 | 24 |

Audit notes:

- Only CSV files are used; unlabeled Arduino `.txt` files are ignored.
- Median observed sampling cadence is about 38.46 Hz, not the intended 50 Hz.
- `standing/30s - r - 1.csv` contains one duplicated/malformed header row and one timestamp reset; the importer skips the bad row and treats it as two segments.
- The current collection is useful for early audit and offline resampling, but it is below the preferred >=100 windows/class target and does not replace the required live on-device trial matrix.

Offline replay on real Arduino CSV data:

| Evaluation Setting | Accuracy | Macro F1 | Samples/Windows | Notes |
|---|---:|---:|---:|---|
| Offline UCI-HAR test, selected INT8 TFLite | 0.9145 | 0.9147 | 2,947 | Public dataset holdout |
| Offline Arduino CSV replay, selected INT8 TFLite | 0.1789 | 0.0582 | 246 | Real Arduino sensor data, laptop TFLite replay |
| Live on-device controlled | TBD | TBD | 20 trials/class required | Hardware run pending |

The Arduino CSV replay confusion matrix is saved at `outputs/arduino_collectdata/metrics/arduino_replay_int8_confusion_matrix.csv`. The dominant failure is severe class collapse: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, and STANDING are predicted as LAYING, while SITTING is predicted as WALKING_UPSTAIRS. This is not acceptable deployment accuracy, but it is valuable evidence of the public-dataset-to-Arduino domain gap that must be addressed before M4.

## R5 Robustness Test

Required final evidence: at least 10 live trials per class under one condition not used for controlled testing.

Recommended robustness condition: different pocket placement, because the provided dataset already distinguishes left and right pocket files and pocket orientation is a realistic source of IMU domain shift.

Alternative acceptable condition: second user, different walking route/staircase, or different indoor environment.

Final report table to fill after hardware testing:

| Condition | Trials/Class | Accuracy | Macro F1 | Main Failure Mode |
|---|---:|---:|---:|---|
| Controlled same pocket/user | 20 | TBD | TBD | TBD |
| Robustness different pocket/environment | 10 | TBD | TBD | TBD |

## R6 Challenges and Lessons Learned

Split and preprocessing reconciliation:

- The active UCI-HAR split is 5,551 train / 1,801 validation / 2,947 test windows.
- The validation split is subject-aware and derived from official UCI-HAR training subjects.
- The active pipeline should not be described as a simple 70/30 split.
- The M3 pass did not change the main model training preprocessing; it added Arduino CSV audit/replay, quantization representative sampling, and a separate posture-feature probe.

Current challenges:

- The collected CSVs do not match the claimed 50 Hz rate; they are closer to 37-38 Hz. The M3 sketch now enforces a 20 ms target interval, but the teammate must verify the live serial cadence.
- Public UCI-HAR smartphone data and Arduino Nano 33 BLE Sense IMU data have a sensor, placement, and unit domain gap. The sketch converts gyroscope deg/s to rad/s to match the UCI-HAR convention more closely.
- Offline replay accuracy on the real Arduino CSV dataset is only 17.89%, mainly due to class collapse toward LAYING. This suggests axis orientation, placement, timing, and/or normalization mismatch rather than a small model-capacity issue. The v1 dataset should be retained for audit but superseded by a clean v2 collection round before final live accuracy is claimed.
- Hardware-only metrics cannot be measured from the laptop. The repo marks these as pending rather than fabricating values.
- The current Arduino data volume is not enough for the professor's suggested >=100 windows/class target, especially upstairs/downstairs.

Architecture details for the deployable model:

| Layer | Output Shape | Parameters |
|---|---:|---:|
| Input | 128 x 6 | 0 |
| Conv1D, 12 filters, kernel 3, no bias | 128 x 12 | 216 |
| BatchNorm + ReLU | 128 x 12 | 48 |
| SeparableConv1D, 24 filters, kernel 5, no bias | 128 x 24 | 348 |
| BatchNorm + ReLU | 128 x 24 | 96 |
| AveragePooling1D, pool 2 | 64 x 24 | 0 |
| SeparableConv1D, 32 filters, kernel 5, no bias | 64 x 32 | 888 |
| BatchNorm + ReLU | 64 x 32 | 128 |
| GlobalAveragePooling1D | 32 | 0 |
| Dense softmax, 6 classes | 6 | 198 |
| Total Keras parameters | - | 1,922 |

Architecture details for the offline stacking baseline:

- Base learners: ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, and CNN-LSTM.
- Meta-learner: XGBoost trained on base-model validation predictions.
- Deployment status: offline reference baseline only, not exported to Arduino.

## R7 Plan for M4

- Complete the M3 hardware run and preserve the serial log/video evidence.
- Collect more Arduino data at verified 50 Hz, targeting at least 2.5 minutes per class for >=100 windows/class with 128-sample windows and 50% overlap.
- Add an Arduino calibration step: record stationary axis orientation, verify accelerometer units, and decide whether to remap/sign-flip axes before normalization.
- Fine-tune or retrain on Arduino-collected windows after calibration; use UCI HAR as pretraining/reference, not as the only training distribution.
- Add posture-focused data or a gravity/orientation feature to reduce SITTING/STANDING confusion.
- If six-class live accuracy is unstable, report a scoped fallback subset while still documenting the six-class failure modes.
- Re-run INT8 conversion and Arduino latency after any model change.

## D4 Repository Evidence

```text
arduino/tinyml_har_m3/tinyml_har_m3.ino
arduino/tinyml_har_m3/model_data.h
src/deployment/convert_lightweight_to_tflite_micro.py
src/deployment/quantization_experiment.py
src/data/arduino_collectdata.py
outputs/deployment/models/lightweight_tiny_cnn_int8.tflite
outputs/deployment/metrics/m2_vs_m3_int8_comparison.csv
outputs/deployment/metrics/lightweight_tiny_cnn_int8_metrics.json
outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.json
outputs/lightweight/experiments/m3_gravity_feature_probe/metrics/gravity_feature_probe_comparison.json
outputs/arduino_collectdata/metrics/arduino_replay_int8_metrics.json
docs/tables/arduino_collectdata_class_counts.csv
docs/tables/arduino_collectdata_file_quality.csv
docs/tables/arduino_collectdata_channel_stats.csv
```
