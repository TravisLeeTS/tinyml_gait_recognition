# Milestone 3 README

Project: Energy-Efficient Human Activity Recognition on the Edge

This file tracks the M3 on-device prototype work. Laptop-side conversion, Arduino sketch prep, Arduino CSV dataset integration, Arduino CSV auditing, and offline replay evaluation are implemented. Final live on-device grading evidence still requires the physical Arduino.

The current Arduino CSV replay result is treated as a **v1 suspect-data domain-gap finding**, not as the final Arduino result. We suspect the v1 collection did not follow the intended 50 Hz protocol correctly, so the next dataset round should replace or supersede the v1 replay numbers after validation.

## Quick Commands

Audit the Arduino CSV collection:

```powershell
python -m src.data.arduino_collectdata
```

Replay the collected Arduino CSV test data through the INT8 model on the laptop:

```powershell
python -m src.deployment.evaluate_arduino_replay
```

Run the quantization experiment and regenerate the selected INT8 Arduino header:

```powershell
python -m src.deployment.quantization_experiment --representative-samples 512 --qat-trigger-drop 0.01
```

## Professor Feedback Checklist For M3

This section tracks the professor's key M3 improvement advice from `M2_FEEDBACK.md`.

| Advice | Required Evidence | Current Status |
|---|---|---|
| Working demo + evidence | Lightweight DS-CNN on Arduino, LSM9DS1 live input, 50 Hz, 128-sample windows, 50% overlap, Serial + LED, stable 2-min run, 60-90s video showing at least 3 classes | Sketch and model header prepared; physical-board run pending |
| Deployment metrics | `.tflite` size, tensor arena, average `Invoke()` latency over >=50 calls using `micros()`, pipeline diagram | `.tflite` size, planned arena, and pipeline documented; real Arduino latency/compile RAM pending |
| Optimization vs M2 | INT8 PTQ before/after table with size, latency, RAM, macro F1, and tradeoff discussion | Quantization experiment done; class-balanced INT8 PTQ selected; hardware latency/RAM pending |
| Second improvement motivated by M2 error analysis | Try focal loss or posture/gravity feature to address SITTING/STANDING confusion | Gravity-direction posture probe completed; it worsened static confusion, so it is documented but not adopted |
| On-device accuracy | Track B Arduino data, >=20 live trials/class, live confusion matrix, offline UCI vs Arduino-live comparison and domain-gap explanation | Real v1 Arduino CSV dataset integrated; live trials pending; v1 replay shows severe domain gap but is not live accuracy |
| Robustness test | New condition, >=10 trials/class, compare against controlled condition, analyze degraded classes | Planned condition: different pocket or environment; live trial data pending |
| Challenges and M4 plan | Reconcile split wording, add per-layer architectures, document M4 fixes | M2 docs updated; M3 draft includes domain-gap and M4 calibration/fine-tuning plan |
| Repo and README | Commit exported `.tflite`, final metrics JSON, `.ino`, conversion script, flash/run README, fallback plan | Artifacts and scripts added; final hardware logs still pending |

## M3 Rubric Status

| Rubric Item | Points | Status |
|---|---:|---|
| D2/D3 working on-device demo and evidence | 25 | Pending physical Arduino run |
| R2 deployment metrics | 15 | Partially done: model size known; real latency/RAM pending |
| R3 optimization with before/after comparison | 20 | Partially done: INT8 before/after offline metrics done; hardware columns pending |
| R4 on-device accuracy | 10 | Pending live 20 trials/class; v1 CSV replay is supporting domain-gap evidence only |
| R5 robustness test | 15 | Pending live 10 trials/class under new condition |
| D4 code and repo quality | 10 | Mostly done: sketch, model file, conversion script, README present |
| Report clarity | 5 | Draft present; must replace pending fields before submission |

## Current M3 Artifacts

```text
arduino/tinyml_har_m3/tinyml_har_m3.ino
arduino/tinyml_har_m3/model_data.h
outputs/deployment/models/lightweight_tiny_cnn_int8.tflite
outputs/deployment/metrics/lightweight_tiny_cnn_int8_metrics.json
outputs/deployment/metrics/m2_vs_m3_int8_comparison.csv
outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.json
docs/milestone3_report_draft.md
docs/milestone3_teammate_message.md
docs/tables/arduino_collectdata_class_counts.csv
docs/tables/arduino_collectdata_file_quality.csv
docs/tables/arduino_collectdata_channel_stats.csv
outputs/arduino_collectdata/metrics/arduino_replay_int8_metrics.json
```

## Deployment Pipeline

```text
LSM9DS1 accelerometer+gyroscope
  -> 50 Hz sampler
  -> 128-sample ring buffer, 64-sample stride
  -> gyroscope deg/s to rad/s conversion
  -> standardize with saved UCI-HAR train mean/std
  -> INT8 input quantization with saved scale/zero-point
  -> TFLite Micro Invoke()
  -> class label on Serial + LED state
```

The sketch uses the lightweight depthwise-separable 1D CNN, not the stacking baseline. The stacking baseline remains an offline reference only.

## Split And Architecture Checks

Split wording is reconciled as:

| Split | Windows | Note |
|---|---:|---|
| Train | 5,551 | Subject-aware subset of official UCI-HAR training subjects |
| Validation | 1,801 | Held out from official training subjects by subject group |
| Test | 2,947 | Official UCI-HAR test split |

Do not describe the active pipeline as a simple 70/30 split.

Layer-level deployable DS-CNN:

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

Stacking baseline architecture remains offline only: ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, and CNN-LSTM base learners feeding an XGBoost meta-learner. It is not exported to Arduino.

## Preprocessing Status

The main UCI-HAR training preprocessing has not been changed in this M3 pass. The current training path still uses:

- UCI-HAR prewindowed 128-sample windows.
- Channels: total acceleration XYZ plus body gyroscope XYZ.
- Subject-aware validation split from the official UCI training split.
- Per-channel standardization fit on training windows only.

What changed is separate from main training preprocessing:

- Arduino CSV import/audit/resampling for Track B real-sensor evaluation.
- Quantization representative sampling for Phase 3.
- A separate gravity-feature probe for sitting/standing error analysis.

Therefore, the paper-reproduction and Phase 2 model-screening results do not need to be rerun solely because of this update. Retrain the lightweight model and rerun quantization if the clean Arduino v2 collection leads to a real input-pipeline change, such as axis remapping, unit correction, added gravity channels in the deployable model, or Arduino fine-tuning.

## Optimization Result

Selected quantization: class-balanced full integer INT8 PTQ. It slightly outperformed random representative PTQ on UCI-HAR macro F1 while keeping the same model size.

| Metric | Phase 2 Lightweight FP32 | Phase 3 Selected INT8 TFLite | Change |
|---|---:|---:|---:|
| Offline UCI-HAR accuracy | 0.9169 | 0.9145 | -0.0024 |
| Offline macro F1 | 0.9173 | 0.9147 | -0.0026 |
| TFLite size | 13,460 bytes | 10,288 bytes | -3,172 bytes |

The INT8 result above is an offline UCI-HAR test-set result. It is not live Arduino accuracy.

Quantization variants tested:

| Variant | Accuracy | Macro F1 | Host TFLite Mean Latency | Selected |
|---|---:|---:|---:|---|
| Full integer INT8 PTQ | 0.9138 | 0.9141 | 0.0391 ms | No |
| Class-balanced full integer INT8 PTQ | 0.9145 | 0.9147 | 0.0430 ms | Yes |

QAT decision: INT8 QAT was not run because the selected PTQ variant stayed within 0.01 macro F1 of the Phase 2 FP32 baseline. If a future model or v2 Arduino-calibrated training run suffers a larger PTQ drop, QAT becomes the next quantization experiment.

Saved quantization outputs:

```text
src/deployment/quantization_experiment.py
outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.csv
outputs/deployment/quantization_experiments/metrics/quantization_experiment_summary.json
outputs/deployment/quantization_experiments/models/lightweight_tiny_cnn_class_balanced_full_integer_int8_ptq.tflite
```

## Second Improvement Experiment: Gravity Feature

The M2 error analysis showed the largest DS-CNN confusion was `SITTING` <-> `STANDING`. To test the professor's suggested posture-discriminating feature, a separate Phase 2 probe was added:

```powershell
python -m src.training.evaluate_posture_gravity_feature --device cpu
```

This probe computes a gravity-direction estimate from each window's mean total acceleration: mean acceleration XYZ, unit direction XYZ, and magnitude. It trains a small logistic-regression classifier on UCI-HAR training-split static windows only, then tests whether that feature can rescue static predictions from the selected DS-CNN.

| Method | Accuracy | Macro F1 | SITTING<->STANDING Confusions | Decision |
|---|---:|---:|---:|---|
| Phase 2 DS-CNN baseline | 0.9169 | 0.9173 | 184 | Keep |
| Gravity static classifier, static subset only | 0.8712 | 0.8684 | 201 | Not enough |
| DS-CNN + gravity static rescue | 0.9111 | 0.9117 | 201 | Do not adopt |
| DS-CNN + thresholded gravity rescue | 0.9111 | 0.9117 | 201 | Do not adopt |

Conclusion: the simple UCI-HAR mean-acceleration gravity feature did not fix the static posture confusion. It is documented as a completed second-improvement attempt, but the deployable M3 model remains the selected class-balanced INT8 PTQ DS-CNN. For M4, posture improvement should use clean Arduino v2 data with verified orientation/axis calibration before adding this feature to the deployable model.

Saved outputs:

```text
src/training/evaluate_posture_gravity_feature.py
outputs/lightweight/experiments/m3_gravity_feature_probe/metrics/gravity_feature_probe_comparison.csv
outputs/lightweight/experiments/m3_gravity_feature_probe/metrics/gravity_feature_probe_comparison.json
```

## Focal Loss Runtime Fix

The focal-loss run was interrupted because it ran too long on CPU. The partial run produced a training log with 16 epochs, but the saved `.keras` file is incomplete and not usable as a reportable test result.

Fix applied:

- `src/training/train_m3_second_improvements.py` now defaults to one experiment, not `all`.
- Defaults are bounded to 16 epochs and patience 4.
- It writes a live CSV log during training.
- TFLite export and host-latency benchmarking are skipped unless `--export-tflite` or `--benchmark-host` is explicitly passed.

Use this bounded command only if we still want a focal-loss result later:

```powershell
python -m src.training.train_m3_second_improvements --experiment focal_loss --epochs 16 --patience 4 --device cpu
```

## Track B Arduino Test Dataset

The attached Arduino CSVs are now kept as the M3 real Arduino test dataset at:

```text
data/raw/arduino_collectdata_v1/
```

This satisfies the Track B requirement to keep real Arduino sensor data available by M3, but the current v1 dataset appears to have collection-quality problems. It should be retained for audit/history and superseded by the next clean collection round before reporting final Arduino accuracy.

It also does not by itself satisfy the separate M3 live on-device demo and live trial requirements.

## Arduino Data Audit

The attached Arduino CSVs were extracted to `data/raw/arduino_collectdata_v1/`. The importer uses CSV files only.

Key findings:

- 32 CSV files, 14,662 valid sensor rows.
- Median observed rate: about 38.46 Hz, not the intended 50 Hz.
- `standing/30s - r - 1.csv` contains one malformed duplicate header row and a timestamp reset; the importer skips the invalid row and splits the file into segments.
- After 50 Hz resampling and 128-sample windows with 50% overlap, the collection has 246 total windows. This is useful for audit/prep, but below the preferred 100 windows/class target.

Because the observed sampling rate is far from the intended 50 Hz and one file contains a duplicated header/timestamp reset, the v1 replay result should be described as a diagnostic failure case. It should not be used as the final M3 accuracy claim.

Current replay-window counts:

| Class | 50 Hz Replay Windows |
|---|---:|
| WALKING | 44 |
| WALKING_UPSTAIRS | 24 |
| WALKING_DOWNSTAIRS | 24 |
| SITTING | 44 |
| STANDING | 66 |
| LAYING | 44 |

## Arduino CSV Offline Replay

The public-data-trained INT8 model was evaluated on the real Arduino CSV dataset by replaying the resampled windows through TensorFlow Lite on the laptop.

| Metric | Value |
|---|---:|
| Replay windows | 246 |
| Accuracy | 0.1789 |
| Macro F1 | 0.0582 |
| Weighted F1 | 0.0625 |

Main failure mode: almost every class collapses to `LAYING`, except `SITTING` which collapses to `WALKING_UPSTAIRS`. This is strong evidence of a domain gap and likely axis/orientation/timing mismatch between UCI HAR smartphone data and this Arduino/pocket collection.

Interpretation: this result likely reflects a combination of incorrect/unstable collection protocol, sampling mismatch, axis/orientation mismatch, and public-dataset-to-Arduino domain gap. It is not enough to conclude that the model architecture is unusable.

Saved outputs:

```text
outputs/arduino_collectdata/metrics/arduino_replay_int8_metrics.json
outputs/arduino_collectdata/metrics/arduino_replay_int8_confusion_matrix.csv
outputs/arduino_collectdata/figures/arduino_replay_int8_confusion_matrix.png
```

This replay result is not live on-device accuracy. The M3 handout still requires live Arduino inference trials.

## New Dataset Round Protocol

When the new Arduino dataset is shared, place it under a new versioned folder instead of overwriting v1:

```text
data/raw/arduino_collectdata_v2/
```

Required collection protocol for the next round:

| Requirement | Target |
|---|---|
| Board | Arduino Nano 33 BLE Sense |
| Sensor | LSM9DS1 accelerometer XYZ + gyroscope XYZ |
| Sampling | Verify actual CSV cadence is close to 50 Hz, ideally median dt near 20 ms |
| Columns | `time_ms, ax, ay, az, gx, gy, gz, label` |
| Classes | `walking`, `walk_up`, `walk_down`, `sitting`, `standing`, `laying` |
| Duration | At least 2.5 minutes per class if possible |
| Windows target | >=100 usable 128-sample windows per class after validation |
| Pocket/orientation | Record placement explicitly, e.g. left pocket or right pocket |
| Users/environments | At least 1 user and 1 environment; second pocket/environment can serve robustness |
| File hygiene | One header row only, monotonic timestamps, one activity label per file |
| Raw evidence | Keep CSV files; `.txt` captures are optional and not used unless labeled |

Immediately after receiving v2:

```powershell
python -m src.data.arduino_collectdata --root data/raw/arduino_collectdata_v2 --output-dir outputs/arduino_collectdata_v2
python -m src.deployment.evaluate_arduino_replay --root data/raw/arduino_collectdata_v2 --output-dir outputs/arduino_collectdata_v2
```

Acceptance checks before using v2 as final M3 Arduino evidence:

- Median effective sampling rate should be close to 50 Hz.
- No malformed duplicate header rows.
- No timestamp resets inside a file unless intentionally split and documented.
- Each class should have enough windows for the planned evaluation.
- Replay predictions should not collapse almost entirely to one class.

If v2 still fails badly, document the failure honestly and use it to motivate M4 domain adaptation: axis calibration/remapping, Arduino-only fine-tuning, more collection diversity, and possibly a scoped fallback class subset.

## Hardware Tasks Still Needed

The teammate with the Arduino must provide:

- Arduino IDE compile summary: flash usage and RAM usage.
- Boot serial lines: `model_bytes` and `tensor_arena_bytes`.
- Stable 2-minute live run.
- 60-90 second video or serial log showing at least 3 classes.
- Average `latency_us` over at least 50 inferences.
- Controlled live confusion matrix: 20 trials per class.
- Robustness test: 10 trials per class in a new condition.

Use `docs/milestone3_live_trial_sheet.csv` as the recording template.
