# Milestone 3 README

Project: Energy-Efficient Human Activity Recognition on the Edge

This README is the Milestone 3 evidence index. The final submission report is
`tinyml_milestone3.pdf`; use this file to find the code, datasets, commands,
metrics, and returned Arduino evidence behind that report.

Current status: Milestone 3 submitted. On-device inference working.

Milestone 3 storyline:

1. We first tried the professor-advised improvements from M2 error analysis:
   focal loss and additional gravity/posture features. The 10-feature gravity
   probe looked promising on offline UCI-HAR posture errors, while focal loss
   did not justify replacing the baseline.
2. We then found that the UCI-only model did not transfer to our real Arduino
   pocket setup, so we collected and audited more self-collected Arduino data:
   V2, V2.1 long adaptation, held-out right-pocket validation, held-out
   left-pocket validation, and final live Serial logs.
3. We solved the deployable path by selecting a mixed UCI + Arduino 6-channel
   DS-CNN, quantizing it to INT8, regenerating the Arduino header, compiling on
   Nano 33 BLE Sense, measuring real `Invoke()` latency, and scoring returned
   live Serial trials. The remaining limitation is stair-direction robustness,
   especially `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS`.

Track status: this is still a Track B open-dataset project. UCI-HAR remains the
primary public source dataset and official held-out benchmark. Arduino data is
used for target-domain adaptation, validation, and live deployment evidence.

## Final M3 Result

Selected deployment model: `m3_v2_1_mixed_6ch`, a lightweight 6-channel
depthwise-separable CNN trained with UCI-HAR source data plus Arduino
target-domain adaptation data, then quantized with full-integer INT8 PTQ.

| Evidence | Result |
|---|---:|
| INT8 TFLite model size | 10,288 bytes / 10.05 KiB |
| Generated `model_data.h` source size | 65,933 bytes / 64.39 KiB |
| Tensor arena | 61,440 bytes / 60.00 KiB |
| Arduino flash usage | 177,504 bytes / 18% |
| Arduino dynamic memory | 111,144 bytes / 108.54 KiB / 42% |
| Average `Invoke()` latency | 34.113 ms over 50 windows |
| Right-pocket live Serial accuracy | 0.9040 |
| Right-pocket live Serial macro F1 | 0.9089 |
| Left-pocket robustness live Serial accuracy | 0.8901 |
| Left-pocket robustness live Serial macro F1 | 0.8826 |

Returned proof evidence is under `outputs/live_evidence/raw_teammate_return/`:

- `2_30_stability_video.mp4`
- `sitting_proof.mp4`
- `standing_proof.mp4`
- `walking_proof.mp4`
- `7_control_trials/`
- `8_robust_trials/`

Large video files should be attached with the submission rather than treated as
normal source-code artifacts.

## Dataset Register

| Dataset / Evidence Set | Path | Role In M3 | Used For Training? | Size / Notes |
|---|---|---|---|---|
| UCI-HAR v1.0 | downloaded by `src.data.inspect_datasets` | Public Track B source dataset and official benchmark | Yes, training split only | Train 5,551 windows, validation 1,801 windows, official test 2,947 windows |
| WISDM classic v1.1 | downloaded by `src.data.inspect_datasets` | Secondary inspection/domain-gap reference | No | Label taxonomy differs from UCI-HAR; not merged into the six-class training set |
| Arduino V2 | `data/raw/arduino_collectdata_v2/` | Corrected 50 Hz target-domain data; split into adaptation train and grouped validation | Partly | 236 windows: 44 walking, 30 upstairs, 30 downstairs, 44 sitting, 44 standing, 44 laying |
| Arduino V2.1 long adaptation | `data/raw/arduino_collectdata_v2_1_long_adaptation/` | Extra target-domain adaptation for walking/sitting/standing/laying | Yes | Four 5-minute captures; each has 233 windows before cap and 180 after cap |
| Right-pocket raw validation | `data/raw/arduino_live_validation/right_60s/` | Held-out controlled raw sensor replay validation | No | 240 replay windows; used for model selection sanity, not final live accuracy |
| Left-pocket raw validation | `data/raw/arduino_live_validation/left_30s/` | Held-out robustness raw sensor replay validation | No | 115 replay windows; used to expose placement/orientation weakness |
| Right-pocket live Serial logs | `outputs/live_evidence/raw_teammate_return/7_control_trials/` | Final controlled on-device evidence | No | 125 scored rows, 0.9040 accuracy, 0.9089 macro F1 |
| Left-pocket live Serial logs | `outputs/live_evidence/raw_teammate_return/8_robust_trials/` | Final robustness on-device evidence | No | 91 scored rows, 0.8901 accuracy, 0.8826 macro F1 |

Important separation:

- Training/adaptation uses UCI-HAR train, Arduino V2 train, and capped V2.1 long
  adaptation windows.
- Normalization and representative INT8 quantization exclude UCI-HAR test,
  Arduino V2 grouped validation, right-pocket raw validation, left-pocket raw
  validation, and all live Serial logs.
- Raw replay validation is laptop evaluation of saved sensor captures. Live
  Serial scoring is true Arduino inference output and is the final M3 on-device
  evidence.

## Professor-Advised Improvements

Professor feedback after M2 asked us to address error-analysis weaknesses such
as static-posture confusion. We tested two lightweight DS-CNN modifications on
the same UCI-HAR held-out test split as the Phase 2 baseline.

Command:

```powershell
python -m src.training.train_m3_second_improvements --experiment all --evaluate-existing --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

| Offline UCI-HAR Test Run | Input | Change | Accuracy | Macro F1 | SITTING<->STANDING Confusions | Decision |
|---|---:|---|---:|---:|---:|---|
| Phase 2 DS-CNN baseline | 128 x 6 | Cross entropy | 0.9169 | 0.9173 | 184 | Keep as UCI reference only |
| M3 focal-loss DS-CNN | 128 x 6 | Sparse focal loss, gamma 2.0 | 0.9074 | 0.9071 | 182 | Reject: accuracy/F1 drop too large |
| M3 gravity-feature DS-CNN | 128 x 10 | Add gravity direction XYZ + acceleration magnitude | 0.9179 | 0.9173 | 173 | Promising probe; not selected for M3 Arduino deployment |

The 10-feature gravity model looked useful for static-posture analysis, but it
added Arduino preprocessing complexity and later did not become the best
right-pocket candidate. For M3, we kept the deployable model 6-channel so the
Arduino path remained stable and measurable.

Saved outputs:

```text
src/training/train_m3_second_improvements.py
outputs/lightweight/experiments/m3_second_improvements_summary.csv
outputs/lightweight/experiments/m3_focal_loss_tiny_cnn/
outputs/lightweight/experiments/m3_gravity_feature_tiny_cnn/
```

## Arduino Data Expansion

The original UCI-trained DS-CNN converted cleanly to INT8, but it failed on
Arduino V2 replay. The V2 replay produced `0.1864` accuracy and `0.0524` macro
F1 on 236 windows, with predictions collapsing to `LAYING`. FP32 and INT8 both
showed the same collapse, so this was not a quantization bug.

Main diagnosis: Arduino pocket captures had a strong domain shift from UCI-HAR,
especially orientation and placement. The audit found Arduino V2 `acc_x` about
`-3.40` standard deviations from the UCI-HAR train-only standardizer mean.

Audit commands:

```powershell
python -m src.data.arduino_collectdata
python -m src.data.audit_arduino_domain
python -m src.deployment.evaluate_arduino_replay --root data\raw\arduino_collectdata_v2 --output-dir outputs\arduino_collectdata_v2_baseline_replay --windowed-output data\processed\arduino_collectdata_v2_baseline_replay_windows_50hz.npz
```

Saved outputs:

```text
docs/tables/arduino_collectdata_class_counts.csv
docs/tables/arduino_collectdata_file_quality.csv
docs/tables/m3_domain_orientation_comparison.csv
outputs/arduino_collectdata/arduino_collectdata_summary.json
outputs/arduino_collectdata_v2_baseline_replay/metrics/arduino_replay_int8_metrics.json
outputs/arduino_domain_audit/m3_domain_standardized_shift.json
```

## Retraining And Model Selection

After the V2 domain-gap result, the deployable DS-CNN family was retrained with
source + target-domain data. The final workflow compares UCI-only, Arduino-only,
fine-tuned, mixed 6-channel, mixed focal 6-channel, and mixed 10-channel gravity
models.

Command:

```powershell
python -m src.training.train_m3_retrained_with_v2_1 --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Final six-model comparison:

| Model | UCI Macro F1 | V2 Group Macro F1 | Right 60s Macro F1 | Left 30s Macro F1 | Decision |
|---|---:|---:|---:|---:|---|
| UCI-only baseline | 0.9041 | 0.0587 | 0.0526 | 0.0535 | Reject: Arduino validation collapses |
| Arduino-only | 0.0920 | 0.4857 | 0.5163 | 0.7300 | Reject: UCI retention collapses |
| UCI pretrain + V2.1 fine-tune | 0.4563 | 0.1986 | 0.2661 | 0.3166 | Reject |
| Mixed 6-channel DS-CNN | 0.7932 | 0.7097 | 0.7647 | 0.3721 | Selected M3 candidate |
| Mixed focal 6-channel DS-CNN | 0.8124 | 0.7651 | 0.5556 | 0.4070 | Reject: weaker controlled validation |
| Mixed 10-channel gravity DS-CNN | 0.8784 | 0.6098 | 0.5548 | 0.6987 | Not selected: better left-pocket replay but weaker controlled right-pocket replay and more complex deployment |

Selected INT8 candidate:

| Split | FP32 Macro F1 | INT8 Macro F1 | INT8 Size |
|---|---:|---:|---:|
| UCI-HAR official test | 0.7932 | 0.7960 | 10,288 bytes |
| Arduino V2 grouped validation | 0.7097 | 0.7309 | 10,288 bytes |
| Held-out right_60s raw replay | 0.7647 | 0.7741 | 10,288 bytes |
| Held-out left_30s raw replay | 0.3721 | 0.3886 | 10,288 bytes |

PTQ did not create a material drop, so QAT was not needed for the selected M3
candidate. The selected INT8 model was packaged into `model_data.h`.

Saved outputs:

```text
outputs/deployment/m3_retrained_with_v2_1/m3_retrained_model_selection.csv
outputs/deployment/m3_retrained_with_v2_1/m3_retrained_validation_summary.csv
outputs/deployment/m3_retrained_with_v2_1/m3_retrained_fp32_vs_int8_summary.csv
outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite
outputs/deployment/m3_retrained_with_v2_1/selected_candidate/
docs/tables/m3_retrained_model_comparison.md
docs/tables/m3_v2_1_model_selection_sanity_check.md
```

## Quantization And Optimization Evidence

This section exists specifically for the handout's R1, R2, and R3 requirements.
The deployment-relevant model size is the `.tflite` / embedded C-array model,
not the training-time Keras model.

Quantization method:

- Type: full-integer INT8 post-training quantization.
- Model: selected `m3_v2_1_mixed_6ch` DS-CNN.
- Toolchain: TensorFlow 2.18.0, Keras 3.8.0, TensorFlow Lite converter, and
  TFLite Micro through Harvard_TinyMLx 1.2.4-Alpha on Arduino.
- Representative data: class-balanced windows from UCI-HAR train, Arduino V2
  train split, and capped V2.1 long-adaptation data.
- Excluded from representative data: UCI-HAR test, Arduino V2 grouped
  validation, right-pocket raw validation, left-pocket raw validation, and all
  live Serial logs.
- QAT decision: not used for M3 because INT8 PTQ did not reduce macro F1 on the
  selected candidate; it slightly improved the measured replay metrics in the
  saved summaries. QAT would add training complexity without solving the main
  remaining live failure, which is stair-direction confusion.

Model-size comparison:

| Artifact | Role | Size |
|---|---|---:|
| `outputs/lightweight/models/lightweight_tiny_cnn.tflite` | M2 UCI-only FP32 DS-CNN reference | 13,460 bytes / 13.14 KiB |
| `outputs/deployment/m3_retrained_with_v2_1/m3_v2_1_mixed_6ch/models/m3_v2_1_mixed_6ch.tflite` | M3 selected FP32 candidate before PTQ | 13,460 bytes / 13.14 KiB |
| `outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite` | M3 selected INT8 deployment model | 10,288 bytes / 10.05 KiB |
| `arduino/tinyml_har_m3/model_data.h` | Generated Arduino C header containing model bytes and normalization constants | 65,933 bytes / 64.39 KiB |

FP32 vs INT8 selected-candidate metrics:

| Split | FP32 Macro F1 | INT8 Macro F1 | Change |
|---|---:|---:|---:|
| UCI-HAR official test | 0.7932 | 0.7960 | +0.0028 |
| Arduino V2 grouped validation | 0.7097 | 0.7309 | +0.0212 |
| Held-out right_60s raw replay | 0.7647 | 0.7741 | +0.0093 |
| Held-out left_30s raw replay | 0.3721 | 0.3886 | +0.0165 |

M2 baseline vs M3 improved deployment comparison:

| Metric | M2 Baseline | M3 Improved | Change / Tradeoff |
|---|---:|---:|---|
| Offline UCI-HAR macro F1 | 0.9173 | 0.7960 | Lower UCI-only score because M3 prioritizes Arduino-domain transfer |
| Arduino V2 grouped macro F1 | 0.0587 | 0.7309 | Large target-domain gain after Arduino adaptation |
| Right-pocket live Serial macro F1 | Not available | 0.9089 | New true on-device evidence from live Arduino predictions |
| Left-pocket robustness live Serial macro F1 | Not available | 0.8826 | New robustness evidence under different pocket placement |
| Deployable TFLite size | 13,460 bytes / 13.14 KiB | 10,288 bytes / 10.05 KiB | 3,172 bytes smaller, about 23.6% reduction |
| Arduino `Invoke()` latency | Not measured in M2 | 34.113 ms | Measured on hardware over 50 calls using `micros()` |
| Tensor arena | Not measured in M2 | 61,440 bytes / 60.00 KiB | Real TFLite Micro allocation reported from board boot output |

Main tradeoff: the final M3 candidate gives much stronger Arduino-domain
behavior and fits the board cleanly, but it sacrifices some UCI-HAR offline
accuracy compared with the M2 UCI-only DS-CNN. This is acceptable for M3 because
the handout prioritizes real on-device inference, deployment metrics, live
accuracy, and robustness testing.

Saved quantization and packaging outputs:

```text
src/training/train_m3_retrained_with_v2_1.py
src/deployment/package_m3_candidate.py
outputs/deployment/m3_retrained_with_v2_1/m3_retrained_fp32_vs_int8_summary.csv
outputs/deployment/m3_retrained_with_v2_1/selected_candidate/m3_candidate_fp32_vs_int8_summary.csv
outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite
arduino/tinyml_har_m3/model_data.h
```

## Deployment Pipeline

Arduino sketch:

```text
arduino/tinyml_har_m3/tinyml_har_m3.ino
arduino/tinyml_har_m3/model_data.h
```

Pipeline:

```text
LSM9DS1 accelerometer + gyroscope
  -> 50 Hz sampler
  -> 128-sample ring buffer, 64-sample stride
  -> gyroscope deg/s to rad/s conversion
  -> per-channel standardization from selected training/adaptation split
  -> INT8 input quantization
  -> TFLite Micro Invoke()
  -> parseable Serial CSV prediction output + LED state
```

Arduino compile/upload command, if Arduino CLI is available:

```powershell
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble arduino\tinyml_har_m3
arduino-cli upload -p COM_PORT --fqbn arduino:mbed_nano:nano33ble arduino\tinyml_har_m3
```

Replace `COM_PORT` with the board port. Final hardware metrics must come from
the real Arduino compile/run, not laptop replay.

Returned hardware environment:

| Item | Value |
|---|---|
| Arduino IDE | 2.3.8 |
| Board package | Arduino Mbed OS Nano Boards 4.5.0 |
| IMU library | Arduino_LSM9DS1 1.1.1 |
| TFLite Micro library | Harvard_TinyMLx 1.2.4-Alpha |

Hardware metrics are saved in `outputs/live_evidence/hardware_metrics.json`.

## Live Serial Evidence

Live Serial logs were scored with:

```powershell
python -m src.reporting.score_live_serial_trials --input outputs\live_evidence\raw_teammate_return\7_control_trials --condition right_pocket_controlled --output-dir outputs\live_evidence
python -m src.reporting.score_live_serial_trials --input outputs\live_evidence\raw_teammate_return\8_robust_trials --condition left_pocket_robustness --output-dir outputs\live_evidence
```

| Condition | Source | Rows | Accuracy | Macro F1 | Main Failure |
|---|---|---:|---:|---:|---|
| Right pocket controlled | Live Arduino Serial | 125 | 0.9040 | 0.9089 | `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |
| Left pocket robustness | Live Arduino Serial | 91 | 0.8901 | 0.8826 | `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |

Per-class live result summary:

| Condition | WALKING | UPSTAIRS | DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---:|---:|---:|---:|---:|---:|
| Right pocket controlled | 20/20 | 13/25 | 20/20 | 20/20 | 20/20 | 20/20 |
| Left pocket robustness | 10/10 | 6/12 | 18/22 | 22/22 | 15/15 | 10/10 |

Saved outputs:

```text
docs/tables/m3_live_serial_results.md
outputs/live_evidence/right_pocket_controlled_metrics.json
outputs/live_evidence/right_pocket_controlled_confusion_matrix.csv
outputs/live_evidence/right_pocket_controlled_per_class_metrics.csv
outputs/live_evidence/left_pocket_robustness_metrics.json
outputs/live_evidence/left_pocket_robustness_confusion_matrix.csv
outputs/live_evidence/left_pocket_robustness_per_class_metrics.csv
outputs/live_evidence/live_controlled_vs_robustness_summary.csv
```

## Challenges And Resolutions

| Challenge | What We Did | Current Status |
|---|---|---|
| UCI-only model failed on Arduino V2 | Audited timing, units, orientation, and replay predictions | Solved diagnostically: failure is domain shift, not quantization |
| Small Arduino dataset | Added V2.1 long adaptation and kept grouped validation separate | Improved controlled-domain behavior, but stair classes still need more data |
| Left/right pocket orientation difference | Kept right-pocket and left-pocket validation/evidence separate | Reported as robustness limitation |
| Quantization risk | Compared FP32 vs INT8 for the selected candidate | Solved for M3: INT8 did not materially reduce macro F1 |
| Hardware latency and memory evidence | Compiled and ran on Nano 33 BLE Sense; measured `Invoke()` over 50 calls | Solved: 34.113 ms average latency, 10,288-byte model, 61,440-byte tensor arena |
| Stair-direction confusion | Preserved per-class live results and failure analysis | Not fully solved; main M4 target |

## Handout Coverage Check

| Handout Item | README Evidence |
|---|---|
| D1 PDF report | `tinyml_milestone3.pdf` is the main report |
| D2 working on-device demo | `arduino/tinyml_har_m3/` uses live LSM9DS1 input and TFLite Micro inference |
| D3 demo evidence | Returned videos and live Serial logs are under `outputs/live_evidence/raw_teammate_return/` |
| D4 updated code repository | Arduino sketch, model header, `.tflite`, conversion/training scripts, scoring script, and README files are present |
| R1 deployment pipeline | Deployment Pipeline section documents sensor, windowing, preprocessing, TFLite Invoke, and output |
| R2 deployment metrics | Final M3 Result and Quantization sections report size, latency, RAM/tensor arena, flash, and library versions |
| R3 optimization | Quantization And Optimization Evidence shows before/after numbers and tradeoffs |
| R4 on-device accuracy | Live Serial Evidence reports right-pocket controlled live accuracy and per-class counts |
| R5 robustness test | Live Serial Evidence reports left-pocket robustness with at least 10 rows per class |
| R6 challenges | Challenges And Resolutions documents domain gap, placement sensitivity, and stair-direction weakness |
| R7 M4 plan | Final report and challenge notes identify more stair data, fixed orientation, calibration, and orientation-aware features |

Placement guidance from returned testing:

- Use right pocket for the controlled demo.
- Keep board orientation fixed within each trial.
- Avoid rotating the board between activities.
- Static postures depend strongly on board angle.
- Excessive leg lifting can look like `WALKING_DOWNSTAIRS`.
- Stair direction remains the main dynamic-class weakness.

## Key Artifacts

```text
tinyml_milestone3.pdf
README_MILESTONE3.md
arduino/tinyml_har_m3/tinyml_har_m3.ino
arduino/tinyml_har_m3/model_data.h
outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite
outputs/deployment/m3_retrained_with_v2_1/selected_candidate/
outputs/live_evidence/hardware_metrics.json
outputs/live_evidence/raw_teammate_return/
docs/tables/m3_retrained_model_comparison.md
docs/tables/m3_live_serial_results.md
docs/tables/m3_live_validation_results.md
docs/m3_submission_checklist.md
docs/m3_handout_compliance_check.md
```

## Quick Commands

Dataset inspection:

```powershell
python -m src.data.inspect_datasets
python -m src.data.arduino_collectdata
python -m src.data.audit_arduino_domain
```

M3 improvement probes:

```powershell
python -m src.training.train_m3_second_improvements --experiment all --evaluate-existing --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Final M3 retraining, INT8 packaging, and header regeneration:

```powershell
python -m src.training.train_m3_retrained_with_v2_1 --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Live Serial scoring:

```powershell
python -m src.reporting.score_live_serial_trials --input outputs\live_evidence\raw_teammate_return\7_control_trials --condition right_pocket_controlled --output-dir outputs\live_evidence
python -m src.reporting.score_live_serial_trials --input outputs\live_evidence\raw_teammate_return\8_robust_trials --condition left_pocket_robustness --output-dir outputs\live_evidence
```

## Final Submission

Submit `tinyml_milestone3.pdf` as the main report and attach the returned
videos/logs with the submission. Keep live Serial logs separate from training
data; do not retrain on them before reporting M3.

Before final upload, check two handout-format details:

- The handout asks for a 3-5 page PDF report. The current
  `tinyml_milestone3.pdf` is 6 pages, so compress it if the instructor enforces
  the page range strictly.
- The handout asks for a 30-90 second video or Serial log. The returned
  stability video is about 156 seconds, so submit the proof clips/Serial logs or
  edit a shorter demo clip if needed.
