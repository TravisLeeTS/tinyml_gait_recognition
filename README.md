# Energy-Efficient Human Activity Recognition on the Edge

Group 3 TinyML HAR project for Arduino Nano 33 BLE Sense.

Team members:

- Lee Ting Sen, B00103724
- Khalifa Alshamsi, B00078654
- Vineetha Addanki, G00111196

## First Read

For Milestone 4, start with:

- `tinyml_milestone4.pdf`: final compiled M4 report PDF for grading.
- `report/main_milestone4.tex`: lightweight Overleaf-ready final report source.
- `README_MILESTONE4.md`: M4 evidence index and grading map.
- `data/README_STANDARDIZED_HAR.md`: standardized Arduino dataset documentation.
- `arduino/tinyml_har_m3/`: Arduino sketch and latest packaged model header.
- `outputs/live_evidence/session05_m4_may14/`: May 14 M4 live Arduino logs, metrics, and compile metadata.

For submission checking, attach the Session05 evidence package together with the final PDF. The GitHub-safe copy is included at `outputs/live_evidence/session05_m4_may14/Session05_live_may14.zip`, with parsed metrics in the same folder.

Milestone 3 live evidence remains available in `tinyml_milestone3.pdf`, `README_MILESTONE3.md`, and `outputs/live_evidence/`.

## 1. Project Title and Team Members

Project title: Energy-Efficient Human Activity Recognition on the Edge.

Track: mixed public and Arduino-collected TinyML HAR system.

Target classes:

- `LAYING`
- `SITTING`
- `STANDING`
- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`

## 2. Project Description

This repository implements a TinyML human activity recognition system. The Arduino Nano 33 BLE Sense reads LSM9DS1 accelerometer and gyroscope samples, builds 128-sample windows at 50 Hz with stride 64, standardizes features, runs an INT8 depthwise-separable CNN with TensorFlow Lite Micro, and reports predictions over Serial plus a simple LED state.

M3 produced the first real live Arduino evidence with a 6-channel INT8 model. M4 adds standardized Arduino sessions, session-aware splitting, training-only augmentation, and a 10-channel gravity/orientation model. The May 14 Session05 live Arduino run scores the packaged M4 model across Person A controlled right pocket, Person A robustness left pocket, and Person B right-pocket robustness.

## 3. Repository Structure

```text
arduino/tinyml_har_m3/                 Arduino sketch and current model_data.h
data/README_STANDARDIZED_HAR.md        Standardized Arduino dataset documentation
docs/                                  Milestone support docs and result tables
outputs/live_evidence/                 Returned M3 and M4 live Arduino evidence
outputs/live_evidence/session05_m4_may14/
                                       M4 Session05 live logs, metrics, and compile metadata
outputs/m4_live_handoff_candidate/     Latest M4 packaged model and replay metrics
outputs/m4_session_split_with_session3_experiment/
                                       M4 session-aware training/evaluation artifacts
report/main_milestone4.tex             Lightweight M4 report source
README_MILESTONE2.md                   M2 background
README_MILESTONE3.md                   M3 evidence index
README_MILESTONE4.md                   M4 evidence index
requirements.txt                       Pinned Python dependencies
scripts/                               Dataset parser and split utilities
src/                                   Data, model, training, deployment code
```

Large raw datasets are not committed unless explicitly required for submission. The standardized HAR v2 zip is documented and parsed by scripts.

## 4. Setup Instructions

Python:

```powershell
python -m pip install -r requirements.txt
```

Arduino:

- Board: Arduino Nano 33 BLE Sense
- Board package: Arduino Mbed OS Nano Boards
- Libraries: `Arduino_LSM9DS1` and TensorFlow Lite Micro / Arduino TensorFlow Lite
- Serial baud: `115200`

The final Arduino sketch is `arduino/tinyml_har_m3/tinyml_har_m3.ino`.

## 5. How to Train

Inspect public and Arduino datasets:

```powershell
python -m src.data.inspect_datasets
```

Reproduce the M3 target-domain candidate comparison:

```powershell
python -m src.training.train_m3_retrained_with_v2_1 --epochs 40 --patience 8 --device cpu --export-tflite
```

Run the final standardized-session M4 experiment:

```powershell
python -m src.training.train_m4_session_split_experiment --standardized-input TINYML_HAR_DATA_STANDARDIZED_v2.zip --include-session3-train --run-augmentation --epochs 8 --patience 3 --batch-size 64 --device cpu --output-dir outputs/m4_session_split_with_session3_experiment
```

Split policy: standardized sessions 3 and 4 are training-side data; standardized session 1 is held out for right-pocket and left-pocket replay evaluation. Session 1 is not used for training, augmentation, model packaging, or representative quantization.

## 6. How to Convert

Package the final M4 model:

```powershell
python -m src.deployment.package_m4_session_split_candidate --standardized-input TINYML_HAR_DATA_STANDARDIZED_v2.zip
```

Current M4 artifacts:

- `outputs/m4_live_handoff_candidate/models/m4_session3_4_gravity_10ch_int8.tflite`
- `arduino/tinyml_har_m3/model_data.h`
- `arduino/tinyml_har_m3/tinyml_har_m3.ino`

The packaged M4 model has been flashed and scored in `outputs/live_evidence/session05_m4_may14/`.

## 7. How to Flash and Run

Open `arduino/tinyml_har_m3/tinyml_har_m3.ino` in Arduino IDE.

Select:

- Board: Arduino Nano 33 BLE Sense
- Port: connected board port
- Serial monitor baud: `115200`

Arduino CLI, if available:

```powershell
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble arduino/tinyml_har_m3
arduino-cli upload -p COM_PORT --fqbn arduino:mbed_nano:nano33ble arduino/tinyml_har_m3
```

Replace `COM_PORT` with the board port.

## 8. Expected Output

Boot metadata should include model bytes, tensor arena bytes, channel count, window size, stride, and quantization scales.

Prediction rows follow:

```text
timestamp_ms,window_id,prediction_id,prediction_label,confidence,latency_us,top1_score,top2_label,top2_score,avg_latency_us
12345,1,3,SITTING,0.9821,34113,0.982,WALKING,0.0042,34113
```

The sketch also emits `latency_summary`, `run_timer`, and `stability_check` lines.

## 9. Key Results Summary

| Evidence | Data source | Accuracy | Macro-F1 | Notes |
|---|---|---:|---:|---|
| M2 baseline | UCI-HAR held-out test | 0.9135 | 0.9128 | Heavy offline baseline |
| M3 deployed 6-channel INT8 | Right-pocket live Arduino Serial | 0.9040 | 0.9089 | Real live evidence, 125 rows |
| M3 deployed 6-channel INT8 | Left-pocket live Arduino Serial | 0.8901 | 0.8826 | Real live robustness evidence, 91 rows |
| M4 broad external replay diagnostic | Fully held-out standardized HAR v2 replay | 0.7692 | 0.6538 | Diagnostic before adding standardized training sessions; stair/posture confusion remained |
| Final M4 standardized-session 10-channel INT8 model | Held-out standardized session 1 right/left pockets | Right: 0.9576; Left: 0.9913 | Right: 0.9472; Left: 0.9876 | Offline INT8 replay, not live evidence |
| Final M4 standardized-session 10-channel INT8 | Person A controlled right live Arduino Serial | 1.0000 | 1.0000 | Session05, 132 rows |
| Final M4 standardized-session 10-channel INT8 | Person A left-pocket live robustness | 1.0000 | 1.0000 | Session05, 137 rows |
| Final M4 standardized-session 10-channel INT8 | Person B right-pocket live robustness | 0.9931 | 0.9928 | Session05, 145 rows; one `DOWNSTAIRS -> UPSTAIRS` |
| Final M4 standardized-session 10-channel INT8 | Combined Session05 live Arduino Serial | 0.9976 | 0.9974 | 414 rows; mean `Invoke()` latency 35.643 ms |

Fair M4 model-selection comparison on the same held-out standardized session 1 right/left pockets:

| M4 variant | Right holdout Acc./F1 | Left holdout Acc./F1 | Selection note |
|---|---:|---:|---|
| Base 6-channel + augmentation | 0.9746 / 0.9663 | 0.7565 / 0.7181 | Strong right-pocket replay, weaker left-pocket transfer |
| Gravity-frame aligned 6-channel + augmentation | 0.7712 / 0.7248 | 0.6087 / 0.6469 | Tested algorithmic placement mitigation, but did not help this split |
| 10-channel gravity/orientation + augmentation, FP32 | 0.7881 / 0.7441 | 0.9826 / 0.9753 | Strong left-pocket replay before INT8 packaging |
| Packaged 10-channel INT8 model | 0.9576 / 0.9472 | 0.9913 / 0.9876 | Best combined right/left INT8 replay, then used for Session05 live testing |

The base 6-channel model uses `ax, ay, az, gx, gy, gz`. The gravity-frame aligned 6-channel model estimates each window's gravity direction and rotates the IMU channels toward a common vertical frame. The 10-channel model keeps the six raw IMU channels and appends gravity direction x/y/z plus acceleration magnitude. Augmentation is training-only: small Gaussian noise, per-channel scaling, and mild time shift. Held-out session 1 is never augmented.

Current packaged M4 model:

- Name: `m4_session3_4_gravity_10ch_int8`
- Size: 10,432 bytes / 10.19 KiB
- Tensor arena in header: 73,728 bytes / 72.00 KiB
- Input: 128 samples by 10 channels
- Arduino compile: 178,264 flash bytes / 18 percent; 123,432 dynamic-memory bytes / 47 percent
- Live latency: 35.643 ms mean `Invoke()` latency over 414 scored Session05 rows

## 10. Known Issues

- The M4 replay scores and M4 live scores are different evidence types and should not be mixed.
- Session05 live evidence covers two people and three supplied conditions; it does not prove broad real-world robustness.
- The supplied Session05 logs include 30-second `run_timer` rows, but no separate 120-second `stability_check` row was present in the imported logs.
- `WALKING_UPSTAIRS` and `WALKING_DOWNSTAIRS` remain the most important failure pair to monitor.
- The standardized dataset has two inferable people, so it does not prove broad user diversity.
- Placement and board angle still matter. The 10-channel model adds gravity/orientation features but does not guarantee robustness.
- This is not a medical device and must not be used for clinical decisions.

## M4 Deliverables

- M4 report source: `report/main_milestone4.tex`
- M4 report PDF: `tinyml_milestone4.pdf`
- M4 evidence index: `README_MILESTONE4.md`
- M4 checklist: `M4_SUBMISSION_CHECKLIST.md`
- Standardized dataset documentation: `data/README_STANDARDIZED_HAR.md`
- Latest Arduino model header: `arduino/tinyml_har_m3/model_data.h`
- Live-test handoff zip: `outputs/m4_live_handoff_candidate/m4_group3_live_handoff.zip`
- Session05 live evidence: `outputs/live_evidence/session05_m4_may14/`
- Session05 evidence package to attach with submission: `outputs/live_evidence/session05_m4_may14/Session05_live_may14.zip`

Publication choices and signatures are intentionally blank in the report. Each student must complete their own row.
