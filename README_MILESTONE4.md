# Milestone 4 Evidence Index

Project: Energy-Efficient Human Activity Recognition on the Edge  
Track: Arduino-collected TinyML HAR with public and self-collected data

This file is the M4 roadmap for graders and teammates. It explains what was completed, what evidence supports the report, how the final live Arduino logs were scored, and where each artifact is located.

## Files To Open First

- `report/main_milestone4.tex`: lightweight Overleaf-ready report source.
- `tinyml_milestone4.pdf`: compiled final M4 report PDF.
- `arduino/tinyml_har_m3/`: Arduino sketch and current `model_data.h`.
- `outputs/live_evidence/session05_m4_may14/`: May 14 M4 live Arduino logs, metrics, confusion matrices, and compile metadata.
- `outputs/live_evidence/session05_m4_may14/Session05_live_may14.zip`: Session05 evidence package to attach with the course submission and keep available in GitHub for checking.
- `outputs/m4_live_handoff_candidate/m4_group3_live_handoff.zip`: lean teammate package for live testing.
- `data/README_STANDARDIZED_HAR.md`: standardized dataset documentation.
- `M4_SUBMISSION_CHECKLIST.md`: final submission checklist.

M3 live evidence remains in `tinyml_milestone3.pdf`, `README_MILESTONE3.md`, and `outputs/live_evidence/`.

## What Was Completed For M4

- Standardized Arduino HAR v2 dataset parsed and documented.
- Leakage-aware session split defined:
  - training side: standardized sessions 3 and 4
  - holdout side: standardized session 1 right and left pocket
- Training-only augmentation tested: small noise, scaling, and mild time shift.
- Placement mitigation tested:
  - baseline 6-channel model
  - gravity-frame aligned 6-channel model
  - 10-channel gravity/orientation model
- Final M4 10-channel INT8 model packaged into `arduino/tinyml_har_m3/model_data.h`.
- Final M4 live Arduino logs imported and scored:
  - Person A controlled right pocket
  - Person A robustness left pocket
  - Person B right-pocket robustness
- M4 report source rewritten as a minimal no-package LaTeX file for Overleaf.
- Publication Intent Disclaimer added with blank student rows.

## M3 Feedback Addressed

| Feedback | M4 response |
|---|---|
| Compress report | `report/main_milestone4.tex` is concise and uses no heavy packages. |
| Remove long appendices | No long review appendix is included. |
| Discuss stair weakness | `WALKING_UPSTAIRS` / `WALKING_DOWNSTAIRS` remains explicitly discussed. |
| Revisit 10-channel gravity model | A final 10-channel gravity/orientation INT8 model is packaged and has Session05 live Arduino evidence. |
| Address placement sensitivity algorithmically | Gravity/orientation features and gravity-frame alignment were tested; controlled placement remains part of the protocol. |
| Increase or document user diversity | Standardized HAR v2 adds Arduino sessions and two inferable people; the report does not overclaim broad user diversity. |
| Document row-count asymmetry | M3 live right/left row-count imbalance is documented before interpreting macro-F1. |
| Keep engineering narrative honest | M2 baseline, M3 DS-CNN deployment, Arduino domain shift, mixed training, INT8 quantization, and live-vs-replay distinction are preserved. |

## Dataset Roles

| Dataset | M4 role | Notes |
|---|---|---|
| UCI-HAR | Public source-domain training and comparison | Official test split is excluded from training and representative quantization. |
| Arduino V2/V2.1 | Target-domain adaptation | Earlier 50 Hz Arduino data from M3 workflow. |
| Old M3 raw replay recordings | Training-side data for final M4 model | Replay data only; not live evidence. |
| Returned M3 live Serial logs | Hardware evidence only | Never used for training or model packaging. |
| Returned M4 Session05 live Serial logs | Final M4 hardware evidence only | Never used for training, quantization, or model packaging. |
| Standardized session 3 | Training-side standardized data | One person, four classes, 5 min, right pocket. |
| Standardized session 4 | Training-side standardized data | Latest/largest standardized session, all classes, right/left pocket, two inferable people. |
| Standardized session 1 | Held-out M4 replay robustness evaluation | Right and left pocket. Never used for training, augmentation, packaging, or representative quantization. |

Standardized HAR v2 summary:

- 84 sensor files
- 275,174 valid rows
- 3 sessions
- 2 inferable people
- right-pocket and left-pocket placements
- about 4,170 raw-row-estimated 128/64 windows

Session 2 is not used because it was removed from the v2 archive and was described as non-50 Hz.

## Final M4 Candidate

Plain-English name: final standardized-session 10-channel INT8 model  
Model artifact name: `m4_session3_4_gravity_10ch_int8`

Training-side data:

- UCI-HAR training data
- Arduino V2/V2.1
- old M3 raw replay recordings
- standardized session 3
- standardized session 4

Held-out evaluation data:

- standardized session 1 right pocket
- standardized session 1 left pocket

Leakage guardrail:

- no standardized session 1 windows are used for training
- no standardized session 1 windows are augmented
- no standardized session 1 windows are used for representative INT8 quantization
- no returned live Serial logs are used for training

Important limitation: this is session-level validation, not strict unseen-person validation.

## Key Results

| Evidence | Accuracy | Macro-F1 | Status |
|---|---:|---:|---|
| M3 right-pocket live Arduino Serial | 0.9040 | 0.9089 | Real live evidence |
| M3 left-pocket live Arduino Serial | 0.8901 | 0.8826 | Real live robustness evidence |
| M4 broad external replay diagnostic | 0.7692 | 0.6538 | Diagnostic before adding standardized training sessions |
| Final M4 standardized-session 10-channel INT8 model | Right: 0.9576; Left: 0.9913 | Right: 0.9472; Left: 0.9876 | Held-out standardized session 1 right/left replay, not live evidence |
| Final M4 10-channel INT8, Person A controlled right | 1.0000 | 1.0000 | Session05 live Arduino Serial, 132 rows |
| Final M4 10-channel INT8, Person A robustness left | 1.0000 | 1.0000 | Session05 live Arduino Serial, 137 rows |
| Final M4 10-channel INT8, Person B robustness right | 0.9931 | 0.9928 | Session05 live Arduino Serial, 145 rows |
| Final M4 10-channel INT8, aggregate Session05 | 0.9976 | 0.9974 | 414 live rows; mean `Invoke()` latency 35.643 ms |

## M4 Model-Selection Comparison

The 0.7692 / 0.6538 diagnostic used the standardized dataset as a broad external replay check. For model selection, the fair comparison is the table below: every model variant is evaluated on the same held-out standardized session 1 right pocket and left pocket sets.

| M4 variant | Features and training change | Right holdout Acc./F1 | Left holdout Acc./F1 | Decision |
|---|---|---:|---:|---|
| Base 6-channel + augmentation | Raw `ax, ay, az, gx, gy, gz`; training-only noise, scaling, and mild time shift | 0.9746 / 0.9663 | 0.7565 / 0.7181 | Strong right-pocket replay, but left-pocket transfer remained weak |
| Gravity-frame aligned 6-channel + augmentation | Same six channels after per-window gravity-frame alignment | 0.7712 / 0.7248 | 0.6087 / 0.6469 | Evaluated algorithmic placement mitigation, but not selected |
| 10-channel gravity/orientation + augmentation, FP32 | Six raw IMU channels plus gravity direction x/y/z and acceleration magnitude | 0.7881 / 0.7441 | 0.9826 / 0.9753 | Improved left-pocket replay, but right-pocket replay was weaker before packaging |
| Packaged 10-channel INT8 model | Same 10-channel features exported to TFLite Micro and `model_data.h` | 0.9576 / 0.9472 | 0.9913 / 0.9876 | Selected from replay evidence and then used for Session05 live Arduino testing |

What M4 changed:

- More standardized Arduino data were moved into the training side: session 3 and session 4.
- Session 1 stayed fully held out for the two final replay robustness checks.
- Augmentation was applied only to training windows, never to validation, test, or held-out session 1.
- Placement sensitivity was addressed through both protocol controls and algorithmic features: gravity-frame alignment was tested, and the selected model adds explicit orientation features.

Current deployment package:

```text
arduino/tinyml_har_m3/tinyml_har_m3.ino
arduino/tinyml_har_m3/model_data.h
outputs/m4_live_handoff_candidate/models/m4_session3_4_gravity_10ch_int8.tflite
outputs/m4_live_handoff_candidate/m4_group3_live_handoff.zip
```

## Confusion Matrix Artifacts

Use these in the report if space allows:

- M4 right-pocket offline holdout: `outputs/m4_live_handoff_candidate/metrics/m4_session3_4_gravity_10ch_int8_session1_right_pocket_int8_confusion_matrix.csv`
- M4 left-pocket offline holdout: `outputs/m4_live_handoff_candidate/metrics/m4_session3_4_gravity_10ch_int8_session1_left_pocket_int8_confusion_matrix.csv`
- M4 Session05 Person A controlled right live: `outputs/live_evidence/session05_m4_may14/metrics/person_a_controlled_right_confusion_matrix.csv`
- M4 Session05 Person A robustness left live: `outputs/live_evidence/session05_m4_may14/metrics/person_a_robustness_left_confusion_matrix.csv`
- M4 Session05 Person B robustness right live: `outputs/live_evidence/session05_m4_may14/metrics/person_b_robustness_right_confusion_matrix.csv`
- M3 right-pocket live Arduino evidence: `outputs/live_evidence/right_pocket_controlled_confusion_matrix.csv`
- M3 left-pocket live Arduino evidence: `outputs/live_evidence/left_pocket_robustness_confusion_matrix.csv`

Rows are true labels and columns are predicted labels.

## M4 Session05 Live Evidence

Main files:

- Summary table: `outputs/live_evidence/session05_m4_may14/session05_m4_live_summary.md`
- Combined metrics JSON: `outputs/live_evidence/session05_m4_may14/session05_m4_live_overall_metrics.json`
- Normalized prediction rows: `outputs/live_evidence/session05_m4_may14/session05_m4_live_predictions_normalized.csv`
- Compile and boot metadata: `outputs/live_evidence/session05_m4_may14/compile_and_boot_metadata.md`
- Original shared zip / submission attachment: `outputs/live_evidence/session05_m4_may14/Session05_live_may14.zip`

Compile and upload evidence:

- Flash: 178,264 bytes / 18 percent.
- Dynamic memory: 123,432 bytes / 47 percent, leaving 138,712 bytes for local variables.
- Written to flash: 178,272 bytes / 44 pages.

Live latency evidence:

- Mean `Invoke()` latency: 35.643 ms over 414 scored rows.
- Median `Invoke()` latency: 35.641 ms.
- Supplied logs include 30-second `run_timer` rows; no separate 120-second `stability_check` row was present in the imported logs.

## Commands

Parse standardized dataset:

```powershell
python scripts/parse_standardized_har_dataset.py --input TINYML_HAR_DATA_STANDARDIZED_v2.zip --output results/standardized_dataset_summary
```

Run final standardized-session experiment:

```powershell
python -m src.training.train_m4_session_split_experiment --standardized-input TINYML_HAR_DATA_STANDARDIZED_v2.zip --include-session3-train --run-augmentation --epochs 8 --patience 3 --batch-size 64 --device cpu --output-dir outputs/m4_session_split_with_session3_experiment
```

Package final M4 model:

```powershell
python -m src.deployment.package_m4_session_split_candidate --standardized-input TINYML_HAR_DATA_STANDARDIZED_v2.zip
```

Flash Arduino:

```powershell
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble arduino/tinyml_har_m3
arduino-cli upload -p COM_PORT --fqbn arduino:mbed_nano:nano33ble arduino/tinyml_har_m3
```

Serial baud: `115200`.

## Remaining Evidence Notes

- The final report now includes Session05 live Arduino results.
- A longer continuous run with an explicit 120-second `stability_check` line would strengthen the stability claim, but it is not present in the imported Session05 logs.
- The M4 live results are strong, but they cover two people and controlled/robustness pocket conditions only; broad production robustness is not claimed.

## Manual Actions

- Each student must fill their own publication choice.
- Each student must sign and date their own row.
- Review the final PDF after compiling `report/main_milestone4.tex`.
