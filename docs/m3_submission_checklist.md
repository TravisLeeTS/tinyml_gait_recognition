# M3 Submission Checklist

Project: Energy-Efficient Human Activity Recognition on the Edge

Selected M3 evidence-collection build: `m3_v2_1_mixed_6ch` INT8, embedded in `arduino/tinyml_har_m3/model_data.h`.

Important wording: this is the M3 candidate deployment build used for returned live evidence. It is not claimed as a fully final M4 model. Held-out `right_60s` and `left_30s` are raw sensor replay validation, not live Arduino Serial accuracy; the returned `7_control_trials` and `8_robust_trials` logs are true live Arduino Serial evidence.

## D1 PDF Report

| Item | Status |
|---|---|
| `docs/milestone3_report_draft.md` follows R1-R7 handout structure only | Done |
| R1 deployment pipeline included | Done |
| R2 deployment metrics table included | Done, hardware values integrated |
| R3 baseline vs improved comparison included | Done |
| R4 on-device accuracy section separates replay from live Serial evidence | Done |
| R5 robustness section documents left-pocket gap | Done |
| R6 challenges and lessons learned included | Done |
| R7 M4 plan included | Done |
| Placement/orientation limitations documented | Done |
| PDF exported from `docs/milestone3_report_draft.md` | Pending |

## D2/D3 Arduino Demo Evidence

| Item | Status |
|---|---|
| Arduino sketch present: `arduino/tinyml_har_m3/tinyml_har_m3.ino` | Done |
| Model header present: `arduino/tinyml_har_m3/model_data.h` | Done |
| Arduino sketch and model header ready for compile/upload | Done |
| Arduino sketch compiles on Nano 33 BLE Sense | Done |
| Stable 2-minute live run | Done, `2_30_stability_video.mp4`, about 156 seconds |
| 30-90 second video or Serial log showing live inference | Done, proof clips returned |
| At least 3 classes visible in demo evidence | Done: sitting, standing, walking |
| Serial timer lines for continuous run (`run_timer`, `stability_check`) | Added to sketch; stability evidence returned |

## R2 Deployment Metrics

| Metric | Status |
|---|---|
| Selected INT8 `.tflite` size: 10,288 bytes | Done |
| `model_data.h` generated and included | Done |
| Tensor arena size: 61,440 bytes | Done, boot Serial metadata |
| Arduino compile flash usage: 177,504 bytes / 18% | Done |
| Arduino compile RAM usage: 111,144 bytes / 42% | Done |
| Average `Invoke()` latency over 50 calls: 34.113 ms | Done |
| Library/board package versions recorded | Done |
| 2-minute stability evidence | Done |

## R3 Optimization Evidence

| Item | Status |
|---|---|
| Six-model V2.1 comparison table | Done |
| INT8 model-selection sanity check | Done |
| FP32 vs INT8 candidate summary | Done |
| Candidate conversion/training script present: `src/training/train_m3_retrained_with_v2_1.py` | Done |
| Live hardware tradeoffs added after teammate run | Done |

## R4 On-Device Accuracy

| Item | Status |
|---|---|
| UCI-HAR official test row | Done |
| Arduino V2 grouped replay row | Done, labeled as raw replay |
| `right_60s` held-out replay row | Done, labeled as raw replay |
| Right-pocket live Serial predictions | Done |
| Right-pocket controlled rows scored: 125 | Done |
| Live right-pocket confusion matrix | Done |
| Right-pocket live accuracy/macro F1: 0.9040 / 0.9089 | Done |

## R5 Robustness Test

| Item | Status |
|---|---|
| `left_30s` held-out replay robustness row | Done, labeled as raw replay |
| Left-pocket live Serial predictions | Done |
| Left-pocket at least 10 trials per class | Done |
| Robustness accuracy/F1 from live Serial predictions: 0.8901 / 0.8826 | Done |
| Degraded classes documented | Done, mainly `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |

## D4 Repository Quality

| Item | Status |
|---|---|
| `README.md` points to M3 details | Done |
| `README_MILESTONE3.md` updated with current selected candidate | Done |
| Arduino README with compile/run guidance | Done |
| Hardware metrics template: `docs/m3_hardware_metrics_template.md` | Done |
| Live trial protocol and sheet | Done |
| Placement guidance added to README/report/Arduino README | Done |
| Handout compliance check: `docs/m3_handout_compliance_check.md` | Done |
| No live evidence used for training | Verified in current workflow documentation |
| Laptop replay not reported as live Arduino accuracy | Verified in current README/report wording |

## Final Before Submission

| Item | Status |
|---|---|
| Receive teammate compile output, Serial logs, video, and completed trial sheet | Done |
| Score live Serial logs with `python -m src.reporting.score_live_serial_trials ...` | Done |
| Update R2/R4/R5 report placeholders with hardware/live results | Done |
| Export final PDF report | Pending |
| Submit zip/repo/report/video or Serial evidence | Pending |
