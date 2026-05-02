# M3 Handout Compliance Check

Project: Energy-Efficient Human Activity Recognition on the Edge

Source of truth: `Project_Milestone3_Handout.pdf`. This check maps the current repository and returned teammate evidence to the handout deliverables and R1-R7 report requirements.

| Requirement | Status | Evidence/File | Notes |
|---|---|---|---|
| D1 PDF report, 3-5 pages | Pending | `docs/milestone3_report_draft.md` | Markdown report is updated and follows R1-R7. Export final PDF before submission. |
| D2 working on-device demo with live sensor input | Complete | `arduino/tinyml_har_m3/tinyml_har_m3.ino`, `outputs/live_evidence/raw_teammate_return/` | Teammate returned Arduino compile/run evidence and live Serial prediction logs. |
| D3 30-90 second video or Serial log showing live inference | Mostly complete | `outputs/live_evidence/raw_teammate_return/2_30_stability_video.mp4`, `sitting_proof.mp4`, `standing_proof.mp4`, `walking_proof.mp4` | Stability video is about 156 seconds, so it is useful supporting evidence but longer than the requested 30-90 seconds. Submit proof clips together or edit a 30-90 second demo clip from them. |
| D4 updated code repository | Complete | `README.md`, `README_MILESTONE3.md`, `src/`, `arduino/tinyml_har_m3/`, `outputs/deployment/m3_retrained_with_v2_1/` | Repo contains training code, packaging/conversion code, Arduino sketch, model header, selected `.tflite`, README instructions, metrics, and report draft. |
| R1 deployment pipeline | Complete | `docs/milestone3_report_draft.md`, `arduino/tinyml_har_m3/README.md` | Report describes Keras -> INT8 TFLite -> C array -> TFLite Micro, LSM9DS1, 50 Hz, 128-sample windows, 64-sample stride, normalization, Serial/LED output. |
| R2 deployment metrics: model size | Complete | `outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite`, `outputs/live_evidence/hardware_metrics.json` | INT8 model size is 10,288 bytes. |
| R2 deployment metrics: latency over >=50 Invoke calls | Complete | `outputs/live_evidence/hardware_metrics.json`, `filling_todolist.docx` | Average `Invoke()` latency is 34,113 us / 34.113 ms over 50 inferences. |
| R2 deployment metrics: RAM/tensor arena | Complete | `outputs/live_evidence/hardware_metrics.json`, `docs/m3_hardware_metrics_template.md` | Tensor arena is 61,440 bytes; compile RAM usage is 111,144 bytes / 42%. |
| R2 compile flash usage | Complete | `outputs/live_evidence/hardware_metrics.json`, `docs/milestone3_report_draft.md` | Sketch uses 177,504 bytes / 18% of program storage. |
| R3 optimization with before/after comparison | Complete | `docs/milestone3_report_draft.md`, `README_MILESTONE3.md`, `docs/tables/m3_retrained_model_comparison.md` | Shows M2 UCI-only DS-CNN versus M3 V2.1 mixed 6-channel INT8 candidate, including target-domain adaptation and INT8 size improvement. |
| R4 live on-device accuracy, controlled condition | Complete | `outputs/live_evidence/right_pocket_controlled_metrics.json`, `outputs/live_evidence/right_pocket_controlled_confusion_matrix.csv` | Right-pocket controlled live Serial: 125 scored rows, 0.9040 accuracy, 0.9089 macro F1. Each class has at least 20 rows; `WALKING_UPSTAIRS` has 25. |
| R5 robustness test under new condition | Complete | `outputs/live_evidence/left_pocket_robustness_metrics.json`, `outputs/live_evidence/left_pocket_robustness_confusion_matrix.csv` | Left-pocket robustness live Serial: 91 scored rows, 0.8901 accuracy, 0.8826 macro F1. Each class has at least 10 rows. |
| R6 challenges and lessons learned | Complete | `docs/milestone3_report_draft.md`, `README_MILESTONE3.md` | Documents UCI-to-Arduino domain shift, placement sensitivity, orientation dependence, stair-direction confusion, and replay versus live Serial mismatch. |
| R7 plan for M4 | Complete | `docs/milestone3_report_draft.md` | Includes more stair data, fixed orientation, calibration pose, orientation-aware features, placement-specific models/calibration, and M4 robustness work. |
| README with compile/flash/run instructions | Complete | `README.md`, `README_MILESTONE3.md`, `arduino/tinyml_har_m3/README.md` | Main README points to M3 docs; Arduino README explains setup, Serial output, placement, and evidence collection. |
| Arduino sketch | Complete | `arduino/tinyml_har_m3/tinyml_har_m3.ino` | Live LSM9DS1 inference sketch with 50 Hz sampling, ring buffer, normalization, INT8 input, TFLite Micro Invoke, Serial output, LED state, timer lines. |
| Model file/header | Complete | `arduino/tinyml_har_m3/model_data.h`, `outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite` | Selected model is `m3_v2_1_mixed_6ch` INT8. |
| Conversion/packaging script | Complete | `src/deployment/package_m3_candidate.py`, `src/training/train_m3_retrained_with_v2_1.py` | Training workflow exports INT8 candidate and regenerates Arduino header. |
| Updated training code | Complete | `src/training/train_m3_retrained_with_v2_1.py`, `src/training/train_m3_target_domain.py`, `src/training/train_m3_mixed_arduino.py` | Contains target-domain adaptation and V2.1 retraining workflows. |
| Live scoring utility | Complete | `src/reporting/score_live_serial_trials.py` | Parses returned raw Serial logs and normalized trial CSVs. Ignores metadata/timer lines and scores by displayed `prediction_label`. |
| Placement/orientation limitations | Complete | `docs/milestone3_report_draft.md`, `README_MILESTONE3.md`, `arduino/tinyml_har_m3/README.md` | Reports right-pocket recommendation, left-pocket orientation matching, static posture angle sensitivity, walking style sensitivity, and stair-direction weakness. |

## Remaining Action Items Before Submission

- Export the final PDF from `docs/milestone3_report_draft.md`.
- Attach or upload the returned demo videos and/or Serial logs with the submission.
- Decide whether to submit the 156-second stability video only as supporting evidence or edit/submit a shorter 30-90 second demo clip from the proof videos.
- Ensure large video files are not committed to git if the course repository should stay lightweight.
