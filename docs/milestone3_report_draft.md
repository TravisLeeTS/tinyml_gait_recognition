# Milestone 3 Report Draft

Project: Energy-Efficient Human Activity Recognition on the Edge

Track: Track B, open dataset. UCI-HAR remains the public source-domain benchmark. Arduino V2 and V2.1 are target-domain adaptation data. Held-out `right_60s` and `left_30s` raw sensor replay results are reported separately from true live Arduino Serial evidence. The teammate returned compile, boot, latency, stability, demo, and live Serial trial evidence for the selected M3 candidate.

The broader study has three model roles. Phase 1 reproduces the paper-style stacked baseline offline. Phase 2 selects a lightweight DS-CNN family for TinyML deployment. Milestone 3 studies UCI-HAR-to-Arduino domain shift, then compares UCI-only, Arduino-only, UCI pretrain plus Arduino fine-tuning, mixed 6-channel, mixed focal 6-channel, and mixed 10-channel gravity candidates before packaging the selected model for live Arduino evidence collection.

## R1 Deployment Pipeline

The M3 candidate is the retrained mixed 6-channel DS-CNN `m3_v2_1_mixed_6ch`. It is selected as an evidence-collection build for controlled right-pocket live trials, not as a fully robust final model.

```text
TensorFlow/Keras training
  -> full-integer INT8 TFLite post-training quantization
  -> TFLite Micro C array in arduino/tinyml_har_m3/model_data.h
  -> Arduino Nano 33 BLE Sense
  -> LSM9DS1 live accelerometer + gyroscope
  -> 50 Hz ring buffer
  -> 128-sample window, 64-sample stride
  -> normalize + INT8 input quantization
  -> TFLite Micro Invoke()
  -> Serial prediction output + simple LED state
```

Pipeline diagram: `LSM9DS1 -> 50 Hz ring buffer -> normalize -> INT8 input -> TFLite Micro Invoke -> Serial/LED output`.

The Arduino sketch reads LSM9DS1 accelerometer and gyroscope samples at a scheduled 50 Hz target cadence. It keeps a 128-sample ring buffer and runs inference every 64 new samples. Gyroscope values are converted from deg/s to rad/s, then the six channels are standardized with constants embedded in `model_data.h`. The sketch quantizes each normalized feature using the exported TFLite input scale and zero point, calls `interpreter->Invoke()`, and prints parseable Serial rows containing timestamp, window id, prediction id, label, confidence, latency, and running average latency.

Conversion and packaging are performed by `src/training/train_m3_retrained_with_v2_1.py`. The deployed files are `arduino/tinyml_har_m3/tinyml_har_m3.ino` and `arduino/tinyml_har_m3/model_data.h`.

## R2 Deployment Metrics

Deployment-relevant model size is measured from the INT8 `.tflite` and generated C header, not from a Keras training file. Hardware metrics below come from the teammate's Arduino IDE compile output, boot Serial metadata, and returned stability evidence.

| Metric | Value | Status / Source |
|---|---:|---|
| INT8 `.tflite` size | 10,288 bytes / 10.05 KB | `outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite` |
| `model_data.h` file size | 65,932 bytes | Generated C array and constants |
| Tensor arena size | 61,440 bytes / 60.00 KB | Boot Serial metadata |
| Arduino compile flash usage | 177,504 bytes / 173.34 KB / 18% of 983,040 bytes | Arduino IDE compile output |
| Arduino compile RAM usage | 111,144 bytes / 108.56 KB / 42% of 262,144 bytes | Arduino IDE compile output |
| Average `Invoke()` latency over 50 calls | 34,113 us / 34.113 ms | Serial `latency_summary,window_count=50` |
| 2-minute stability | Passed, about 156 seconds | `2_30_stability_video.mp4` |

Environment recorded by teammate: Arduino IDE 2.3.8, Arduino Mbed OS Nano Boards 4.5.0, Arduino_LSM9DS1 1.1.1, and Harvard_TinyMLx 1.2.4-Alpha. Laptop latency and raw sensor replay validation are not Arduino hardware metrics.

## R3 Optimization: Baseline vs Improved

The M3 optimization is target-domain adaptation plus INT8 deployment packaging. UCI-HAR is still the source benchmark, but the demo environment is closer to the Arduino V2/V2.1 pocket captures.

| Metric | M2 UCI-only DS-CNN | M3 V2.1 Mixed 6-channel INT8 | Change / Tradeoff |
|---|---:|---:|---|
| UCI-HAR test accuracy | 0.9169 | 0.7984 | Lower source-domain score |
| UCI-HAR test macro F1 | 0.9173 | 0.7960 | Lower source-domain score |
| Arduino V2 grouped macro F1 | UCI-only replay collapsed | 0.7309 | Large Arduino-domain improvement |
| Held-out right_60s replay macro F1 | Not measured for M2 | 0.7741 | Controlled replay candidate |
| Held-out left_30s replay macro F1 | Not measured for M2 | 0.3886 | Major robustness gap |
| TFLite size | 13,460 bytes | 10,288 bytes | INT8 reduces deployment model size |

Data improvement: V2.1 added four 5-minute adaptation captures for `WALKING`, `SITTING`, `STANDING`, and `LAYING`. The training pipeline uses mixed UCI-HAR plus Arduino adaptation data and caps long-session windows so the new non-stair data does not dominate stair classes.

Model-selection sanity check: `m3_v2_1_mixed_6ch` is selected because it is the strongest controlled-demo candidate among UCI-retaining mixed models and has `0.7741` right_60s INT8 macro F1. Arduino-only reaches `0.8050` right_60s INT8 macro F1, but is rejected because UCI-HAR INT8 macro F1 collapses to `0.1082` and static-posture failures remain. The 10-channel gravity model is a robustness candidate with stronger left_30s INT8 macro F1 (`0.6856`), but it is not selected for M3 deployment because right_60s is lower (`0.5655`) and the Arduino path is more complex.

| V2.1 Candidate | UCI INT8 Macro F1 | V2 Group INT8 Macro F1 | right_60s INT8 Macro F1 | left_30s INT8 Macro F1 | Decision |
|---|---:|---:|---:|---:|---|
| UCI-only baseline | 0.9025 | 0.0587 | 0.0526 | 0.0535 | Reject: Arduino replay collapses |
| Arduino-only | 0.1082 | 0.4686 | 0.8050 | 0.5094 | Reject: source benchmark retention collapses |
| UCI pretrain + V2.1 fine-tune | 0.4630 | 0.1986 | 0.2582 | 0.3442 | Reject: weak target-domain validation |
| Mixed 6-channel DS-CNN | 0.7960 | 0.7309 | 0.7741 | 0.3886 | Selected controlled-demo evidence build |
| Mixed focal 6-channel DS-CNN | 0.7396 | 0.7099 | 0.7357 | 0.5643 | Intermediate: weaker right_60s |
| Mixed 10-channel gravity DS-CNN | 0.8832 | 0.6189 | 0.5655 | 0.6856 | Robustness candidate; not selected for M3 deployment |

The selected model is now confirmed to compile and run on Arduino with live LSM9DS1 input. The earlier raw left-pocket replay was weak, but the returned live Serial robustness result is stronger. Stair direction remains the main weakness.

## R4 On-Device Accuracy

The handout requires live Arduino Serial predictions for on-device accuracy. The teammate returned right-pocket controlled Serial logs, which were scored by `src/reporting/score_live_serial_trials.py`. The rows below separate public benchmark, raw replay validation, and true live evidence.

| Evaluation Setting | Accuracy | Macro F1 | Weighted F1 | Evidence Type |
|---|---:|---:|---:|---|
| UCI-HAR official test, M2 baseline | 0.9169 | 0.9173 | 0.9168 | Laptop benchmark |
| UCI-HAR official test, M3 INT8 candidate | 0.7984 | 0.7960 | - | Laptop TFLite benchmark |
| Arduino V2 grouped replay, M3 INT8 | 0.7573 | 0.7309 | - | Held-out grouped raw sensor replay |
| right_60s replay, M3 INT8 | 0.8708 | 0.7741 | - | Held-out controlled raw sensor replay |
| Right-pocket controlled live Serial | 0.9040 | 0.9089 | 0.8999 | Arduino live inference, 125 scored rows |

`right_60s` is held-out raw Arduino sensor replay validation and is not live Arduino Serial accuracy. The true live row comes from Arduino Serial predictions produced by the sketch running on live LSM9DS1 input. The controlled live confusion matrix is saved at `outputs/live_evidence/right_pocket_controlled_confusion_matrix.csv`; the main failure is `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` with 12 of 25 upstairs windows predicted as downstairs.

The raw controlled Serial logs are preserved in `outputs/live_evidence/raw_teammate_return/7_control_trials/` and included in the report-finalization zip.

Right-pocket controlled live confusion matrix:

| Actual \ Predicted | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---:|---:|---:|---:|---:|---:|
| WALKING | 20 | 0 | 0 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 0 | 13 | 12 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 0 | 0 | 20 | 0 | 0 | 0 |
| SITTING | 0 | 0 | 0 | 20 | 0 | 0 |
| STANDING | 0 | 0 | 0 | 0 | 20 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 20 |

Right-pocket controlled per-class metrics:

| Class | Precision | Recall | F1 | Support | Dominant Confusion |
|---|---:|---:|---:|---:|---|
| WALKING | 1.0000 | 1.0000 | 1.0000 | 20 | None |
| WALKING_UPSTAIRS | 1.0000 | 0.5200 | 0.6842 | 25 | WALKING_DOWNSTAIRS, 12 |
| WALKING_DOWNSTAIRS | 0.6250 | 1.0000 | 0.7692 | 20 | None |
| SITTING | 1.0000 | 1.0000 | 1.0000 | 20 | None |
| STANDING | 1.0000 | 1.0000 | 1.0000 | 20 | None |
| LAYING | 1.0000 | 1.0000 | 1.0000 | 20 | None |

## R5 Robustness Test

The robustness condition is left pocket, which differs from the controlled right-pocket condition. The held-out raw replay predicted a major gap, but the returned live Serial robustness result is much stronger.

| Condition | Accuracy | Macro F1 | Evidence Type | Main Failure |
|---|---:|---:|---|---|
| right_60s controlled replay, INT8 | 0.8708 | 0.7741 | Raw sensor replay | `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |
| left_30s robustness replay, INT8 | 0.4348 | 0.3886 | Raw sensor replay | `SITTING -> LAYING`, `LAYING -> STANDING` |
| Right-pocket controlled live Serial | 0.9040 | 0.9089 | Arduino live inference, 125 rows | `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |
| Left-pocket live Serial robustness | 0.8901 | 0.8826 | Arduino live inference, 91 rows | `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |

The live robustness score is close to the controlled score, and static postures were strong in the returned live logs: `SITTING`, `STANDING`, and `LAYING` were all correct in the robustness files. The remaining R5 limitation is stair-direction separation: `WALKING_UPSTAIRS` is often predicted as `WALKING_DOWNSTAIRS`. M4 should collect more stair examples and test features that better separate upstairs from downstairs motion.

The raw robustness Serial logs are preserved in `outputs/live_evidence/raw_teammate_return/8_robust_trials/` and included in the report-finalization zip.

Left-pocket robustness live confusion matrix:

| Actual \ Predicted | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---:|---:|---:|---:|---:|---:|
| WALKING | 10 | 0 | 0 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 0 | 6 | 6 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 2 | 0 | 18 | 0 | 2 | 0 |
| SITTING | 0 | 0 | 0 | 22 | 0 | 0 |
| STANDING | 0 | 0 | 0 | 0 | 15 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 10 |

Left-pocket robustness per-class metrics:

| Class | Precision | Recall | F1 | Support | Dominant Confusion |
|---|---:|---:|---:|---:|---|
| WALKING | 0.8333 | 1.0000 | 0.9091 | 10 | None |
| WALKING_UPSTAIRS | 1.0000 | 0.5000 | 0.6667 | 12 | WALKING_DOWNSTAIRS, 6 |
| WALKING_DOWNSTAIRS | 0.7500 | 0.8182 | 0.7826 | 22 | WALKING, 2 and STANDING, 2 |
| SITTING | 1.0000 | 1.0000 | 1.0000 | 22 | None |
| STANDING | 0.8824 | 1.0000 | 0.9375 | 15 | None |
| LAYING | 1.0000 | 1.0000 | 1.0000 | 10 | None |

Deployment placement guidance from teammate testing: the model is highly sensitive to board placement and orientation. The recommended controlled setup is the right pocket. If the left pocket is used, the board should be oriented to resemble the right-pocket orientation as closely as possible. Static postures are strongly tied to gravity/board angle: standing worked best with the board facing downward at about 90 degrees, sitting with a diagonal angle around 135 degrees, and laying with the board flatter/horizontal around 180 degrees. Walking should be straight and natural; excessive leg lifting can shift predictions toward `WALKING_DOWNSTAIRS`. These are real-world deployment constraints and help explain both the confusion matrices and the robustness limitations.

## R6 Challenges and Lessons Learned

The main challenge is public-dataset-to-Arduino domain shift. The UCI-HAR trained baseline performed well on the public held-out split, but Arduino V2 replay collapsed to `LAYING`. This happened in both FP32 and INT8, so quantization was not the root cause.

Likely causes are orientation and preprocessing mismatch, pocket placement, clothing, sensor hardware differences, and limited target-domain data. The domain audit found Arduino V2 `acc_x` about `-3.40` standard deviations from the UCI-HAR train standardizer mean. V2.1 long adaptation improved controlled replay, and the live Serial results improved substantially over the earlier left-pocket raw replay. This mismatch between raw replay and live Serial evidence reinforces why final M3 claims must use live Arduino Serial predictions rather than laptop replay alone.

The largest deployment challenge was not model size or latency; it was placement sensitivity. Static postures can be classified largely from gravity and board orientation, which helps `SITTING`, `STANDING`, and `LAYING` when the board angle is consistent, but makes the system sensitive to rotations between trials. Dynamic classes are also gait-sensitive: walking with exaggerated leg lifting can look like downstairs motion. Stair direction is not reliably separated yet, especially `WALKING_UPSTAIRS` versus `WALKING_DOWNSTAIRS`.

## R7 Plan for M4

Next steps for M4:

1. Collect more balanced live stair-up and stair-down examples under fixed board orientation.
2. Refine stair-direction separation with better features, calibration, or model training.
3. Repeat live testing with more users and environments.
4. Keep monitoring sitting/laying across pocket sides because earlier raw replay showed weakness even though the returned live logs improved.
5. Test a short calibration pose or orientation-aware preprocessing.
6. Explore orientation-invariant features or gravity-frame alignment.
7. Consider separate placement-specific models or a placement-calibration mode.
8. Keep right-pocket orientation as the default demo setup until robustness improves.
9. Revisit the 10-channel gravity model as a robustness candidate after more live evidence is available.
10. Use QAT only if future PTQ becomes unstable or materially degrades live validation.
