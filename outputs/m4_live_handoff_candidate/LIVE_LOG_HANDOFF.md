# M4 Live Log Handoff

This package contains the M4 Arduino model that was handed off for live testing. The May 14 Session05 live logs have now been received and scored under `outputs/live_evidence/session05_m4_may14/`.

## What Is Packaged

- Arduino sketch: `arduino/tinyml_har_m3/tinyml_har_m3.ino`
- Updated model header: `arduino/tinyml_har_m3/model_data.h`
- INT8 TFLite model: `outputs/m4_live_handoff_candidate/models/m4_session3_4_gravity_10ch_int8.tflite`
- Packaging metadata: `outputs/m4_live_handoff_candidate/m4_live_candidate_metadata.json`
- Session-1 holdout metrics and confusion matrices under `outputs/m4_live_handoff_candidate/metrics/`

## Current Candidate

Candidate: `m4_session3_4_gravity_10ch_int8`

Training-side data:

- UCI-HAR training split
- Existing Arduino V2/V2.1 adaptation data
- Old M3 raw replay validation recordings, moved into training only after standardized HAR v2 became available
- Standardized HAR v2 session 3
- Standardized HAR v2 session 4

Held-out data:

- Standardized HAR v2 session 1 right pocket
- Standardized HAR v2 session 1 left pocket
- Returned live Serial logs
- UCI-HAR official test split

Leakage guardrail: standardized session 1 was not used for training, augmentation, representative quantization, or packaging.

Important caveat: the high offline scores are session-level holdout evidence, not strict unseen-person evidence. Report the Session05 results separately as live Arduino evidence.

## Offline INT8 Holdout Results

| Condition | Windows | Accuracy | Macro-F1 | Main failure |
|---|---:|---:|---:|---|
| Session 1 right pocket | 118 | 0.9576 | 0.9472 | `WALKING_DOWNSTAIRS -> WALKING_UPSTAIRS` |
| Session 1 left pocket | 115 | 0.9913 | 0.9876 | `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |

## Received Session05 Live Results

| Condition | Rows | Accuracy | Macro-F1 | Main failure |
|---|---:|---:|---:|---|
| Person A controlled right pocket | 132 | 1.0000 | 1.0000 | none |
| Person A robustness left pocket | 137 | 1.0000 | 1.0000 | none |
| Person B robustness right pocket | 145 | 0.9931 | 0.9928 | one `WALKING_DOWNSTAIRS -> WALKING_UPSTAIRS` |
| Aggregate Session05 | 414 | 0.9976 | 0.9974 | one total stair-direction error |

Mean `Invoke()` latency over the 414 scored rows is 35.643 ms. The compile output reports 178,264 flash bytes / 18 percent and 123,432 dynamic-memory bytes / 47 percent.

## Flash Instructions

Use Arduino IDE or Arduino CLI with:

- Board: Arduino Nano 33 BLE Sense
- Board package: Arduino Mbed OS Nano Boards
- Libraries: `Arduino_LSM9DS1` and Arduino TensorFlow Lite / TensorFlow Lite Micro
- Serial monitor: `115200` baud

Arduino CLI commands, if available:

```powershell
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble arduino/tinyml_har_m3
arduino-cli upload -p COM_PORT --fqbn arduino:mbed_nano:nano33ble arduino/tinyml_har_m3
```

Replace `COM_PORT` with the board port shown by Arduino IDE or `arduino-cli board list`.

## Live Logs Collected

The received logs cover:

- Person A controlled right-pocket trials.
- Person A left-pocket robustness trials.
- Person B right-pocket robustness trials.
- More than 50 inference rows for average `latency_us`.
- Boot metadata lines showing `model_bytes`, `tensor_arena_bytes`, `kChannelCount`, `input_scale`, and `input_zero_point`.
- Arduino IDE compile output for flash and dynamic memory usage.

Serial header:

```text
timestamp_ms,window_id,prediction_id,prediction_label,confidence,latency_us,top1_score,top2_label,top2_score,avg_latency_us
```

The supplied logs include 30-second `run_timer` rows. A separate 120-second `stability_check` line was not present in the imported logs.
