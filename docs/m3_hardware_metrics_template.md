# M3 Hardware Metrics Template

Use this file to record the candidate sketch run on the Arduino Nano 33 BLE Sense. These values come from the returned Arduino build/run evidence, not from laptop replay.

Candidate build:

```text
Model: retrained mixed 6-channel DS-CNN INT8 with V2.1 long adaptation
Header: arduino/tinyml_har_m3/model_data.h
TFLite: outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite
Status: M3 candidate deployment build for live evidence collection, not fully validated
```

## Compile Metrics

| Metric | Value | Source / Screenshot |
|---|---:|---|
| Sketch flash usage | 177,504 bytes / 173.34 KB / 18% of 983,040 bytes | Arduino IDE compile output |
| Sketch RAM usage | 111,144 bytes / 108.56 KB / 42% of 262,144 bytes | Arduino IDE compile output |
| Model byte size | 10,288 bytes | `g_tinyml_har_model_len` / `model_data.h` |
| Tensor arena size | 61,440 bytes | `kTensorArenaSize` boot print |

## Runtime Metrics

| Metric | Value | Source |
|---|---:|---|
| Average `Invoke()` latency over >=50 windows | 34,113 us / 34.113 ms over 50 windows | Serial `latency_summary` line |
| Median / typical confidence | Captured in Serial logs | Serial CSV output |
| 2-minute stability result | Passed, about 156-second returned video | `2_30_stability_video.mp4` |
| Crash/reset observed? | None reported in returned evidence | Teammate return |

## Required Serial Boot Lines

Record these from the Serial Monitor:

```text
model_bytes=10288
tensor_arena_bytes=61440
kChannelCount=6
kWindowSize=128
kStride=64
normalization_source=UCI-HAR train + Arduino V2 train + capped V2.1 long adaptation
input_scale=0.136206716
input_zero_point=8
output_scale=0.003906250
output_zero_point=-128
```

## Evidence Checklist

| Evidence | Status |
|---|---|
| Arduino sketch compiles | Done |
| Arduino uploads successfully | Done by teammate evidence |
| Stable 2-minute run | Done |
| At least 50 inference windows for latency average | Done |
| 60-90 second video or Serial log showing live sensor inference | Done with proof clips |
| Right-pocket controlled trial sheet completed | Done, scored from Serial logs |
| Left-pocket robustness trial sheet completed | Done, scored from Serial logs |
