# Milestone 3 Arduino Sketch

This folder contains the live-sensor Arduino sketch for the M3 on-device prototype.

## Files

```text
tinyml_har_m3.ino   Arduino Nano 33 BLE Sense sketch
model_data.h        generated INT8 TFLite model C array and normalization constants
```

Regenerate `model_data.h` from the repo root:

```powershell
python -m src.deployment.quantization_experiment --representative-samples 512 --qat-trigger-drop 0.01
```

## Arduino Setup

Use Arduino IDE or Arduino CLI with:

- Board: Arduino Nano 33 BLE Sense
- Board package: Arduino Mbed OS Nano Boards
- Libraries: `Arduino_LSM9DS1` and Arduino TensorFlow Lite / TensorFlow Lite Micro
- Serial monitor: `115200` baud

Record the exact library and board-package versions from the teammate's machine in the M3 report after the first successful compile.

## Demo Output

The sketch reads the onboard IMU at a 50 Hz target cadence, fills 128-sample windows with 50% overlap, normalizes with the saved UCI-HAR training statistics, invokes the INT8 TFLite model, and prints one line per inference:

```text
window=1,pred=WALKING,score=0.742,latency_us=12345,avg_latency_us=12345,arena_bytes=61440
```

The built-in LED is on for the three movement classes and off for static posture classes. Serial output is the primary class evidence.

## Required Hardware Measurements

For M3, capture these from the real board:

- Stable 2-minute run with live IMU input and no crash.
- 60-90 second video or serial log showing at least 3 classes.
- Average `latency_us` over at least 50 inference lines.
- `model_bytes` and `tensor_arena_bytes` from boot output.
- Arduino IDE compile summary for total sketch flash and RAM usage.
- Live accuracy: 20 trials per class under controlled conditions.
- Robustness: 10 trials per class in a new condition, such as the other pocket or a different user/environment.
