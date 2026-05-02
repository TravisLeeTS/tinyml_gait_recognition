# Milestone 3 Arduino Sketch

This folder contains the live-sensor Arduino sketch for the M3 on-device candidate build. The current `model_data.h` is the retrained mixed 6-channel INT8 candidate built with UCI-HAR train, Arduino V2 train, and capped V2.1 long-adaptation data. It was used for returned live M3 evidence. It is not a fully final M4 model because stair-direction separation remains the main live weakness and placement/orientation sensitivity is high.

## Files

```text
tinyml_har_m3.ino   Arduino Nano 33 BLE Sense sketch
model_data.h        generated INT8 TFLite model C array and normalization constants
```

Regenerate `model_data.h` from the latest V2.1 candidate workflow with:

```powershell
python -m src.training.train_m3_retrained_with_v2_1 --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

## Arduino Setup

Use Arduino IDE or Arduino CLI with:

- Board: Arduino Nano 33 BLE Sense
- Board package: Arduino Mbed OS Nano Boards
- Libraries: `Arduino_LSM9DS1` and Arduino TensorFlow Lite / TensorFlow Lite Micro
- Serial monitor: `115200` baud

Record the exact library and board-package versions from the teammate's machine in the M3 report after the first successful compile.

## Demo Output

The sketch reads the onboard IMU at a 50 Hz target cadence, fills 128-sample windows with 50% overlap, normalizes with the constants embedded in `model_data.h`, invokes the TFLite model, and prints one line per inference:

```text
timestamp_ms,window_id,prediction_id,prediction_label,confidence,latency_us,top1_score,top2_label,top2_score,avg_latency_us
12345,1,0,WALKING,0.7420,12345,0.742,WALKING_UPSTAIRS,0.1000,12345
```

The built-in LED is on for the three movement classes and off for static posture classes. Serial output is the primary class evidence.

The sketch runs live inference continuously until the board is unplugged or reset. It also prints timer lines so the stability run can be documented:

```text
run_timer,elapsed_ms=30000,elapsed_s=30,window_count=22,status=running
stability_check,elapsed_ms=120000,window_count=92,status=passed_2min_if_no_crash
```

The sketch supports the current 6-channel model and a future 10-channel gravity-feature model. If `model_data.h` is regenerated with `kChannelCount = 10`, the sketch computes `gravity_dir_x`, `gravity_dir_y`, `gravity_dir_z`, and `acc_magnitude` from the 128-sample acceleration window before normalization.

## How To Wear / Place The Board

Use right pocket for the controlled demo. Keep the board orientation consistent with the training/testing setup and avoid rotating the board between trials.

If using left pocket, mirror or approximate the right-pocket board orientation as closely as possible. The model is sensitive to pocket side and board angle.

Static postures are orientation-sensitive:

- `STANDING`: board facing downward, about 90 degrees.
- `SITTING`: board diagonal, about 135 degrees.
- `LAYING`: board flat/horizontal, about 180 degrees.

For `WALKING`, walk straight and naturally. Excessive leg lifting can be misclassified as `WALKING_DOWNSTAIRS`.

For stair tests, note that `WALKING_UPSTAIRS` and `WALKING_DOWNSTAIRS` may be confused. In the returned live evidence, stair-direction confusion was the main remaining failure mode.

## Required Hardware Measurements

For M3, these were captured from the real board:

- Stable 2-minute run with live IMU input and no crash.
- Proof videos showing sitting, standing, and walking.
- Average `latency_us` over at least 50 inference lines.
- `model_bytes` and `tensor_arena_bytes` from boot output.
- Arduino IDE compile summary for total sketch flash and RAM usage.
- Live accuracy: right-pocket controlled Serial logs.
- Robustness: left-pocket Serial logs.
