# Arduino Sketch for M3/M4 TinyML HAR

This folder contains the live-sensor Arduino sketch for the Nano 33 BLE Sense HAR deployment. The sketch was used for the returned M3 live evidence. For M4, `model_data.h` has been updated to the latest packaged session3+4 gravity/orientation INT8 model and was tested in the May 14 Session05 live run.

Important distinction:

- The M3 mixed 6-channel INT8 model has real returned live Arduino evidence.
- The M4 session3+4 10-channel INT8 model now has Session05 live Arduino evidence in `outputs/live_evidence/session05_m4_may14/`.

## Files

```text
tinyml_har_m3.ino   Arduino Nano 33 BLE Sense sketch
model_data.h        generated INT8 TFLite model C array and normalization constants
```

Regenerate the M3 evidence model with:

```powershell
python -m src.training.train_m3_retrained_with_v2_1 --experiment all --epochs 40 --patience 8 --device cpu --export-tflite --benchmark-host
```

Regenerate the current M4 packaged model with:

```powershell
python -m src.deployment.package_m4_session_split_candidate --standardized-input TINYML_HAR_DATA_STANDARDIZED_v2.zip
```

## Arduino Setup

Use Arduino IDE or Arduino CLI with:

- Board: Arduino Nano 33 BLE Sense
- Board package: Arduino Mbed OS Nano Boards
- Libraries: `Arduino_LSM9DS1` and Arduino TensorFlow Lite / TensorFlow Lite Micro
- Serial monitor: `115200` baud

Compile and upload with Arduino CLI, if available:

```powershell
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble arduino\tinyml_har_m3
arduino-cli upload -p COM_PORT --fqbn arduino:mbed_nano:nano33ble arduino\tinyml_har_m3
```

Replace `COM_PORT` with the board port shown by Arduino IDE or `arduino-cli board list`.

The returned M3 evidence used Arduino IDE 2.3.8, Arduino Mbed OS Nano Boards 4.5.0, Arduino_LSM9DS1 1.1.1, and Harvard_TinyMLx 1.2.4-Alpha.

## Quantization

The current sketch uses a full-integer INT8 post-training-quantized TFLite model. The packaged M4 INT8 `.tflite` is 10,432 bytes / 10.19 KiB. The generated `model_data.h` source file is larger because it stores the model as C-array text plus normalization constants.

QAT was not used. INT8 PTQ preserved a compact deployment size. The main remaining technical risk is stair-direction confusion under broader users and placements, not only quantization loss.

## Current M4 Candidate

| Field | Value |
|---|---|
| Candidate | `m4_session3_4_gravity_10ch_int8` |
| Training policy | UCI-HAR train + Arduino V2/V2.1 + old M3 raw replay validation + standardized sessions 3 and 4 |
| Excluded from training | Standardized session 1 holdout, UCI-HAR official test, returned live Serial logs |
| Augmentation | Training-only noise, scaling, and mild time shift |
| Input channels | 10: accelerometer, gyroscope, gravity direction, acceleration magnitude |
| INT8 size | 10,432 bytes |
| Session 1 right-pocket INT8 accuracy / macro-F1 | 0.9576 / 0.9472 |
| Session 1 left-pocket INT8 accuracy / macro-F1 | 0.9913 / 0.9876 |
| Session05 live aggregate accuracy / macro-F1 | 0.9976 / 0.9974 over 414 rows |
| Session05 mean latency | 35.643 ms |
| Session05 compile flash / dynamic memory | 178,264 bytes / 18 percent; 123,432 bytes / 47 percent |
| Main held-out failure | `WALKING_DOWNSTAIRS -> WALKING_UPSTAIRS` on right pocket; one `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` on left pocket |

Leakage guardrail: standardized session 1 was not used for training, augmentation, representative quantization, or live-model packaging. The high holdout scores are session-level replay evidence, not strict unseen-person evidence. Session05 is separate live hardware evidence.

The supplied Session05 logs include 30-second `run_timer` rows. A separate 120-second `stability_check` row was not present in the imported logs.

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

The current `model_data.h` sets `kChannelCount = 10`. The sketch computes `gravity_dir_x`, `gravity_dir_y`, `gravity_dir_z`, and `acc_magnitude` from the 128-sample acceleration window before normalization.

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

For M4 Session05, these were captured from the real board:

- Compile summary: 178,264 flash bytes / 18 percent; 123,432 dynamic-memory bytes / 47 percent.
- Boot metadata: `model_bytes=10432`, `tensor_arena_bytes=73728`, `kChannelCount=10`, `kWindowSize=128`, `kStride=64`.
- Live accuracy: Person A controlled right pocket, Person A robustness left pocket, and Person B right-pocket robustness.
- Average `latency_us`: 35.643 ms over 414 scored rows.
