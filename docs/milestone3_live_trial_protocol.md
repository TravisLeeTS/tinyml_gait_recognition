# Milestone 3 Live Trial Protocol

This protocol is for final M3 evidence collection on the Arduino. Do not use these live evidence files for training before reporting M3 results.

## Candidate Build

- Model: retrained mixed 6-channel DS-CNN INT8 with V2.1 long adaptation.
- Input: 128 x 6 IMU window.
- Sampling: 50 Hz.
- Stride: 64 samples.
- Sensor: Arduino Nano 33 BLE Sense LSM9DS1 accelerometer and gyroscope.
- Status: M3 candidate deployment build for evidence collection, not fully scientifically validated.

## Conditions

| Condition | Folder | Trials |
|---|---|---:|
| Controlled | `data/raw/arduino_live_right_pocket_controlled/` | 20 trials per class |
| Robustness / new condition | `data/raw/arduino_live_left_pocket_new_condition/` | At least 10 trials per class |

Classes:

```text
WALKING
WALKING_UPSTAIRS
WALKING_DOWNSTAIRS
SITTING
STANDING
LAYING
```

## Procedure

1. Upload `arduino/tinyml_har_m3/tinyml_har_m3.ino` with the current `model_data.h`.
2. Open Serial Monitor at 115200 baud.
3. Save the boot metadata and CSV prediction lines.
4. Keep board orientation fixed within each condition.
5. Run the controlled condition in the right pocket: 20 trials per class.
6. Run the robustness condition in the left pocket: at least 10 trials per class.
7. Record a 60-90 second video or Serial log showing live sensor inference for at least 3 classes.
8. Complete `docs/milestone3_live_trial_sheet.csv`.
9. Score the trial sheet with:

```powershell
python -m src.reporting.score_live_serial_trials docs\milestone3_live_trial_sheet.csv
```

## Serial Output

The sketch prints:

```text
timestamp_ms,window_id,prediction_id,prediction_label,confidence,latency_us,top1_score,top2_label,top2_score,avg_latency_us
```

The sketch also prints timer/status lines while live inference continues:

```text
run_timer,elapsed_ms=30000,elapsed_s=30,window_count=22,status=running
stability_check,elapsed_ms=120000,window_count=92,status=passed_2min_if_no_crash
```

Copy the prediction label, timestamp, confidence, latency, and window id into the trial sheet. Use `timestamp_ms` and `run_timer` lines to mark trial start/end times. Evidence must come from Arduino Serial predictions, not laptop replay.

## Separation Rule

Do not retrain using:

```text
data/raw/arduino_live_right_pocket_controlled/
data/raw/arduino_live_left_pocket_new_condition/
```

before reporting M3 live results. These folders are held-out live evidence only.
