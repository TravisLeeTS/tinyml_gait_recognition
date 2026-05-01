# Message To Send Teammate With The Arduino

Please pull the latest repo and test `arduino/tinyml_har_m3/tinyml_har_m3.ino` on the Arduino Nano 33 BLE Sense.

Steps:

1. Open `arduino/tinyml_har_m3/tinyml_har_m3.ino` in Arduino IDE.
2. Select board `Arduino Nano 33 BLE Sense` and port.
3. Install/confirm `Arduino_LSM9DS1` and Arduino TensorFlow Lite / TensorFlow Lite Micro.
4. Compile and upload.
5. Open Serial Monitor at `115200`.
6. Send me the boot lines that include `model_bytes` and `tensor_arena_bytes`.
7. Let it run for 2 minutes and save the full serial log.
8. Record a 60-90 second video showing the board and Serial Monitor while demonstrating at least 3 classes.
9. For controlled accuracy, run 20 trials per class: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`.
10. For robustness, run 10 trials per class in a different pocket, user, route, or environment.
11. For each trial, record true label, predicted label, score, and latency_us. The template is `docs/milestone3_live_trial_sheet.csv`.

Important notes from the CSV audit:

- The uploaded CSVs are closer to 37-38 Hz, not 50 Hz. Please verify the new sketch prints stable inference lines and uses the updated 50 Hz sampler.
- `standing/30s - r - 1.csv` has a duplicated/malformed header row in the middle; the repo importer skips it, but future captures should save one clean CSV per session.
- Current CSV volume is below 100 windows/class after 50 Hz resampling. If there is time, collect at least 2.5 minutes per class to reach 100+ windows/class.
- Offline replay of the selected class-balanced INT8 model on the current Arduino CSVs is very poor: 17.89% accuracy and 0.0582 macro F1. Please help verify live axis orientation and sampling cadence on the board; we need a clean next-round dataset before claiming final Arduino-live accuracy.

Please also send the Arduino IDE compile summary showing sketch flash usage and RAM usage. We need those numbers for the M3 report.
