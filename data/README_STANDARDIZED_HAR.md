# Standardized Arduino HAR Dataset v2

This document describes `TINYML_HAR_DATA_STANDARDIZED_v2.zip`, which is kept outside the repository unless explicitly attached for submission.

## File Format

The dataset contains semicolon-separated `.txt` sensor files with this header:

```text
time_ms; ax; ay; az; gx; gy; gz
```

Columns:

- `time_ms`: elapsed sample timestamp in milliseconds.
- `ax`, `ay`, `az`: accelerometer channels.
- `gx`, `gy`, `gz`: gyroscope channels.

The parser expects one activity per file. Labels are inferred from the directory/file name, not from a label column.

## Naming Convention Observed

The inspected zip contains:

- `session_01_1person_allclass_30s`: one person, all six classes, right and left pocket, about 30 seconds per non-stair class and short repeated stair trials.
- `session_03_1person_4class_5min`: one person, four non-stair/static-walking classes, right pocket, about five minutes each.
- `session_04_2person_allclass_mixed_duration`: latest/largest session, two inferable people, all six classes, right and left pocket, mixed duration.

Session 2 is not present in this v2 archive. The team notes it was removed
because it was not 50 Hz, so it is not used in the final M4 split.

Inferable people:

- `person_A`
- `person_B`

Inferable placements:

- `left_pocket`
- `right_pocket`

Observed activity folders map to the six UCI-style classes:

| Folder label | Report class |
|---|---|
| `lay` | `LAYING` |
| `sit` | `SITTING` |
| `stand` | `STANDING` |
| `walk` | `WALKING` |
| `walkup` | `WALKING_UPSTAIRS` |
| `walkdown` | `WALKING_DOWNSTAIRS` |

## Parsed Summary

Generated with:

```powershell
python scripts\parse_standardized_har_dataset.py --input TINYML_HAR_DATA_STANDARDIZED_v2.zip --output results\standardized_dataset_summary
```

Current parsed summary:

| Item | Value |
|---|---:|
| Sensor files | 84 |
| Valid rows | 275,174 |
| Invalid rows | 0 |
| Median effective sampling rate | 50.0 Hz |
| Raw-row-estimated 128-sample/64-stride windows | 4,170 |
| Resampled windows used by M4 experiment | 4,121 |
| Sessions | 3 |
| Inferable people | 2 |
| Placements | right pocket, left pocket |

Class distribution:

| Class | Files | Valid rows | Estimated windows |
|---|---:|---:|---:|
| LAYING | 7 | 66,818 | 1,034 |
| SITTING | 9 | 60,716 | 936 |
| STANDING | 8 | 57,704 | 890 |
| WALKING | 7 | 54,646 | 844 |
| WALKING_DOWNSTAIRS | 26 | 16,461 | 215 |
| WALKING_UPSTAIRS | 27 | 18,829 | 251 |

## Intended M4 Use

For the final M4 report, the first grading-facing policy is Strategy A: keep this standardized dataset fully held out as external validation. This avoids leakage from overlapping windows and avoids claiming robustness from data that was also used for model selection.

The final additional experiments use session-aware splits:

- The first run adds standardized session 4 because it is the largest all-class standardized session.
- The second run adds session 3 plus session 4. Session 3 contributes one-person, four-class, five-minute right-pocket data.
- Evaluation holds out standardized session 1 completely in both runs.
- Robustness is reported separately on session 1 right pocket and session 1 left pocket.
- Session 2 is not used because it is not present in v2 and was described by the team as non-50 Hz.

If the team chooses to retrain with part of this dataset, use Strategy B or C only when a clean final holdout remains:

- Do not split randomly at the window level.
- Do not place windows from one raw recording in both train and validation/test.
- Hold out a full person, session, or placement for final evaluation.
- Generate overlapping windows only after the recording-level split is fixed.
- Never augment validation or test windows.

Split plans can be generated with:

```powershell
python scripts\prepare_m4_splits.py --strategy A
python scripts\prepare_m4_splits.py --strategy C --holdout-person person_B --output results\standardized_dataset_summary\m4_strategy_c_person_b_holdout_split_plan.csv
```

The bounded M4 augmentation experiment keeps this dataset held out:

```powershell
python -m src.training.train_m4_strategy_a_experiment --standardized-input TINYML_HAR_DATA_STANDARDIZED_v2.zip --include-old-m3-raw-holdouts-in-train --run-augmentation --epochs 8 --patience 3 --batch-size 64 --device cpu --output-dir outputs\m4_strategy_a_experiment
```

Result summary:

| Experiment | Standardized external accuracy | Standardized external macro-F1 | Decision |
|---|---:|---:|---|
| Expanded training, no augmentation | 0.7256 | 0.6347 | Baseline external experiment |
| Expanded training, training-only augmentation | 0.7663 | 0.6509 | Better external score, but stair confusion remains |
| Packaged INT8 candidate | 0.7692 | 0.6538 | Written to Arduino header for re-test |

Session-aware final experiment summary:

Training on session 4 only:

| Candidate | Session 1 right accuracy | Session 1 right macro-F1 | Session 1 left accuracy | Session 1 left macro-F1 |
|---|---:|---:|---:|---:|
| Base 6-channel + augmentation | 0.7881 | 0.7441 | 0.7652 | 0.7169 |
| Gravity-frame aligned 6-channel + augmentation | 0.7881 | 0.7441 | 0.7913 | 0.7649 |
| 10-channel gravity/orientation + augmentation | 0.7797 | 0.7349 | 0.7130 | 0.6289 |

Training on session 3 plus session 4:

| Candidate | Session 1 right accuracy | Session 1 right macro-F1 | Session 1 left accuracy | Session 1 left macro-F1 |
|---|---:|---:|---:|---:|
| Base 6-channel + augmentation | 0.9746 | 0.9663 | 0.7565 | 0.7181 |
| Gravity-frame aligned 6-channel + augmentation | 0.7712 | 0.7248 | 0.6087 | 0.6469 |
| 10-channel gravity/orientation + augmentation | 0.7881 | 0.7441 | 0.9826 | 0.9753 |

Packaged INT8 live-test candidate from the 10-channel session3+4 run:

| Candidate | Session 1 right accuracy | Session 1 right macro-F1 | Session 1 left accuracy | Session 1 left macro-F1 | Notes |
|---|---:|---:|---:|---:|---|
| `m4_session3_4_gravity_10ch_int8` | 0.9576 | 0.9472 | 0.9913 | 0.9876 | Latest `arduino/tinyml_har_m3/model_data.h`; session-level holdout only, not live evidence |

The packaged INT8 result is high, so it should be interpreted carefully. The
split prevents overlapping-window leakage because standardized session 1 is
excluded from training, augmentation, representative quantization, and model
packaging. However, the holdout is session-level, not a strict unseen-person
evaluation, so the live demo logs are still required before making any new M4
deployment-performance claim.

## Gravity-Frame Aligned 6-Channel + Augmentation

The gravity-frame aligned 6-channel experiment keeps the model input at six
channels: `ax`, `ay`, `az`, `gx`, `gy`, and `gz`. For each 128-sample window,
the preprocessing estimates the mean acceleration vector as the local gravity
direction, computes a rotation that maps that vector to a common vertical axis,
and applies the same rotation to accelerometer and gyroscope samples. The goal
is to reduce board-angle and pocket-orientation sensitivity without adding
extra input channels.

This differs from the 10-channel gravity/orientation experiment. The 10-channel
variant keeps the six IMU channels and appends gravity direction components and
acceleration magnitude features, so it is a larger feature representation. In
the session3+4 experiment, 10-channel features produced the best session-1
left-pocket holdout result, while base 6-channel features produced the best
session-1 right-pocket holdout result.

Augmentation is applied only to training windows after the session-level split.
The bounded augmentation uses small Gaussian noise, per-channel scaling, and
mild time shifts. Session 1 holdout windows are never augmented and never mixed
into training, which prevents overlapping-window leakage into the right-pocket
or left-pocket robustness checks.
