# Arduino Collect Data v1

Source archive: `TinyML_arduino_collectdata_v1.zip` supplied by teammate.

Only CSV files were extracted into this folder. The unlabeled Arduino `.txt` files from the source archive are intentionally not used.

This folder is the real Arduino sensor dataset for the Track B Milestone 3 evaluation path. It is kept in the repository because Track B requires Arduino-collected data by M3, even though public UCI HAR remains the training dataset.

## Data Card

| Field | Entry |
|---|---|
| Dataset role | Real Arduino-collected test/evaluation dataset for Track B M3 |
| Source archive | `TinyML_arduino_collectdata_v1.zip` |
| Archive received locally | 2026-05-01 |
| Exact collection date | Not embedded in files; teammate confirmation needed before final report |
| Board/sensor source | Arduino IMU collection described by teammate |
| Channels | `time_ms`, `ax`, `ay`, `az`, `gx`, `gy`, `gz`, `label` |
| Sensor configuration | Accelerometer XYZ + gyroscope XYZ |
| Intended sampling rate | 50 Hz |
| Observed sampling rate | Median about 38.46 Hz from CSV timestamps |
| Classes | walking, walk_up, walk_down, sitting, standing, laying |
| Pocket/orientation variation | Left and right pocket files are present |
| Users/environments | Not encoded in files; current evidence supports at least one collector and two pocket placements |
| Raw rows used | 14,662 valid rows; one malformed duplicate header row skipped |
| Windowing for offline replay | Resample to 50 Hz, 128-sample windows, 64-sample stride |
| Offline replay windows | 246 total windows |
| Limitation | Offline replay is not live on-device accuracy; M3 still requires live Arduino trials |

Expected CSV columns:

```text
time_ms, ax, ay, az, gx, gy, gz, label
```

Folder-to-UCI mapping:

| Folder | UCI HAR Class |
|---|---|
| `walking` | `WALKING` |
| `walk_up` | `WALKING_UPSTAIRS` |
| `walk_down` | `WALKING_DOWNSTAIRS` |
| `sitting` | `SITTING` |
| `standing` | `STANDING` |
| `laying` | `LAYING` |

Inspect this dataset from the repo root:

```powershell
python -m src.data.arduino_collectdata
```

Known audit findings are saved in:

```text
docs/tables/arduino_collectdata_class_counts.csv
docs/tables/arduino_collectdata_file_quality.csv
outputs/arduino_collectdata/arduino_collectdata_summary.json
outputs/arduino_collectdata/metrics/arduino_replay_int8_metrics.json
```

## Current Class Counts

| Folder | UCI HAR Class | Valid Rows | Duration (s) | 50 Hz Replay Windows |
|---|---|---:|---:|---:|
| `laying` | `LAYING` | 2,245 | 59.927 | 44 |
| `sitting` | `SITTING` | 2,249 | 59.925 | 44 |
| `standing` | `STANDING` | 3,397 | 89.898 | 66 |
| `walking` | `WALKING` | 2,228 | 59.922 | 44 |
| `walk_down` | `WALKING_DOWNSTAIRS` | 2,271 | 59.604 | 24 |
| `walk_up` | `WALKING_UPSTAIRS` | 2,272 | 59.619 | 24 |

## Offline Replay Result

The current public-data-trained INT8 model was replayed on this Arduino CSV dataset on the laptop:

```powershell
python -m src.deployment.evaluate_arduino_replay
```

Result:

- Accuracy: `0.1789`
- Macro F1: `0.0596`
- Weighted F1: `0.0640`

This poor result is useful evidence of the public-dataset-to-Arduino domain gap. It should not be reported as live on-device accuracy.
