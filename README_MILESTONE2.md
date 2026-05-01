# TinyML HAR Package

Project: Energy-Efficient Human Activity Recognition on the Edge

Team:

- Lee Ting Sen, B00103724
- Khalifa Alshamsi, B00078654
- Vineetha Addanki, G00111196

> **Milestone 2 grading note:**  
> This file is the detailed Milestone 2 README. Please read it together with the general [`README.md`](README.md). The general README gives the project overview, while this file follows the Milestone 2 handout structure and explains D2, D3, D4, dataset handling, preprocessing, baseline training, held-out evaluation, error analysis, and the M3 Arduino data collection plan. Please also refer to [`tinyml_milestone2.pdf`](tinyml_milestone2.pdf), the Milestone 2 handout PDF, as the main grading reference.

## Milestone 2 Status

This repository is the Milestone 2 package for TinyML human activity recognition targeting future Arduino Nano 33 BLE Sense deployment. The current training track uses public/open datasets, so this is a Track B submission: the model is trained and evaluated on a held-out public dataset split at M2, and real Arduino sensor data collection is planned for M3/M4 on-device evaluation.

Current status: Milestone 2 baseline training and evaluation are implemented. The repo contains runnable data inspection, preprocessing, training, evaluation, metrics, confusion matrix generation, and README documentation. The full public datasets are intentionally not committed to Git because the archives and extracted raw files are too large for normal repository submission and are fully reproducible from the included code. The commands below download the data automatically and run the pipeline end to end.

M2 feedback follow-up applied in the repository docs:

- The UCI split is described consistently as 5,551 train / 1,801 validation / 2,947 test windows. The validation split is subject-aware and comes only from the official UCI training subjects.
- WISDM per-class raw counts are surfaced below instead of only storing them in `docs/tables/wisdm_classic_raw_counts.csv`.
- Layer-level architecture details are listed for the paper-reproduction learners and the lightweight model.
- The Arduino M3 collection target is reconciled to at least 100 windows per class where feasible.
- Public dataset archives are ignored by Git and should be regenerated through `python -m src.data.inspect_datasets`; trained model and metric artifacts are present under `outputs/`.

## Quick Reproduction

From the repository root, use Python 3.12 and install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the dataset download and inspection step:

```powershell
python -m src.data.inspect_datasets
```

Train and evaluate the paper reproduction baseline:

```powershell
python -m src.training.train_reproduction --epochs 20 --patience 5 --device cpu --normalization standard --fast-xgb-grid
```

Train and evaluate the lightweight TinyML baseline:

```powershell
python -m src.training.train_lightweight --epochs 40 --patience 8 --device cpu
```

These commands create or update:

```text
data/raw/                         downloaded public datasets, ignored by Git
outputs/datacards/                dataset inspection JSON files
docs/tables/                      class-count tables
outputs/reproduction/metrics/     baseline metrics and confusion matrices
outputs/reproduction/figures/     baseline confusion matrix figures
outputs/reproduction/logs/        training histories and XGBoost search logs
outputs/lightweight/metrics/      lightweight metrics and confusion matrices
outputs/lightweight/figures/      lightweight confusion matrix figures
outputs/lightweight/logs/         lightweight training history
```

If the official UCI download endpoint is temporarily unavailable, the downloader falls back to a public mirror defined in `src/config.py`.

## M2 Data Track

This project uses Track B, Open/Public Dataset, for M2.

Primary dataset:

- UCI HAR v1.0, Human Activity Recognition Using Smartphones
- URL: `https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`
- Sensors: smartphone accelerometer and gyroscope
- Sampling rate: 50 Hz
- Input used in this repo: prewindowed inertial-signal files
- Model input shape: `[128, 6]`

Secondary dataset:

- Fordham WISDM Activity Prediction v1.1
- URL: `https://www.cis.fordham.edu/wisdm/dataset.php`
- Use in this repo: inspection, timestamp audit, and domain-gap planning only
- Important note: WISDM is not merged with UCI HAR because its sensor channels and label taxonomy differ.

## R1/R2 Dataset Description And Data Card

| Field | Entry |
|---|---|
| Task type | Multiclass human activity classification |
| Primary dataset | UCI HAR v1.0 |
| Number of classes | 6 |
| Classes | `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING` |
| Sensors used | Total acceleration XYZ and body gyroscope XYZ |
| Sampling rate | 50 Hz |
| Window length | 128 samples, 2.56 seconds |
| Window overlap | 50% |
| Input tensor | `[window_count, 128, 6]` |
| Features extracted | No handcrafted features for neural models; raw sequence windows are fed directly after normalization |
| Split method | Official UCI train/test split, plus subject-aware validation split from official training data |
| Split sizes | 5,551 train / 1,801 validation / 2,947 test windows |
| Split ratio | 53.9% / 17.5% / 28.6% of the 10,299 UCI windows; official test subjects remain held out |
| Test set | Official UCI HAR test split |
| Test windows | 2,947 |
| Number of subjects | 30 total in UCI HAR; official train/test subjects are kept separated |
| Known limitations | Public smartphone dataset, not Arduino Nano 33 BLE Sense data; hardware and placement domain gap must be tested in M3/M4 |

Class-count outputs are generated at:

```text
docs/tables/uci_har_class_counts.csv
outputs/reproduction/metrics/uci_har_class_counts.csv
outputs/lightweight/metrics/uci_har_class_counts.csv
```

Current train/validation/test class counts from the subject-aware split:

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| WALKING | 888 | 338 | 496 | 1722 |
| WALKING_UPSTAIRS | 797 | 276 | 471 | 1544 |
| WALKING_DOWNSTAIRS | 744 | 242 | 420 | 1406 |
| SITTING | 993 | 293 | 491 | 1777 |
| STANDING | 1053 | 321 | 532 | 1906 |
| LAYING | 1076 | 331 | 537 | 1944 |

WISDM classic v1.1 is not merged into the UCI HAR label set, but its raw label distribution is inspected for data-card completeness:

| WISDM Raw Class | Count |
|---|---:|
| Walking | 424398 |
| Jogging | 342179 |
| Upstairs | 122869 |
| Downstairs | 100427 |
| Sitting | 59939 |
| Standing | 48395 |

## R3 Preprocessing Pipeline

The reported training and evaluation pipeline uses the UCI HAR inertial-signal files. UCI HAR already provides fixed-length sliding windows, so this repo does not rebuild the original raw-signal filtering pipeline from unwindowed phone recordings.

Segmentation:

- Sampling rate: 50 Hz
- Window length: 128 samples
- Window duration: 2.56 seconds
- Overlap: 50%
- Effective stride: 64 samples, or 1.28 seconds
- Triggering mechanism: fixed periodic sliding windows, not event-triggered windows

The six selected input channels are:

```text
total_acc_x
total_acc_y
total_acc_z
body_gyro_x
body_gyro_y
body_gyro_z
```

Feature extraction:

- No handcrafted time-domain or frequency-domain features are computed for the neural-network inputs.
- The project does not use the UCI HAR `X_train.txt` and `X_test.txt` 561-feature vectors as neural-network input.
- The model input is the normalized raw sequence window with shape `[128, 6]`.
- For the paper reproduction stacking baseline only, class-probability outputs from the five base neural models are concatenated and used as XGBoost meta-learner features.

Normalization:

- Normalization statistics are fitted on the training split only.
- Validation and test windows are transformed using the training-derived parameters.
- The lightweight baseline uses per-channel standardization:

```text
x_normalized = (x - training_mean_per_channel) / training_std_per_channel
```

- The reproduction script supports `standard` and `minmax`; the current bounded reproduction artifact uses `standard`.
- Normalizer files are saved under `outputs/*/models/`, but model artifacts are ignored by Git because they are reproducible.

Data augmentation:

- No augmentation is used in the reported M2 baselines.
- No noise injection, random shifting, time warping, random scaling, cropping, mixup, or synthetic oversampling is applied.

## Where Are `x_train`, `y_train`, `x_val`, And `x_test`?

There are no committed files named `x_train`, `x_val`, `x_test`, `y_train`, `y_val`, or `y_test`. They are created in memory by the loader.

UCI HAR stores each sensor axis separately. The project reads files like:

```text
data/raw/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt
data/raw/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt
data/raw/UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt
data/raw/UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt
```

The labels are read from:

```text
data/raw/UCI HAR Dataset/train/y_train.txt
data/raw/UCI HAR Dataset/test/y_test.txt
```

The validation set is not an official UCI file. It is generated from the official training split using subject-aware grouping.

To inspect the arrays directly:

```powershell
@'
from src.data.uci_har import load_uci_har_sequence_dataset
from src.utils.normalization import SequenceStandardizer

data = load_uci_har_sequence_dataset(download=True)

x_train = data.train.x
y_train = data.train.y
x_val = data.val.x
y_val = data.val.y
x_test = data.test.x
y_test = data.test.y

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_val", x_val.shape)
print("y_val", y_val.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)
print("classes", data.class_names)
print("validation subjects", sorted(set(data.val.subjects.tolist())))

scaler = SequenceStandardizer.fit(x_train)
print("normalized x_train", scaler.transform(x_train).shape)
'@ | python -
```

If a saved processed NumPy file is required for local inspection, it can be generated reproducibly:

```powershell
@'
from pathlib import Path
import numpy as np
from src.data.uci_har import load_uci_har_sequence_dataset

data = load_uci_har_sequence_dataset(download=True)
out = Path("data/processed/uci_har_v1_train_val_test.npz")
out.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    out,
    x_train=data.train.x,
    y_train=data.train.y,
    train_subjects=data.train.subjects,
    x_val=data.val.x,
    y_val=data.val.y,
    val_subjects=data.val.subjects,
    x_test=data.test.x,
    y_test=data.test.y,
    test_subjects=data.test.subjects,
    class_names=np.array(data.class_names),
)

print("saved", out)
'@ | python -
```

`data/processed/` is ignored by Git because this file is generated and can become large. The command above is the reproducible way to create it.

## R4 Baseline Models

This repo has two baseline roles.

Paper Reproduction Baseline:

- Purpose: offline reference baseline, not intended for Arduino deployment
- Models: ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, CNN-LSTM
- Meta-learner: XGBoost stacking classifier
- Input shape: `[128, 6]`
- Output: 6-class softmax for each neural base learner
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 50
- Loss: cross-entropy
- Early stopping: validation loss with patience set by command-line argument

Lightweight TinyML-Oriented Baseline:

- Purpose: smaller path for later M3 quantization and Arduino deployment
- Model: depthwise separable 1D CNN
- Input shape: `[128, 6]`
- Output: 6-class softmax
- Parameters: 1,922 Keras parameters in the current run
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 50
- Loss: cross-entropy
- Early stopping: validation loss
- TFLite export: generated by the training script

Model definitions are in:

```text
src/models/reproduction/keras_models.py
src/models/lightweight/tiny_cnn.py
```

Layer-level architecture details:

Paper reproduction baseline:

- Shared input: `[128, 6]`, reshaped to 4 subsequences of 32 timesteps each.
- ConvLSTM: `ConvLSTM1D(filters=64, kernel_size=3, padding=same, activation=relu, dropout=0.5)` -> dropout 0.5 -> flatten -> dense 100 ReLU -> dropout 0.5 -> dense 6 softmax.
- CNN-GRU: time-distributed Conv1D 32 kernel 3 ReLU -> dropout 0.5 -> time-distributed Conv1D 128 kernel 3 ReLU -> dropout 0.5 -> max pool 1D size 2 -> flatten -> GRU 100 -> dropout 0.5 -> dense 100 ReLU -> dropout 0.5 -> dense 6 softmax.
- CNN-BiGRU: same CNN front end as CNN-GRU, with bidirectional GRU 100.
- CNN-LSTM: time-distributed Conv1D 64 kernel 3 ReLU -> dropout 0.5 -> time-distributed Conv1D 64 kernel 3 ReLU -> dropout 0.5 -> max pool 1D size 2 -> flatten -> LSTM 100 -> dropout 0.5 -> dense 100 ReLU -> dropout 0.5 -> dense 6 softmax.
- CNN-BiLSTM: same CNN front end as CNN-LSTM, with bidirectional LSTM 100.
- Stacking meta-learner: XGBoost classifier trained on concatenated class-probability outputs from the five base learners; the bounded run selected `max_depth=2`, `n_estimators=100`, `learning_rate=0.1`, `gamma=0.0`.

Lightweight TinyML-oriented baseline:

- Input `[128, 6]`.
- Conv1D 12 filters, kernel 3, same padding, no bias -> batch normalization -> ReLU.
- SeparableConv1D 24 filters, kernel 5, same padding, no bias -> batch normalization -> ReLU -> average pooling size 2.
- SeparableConv1D 32 filters, kernel 5, same padding, no bias -> batch normalization -> ReLU.
- Global average pooling -> dense 6 softmax.
- Current run: 1,922 Keras parameters, 1,786 trainable parameters.

## R5 Held-Out Test Results

The reported numbers below are from the held-out UCI HAR test set, not training accuracy.

Paper reproduction bounded CPU artifact:

- Model: XGBoost stacking meta-learner over five neural base models
- Test windows: 2,947
- Accuracy: 0.9135
- Macro F1: 0.9128
- Weighted F1: 0.9137
- Metrics: `outputs/reproduction/metrics/paper_reproduction_stacking_metrics.json`
- Per-class metrics: `outputs/reproduction/metrics/paper_reproduction_stacking_per_class_metrics.csv`
- Confusion matrix CSV: `outputs/reproduction/metrics/paper_reproduction_stacking_confusion_matrix.csv`
- Confusion matrix figure: `outputs/reproduction/figures/paper_reproduction_stacking_confusion_matrix.png`

Lightweight TinyML baseline:

- Model: TinyDepthwiseSeparableCnn
- Test windows: 2,947
- Accuracy: 0.9169
- Macro F1: 0.9173
- Weighted F1: 0.9168
- TFLite size: 13,460 bytes
- Host CPU Keras latency proxy: mean 62.65 ms over 20 runs
- Metrics: `outputs/lightweight/metrics/lightweight_tiny_cnn_metrics.json`
- Per-class metrics: `outputs/lightweight/metrics/lightweight_tiny_cnn_per_class_metrics.csv`
- Confusion matrix CSV: `outputs/lightweight/metrics/lightweight_tiny_cnn_confusion_matrix.csv`
- Confusion matrix figure: `outputs/lightweight/figures/lightweight_tiny_cnn_confusion_matrix.png`

The host latency number is not an Arduino latency claim. It is only a development-machine proxy until M3/M4 on-device testing.

## R6 Error Analysis Summary

Top confusion pairs from the paper reproduction stacking baseline:

| True class | Predicted class | Count |
|---|---|---:|
| STANDING | SITTING | 101 |
| WALKING | WALKING_DOWNSTAIRS | 43 |
| SITTING | STANDING | 38 |
| WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | 27 |
| SITTING | WALKING_UPSTAIRS | 22 |

Top confusion pairs from the lightweight baseline:

| True class | Predicted class | Count |
|---|---|---:|
| SITTING | STANDING | 103 |
| STANDING | SITTING | 81 |
| WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | 21 |
| WALKING_DOWNSTAIRS | WALKING_UPSTAIRS | 19 |
| WALKING | WALKING_DOWNSTAIRS | 13 |

The main observed failure mode is confusion between static postures, especially sitting and standing. This is expected because both classes can have similar low-motion inertial signatures and may depend heavily on gravity direction and phone placement. The second failure mode is confusion among walking-related classes, especially upstairs and downstairs, where short windows can contain overlapping gait patterns.

M3 next steps:

- Collect real Arduino Nano 33 BLE Sense IMU data to measure the public-dataset-to-Arduino domain gap.
- Keep accelerometer plus gyroscope as the main model path for M3; do not split the plan into a separate accelerometer-only sensor-ablation phase.
- Quantize the lightweight model and evaluate its accuracy, memory size, latency, and energy behavior on-device.
- Add robustness testing with at least one new user or environment.

## R7 Arduino Data Collection Plan For M3

Because this is a Track B public-dataset submission, real Arduino data is required by M3 for on-device evaluation.

Planned M3 collection:

- Board: Arduino Nano 33 BLE Sense
- Sensor configuration: onboard IMU, accelerometer plus gyroscope where available
- Sampling target: match the model pipeline as closely as practical, targeting 50 Hz and 2.56-second windows
- Windowing target: 128 samples per window with 50% overlap
- Classes: same six UCI HAR-style classes where feasible: walking, walking upstairs, walking downstairs, sitting, standing, laying
- Target amount: at least 100 Arduino windows per class where feasible; 128-sample windows with 50% overlap require about 2.2 minutes of clean 50 Hz data per class, so 2.5 minutes per class is the practical collection target
- Users/environments: at least one user and one indoor environment; more users/environments will be added if time allows
- Domain adaptation: first evaluate the public-data-trained lightweight model directly, then fine-tune or retrain if the domain gap is too large
- Fallback plan: if six-class Arduino performance is weak, narrow the M3 scope to a smaller reliable class subset while documenting the limitation

Actual M3 Arduino dataset received and added:

- Location: `data/raw/arduino_collectdata_v1/`
- Source archive: `TinyML_arduino_collectdata_v1.zip`
- Files used: CSV files only; unlabeled `.txt` files from the source archive are ignored.
- Channels: `time_ms`, accelerometer XYZ, gyroscope XYZ, label.
- Labels collected: walking, walk_up, walk_down, sitting, standing, laying.
- Pocket variation: left and right pocket captures are present.
- Exact collection date/environment: not embedded in the files; teammate confirmation is needed before a final report claim.
- Observed sampling: median about 38.46 Hz, not the intended 50 Hz.
- Windowed replay dataset: 246 windows after 50 Hz resampling, 128-sample windows, and 64-sample stride.
- Offline replay result with the current selected INT8 model: accuracy 0.1789 and macro F1 0.0582, indicating a severe public-dataset-to-Arduino domain gap.

This dataset satisfies the requirement to keep real Arduino sensor data for M3 evaluation, but it does not replace the M3 handout requirement for live on-device trials and a live confusion matrix.

## D2 Dataset V1

The public dataset deliverable is reproducible rather than fully committed.

The public dataset files are too large for normal Git submission and can be regenerated exactly by the included scripts. Therefore:

- `data/raw/` is ignored by Git.
- The professor or evaluator should run `python -m src.data.inspect_datasets` to download and inspect the datasets.
- Dataset metadata and count summaries are saved to `outputs/datacards/` and `docs/tables/`.
- The UCI HAR loader uses relative paths through `src/config.py`.
- No machine-specific absolute data path is required.

The M3 Arduino CSV collection is stored separately under `data/raw/arduino_collectdata_v1/` and inspected by:

```powershell
python -m src.data.arduino_collectdata
```

This importer uses CSV files only and ignores the unlabeled raw `.txt` captures.

D2 evidence in this repo:

```text
src/data/uci_har.py
src/data/wisdm.py
src/data/inspect_datasets.py
src/utils/download.py
outputs/datacards/uci_har_inspection.json
outputs/datacards/wisdm_classic_inspection.json
outputs/datacards/dataset_inspection_summary.json
docs/tables/uci_har_class_counts.csv
docs/tables/wisdm_classic_raw_counts.csv
```

## D3 Training Code Or Notebook

The baseline deliverable is script-based rather than notebook-based. The Milestone 2 handout allows a script or a Jupyter notebook.

Main baseline scripts:

```text
src/training/train_reproduction.py
src/training/train_lightweight.py
```

Supporting code:

```text
src/data/uci_har.py
src/utils/normalization.py
src/utils/metrics.py
src/training/tf_common.py
src/models/reproduction/keras_models.py
src/models/lightweight/tiny_cnn.py
```

Library versions are pinned in:

```text
requirements.txt
```

The notebook at `notebooks/phase2_model_screening_lab.ipynb` is for Phase 2 architecture screening. It is not the main paper reproduction baseline. The paper reproduction baseline is implemented in `src/training/train_reproduction.py`.

## D4 README Update

This README is the D4 update. It includes:

- Project title and team members
- M2 status
- Track B dataset explanation
- Repository structure
- End-to-end setup, download, training, and evaluation commands
- Clarification that `x_train/y_train/x_val/x_test` are generated in memory
- Baseline model descriptions
- Held-out test results
- D2/D3/D4 deliverable mapping
- M3 Arduino data collection plan

## Repository Layout

```text
data/
  raw/                 downloaded public datasets, ignored by Git
  interim/             reserved for intermediate generated files
  processed/           optional generated NumPy files, ignored by Git
  sample/              placeholder for small examples if needed
docs/
  tables/              dataset class-count tables
notebooks/
  phase2_model_screening_lab.ipynb
src/
  config.py
  data/                dataset download, parsing, splitting, inspection
  models/
    reproduction/      paper reproduction model families
    lightweight/       TinyML-oriented model
  training/            runnable training/evaluation scripts
  utils/               normalization, metrics, reproducibility, download helpers
outputs/
  datacards/           dataset inspection outputs
  reproduction/        paper reproduction metrics, figures, logs
  lightweight/         lightweight metrics, figures, logs
```

## Arduino / PlatformIO Starter

The original Wokwi/PlatformIO firmware remains available for embedded bring-up while waiting for Nano 33 BLE Sense hardware:

```powershell
pio run
```

Main firmware file:

```text
src/main.cpp
```

The current firmware is a simulation starter, not the final M3/M4 TFLite Micro deployment.
