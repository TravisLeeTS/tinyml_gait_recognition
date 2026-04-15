# Reproduction Notes

Project: Energy-Efficient Human Activity Recognition on the Edge

Reference paper: Tarekegn et al., "Efficient Human Gait Activity Recognition Based on Sensor Fusion and Intelligent Stacking Framework," IEEE Sensors Journal, 2023.

## Source Priority Applied

1. `Project_Milestone2_Handout.pdf`: grading requirements and deliverable structure.
2. `Revised_Milestone_1_Proposal_v2.docx`: project scope, UCI HAR as primary benchmark, WISDM as secondary validation, Arduino Nano 33 BLE Sense target.
3. `Efficient_Human_Gait_Activity_Recognition_Based_on_Sensor_Fusion_and_Intelligent_Stacking_Framework.pdf`: mandatory Phase 1 reproduction target.
4. `Preparing WISDM and UCI HAR for Gait Activity Recognition.pdf`: timestamp, split, preprocessing, and leakage guidance.

## Exact Paper-Specified Choices Implemented

- Level-0 model families:
  - ConvLSTM
  - CNN-GRU
  - CNN-BiGRU
  - CNN-BiLSTM
  - CNN-LSTM
- Level-1 meta-learner:
  - XGBoost
- Classification loss:
  - Cross entropy, equivalent to categorical cross-entropy for integer labels.
- Optimizer:
  - Adam.
- Initial learning rate:
  - 0.001.
- Learning-rate decay:
  - Implemented as inverse-time decay: `lr = 0.001 / (1 + 0.01 * epoch_index)`.
- Batch size:
  - 50.
- Early stopping:
  - Validation loss monitor.
- XGBoost tuning:
  - `GridSearchCV` with 5-fold cross-validation.
- UCI HAR input:
  - 128-timestep prewindowed inertial signal rows.
- Output layer:
  - Six-class softmax-equivalent probability output for all base models.
- Output separation:
  - Paper reproduction artifacts are under `outputs/reproduction/`.
  - Lightweight TinyML artifacts are under `outputs/lightweight/`.

## Approximate or Ambiguous Choices

- Framework: the paper reports TensorFlow 2.9. This repo now uses TensorFlow 2.18.0 with Keras 3.8.0 so the M2 code can feed directly into TensorFlow Lite and TensorFlow Lite Micro work in M3.
- ConvLSTM: the paper does not provide enough layer-level detail for an exact layer-by-layer recreation. This package uses Keras `ConvLSTM1D` over four subsequences as the closest maintainable TensorFlow approximation.
- Time-distributed subsequences: the paper says the hybrid CNN-RNN models read subsequences as blocks and use a time-distributed wrapper. This package uses four 32-sample subsequences for UCI HAR's 128-sample windows.
- CNN-LSTM and CNN-BiLSTM: implemented with two Conv1D layers with 64 filters, kernel size 3, dropout 0.5, max-pooling, LSTM/BiLSTM hidden size 100, dense size 100, and six-class output.
- CNN-GRU and CNN-BiGRU: implemented with two Conv1D layers with 32 and 128 filters, kernel size 3, dropout 0.5, max-pooling, GRU/BiGRU hidden size 100, dense size 100, and six-class output.
- Stacking protocol: the paper's algorithm text mentions 5 subsets, while the experimental text says base model outputs are stacked and XGBoost is tuned with fivefold CV. This package uses the subject-aware training split for base models, a subject-held-out validation split to train the XGBoost meta-learner, and the official held-out UCI test subjects for final evaluation.
- UCI split: the UCI release uses a subject-disjoint 70/30 volunteer split. The paper says 80/20 train/test after segmentation. The package keeps the official subject-disjoint test set to reduce leakage and carves validation subjects only from the UCI train subjects.
- Normalization: the paper describes min-max normalization but also refers to mean and standard deviation. The reproduction script supports both; the current static-posture fix uses train-only standardization with total acceleration plus gyroscope signals.
- Signal view: the initial implementation used body acceleration plus gyroscope. That removed gravity/posture cues and hurt SITTING, STANDING, and LAYING. The maintained default now uses total acceleration plus gyroscope, which is closer to Arduino IMU collection and improved static posture performance.

## Dataset Access Notes

- `ucimlrepo.fetch_ucirepo(id=240)` was attempted during setup and returned HTTP 502 from the UCI API on April 12, 2026.
- The official UCI HAR zip endpoints also returned HTTP 502 from this machine during setup.
- The package therefore keeps the official UCI URL as the primary target and uses a public UCI HAR mirror as a fallback.
- Fordham WISDM classic v1.1 downloaded successfully from the WISDM Lab URL and was inspected programmatically.

## Current Executed Runs

- Lightweight TinyML-oriented baseline:
  - Script: `python -m src.training.train_lightweight --epochs 40 --patience 8 --device cpu`
  - Result: accuracy 0.9169, macro F1 0.9173 on the UCI HAR held-out test set.
  - TensorFlow Lite size: 13,460 bytes.
- Paper reproduction baseline:
  - Script: `python -m src.training.train_reproduction --epochs 20 --patience 5 --device cpu --normalization standard --fast-xgb-grid`
  - Result: accuracy 0.9135, macro F1 0.9128 on the UCI HAR held-out test set.
  - Interpretation: this is a bounded M2 run for artifact generation. For the closest reproduction attempt, rerun with `--epochs 64` and remove `--fast-xgb-grid`.

## Static-Posture Diagnosis

The earlier run failed mainly on SITTING, STANDING, and LAYING because the input view used body acceleration plus gyroscope. UCI HAR body acceleration has gravity removed, which removes a key posture cue. After switching to total acceleration plus gyroscope and moving to TensorFlow/Keras:

- Lightweight baseline improved from 0.8700 to 0.9169 accuracy.
- Paper reproduction bounded run improved from 0.7075 to 0.9135 accuracy.
- LAYING F1 improved to 1.0000 for both maintained TensorFlow runs.
- The remaining dominant static error is SITTING versus STANDING, which is expected because both classes have low motion and can be separated mostly by gravity/orientation posture cues at the waist.

Next targeted improvements should compare total acceleration plus gyroscope against a 9-channel view with total acceleration, body acceleration, and gyroscope, then check whether the added body-motion channels improve walking/stairs without hurting Arduino feasibility.

## Not Claimed

- The current paper reproduction run is not claimed to match the paper's reported 97.52% UCIHAR result.
- WISDM model results are not claimed in this M2 report. WISDM was downloaded and inspected, and its timestamp irregularity was measured, but the main executed model benchmark is UCI HAR.
- No Arduino Nano 33 BLE Sense deployment metrics are claimed for M2; those are planned for M3.
