# AGENTS.md

## Project
TinyML: Energy-Efficient Human Activity Recognition on the Edge

## Source priority
1. Project_Milestone3_Handout.pdf
2. M2_FEEDBACK.md
3. Project_Milestone2_Handout.pdf
4. Project_Handout.pdf
5. Revised_Milestone_1_Proposal_v2.docx
6. Efficient_Human_Gait_Activity_Recognition_Based_on_Sensor_Fusion_and_Intelligent_Stacking_Framework.pdf
7. Preparing WISDM and UCI HAR for Gait Activity Recognition.pdf

## Core requirement
This project has two distinct baseline roles:

### 1) Paper Reproduction Baseline
Reproduce the Sensors Journal paper as closely as feasible:
- ConvLSTM
- CNN-GRU
- CNN-BiGRU
- CNN-BiLSTM
- CNN-LSTM
- XGBoost stacking meta-learner

This is:
- mandatory
- offline reference baseline
- not intended for Arduino deployment

### 2) Lightweight TinyML-Oriented Baseline
Build a smaller model path suitable for later M3 deployment or quantization.
This does not replace the reproduction baseline.

## Non-negotiable rules
- The project handout is the grading source of truth.
- Do not fabricate dataset metadata, licenses, paper details, or results.
- Do not fabricate Arduino hardware metrics, live accuracy, robustness results, video evidence, or compile memory usage.
- If the paper is ambiguous, document assumptions explicitly.
- Use deterministic seeds and relative paths only.
- Use training-split statistics only for normalization.
- Prefer subject-aware or grouped split logic where feasible.
- Do not force invalid label merges between UCI HAR and WISDM.
- Save metrics, confusion matrices, tables, logs, and figures to disk.
- Keep reproduction outputs separate from lightweight-model outputs.
- Keep laptop/offline metrics separate from real Arduino on-device metrics.
- Use Arduino CSV files only from `data/raw/arduino_collectdata_v1`; ignore unlabeled `.txt` captures unless explicitly asked.

## M2 feedback fixes to preserve
- State UCI HAR split consistently as 5,551 train / 1,801 validation / 2,947 test windows; validation is subject-aware and derived only from official training subjects.
- Include WISDM per-class raw counts when WISDM is listed in a Data Card.
- Include layer-level architecture details for both paper reproduction and lightweight baselines.
- Reconcile the M3 Arduino data target to at least 100 windows per class where feasible.
- Do not commit public dataset archives; use download scripts for UCI HAR and WISDM public data.

## M3 implementation status and constraints
- Phase 2 covers model/feature experiments. Phase 3 is quantization strategy selection; do not reintroduce the removed accelerometer-only sensor-ablation phase.
- Primary M3 model path is the lightweight DS-CNN exported with class-balanced full-integer INT8 PTQ selected by `python -m src.deployment.quantization_experiment`.
- Generated deployment artifacts live under `outputs/deployment/` and `arduino/tinyml_har_m3/`.
- The attached Arduino CSV archive is kept as a first-class Track B M3 test dataset under `data/raw/arduino_collectdata_v1/`.
- Offline replay on the real Arduino CSV dataset is implemented with `python -m src.deployment.evaluate_arduino_replay`; current result is accuracy 0.1789 and macro F1 0.0582 on 246 replay windows, so document a severe domain gap rather than hiding it.
- The main UCI-HAR training preprocessing was not changed in the current M3 pass. Do not claim all models were retrained unless that is actually done.
- The gravity-direction posture feature probe is implemented with `python -m src.training.evaluate_posture_gravity_feature`; current result did not improve SITTING/STANDING confusion, so do not adopt it into the deployable M3 model.
- Arduino sketch must use live LSM9DS1 input, 50 Hz target sampling, 128-sample windows, 50% overlap, saved training normalization, and Serial output.
- Required hardware evidence still needs the physical Arduino: stable 2-minute run, 30-90 second video or serial log, model/sketch memory numbers, average `Invoke()` latency over at least 50 calls, 20 live trials per class, and 10 robustness trials per class.
- The current teammate CSV collection audits at about 37-38 Hz, not 50 Hz. `standing/30s - r - 1.csv` has one malformed duplicate header row and a timestamp reset. Document these as data-quality findings and expect a clean v2 collection round.

## Done means
- R1 to R7 are covered
- D2, D3, and D4 are present
- README explains setup and run steps
- code is runnable
- Paper Reproduction Baseline is implemented or clearly documented with exact vs approximate parts
- held-out test results, per-class metrics, and confusion matrices are produced
- report draft is PDF-ready in Markdown
- M3 report draft has all hardware-only fields marked pending until measured on the Arduino
