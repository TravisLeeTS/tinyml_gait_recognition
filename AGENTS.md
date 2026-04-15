# AGENTS.md

## Project
TinyML: Energy-Efficient Human Activity Recognition on the Edge

## Source priority
1. Project_Handout.pdf
2. Revised_Milestone_1_Proposal_v2.docx
3. Efficient_Human_Gait_Activity_Recognition_Based_on_Sensor_Fusion_and_Intelligent_Stacking_Framework.pdf
4. Preparing WISDM and UCI HAR for Gait Activity Recognition.pdf

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
- If the paper is ambiguous, document assumptions explicitly.
- Use deterministic seeds and relative paths only.
- Use training-split statistics only for normalization.
- Prefer subject-aware or grouped split logic where feasible.
- Do not force invalid label merges between UCI HAR and WISDM.
- Save metrics, confusion matrices, tables, logs, and figures to disk.
- Keep reproduction outputs separate from lightweight-model outputs.

## Done means
- R1 to R7 are covered
- D2, D3, and D4 are present
- README explains setup and run steps
- code is runnable
- Paper Reproduction Baseline is implemented or clearly documented with exact vs approximate parts
- held-out test results, per-class metrics, and confusion matrices are produced
- report draft is PDF-ready in Markdown
