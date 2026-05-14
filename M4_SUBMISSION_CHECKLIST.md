# M4 Submission Checklist

## Report
- [x] Final project report complete
- [x] Final project report PDF available as `tinyml_milestone4.pdf`
- [x] R1 Problem Statement present
- [x] R2 Data Card updated
- [x] R3 Model progression table present
- [x] R4 Deployment metrics present
- [x] R5 Final evaluation present
- [x] R6 Ethics and limitations present
- [x] R7 Lessons learned present
- [x] Publication Intent Disclaimer added to project report

## Paper
- [x] Publication Intent Disclaimer added to paper manuscript
- [x] References preserved
- [x] No obvious placeholders

## Reproducibility
- [x] README complete with 10 required sections
- [x] Dataset documented
- [x] Standardized dataset parser available
- [x] Results tables present
- [x] Confusion matrices present
- [x] Session05 M4 live Arduino logs integrated
- [x] Arduino sketch present
- [x] Model files present
- [x] requirements.txt pinned
- [x] No hardcoded absolute paths in M4-facing artifacts
- [x] No credentials or API keys found in M4-facing artifacts
- [x] PDF generated or build instructions provided

## Manual actions
- [ ] Each student must fill their own publication choice
- [ ] Each student must sign and date their own row
- [ ] Final PDF must be reviewed before submission
- [ ] Attach `outputs/live_evidence/session05_m4_may14/Session05_live_may14.zip` with the course submission for evidence checking
- [ ] Optional: collect a longer run with an explicit 120-second `stability_check` line if the instructor asks for stronger stability evidence

## M3 Feedback Resolution
- [x] M4 report is compressed to the required page range
- [x] No long Final Review Highlights appendix included
- [x] WALKING_UPSTAIRS to WALKING_DOWNSTAIRS failure mode discussed
- [x] Stair per-class F1 reported where available
- [x] More stair evidence from standardized HAR v2 documented without claiming the gap is solved
- [x] 10-channel gravity model's 0.6856 left-pocket replay F1 documented as final experiment evidence, not a live M4 claim
- [x] Session 4 standardized data used in a final training experiment with session 1 right/left held out
- [x] Session 3 standardized right-pocket 5-minute data utilized in an additional session3+4 training experiment with session 1 right/left held out
- [x] Gravity-frame aligned 6-channel and 10-channel orientation-aware variants tested for placement mitigation
- [x] Latest session3+4 10-channel INT8 candidate packaged into `arduino/tinyml_har_m3/model_data.h` and Session05 live re-test scored
- [x] Right-pocket versus left-pocket placement sensitivity discussed
- [x] Calibration pose, gravity-frame alignment, orientation-aware features, more stair data, and controlled placement protocol listed as mitigations
- [x] User diversity documented without overclaiming
- [x] Live confusion-matrix row-count asymmetry documented
- [x] M2 heavy baseline to M3/M4 deployable DS-CNN engineering narrative preserved
