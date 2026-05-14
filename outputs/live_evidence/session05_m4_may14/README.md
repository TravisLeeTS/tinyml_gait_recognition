# Session05 M4 Live Arduino Evidence

This folder contains the May 14 live Arduino run for the final M4 10-channel INT8 HAR model.

## Conditions

| Condition key | Description | Rows | Accuracy | Macro-F1 | Main failure |
|---|---|---:|---:|---:|---|
| `person_a_controlled_right` | Person A controlled right pocket | 132 | 1.0000 | 1.0000 | none |
| `person_a_robustness_left` | Person A robustness left pocket | 137 | 1.0000 | 1.0000 | none |
| `person_b_robustness_right` | Person B right-pocket robustness | 145 | 0.9931 | 0.9928 | one `WALKING_DOWNSTAIRS -> WALKING_UPSTAIRS` |
| aggregate | All Session05 live rows | 414 | 0.9976 | 0.9974 | one total stair-direction error |

Mean `Invoke()` latency across all scored rows is 35.643 ms. Median latency is 35.641 ms.

## Files

- `Session05_live_may14.zip`: original shared live-log archive.
- `raw/`: extracted raw Serial text logs.
- `compile_and_boot_metadata.md`: Arduino compile/upload output and model boot metadata.
- `session05_m4_live_summary.md`: condition-level metric summary.
- `session05_m4_live_summary.csv`: CSV version of the same summary.
- `session05_m4_live_overall_metrics.json`: aggregate metrics.
- `session05_m4_live_predictions_normalized.csv`: normalized scored prediction rows.
- `metrics/*_confusion_matrix.csv`: condition-level confusion matrices.
- `metrics/*_per_class_metrics.csv`: condition-level per-class precision/recall/F1.

## Evidence Notes

These live Serial logs are hardware evidence only. They were not used for training, augmentation, quantization calibration, model selection, or packaging.

The supplied logs include 30-second `run_timer` rows. A separate 120-second `stability_check` row was not present in the imported logs.
