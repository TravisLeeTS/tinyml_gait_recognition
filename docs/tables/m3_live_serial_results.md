# M3 Live Serial Results

These results come from true Arduino Serial prediction logs returned by the teammate. They are separate from UCI-HAR laptop benchmarks and raw sensor replay validation.

| Condition | Source | Rows | Accuracy | Macro F1 | Main Failure |
|---|---|---:|---:|---:|---|
| Right pocket controlled | Live Arduino Serial | 125 | 0.9040 | 0.9089 | WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS |
| Left pocket robustness | Live Arduino Serial | 91 | 0.8901 | 0.8826 | WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS |

Live Serial robustness is much better than the earlier `left_30s` raw replay result. The remaining live failure mode is mainly stair-direction confusion, while static postures are strong in the returned live Serial trials.

Parsing notes: boot metadata, headers, blank lines, `run_timer` lines, and other non-prediction lines were ignored. Scoring uses the displayed `prediction_label` field rather than `prediction_id`; this handles the one robustness row where those fields disagree. A `top2_label` typo does not affect top-1 scoring.

## Controlled Right Pocket

- `WALKING`: 20/20 correct.
- `WALKING_UPSTAIRS`: 13/25 correct; 12 predicted as `WALKING_DOWNSTAIRS`.
- `WALKING_DOWNSTAIRS`: 20/20 correct.
- `SITTING`: 20/20 correct.
- `STANDING`: 20/20 correct.
- `LAYING`: 20/20 correct.

## Robustness Left Pocket

- `WALKING`: 10/10 correct.
- `WALKING_UPSTAIRS`: 6/12 correct; 6 predicted as `WALKING_DOWNSTAIRS`.
- `WALKING_DOWNSTAIRS`: 18/22 correct; remaining predictions are mainly `WALKING` or `STANDING`.
- `SITTING`: 22/22 correct.
- `STANDING`: 15/15 correct.
- `LAYING`: 10/10 correct.

Saved scoring artifacts:

```text
outputs/live_evidence/right_pocket_controlled_metrics.json
outputs/live_evidence/right_pocket_controlled_confusion_matrix.csv
outputs/live_evidence/right_pocket_controlled_per_class_metrics.csv
outputs/live_evidence/left_pocket_robustness_metrics.json
outputs/live_evidence/left_pocket_robustness_confusion_matrix.csv
outputs/live_evidence/left_pocket_robustness_per_class_metrics.csv
outputs/live_evidence/live_controlled_vs_robustness_summary.csv
```
