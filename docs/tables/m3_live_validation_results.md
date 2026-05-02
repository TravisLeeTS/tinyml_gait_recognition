# M3 Held-Out Raw Sensor Validation Results

These results are held-out raw sensor replay/validation results. They are not live on-device Serial prediction results. The `right_60s` and `left_30s` files were not used for training, normalization fitting, representative quantization, or early stopping.

## Selected Model

Selected candidate: `m3_v2_1_mixed_6ch`

Selected INT8 TFLite: `outputs/deployment/m3_retrained_with_v2_1/models/m3_v2_1_mixed_6ch_int8.tflite`

## Selected FP32 Summary

| split | accuracy | macro_f1 | f1_walking | f1_walking_upstairs | f1_walking_downstairs | f1_sitting | f1_standing | f1_laying |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uci_test | 0.7957 | 0.7932 | 0.6980 | 0.8743 | 0.6331 | 0.7598 | 0.8352 | 0.9589 |
| arduino_v2_grouped | 0.7379 | 0.7097 | 0.9000 | 0.9091 | 0.7826 | 0.0000 | 1.0000 | 0.6667 |
| right_60s | 0.8583 | 0.7647 | 0.9655 | 0.0000 | 0.6452 | 0.9888 | 1.0000 | 0.9890 |
| left_30s | 0.4261 | 0.3721 | 0.3478 | 0.7179 | 0.5000 | 0.0000 | 0.6667 | 0.0000 |


## Selected FP32 Confusion Matrices


### right_60s Controlled Raw Replay

| Actual | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
| --- | --- | --- | --- | --- | --- | --- |
| WALKING | 42 | 0 | 3 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 0 | 0 | 30 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 0 | 0 | 30 | 0 | 0 | 0 |
| SITTING | 0 | 0 | 0 | 44 | 0 | 1 |
| STANDING | 0 | 0 | 0 | 0 | 45 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 45 |

### left_30s Robustness Raw Replay

| Actual | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
| --- | --- | --- | --- | --- | --- | --- |
| WALKING | 4 | 4 | 11 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 0 | 14 | 1 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 0 | 6 | 9 | 0 | 0 | 0 |
| SITTING | 0 | 0 | 0 | 0 | 0 | 22 |
| STANDING | 0 | 0 | 0 | 0 | 22 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 22 | 0 |

### Arduino V2 Grouped Validation

| Actual | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
| --- | --- | --- | --- | --- | --- | --- |
| WALKING | 18 | 0 | 4 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 0 | 5 | 1 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 0 | 0 | 9 | 0 | 0 | 0 |
| SITTING | 0 | 0 | 0 | 0 | 0 | 22 |
| STANDING | 0 | 0 | 0 | 0 | 22 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 22 |


## Interpretation

- The selected candidate is better than the UCI-only all-LAYING collapse on Arduino-domain data.
- `right_60s` is the strongest validation condition, but `WALKING_UPSTAIRS` is still confused with `WALKING_DOWNSTAIRS`.
- `left_30s` remains weak because SITTING collapses to LAYING and LAYING collapses to STANDING for the selected FP32 model.
- Live on-device Serial evidence is still required separately for final M3 hardware claims.
