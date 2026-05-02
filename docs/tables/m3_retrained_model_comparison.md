# M3 V2.1 Six-Model Retraining Comparison

Training/adaptation data: UCI-HAR train, Arduino V2 train split, and capped V2.1 long adaptation files. Held-out `right_60s` and `left_30s` were not used for training, normalization, representative quantization, or early stopping.

## Final Six-Model Comparison

| Model | UCI Acc | UCI Macro F1 | V2 Group Acc | V2 Group Macro F1 | Right 60s Acc | Right 60s Macro F1 | Left 30s Acc | Left 30s Macro F1 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UCI-only baseline | 0.9050 | 0.9041 | 0.2136 | 0.0587 | 0.1875 | 0.0526 | 0.1913 | 0.0535 | Rejected: UCI benchmark only; Arduino validation collapses to LAYING |
| Arduino-only | 0.1812 | 0.0920 | 0.5340 | 0.4857 | 0.6042 | 0.5163 | 0.7739 | 0.7300 | Rejected: right/left tradeoff and UCI retention too weak |
| UCI pretrain + V2.1 fine-tune | 0.5124 | 0.4563 | 0.4175 | 0.1986 | 0.3708 | 0.2661 | 0.3826 | 0.3166 | Rejected: weak target-domain validation |
| Mixed 6-channel DS-CNN | 0.7957 | 0.7932 | 0.7379 | 0.7097 | 0.8583 | 0.7647 | 0.4261 | 0.3721 | Selected M3 candidate build; not fully validated due left static-posture collapse |
| Mixed focal 6-channel DS-CNN | 0.8161 | 0.8124 | 0.7767 | 0.7651 | 0.6875 | 0.5556 | 0.4522 | 0.4070 | Intermediate/rejected: good V2 and UCI, weaker right_60s than selected candidate |
| Mixed 10-channel gravity DS-CNN | 0.8792 | 0.8784 | 0.6699 | 0.6098 | 0.6667 | 0.5548 | 0.7478 | 0.6987 | Intermediate/rejected: better left_30s, but right_60s weaker and 10-channel deployment is more complex |


## Selection Interpretation

Selected candidate: `m3_v2_1_mixed_6ch`. It is selected by the requested primary metric, `right_60s` macro F1. It is still not scientifically final because left-side SITTING/LAYING remains weak and true live Arduino Serial evidence is pending.

## Failure-Mode Summary

| Model | Right-Left Macro F1 Gap | Right Downstairs Recall | Left Downstairs Recall | Right Sitting Recall | Left Sitting Recall | Right Laying Recall | Left Laying Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UCI-only baseline | -0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Arduino-only | -0.2137 | 0.3333 | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| UCI pretrain + V2.1 fine-tune | -0.0505 | 0.0000 | 0.4000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Mixed 6-channel DS-CNN | 0.3927 | 1.0000 | 0.6000 | 0.9778 | 0.0000 | 1.0000 | 0.0000 |
| Mixed focal 6-channel DS-CNN | 0.1485 | 1.0000 | 0.6000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Mixed 10-channel gravity DS-CNN | -0.1438 | 0.9667 | 0.6000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |


## FP32 vs INT8 For Selected Candidate

| run_name | experiment | split | accuracy | macro_f1 | weighted_f1 | model_size_bytes | precision_walking | recall_walking | f1_walking | precision_walking_upstairs | recall_walking_upstairs | f1_walking_upstairs | precision_walking_downstairs | recall_walking_downstairs | f1_walking_downstairs | precision_sitting | recall_sitting | f1_sitting | precision_standing | recall_standing | f1_standing | precision_laying | recall_laying | f1_laying | model_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m3_v2_1_mixed_6ch | mixed_6ch_v2_1 | uci_test | 0.7957 | 0.7932 | 0.7995 | 13460 | 0.7632 | 0.6431 | 0.6980 | 0.9182 | 0.8344 | 0.8743 | 0.5570 | 0.7333 | 0.6331 | 0.7500 | 0.7699 | 0.7598 | 0.8321 | 0.8383 | 0.8352 | 0.9862 | 0.9330 | 0.9589 | FP32 |
| m3_v2_1_mixed_6ch | mixed_6ch_v2_1 | arduino_v2_grouped | 0.7379 | 0.7097 | 0.6696 | 13460 | 1.0000 | 0.8182 | 0.9000 | 1.0000 | 0.8333 | 0.9091 | 0.6429 | 1.0000 | 0.7826 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 0.6667 | FP32 |
| m3_v2_1_mixed_6ch | mixed_6ch_v2_1 | right_60s | 0.8583 | 0.7647 | 0.8200 | 13460 | 1.0000 | 0.9333 | 0.9655 | 0.0000 | 0.0000 | 0.0000 | 0.4762 | 1.0000 | 0.6452 | 1.0000 | 0.9778 | 0.9888 | 1.0000 | 1.0000 | 1.0000 | 0.9783 | 1.0000 | 0.9890 | FP32 |
| m3_v2_1_mixed_6ch | mixed_6ch_v2_1 | left_30s | 0.4261 | 0.3721 | 0.3439 | 13460 | 1.0000 | 0.2105 | 0.3478 | 0.5833 | 0.9333 | 0.7179 | 0.4286 | 0.6000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | FP32 |
| m3_v2_1_mixed_6ch_int8 | mixed_6ch_v2_1 | uci_test | 0.7984 | 0.7960 | 0.8021 | 10288 | 0.7604 | 0.6653 | 0.7097 | 0.9215 | 0.8471 | 0.8827 | 0.5706 | 0.7214 | 0.6372 | 0.7436 | 0.7739 | 0.7585 | 0.8318 | 0.8365 | 0.8341 | 0.9861 | 0.9236 | 0.9538 | INT8 |
| m3_v2_1_mixed_6ch_int8 | mixed_6ch_v2_1 | arduino_v2_grouped | 0.7573 | 0.7309 | 0.6873 | 10288 | 1.0000 | 0.9091 | 0.9524 | 1.0000 | 0.8333 | 0.9091 | 0.7500 | 1.0000 | 0.8571 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 0.6667 | INT8 |
| m3_v2_1_mixed_6ch_int8 | mixed_6ch_v2_1 | right_60s | 0.8708 | 0.7741 | 0.8292 | 10288 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 0.9778 | 0.9888 | 1.0000 | 1.0000 | 1.0000 | 0.9783 | 1.0000 | 0.9890 | INT8 |
| m3_v2_1_mixed_6ch_int8 | mixed_6ch_v2_1 | left_30s | 0.4348 | 0.3886 | 0.3614 | 10288 | 1.0000 | 0.3158 | 0.4800 | 0.5600 | 0.9333 | 0.7000 | 0.4444 | 0.5333 | 0.4848 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | INT8 |


Detailed per-class metrics and confusion matrices are in `outputs/deployment/m3_retrained_with_v2_1/<run>/metrics/` and `outputs/deployment/m3_retrained_with_v2_1/confusion_matrices/`.
