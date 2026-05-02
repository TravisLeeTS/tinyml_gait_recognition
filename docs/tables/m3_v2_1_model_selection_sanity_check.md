# V2.1 Model Selection Sanity Check

Selection priority for the M3 controlled demo is `right_60s` macro F1 first, then `left_30s` macro F1, then degraded-class recall. INT8 rows use representative samples from training/adaptation data only; held-out `right_60s` and `left_30s` are evaluation only.

| Model | UCI INT8 Macro F1 | V2 Group INT8 Macro F1 | Right 60s INT8 Macro F1 | Left 30s INT8 Macro F1 | Right-Left Gap | Worst-Class Recall | Main Failure | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| UCI-only baseline | 0.9025 | 0.0587 | 0.0526 | 0.0535 | -0.0009 | 0.0000 | right_60s: WALKING->LAYING (45); right_60s: SITTING->LAYING (45) | Rejected: source benchmark only; Arduino replay collapses |
| Arduino-only | 0.1082 | 0.4686 | 0.8050 | 0.5094 | 0.2956 | 0.0000 | right_60s: SITTING->LAYING (37); left_30s: SITTING->LAYING (22) | Rejected: high right replay after PTQ, but right SITTING recall is weak, left static postures collapse, and UCI retention is too poor |
| UCI pretrain + V2.1 fine-tune | 0.4630 | 0.1986 | 0.2582 | 0.3442 | -0.0860 | 0.0000 | right_60s: STANDING->WALKING (45); right_60s: LAYING->WALKING_DOWNSTAIRS (45) | Rejected: weak target-domain validation |
| Mixed 6-channel DS-CNN | 0.7960 | 0.7309 | 0.7741 | 0.3886 | 0.3855 | 0.0000 | right_60s: WALKING_UPSTAIRS->WALKING_DOWNSTAIRS (30); left_30s: SITTING->LAYING (22) | Selected controlled-demo candidate; robustness warning for left static postures |
| Mixed focal 6-channel DS-CNN | 0.7396 | 0.7099 | 0.7357 | 0.5643 | 0.1714 | 0.0000 | right_60s: WALKING_UPSTAIRS->WALKING_DOWNSTAIRS (29); left_30s: SITTING->LAYING (22) | Intermediate/rejected: good V2 grouped, weaker right controlled replay |
| Mixed 10-channel gravity DS-CNN | 0.8832 | 0.6189 | 0.5655 | 0.6856 | -0.1201 | 0.0000 | right_60s: SITTING->LAYING (45); right_60s: WALKING_UPSTAIRS->WALKING_DOWNSTAIRS (26) | Robustness candidate; not selected due weaker right replay and 10-channel cost |

## FP32 Reference

| Model | UCI FP32 Macro F1 | V2 Group FP32 Macro F1 | Right 60s FP32 Macro F1 | Left 30s FP32 Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| UCI-only baseline | 0.9041 | 0.0587 | 0.0526 | 0.0535 |
| Arduino-only | 0.0920 | 0.4857 | 0.5163 | 0.7300 |
| UCI pretrain + V2.1 fine-tune | 0.4563 | 0.1986 | 0.2661 | 0.3166 |
| Mixed 6-channel DS-CNN | 0.7932 | 0.7097 | 0.7647 | 0.3721 |
| Mixed focal 6-channel DS-CNN | 0.8124 | 0.7651 | 0.5556 | 0.4070 |
| Mixed 10-channel gravity DS-CNN | 0.8784 | 0.6098 | 0.5548 | 0.6987 |

## Decision Notes

- `m3_v2_1_mixed_6ch` remains selected because it is the strongest controlled-demo candidate among UCI-retaining mixed models and has much more stable right-pocket static-posture behavior than Arduino-only.
- The same model is not robust under the left-pocket condition: left_30s INT8 macro F1 is only `0.3886`, with `SITTING -> LAYING` and `LAYING -> STANDING` collapse.
- Arduino-only is not selected despite `0.8050` right_60s INT8 macro F1 because it is a target-only diagnostic, has only `0.1082` UCI-HAR INT8 macro F1, and right-pocket `SITTING` still mostly predicts as `LAYING`.
- The 10-channel gravity model is a robustness candidate because it has much better left_30s macro F1 (`0.6856` INT8), but it is not selected for M3 deployment because controlled right_60s is lower (`0.5655` INT8) and it requires a more complex 10-channel Arduino path.
- Live Arduino Serial accuracy is still pending; these are raw sensor replay validations.
