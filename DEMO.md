# TinyML HAR Live Demo Guide

Project: Energy-Efficient Human Activity Recognition on the Edge  
Board: Arduino Nano 33 BLE Sense  
Model: INT8 mixed 6-channel DS-CNN  
Goal: show live on-device human activity recognition from real IMU input

## Before Screen Share

Open these before the demo starts:

| Item | Use During Demo |
|---|---|
| Arduino Serial Monitor | Main live inference evidence |
| Webcam / phone camera | Show board placement and movement |
| GitHub repo tab | Reproducibility proof near the end |
| Backup video or Serial log | Use only if hardware fails |
| Metric note | Accuracy, F1, model size, latency, RAM |

Serial Monitor should show prediction lines with:

```text
timestamp_ms,window_id,prediction_id,prediction_label,confidence,latency_us,...
```

## 0:00-0:30 Task, Board, Classes

**Show:** Serial Monitor and Arduino Nano 33 BLE Sense.

**Say:**

Good evening. Our project is an on-device TinyML human activity recognition system using the Arduino Nano 33 BLE Sense.

The model uses live accelerometer and gyroscope data to classify six activities:

| Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 |
|---|---|---|---|---|---|
| Walking | Walking upstairs | Walking downstairs | Sitting | Standing | Laying |

The goal is to run inference directly on the Arduino, so raw motion data does not need to be sent to the cloud.

## 0:30-1:20 Project Story

**Show:** Serial Monitor or this section. Do not open GitHub yet.

**Say:**

Our original baseline was a strong ensemble, not a tiny model. We reproduced a paper-style ensemble with five neural sequence models: ConvLSTM, CNN-GRU, CNN-BiGRU, CNN-BiLSTM, and CNN-LSTM, then stacked their outputs with XGBoost.

That gave a strong public-dataset result, but it was too heavy and complex for Arduino deployment.

We then tested a lightweight depthwise-separable 1D CNN. It achieved similar accuracy on the public UCI HAR dataset while being much smaller.

This changed the project direction: instead of making the ensemble smaller, we focused on making the lightweight model robust enough for real Arduino data.

**Key point:** the main challenge became domain shift, not public-dataset accuracy.

## 1:20-2:15 Dataset And Domain Shift

**Show:** Dataset table.

| Dataset / Split | Role | Windows |
|---|---|---:|
| UCI HAR train | Public source training | 5,551 |
| UCI HAR validation | Subject-aware validation from official train split | 1,801 |
| UCI HAR test | Public held-out test | 2,947 |
| Arduino V2 | Corrected 50 Hz target-domain data | 236 |
| Arduino V2 train | Group-wise adaptation training | 133 |
| Arduino V2 validation | Group-wise validation | 103 |
| V2.1 long adaptation | Walking, sitting, standing, laying | 932 before cap / 720 after cap |
| Right-pocket live logs | Final controlled evaluation | 125 rows |
| Left-pocket live logs | Final robustness evaluation | 91 rows |

**Say:**

We used UCI HAR as the public source dataset. One model input is a 128 by 6 window: 128 timesteps and six IMU channels, acceleration X/Y/Z and gyroscope X/Y/Z. At 50 Hz, each window covers 2.56 seconds.

The UCI HAR dataset has 10,299 windows. We used 5,551 for training, 1,801 for validation, and 2,947 for public test evaluation. The UCI train split was separated into train and validation using a subject-aware split, while the official test set stayed held out.

When we collected Arduino data, the distribution shifted because UCI HAR came from a smartphone setup, while our data came from Arduino pocket placement and board orientation. The UCI-only model did not transfer well.

To adapt, we used Arduino V2 as corrected 50 Hz target-domain data. V2.1 added four 5-minute recordings for walking, sitting, standing, and laying. Right-pocket and left-pocket live data were kept for final evaluation, not training.

## 2:15-3:05 Six Approaches Tried

**Show:** Model comparison table.

| Attempt | Result | Reason |
|---|---|---|
| UCI-only | Failed on Arduino | Learned public smartphone distribution, not Arduino pocket placement |
| Arduino-only | Not selected | Too little target data; high overfit risk |
| UCI pretrain then Arduino fine-tune | Not selected | Fine-tuning was unstable with small target data |
| Mixed UCI + Arduino 6-channel | Selected | Best balance of source knowledge and Arduino adaptation |
| Mixed focal 6-channel | Useful but not final | Helped some hard cases, but weaker controlled validation |
| Mixed 10-channel gravity | Future candidate | Adds orientation features, but did not justify replacing 6-channel model |

**Say:**

The selected final model is the mixed 6-channel DS-CNN. It keeps the deployment pipeline simple and adapts the model to the Arduino distribution without relying only on small Arduino data.

The final workflow combines UCI training data, Arduino V2 training data, and capped V2.1 long-adaptation data, then evaluates on UCI test, Arduino grouped validation, right-pocket, and left-pocket splits.

## 3:05-3:35 Preprocessing Pipeline

**Show:** Pipeline.

```text
LSM9DS1 IMU
  -> 50 Hz sampling
  -> 128-sample window
  -> 50% overlap, new prediction every 64 samples
  -> per-channel standardization
  -> INT8 input quantization
  -> TFLite Micro Invoke()
  -> Serial prediction label, confidence, latency
```

**Say:**

The preprocessing pipeline reads IMU data at 50 Hz, collects 128 samples, uses 50% overlap, normalizes each channel using the training standardizer, then runs TensorFlow Lite Micro inference.

For the selected model, the input has six channels: acceleration X/Y/Z and gyroscope X/Y/Z.

We also tested a 10-channel gravity version. It adds gravity direction X/Y/Z and acceleration magnitude. Because it did not validate better than the simpler 6-channel model, we kept the 6-channel model for the live demo.

## 3:35-5:05 Live Class Demonstration

**Show:** Arduino Serial Monitor.  
**Camera:** show board placement and movement.

**Say:**

Now I will demonstrate the model live. I will use the controlled right-pocket placement, because that is the validated setup for our final demo. Please watch the Serial Monitor prediction label, confidence, and latency.

| Demo Step | What To Do | What To Say |
|---|---|---|
| Standing | Stand still, fixed board orientation | First is standing. I will keep the board still and maintain the same orientation. |
| Sitting | Sit with consistent board angle | Next is sitting. Static posture classes depend strongly on board angle, so I keep the orientation consistent. |
| Walking | Walk naturally | Now walking. This is live IMU input from the Arduino, not laptop replay. |
| Walking upstairs | Perform validated upstairs movement | Now walking upstairs. This is one of the hardest classes because upstairs and downstairs have similar periodic leg motion. |
| Walking downstairs | Perform validated downstairs movement | Now walking downstairs. Please watch whether the model separates downstairs from upstairs. |
| Laying | Place board in laying posture | Finally, laying. This is another static posture class and usually works well when the board orientation is consistent. |

## 5:05-5:45 Robustness Scenario

**Show:** Serial Monitor.  
**Camera:** show placement change.

**Say:**

Now I will show one robustness scenario. I will change from the controlled right-pocket setup to a shifted placement or left-pocket setup.

This tests a realistic deployment issue: accelerometer and gyroscope signals change when the board position or orientation changes.

If it works:

> The prediction remains reasonable, although confidence may change. This matches our robustness result, where the left-pocket live condition remained close to the controlled setup.

If it fails:

> This is also useful evidence. It shows our main deployment limitation: the model is sensitive to placement and orientation shift.

## 5:45-6:20 Final Metrics And Limitation

**Show:** Metrics table.

| Metric | Result |
|---|---:|
| Controlled right-pocket live accuracy | 90.40% |
| Controlled right-pocket macro F1 | 0.9089 |
| Right-pocket prediction rows | 125 |
| Left-pocket robustness live accuracy | 89.01% |
| Left-pocket robustness macro F1 | 0.8826 |
| Left-pocket prediction rows | 91 |
| INT8 model size | 10,288 bytes / 10.05 KiB |
| Tensor arena | 61,440 bytes / 60.00 KiB |
| Average Arduino `Invoke()` latency | 34.113 ms over 50 calls |
| Main failure mode | `WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS` |

**Say:**

Our final deployable model is the INT8 mixed 6-channel DS-CNN.

In true live Arduino Serial trials, the controlled right-pocket condition achieved 90.40% accuracy and 0.9089 macro F1 over 125 prediction rows.

The left-pocket robustness condition achieved 89.01% accuracy and 0.8826 macro F1 over 91 prediction rows.

The INT8 model size is 10,288 bytes, the tensor arena is 61,440 bytes, and the average Arduino Invoke latency is 34.113 milliseconds over 50 inferences.

The main limitation is stair-direction confusion. Walking, walking downstairs, sitting, standing, and laying were strong, but walking upstairs was often predicted as walking downstairs.

## 6:20-6:45 GitHub Reproducibility Package

**Show:** GitHub repository. Open only the high-level files and folders.

```text
README.md
README_MILESTONE3.md
DEMO.md
arduino/
src/
outputs/
docs/tables/
requirements.txt
tinyml_milestone3.pdf
```

**Say:**

This is our GitHub reproducibility package. It includes the training code, preprocessing, quantization workflow, Arduino sketch, model header, live evidence logs, result tables, and documentation.

The repository allows someone to reproduce the path from data, to training, to TFLite conversion, to Arduino deployment.

## 6:45-7:00 Closing

**Show:** Return to Serial Monitor if possible.

**Say:**

To summarize, public-dataset accuracy was not enough. The ensemble and lightweight model both worked on UCI HAR, but real Arduino deployment introduced domain shift from sensor placement and orientation.

Our final solution uses mixed source and target-domain training, INT8 quantization, and live Arduino inference. The system works well in the controlled setup, remains reasonable under left-pocket robustness testing, and the main remaining limitation is stair-direction separation.

Thank you. I am ready for questions.

## Backup Plan

If hardware or Serial Monitor fails:

1. Say: "The live hardware path normally reads from the onboard IMU and prints prediction labels, confidence, and latency through Serial Monitor."
2. Show the 30-60 second backup video or returned Serial logs.
3. Continue with the metrics and GitHub reproducibility sections.

## Q&A Quick Answers

| Question | Short Answer |
|---|---|
| Why start with an ensemble? | It gave a strong reference benchmark. It was too heavy for Arduino, so it became the offline baseline, not the deployment model. |
| Why did UCI-only fail on Arduino? | UCI HAR is smartphone-based; our Arduino data has different placement, orientation, sensor behavior, and movement patterns. |
| Why mixed training? | It keeps broad UCI HAR activity knowledge while adapting to Arduino-specific distribution. |
| Why not Arduino-only? | The Arduino dataset is small, especially for stair classes, so Arduino-only training can overfit. |
| Why cap V2.1? | V2.1 only has four classes. Capping prevents walking/sitting/standing/laying from dominating stair classes. |
| Why is upstairs weakest? | Upstairs and downstairs have similar periodic leg motion, and V2.1 did not include extra stair adaptation data. |
| What is one input sample? | A 128 x 6 IMU window: 128 timesteps, acceleration XYZ, gyroscope XYZ, covering 2.56 seconds at 50 Hz. |
| How was data split? | UCI test stayed official and held out; UCI train was split subject-aware; Arduino V2 was split group-wise; V2.1 was training adaptation; right/left live logs were final evaluation only. |
| What preprocessing is used? | 50 Hz sampling, 128-sample windows, 50% overlap, six IMU channels, training standardization, INT8 TFLite Micro inference. |
| Why not deploy 10-channel gravity? | It is useful for orientation features, but it did not beat the 6-channel model in the selected validation path and adds deployment complexity. |
| What is the architecture? | A small depthwise-separable 1D CNN designed for TinyML windowed IMU classification. |
| What quantization did we use? | Full-integer INT8 post-training quantization with representative data from training/adaptation windows only. |
| Offline vs live accuracy? | Offline uses stored windows or replay; live accuracy comes from physical Arduino Serial predictions during movement. |
| Is this medical? | No. It is a TinyML activity-recognition deployment demo, not a medical or diagnostic device. |
| Main limitations? | Stair-direction confusion, placement/orientation sensitivity, small Arduino target-domain data, and limited user diversity. |
| Four more weeks? | Collect more upstairs/downstairs data, add users and placements, retry orientation-aware features, and add confidence smoothing or rejection. |
| Why TinyML? | It reduces latency, keeps data on-device, avoids network dependence, and supports low-power wearable-style sensing. |
| What does Serial Monitor prove? | It shows the deployed Arduino model producing real-time labels, confidence, and latency from live IMU input. |
| What if demo fails? | Use backup video or Serial logs, explain the normal live path, and continue Q&A. |

## One-Sentence Summary

The main contribution is showing that TinyML HAR deployment is not just model compression; it is a domain adaptation problem where public-dataset accuracy must be reconciled with real Arduino placement, orientation, quantization, latency, memory, and live robustness constraints.
