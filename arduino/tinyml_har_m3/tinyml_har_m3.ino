#include <Arduino.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>

#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
constexpr uint32_t kSampleIntervalUs = 20000;  // 50 Hz
constexpr uint32_t kRunTimerReportIntervalMs = 30000;
constexpr uint32_t kStabilityTargetMs = 120000;
constexpr int kLedPin = LED_BUILTIN;

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];
float imu_window[kWindowSize][6];
int samples_in_window = 0;
uint32_t next_sample_us = 0;
uint32_t inference_count = 0;
uint64_t latency_sum_us = 0;
uint32_t run_start_ms = 0;
uint32_t next_run_timer_report_ms = kRunTimerReportIntervalMs;
bool stability_reported = false;
}  // namespace

bool addOpStatus(TfLiteStatus status, const char* op_name) {
  if (status == kTfLiteOk) {
    return true;
  }
  Serial.print("Failed to register op: ");
  Serial.println(op_name);
  return false;
}

int8_t quantizeInput(float value) {
  const int32_t quantized = static_cast<int32_t>(roundf(value / kInputScale)) + kInputZeroPoint;
  if (quantized < -128) {
    return -128;
  }
  if (quantized > 127) {
    return 127;
  }
  return static_cast<int8_t>(quantized);
}

void writeInputTensor() {
  float gravity_dir_x = 0.0f;
  float gravity_dir_y = 0.0f;
  float gravity_dir_z = 0.0f;
  float acc_magnitude = 0.0f;
  if (kChannelCount >= 10) {
    float mean_ax = 0.0f;
    float mean_ay = 0.0f;
    float mean_az = 0.0f;
    for (int t = 0; t < kWindowSize; ++t) {
      mean_ax += imu_window[t][0];
      mean_ay += imu_window[t][1];
      mean_az += imu_window[t][2];
    }
    mean_ax /= static_cast<float>(kWindowSize);
    mean_ay /= static_cast<float>(kWindowSize);
    mean_az /= static_cast<float>(kWindowSize);
    acc_magnitude = sqrtf(mean_ax * mean_ax + mean_ay * mean_ay + mean_az * mean_az);
    const float denom = acc_magnitude > 1.0e-6f ? acc_magnitude : 1.0e-6f;
    gravity_dir_x = mean_ax / denom;
    gravity_dir_y = mean_ay / denom;
    gravity_dir_z = mean_az / denom;
  }

  for (int t = 0; t < kWindowSize; ++t) {
    for (int c = 0; c < kChannelCount; ++c) {
      float raw = 0.0f;
      if (c < 3) {
        raw = imu_window[t][c];
      } else if (c < 6) {
        raw = imu_window[t][c];
        raw *= kGyroDegToRad;
      } else if (c == 6) {
        raw = gravity_dir_x;
      } else if (c == 7) {
        raw = gravity_dir_y;
      } else if (c == 8) {
        raw = gravity_dir_z;
      } else if (c == 9) {
        raw = acc_magnitude;
      }
      const float normalized = (raw - kFeatureMean[c]) / kFeatureStd[c];
      const int idx = t * kChannelCount + c;
      if (input->type == kTfLiteInt8) {
        input->data.int8[idx] = quantizeInput(normalized);
      } else {
        input->data.f[idx] = normalized;
      }
    }
  }
}

float outputScore(int class_index) {
  if (output->type == kTfLiteInt8) {
    return (static_cast<int>(output->data.int8[class_index]) - output->params.zero_point) * output->params.scale;
  }
  return output->data.f[class_index];
}

void shiftWindow() {
  for (int i = 0; i < kWindowStride; ++i) {
    for (int c = 0; c < 6; ++c) {
      imu_window[i][c] = imu_window[i + kWindowStride][c];
    }
  }
  samples_in_window = kWindowStride;
}

void updateLed(int predicted) {
  // One built-in LED is available, so use a simple dynamic/static state.
  digitalWrite(kLedPin, predicted <= 2 ? HIGH : LOW);
}

void reportRunTimer() {
  const uint32_t elapsed_ms = millis() - run_start_ms;
  if (elapsed_ms >= next_run_timer_report_ms) {
    Serial.print("run_timer,elapsed_ms=");
    Serial.print(elapsed_ms);
    Serial.print(",elapsed_s=");
    Serial.print(elapsed_ms / 1000);
    Serial.print(",window_count=");
    Serial.print(inference_count);
    Serial.println(",status=running");
    next_run_timer_report_ms += kRunTimerReportIntervalMs;
  }

  if (!stability_reported && elapsed_ms >= kStabilityTargetMs) {
    stability_reported = true;
    Serial.print("stability_check,elapsed_ms=");
    Serial.print(elapsed_ms);
    Serial.print(",window_count=");
    Serial.print(inference_count);
    Serial.println(",status=passed_2min_if_no_crash");
  }
}

void runInference() {
  writeInputTensor();

  const uint32_t timestamp_ms = millis();
  const uint32_t start_us = micros();
  const TfLiteStatus invoke_status = interpreter->Invoke();
  const uint32_t latency_us = micros() - start_us;
  if (invoke_status != kTfLiteOk) {
    Serial.println("invoke_status=failed");
    return;
  }

  int predicted = 0;
  float best_score = outputScore(0);
  int second_predicted = 1;
  float second_score = outputScore(1);
  if (second_score > best_score) {
    second_predicted = 0;
    second_score = best_score;
    predicted = 1;
    best_score = outputScore(1);
  }
  for (int i = 2; i < kClassCount; ++i) {
    const float score = outputScore(i);
    if (score > best_score) {
      second_score = best_score;
      second_predicted = predicted;
      best_score = score;
      predicted = i;
    } else if (score > second_score) {
      second_score = score;
      second_predicted = i;
    }
  }

  ++inference_count;
  latency_sum_us += latency_us;
  const uint32_t avg_latency_us = static_cast<uint32_t>(latency_sum_us / inference_count);

  updateLed(predicted);

  Serial.print(timestamp_ms);
  Serial.print(",");
  Serial.print(inference_count);
  Serial.print(",");
  Serial.print(predicted);
  Serial.print(",");
  Serial.print(kClassNames[predicted]);
  Serial.print(",");
  Serial.print(best_score, 4);
  Serial.print(",");
  Serial.print(latency_us);
  Serial.print(",");
  Serial.print(best_score, 3);
  Serial.print(",");
  Serial.print(kClassNames[second_predicted]);
  Serial.print(",");
  Serial.print(second_score, 4);
  Serial.print(",");
  Serial.print(avg_latency_us);
  Serial.println();

  if (inference_count >= 50 && inference_count % 50 == 0) {
    Serial.print("latency_summary,window_count=");
    Serial.print(inference_count);
    Serial.print(",avg_invoke_latency_us=");
    Serial.println(avg_latency_us);
  }
}

void appendSample(float ax, float ay, float az, float gx, float gy, float gz) {
  if (samples_in_window >= kWindowSize) {
    return;
  }
  imu_window[samples_in_window][0] = ax;
  imu_window[samples_in_window][1] = ay;
  imu_window[samples_in_window][2] = az;
  imu_window[samples_in_window][3] = gx;
  imu_window[samples_in_window][4] = gy;
  imu_window[samples_in_window][5] = gz;
  ++samples_in_window;

  if (samples_in_window == kWindowSize) {
    runInference();
    shiftWindow();
  }
}

void setupModel() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_tinyml_har_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema version mismatch");
    while (true) {
      delay(1000);
    }
  }

  static tflite::MicroMutableOpResolver<8> resolver;
  bool ok = true;
  ok &= addOpStatus(resolver.AddReshape(), "RESHAPE");
  ok &= addOpStatus(resolver.AddConv2D(), "CONV_2D");
  ok &= addOpStatus(resolver.AddDepthwiseConv2D(), "DEPTHWISE_CONV_2D");
  ok &= addOpStatus(resolver.AddAveragePool2D(), "AVERAGE_POOL_2D");
  ok &= addOpStatus(resolver.AddMean(), "MEAN");
  ok &= addOpStatus(resolver.AddFullyConnected(), "FULLY_CONNECTED");
  ok &= addOpStatus(resolver.AddSoftmax(), "SOFTMAX");
  if (!ok) {
    while (true) {
      delay(1000);
    }
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed; increase kTensorArenaSize");
    while (true) {
      delay(1000);
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("model_bytes=");
  Serial.println(g_tinyml_har_model_len);
  Serial.print("tensor_arena_bytes=");
  Serial.println(kTensorArenaSize);
  Serial.print("kChannelCount=");
  Serial.println(kChannelCount);
  Serial.print("kWindowSize=");
  Serial.println(kWindowSize);
  Serial.print("kStride=");
  Serial.println(kStride);
  Serial.print("normalization_source=");
  Serial.println(kNormalizationSource);
  Serial.print("input_scale=");
  Serial.println(kInputScale, 9);
  Serial.print("input_zero_point=");
  Serial.println(kInputZeroPoint);
  Serial.print("output_scale=");
  Serial.println(kOutputScale, 9);
  Serial.print("output_zero_point=");
  Serial.println(kOutputZeroPoint);
}

void setup() {
  pinMode(kLedPin, OUTPUT);
  digitalWrite(kLedPin, LOW);

  Serial.begin(115200);
  const uint32_t serial_start_ms = millis();
  while (!Serial && millis() - serial_start_ms < 5000) {
    delay(10);
  }

  Serial.println("tinyml_har_m3_boot");
  if (!IMU.begin()) {
    Serial.println("IMU.begin failed");
    while (true) {
      delay(1000);
    }
  }

  setupModel();
  run_start_ms = millis();
  next_run_timer_report_ms = kRunTimerReportIntervalMs;
  stability_reported = false;
  next_sample_us = micros();
  Serial.println("status=ready,sampling_hz=50,window=128,stride=64");
  Serial.println("timer_output=run_timer every 30s; stability_check after 120s; inference continues until unplugged/reset");
  Serial.println("placement_guidance=right pocket recommended; keep fixed orientation; stair direction may be confused; see README");
  Serial.println("timestamp_ms,window_id,prediction_id,prediction_label,confidence,latency_us,top1_score,top2_label,top2_score,avg_latency_us");
}

void loop() {
  reportRunTimer();

  const uint32_t now_us = micros();
  if (static_cast<int32_t>(now_us - next_sample_us) < 0) {
    return;
  }
  next_sample_us += kSampleIntervalUs;

  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    return;
  }

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);
  appendSample(ax, ay, az, gx, gy, gz);
}
