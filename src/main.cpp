#include <Arduino.h>

namespace {
constexpr uint32_t kSampleIntervalMs = 1000;
constexpr uint8_t kStatusLedPin = 2;
uint32_t lastSampleMs = 0;
uint32_t sampleCount = 0;
}  // namespace

void setup() {
  pinMode(kStatusLedPin, OUTPUT);
  Serial.begin(115200);

  while (!Serial) {
    delay(10);
  }

  Serial.println();
  Serial.println("TinyML ESP32 simulation starter");
}

void loop() {
  const uint32_t now = millis();
  if (now - lastSampleMs >= kSampleIntervalMs) {
    lastSampleMs = now;
    ++sampleCount;

    digitalWrite(kStatusLedPin, !digitalRead(kStatusLedPin));

    const int pseudoSensorValue = random(300, 900);
    Serial.print("sample=");
    Serial.print(sampleCount);
    Serial.print(", value=");
    Serial.println(pseudoSensorValue);
  }
}
