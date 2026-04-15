# TinyML Project Copilot Instructions

## Project context

- Primary classroom board: Arduino Nano 33 BLE Sense (no camera).
- Current simulation target: ESP32 in Wokwi.
- Development environment: VS Code + PlatformIO.
- Do not suggest Arduino IDE workflows unless explicitly asked.

## Implementation guidance

1. Prefer PlatformIO-compatible Arduino framework C++ code.
2. Keep hardware access isolated behind small adapter functions or classes.
3. Write code so it can move from ESP32 simulation to BLE Sense hardware with minimal edits.
4. Prefer non-blocking timing based on millis() for periodic tasks.
5. Use Serial logging at 115200 for observability during simulation.
6. Keep changes incremental and testable in Wokwi.
7. When suggesting libraries, prefer packages available through the PlatformIO registry.
8. Add comments only when logic is not immediately clear.

## Output expectations

- Prioritize compile-ready code for the env:esp32dev target in platformio.ini.
- If a feature depends on real sensors unavailable in simulation, provide a simulation fallback.
- Keep file organization simple: entry logic in src/main.cpp and reusable code in include/ and src/.
