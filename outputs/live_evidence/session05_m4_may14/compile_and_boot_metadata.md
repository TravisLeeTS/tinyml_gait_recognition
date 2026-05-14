# Session05 M4 Arduino Compile And Boot Metadata

Source: live Arduino run shared on May 14, 2026.

## Arduino Compile / Upload Output

```text
Sketch uses 178264 bytes (18%) of program storage space. Maximum is 983040 bytes.
Global variables use 123432 bytes (47%) of dynamic memory, leaving 138712 bytes for local variables. Maximum is 262144 bytes.
Device       : nRF52840-QIAA
Version      : Arduino Bootloader (SAM-BA extended) 2.0 [Arduino:IKXYZ]
Address      : 0x0
Pages        : 256
Page Size    : 4096 bytes
Total Size   : 1024KB
Planes       : 1
Lock Regions : 0
Locked       : none
Security     : false
Erase flash  Done in 0.001 seconds
Write 178272 bytes to flash (44 pages)
```

## Boot Metadata

This boot block appears in the live Serial logs.

```text
tinyml_har_m3_boot
model_bytes=10432
tensor_arena_bytes=73728
kChannelCount=10
kWindowSize=128
kStride=64
normalization_source=UCI-HAR train + Arduino V2/V2.1 + old M3 raw replay validation + standardized sessions 3 and 4; standardized session 1 and live Serial logs excluded
input_scale=0.147774130
input_zero_point=11
output_scale=0.003906250
output_zero_point=-128
status=ready,sampling_hz=50,window=128,stride=64
timer_output=run_timer every 30s; stability_check after 120s; inference continues until unplugged/reset
placement_guidance=right pocket recommended; keep fixed orientation; stair direction may be confused; see README
timestamp_ms,window_id,prediction_id,prediction_label,confidence,latency_us,top1_score,top2_label,top2_score,avg_latency_us
```

## Evidence Notes

- Raw logs are under `outputs/live_evidence/session05_m4_may14/raw/`.
- Scored metrics are under `outputs/live_evidence/session05_m4_may14/metrics/`.
- Summary tables are `session05_m4_live_summary.csv` and `session05_m4_live_summary.md`.
- This is live Arduino Serial evidence, not laptop replay.
