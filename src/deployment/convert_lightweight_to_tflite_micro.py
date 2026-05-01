from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import OUTPUTS_DIR, SEED
from src.data.uci_har import load_uci_har_sequence_dataset
from src.utils.metrics import classification_outputs, save_classification_outputs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import set_global_seed


def representative_dataset(x_train: np.ndarray, count: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_train), size=min(count, len(x_train)), replace=False)

    def _generator():
        for idx in indices:
            yield [x_train[idx : idx + 1].astype(np.float32)]

    return _generator


def convert_int8(
    model: keras.Model,
    x_train: np.ndarray,
    representative_count: int,
    seed: int,
) -> bytes:
    input_shape = [1, *x_train.shape[1:]]

    @tf.function
    def serve(inputs):
        return model(inputs, training=False)

    concrete_func = serve.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(x_train, representative_count, seed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def quantize_input(x: np.ndarray, input_detail: dict) -> np.ndarray:
    scale, zero_point = input_detail["quantization"]
    if input_detail["dtype"] == np.float32:
        return x.astype(np.float32)
    if not scale:
        raise ValueError("Quantized input tensor is missing scale")
    q = np.round(x / scale + zero_point)
    info = np.iinfo(input_detail["dtype"])
    return np.clip(q, info.min, info.max).astype(input_detail["dtype"])


def dequantize_output(y: np.ndarray, output_detail: dict) -> np.ndarray:
    scale, zero_point = output_detail["quantization"]
    if output_detail["dtype"] == np.float32 or not scale:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero_point) * scale


def run_tflite(model_path: Path, x: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    outputs = []
    for sample in x:
        interpreter.set_tensor(input_detail["index"], quantize_input(sample[None, ...], input_detail))
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])
        outputs.append(dequantize_output(raw, output_detail)[0])
    raw_ops = [op["op_name"] for op in interpreter._get_ops_details()]
    metadata = {
        "input_dtype": str(input_detail["dtype"]),
        "output_dtype": str(output_detail["dtype"]),
        "input_scale": float(input_detail["quantization"][0]),
        "input_zero_point": int(input_detail["quantization"][1]),
        "output_scale": float(output_detail["quantization"][0]),
        "output_zero_point": int(output_detail["quantization"][1]),
        "ops": [op for op in raw_ops if op != "DELEGATE"],
        "host_delegate_ops": int(sum(1 for op in raw_ops if op == "DELEGATE")),
    }
    return np.asarray(outputs, dtype=np.float32), metadata


def benchmark_tflite_ms(model_path: Path, sample: np.ndarray, runs: int = 50, warmup: int = 5) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    q_sample = quantize_input(sample[None, ...], input_detail)

    for _ in range(warmup):
        interpreter.set_tensor(input_detail["index"], q_sample)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])

    timings = []
    for _ in range(runs):
        interpreter.set_tensor(input_detail["index"], q_sample)
        start = time.perf_counter()
        interpreter.invoke()
        timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "runs": int(runs),
        "device_note": "Host TensorFlow Lite Interpreter latency only; M3 requires Arduino micros() latency.",
    }


def format_float_array(values: list[float]) -> str:
    return ", ".join(f"{value:.9g}f" for value in values)


def write_model_header(
    *,
    model_bytes: bytes,
    header_path: Path,
    standardizer: SequenceStandardizer,
    class_names: list[str],
    input_scale: float,
    input_zero_point: int,
    tensor_arena_size: int,
) -> None:
    header_path.parent.mkdir(parents=True, exist_ok=True)
    byte_values = list(model_bytes)
    lines = []
    for start in range(0, len(byte_values), 12):
        chunk = byte_values[start : start + 12]
        lines.append("  " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")

    class_lines = ", ".join(f'"{name}"' for name in class_names)
    content = f"""#ifndef TINYML_HAR_MODEL_DATA_H_
#define TINYML_HAR_MODEL_DATA_H_

#include <cstdint>

#if defined(ARDUINO_ARCH_AVR)
#include <avr/pgmspace.h>
#define TINYML_PROGMEM PROGMEM
#else
#define TINYML_PROGMEM
#endif

constexpr int kWindowSize = 128;
constexpr int kWindowStride = 64;
constexpr int kChannelCount = 6;
constexpr int kClassCount = {len(class_names)};
constexpr int kTensorArenaSize = {tensor_arena_size};
constexpr float kGyroDegToRad = 0.01745329252f;
constexpr float kInputScale = {input_scale:.9g}f;
constexpr int kInputZeroPoint = {input_zero_point};

constexpr float kFeatureMean[kChannelCount] = {{{format_float_array(standardizer.mean)}}};
constexpr float kFeatureStd[kChannelCount] = {{{format_float_array(standardizer.std)}}};
const char* const kClassNames[kClassCount] = {{{class_lines}}};

alignas(8) const unsigned char g_tinyml_har_model[] TINYML_PROGMEM = {{
{chr(10).join(lines)}
}};
const unsigned int g_tinyml_har_model_len = {len(model_bytes)};

#endif  // TINYML_HAR_MODEL_DATA_H_
"""
    header_path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_comparison(
    *,
    baseline_metrics_path: Path,
    int8_outputs: dict,
    fp32_tflite_path: Path,
    int8_tflite_path: Path,
    output_csv: Path,
) -> dict:
    baseline = load_json(baseline_metrics_path)
    rows = [
        {
            "metric": "offline_accuracy",
            "m2_lightweight_keras": baseline["accuracy"],
            "m3_int8_tflite": int8_outputs["accuracy"],
            "change": int8_outputs["accuracy"] - baseline["accuracy"],
        },
        {
            "metric": "offline_macro_f1",
            "m2_lightweight_keras": baseline["macro_f1"],
            "m3_int8_tflite": int8_outputs["macro_f1"],
            "change": int8_outputs["macro_f1"] - baseline["macro_f1"],
        },
        {
            "metric": "model_size_bytes",
            "m2_lightweight_keras": fp32_tflite_path.stat().st_size,
            "m3_int8_tflite": int8_tflite_path.stat().st_size,
            "change": int8_tflite_path.stat().st_size - fp32_tflite_path.stat().st_size,
        },
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return {"baseline_metrics": baseline, "comparison_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the M2 lightweight model for M3 TFLite Micro deployment.")
    parser.add_argument("--keras-model", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_tiny_cnn.keras")
    parser.add_argument("--standardizer", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_standardizer.json")
    parser.add_argument("--baseline-metrics", type=Path, default=OUTPUTS_DIR / "lightweight/metrics/lightweight_tiny_cnn_metrics.json")
    parser.add_argument("--fp32-tflite", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_tiny_cnn.tflite")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "deployment")
    parser.add_argument("--header", type=Path, default=Path("arduino/tinyml_har_m3/model_data.h"))
    parser.add_argument("--representative-samples", type=int, default=512)
    parser.add_argument("--tensor-arena-size", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    standardizer = SequenceStandardizer.load(args.standardizer)
    x_train = standardizer.transform(data.train.x)
    x_test = standardizer.transform(data.test.x)

    model = keras.models.load_model(args.keras_model, compile=False)
    model_bytes = convert_int8(model, x_train, args.representative_samples, args.seed)

    models_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    int8_tflite_path = models_dir / "lightweight_tiny_cnn_int8.tflite"
    int8_tflite_path.write_bytes(model_bytes)

    probs, tflite_metadata = run_tflite(int8_tflite_path, x_test)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(data.test.y, y_pred, data.class_names, "TinyDepthwiseSeparableCnnInt8Tflite")
    outputs["model_info"] = {
        "baseline_role": "M3 INT8 optimized TinyML deployment candidate",
        "source_keras_model": str(args.keras_model),
        "source_standardizer": str(args.standardizer),
        "tflite_model_path": str(int8_tflite_path),
        "tflite_size_bytes": int8_tflite_path.stat().st_size,
        "quantization": "full integer post-training quantization",
        "representative_samples": args.representative_samples,
        "input_shape": list(x_test.shape[1:]),
        "output_classes": data.class_names,
        "tflite_metadata": tflite_metadata,
        "host_tflite_latency": benchmark_tflite_ms(int8_tflite_path, x_test[0], runs=50, warmup=5),
        "hardware_metrics_status": "pending Arduino measurement",
        "tensor_arena_size_bytes_planned": args.tensor_arena_size,
    }
    save_classification_outputs(outputs, metrics_dir, figures_dir, "lightweight_tiny_cnn_int8")

    comparison = write_comparison(
        baseline_metrics_path=args.baseline_metrics,
        int8_outputs=outputs,
        fp32_tflite_path=args.fp32_tflite,
        int8_tflite_path=int8_tflite_path,
        output_csv=metrics_dir / "m2_vs_m3_int8_comparison.csv",
    )
    comparison["int8_metrics"] = {
        "accuracy": outputs["accuracy"],
        "macro_f1": outputs["macro_f1"],
        "weighted_f1": outputs["weighted_f1"],
        "tflite_size_bytes": int8_tflite_path.stat().st_size,
    }
    (metrics_dir / "m2_vs_m3_int8_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    write_model_header(
        model_bytes=model_bytes,
        header_path=args.header,
        standardizer=standardizer,
        class_names=data.class_names,
        input_scale=tflite_metadata["input_scale"],
        input_zero_point=tflite_metadata["input_zero_point"],
        tensor_arena_size=args.tensor_arena_size,
    )

    print(
        json.dumps(
            {
                "int8_tflite": str(int8_tflite_path),
                "header": str(args.header),
                "accuracy": outputs["accuracy"],
                "macro_f1": outputs["macro_f1"],
                "size_bytes": int8_tflite_path.stat().st_size,
                "ops": tflite_metadata["ops"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
