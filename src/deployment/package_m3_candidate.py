from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import OUTPUTS_DIR, RAW_DATA_DIR, SEED
from src.data.arduino_collectdata import UCI_CLASS_NAMES, build_windowed_dataset
from src.data.uci_har import load_uci_har_sequence_dataset
from src.deployment.convert_lightweight_to_tflite_micro import benchmark_tflite_ms, run_tflite
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn
from src.training.tf_common import predict_proba
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import set_global_seed


DEFAULT_RUN_DIR = OUTPUTS_DIR / "lightweight" / "experiments" / "m3_target_domain" / "m3_target_mixed_focal_6ch"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "deployment" / "m3_candidate_mixed_focal_6ch"
DEFAULT_HEADER = Path("arduino/tinyml_har_m3/model_data.h")


def _source_group(source: str) -> str:
    return str(source).split("#segment", maxsplit=1)[0]


def _representative_dataset(x: np.ndarray, indices: np.ndarray):
    def _generator():
        for idx in indices:
            yield [x[int(idx) : int(idx) + 1].astype(np.float32)]

    return _generator


def _class_balanced_indices(y: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    classes = np.unique(y)
    per_class = max(1, int(np.ceil(count / len(classes))))
    for cls in classes:
        candidates = np.flatnonzero(y == cls)
        take = min(per_class, len(candidates))
        selected.append(rng.choice(candidates, size=take, replace=False))
    indices = np.concatenate(selected)
    if len(indices) > count:
        indices = rng.choice(indices, size=count, replace=False)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def _convert_int8(model: tf.keras.Model, representative_x: np.ndarray, indices: np.ndarray) -> bytes:
    input_shape = [1, *representative_x.shape[1:]]

    @tf.function
    def serve(inputs):
        return model(inputs, training=False)

    concrete_func = serve.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(representative_x, indices)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _format_float_array(values: list[float]) -> str:
    return ", ".join(f"{value:.9g}f" for value in values)


def _write_model_header(
    *,
    model_bytes: bytes,
    header_path: Path,
    standardizer: SequenceStandardizer,
    class_names: list[str],
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    tensor_arena_size: int,
    model_description: str = "mixed focal 6-channel DS-CNN, selected by Arduino V2 validation macro F1",
    normalization_source: str = "mixed UCI-HAR train + Arduino V2 train/adaptation split",
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

// M3 candidate deployment model for live evidence collection.
// Model: {model_description}.
// This is not claimed as a fully scientifically validated final model.
// Check the M3 report for remaining held-out validation failure modes.

#include <cstdint>

#if defined(ARDUINO_ARCH_AVR)
#include <avr/pgmspace.h>
#define TINYML_PROGMEM PROGMEM
#else
#define TINYML_PROGMEM
#endif

constexpr int kWindowSize = 128;
constexpr int kWindowStride = 64;
constexpr int kStride = kWindowStride;
constexpr int kChannelCount = 6;
constexpr int kClassCount = {len(class_names)};
constexpr int kTensorArenaSize = {tensor_arena_size};
constexpr float kGyroDegToRad = 0.01745329252f;
constexpr float kInputScale = {input_scale:.9g}f;
constexpr int kInputZeroPoint = {input_zero_point};
constexpr float kOutputScale = {output_scale:.9g}f;
constexpr int kOutputZeroPoint = {output_zero_point};
const char* const kNormalizationSource = "{normalization_source}";

constexpr float kFeatureMean[kChannelCount] = {{{_format_float_array(standardizer.mean)}}};
constexpr float kFeatureStd[kChannelCount] = {{{_format_float_array(standardizer.std)}}};
const char* const kClassNames[kClassCount] = {{{class_lines}}};

alignas(8) const unsigned char g_tinyml_har_model[] TINYML_PROGMEM = {{
{chr(10).join(lines)}
}};
const unsigned int g_tinyml_har_model_len = {len(model_bytes)};

#endif  // TINYML_HAR_MODEL_DATA_H_
"""
    header_path.write_text(content, encoding="utf-8")


def _evaluate_fp32(model, x, y, class_names, run_name, split_name, out_dir):
    probs = predict_proba(model, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, class_names, f"{run_name}_{split_name}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), class_names)
    save_classification_outputs(outputs, out_dir / "metrics", out_dir / "figures", f"{run_name}_{split_name}")
    return outputs


def _evaluate_tflite(model_path: Path, x, y, class_names, run_name, split_name, out_dir):
    probs, metadata = run_tflite(model_path, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, class_names, f"{run_name}_{split_name}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), class_names)
    outputs["model_info"] = {
        "tflite_path": str(model_path),
        "tflite_size_bytes": model_path.stat().st_size,
        "tflite_metadata": metadata,
        "host_tflite_latency": benchmark_tflite_ms(model_path, x[0], runs=50, warmup=5),
    }
    save_classification_outputs(outputs, out_dir / "metrics", out_dir / "figures", f"{run_name}_{split_name}")
    return outputs, metadata


def _metric_row(model_type: str, split: str, outputs: dict, size_bytes: int | None) -> dict:
    row = {
        "model_type": model_type,
        "split": split,
        "accuracy": outputs["accuracy"],
        "macro_f1": outputs["macro_f1"],
        "weighted_f1": outputs["weighted_f1"],
        "model_size_bytes": size_bytes,
    }
    for class_name in outputs["class_names"]:
        row[f"f1_{class_name.lower()}"] = outputs["classification_report"][class_name]["f1-score"]
    return row


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    for path in [models_dir, metrics_dir, output_dir / "figures"]:
        path.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    standardizer = SequenceStandardizer.load(Path(summary["normalization"]["standardizer_path"]))
    model = build_tiny_ds_cnn(input_shape=tuple(summary["input_shape"]), num_classes=len(UCI_CLASS_NAMES))
    model.load_weights(run_dir / "models" / f"{summary['run_name']}.weights.h5")

    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    arduino_x, arduino_y, arduino_sources = build_windowed_dataset(Path(args.arduino_root), hz=50.0)
    val_groups = set(summary["data_sources"]["arduino_v2_val_groups"])
    source_groups = np.asarray([_source_group(source) for source in arduino_sources])
    val_mask = np.asarray([group in val_groups for group in source_groups])
    train_mask = ~val_mask

    x_uci_train = standardizer.transform(data.train.x)
    x_uci_test = standardizer.transform(data.test.x)
    x_arduino_train = standardizer.transform(arduino_x[train_mask])
    y_arduino_train = arduino_y[train_mask]
    x_arduino_val = standardizer.transform(arduino_x[val_mask])
    y_arduino_val = arduino_y[val_mask]

    representative_x = np.concatenate([x_uci_train, x_arduino_train], axis=0)
    representative_y = np.concatenate([data.train.y, y_arduino_train], axis=0)
    indices = _class_balanced_indices(representative_y, args.representative_samples, args.seed)

    fp32_tflite = run_dir / "models" / f"{summary['run_name']}.tflite"
    selected_fp32_copy = models_dir / "m3_candidate_mixed_focal_6ch_fp32.tflite"
    if fp32_tflite.exists():
        shutil.copyfile(fp32_tflite, selected_fp32_copy)

    int8_bytes = _convert_int8(model, representative_x, indices)
    int8_path = models_dir / "m3_candidate_mixed_focal_6ch_int8.tflite"
    int8_path.write_bytes(int8_bytes)

    fp32_uci = _evaluate_fp32(model, x_uci_test, data.test.y, data.class_names, "m3_candidate_fp32", "uci_test", output_dir)
    fp32_v2 = _evaluate_fp32(model, x_arduino_val, y_arduino_val, data.class_names, "m3_candidate_fp32", "arduino_v2_val", output_dir)
    int8_uci, tflite_metadata = _evaluate_tflite(int8_path, x_uci_test, data.test.y, data.class_names, "m3_candidate_int8", "uci_test", output_dir)
    int8_v2, _ = _evaluate_tflite(int8_path, x_arduino_val, y_arduino_val, data.class_names, "m3_candidate_int8", "arduino_v2_val", output_dir)

    rows = [
        _metric_row("FP32", "UCI-HAR official test", fp32_uci, selected_fp32_copy.stat().st_size if selected_fp32_copy.exists() else None),
        _metric_row("FP32", "Arduino V2 validation", fp32_v2, selected_fp32_copy.stat().st_size if selected_fp32_copy.exists() else None),
        _metric_row("INT8", "UCI-HAR official test", int8_uci, int8_path.stat().st_size),
        _metric_row("INT8", "Arduino V2 validation", int8_v2, int8_path.stat().st_size),
    ]
    pd.DataFrame(rows).to_csv(metrics_dir / "m3_candidate_fp32_vs_int8_summary.csv", index=False)

    selected_metadata = {
        "candidate_name": "m3_candidate_mixed_focal_6ch",
        "candidate_status": "M3 candidate deployment build for live evidence collection; not fully scientifically validated.",
        "source_run_dir": str(run_dir),
        "source_weights": str(run_dir / "models" / f"{summary['run_name']}.weights.h5"),
        "source_fp32_tflite": str(selected_fp32_copy) if selected_fp32_copy.exists() else str(fp32_tflite),
        "selected_int8_tflite": str(int8_path),
        "normalization_source": "mixed UCI-HAR train + Arduino V2 train/adaptation split",
        "standardizer_path": str(summary["normalization"]["standardizer_path"]),
        "standardizer": {"mean": standardizer.mean, "std": standardizer.std},
        "class_order": {str(i): name for i, name in enumerate(data.class_names)},
        "input_shape": [128, 6],
        "window_size": 128,
        "stride": 64,
        "representative_dataset": {
            "policy": "class-balanced samples from UCI-HAR train plus Arduino V2 training/adaptation split only",
            "samples": int(len(indices)),
            "excluded": ["UCI-HAR official test", "Arduino V2 validation split", "future live evidence folders"],
            "class_counts": {
                data.class_names[int(cls)]: int((representative_y[indices] == cls).sum()) for cls in np.unique(representative_y[indices])
            },
        },
        "known_failures": ["WALKING_DOWNSTAIRS predicted as WALKING_UPSTAIRS", "SITTING predicted as LAYING"],
        "tflite_metadata": tflite_metadata,
        "metrics_summary_csv": str(metrics_dir / "m3_candidate_fp32_vs_int8_summary.csv"),
    }
    (output_dir / "m3_candidate_metadata.json").write_text(json.dumps(selected_metadata, indent=2), encoding="utf-8")
    (models_dir / "m3_candidate_standardizer.json").write_text(
        json.dumps({"mean": standardizer.mean, "std": standardizer.std}, indent=2),
        encoding="utf-8",
    )

    _write_model_header(
        model_bytes=int8_bytes,
        header_path=Path(args.header),
        standardizer=standardizer,
        class_names=data.class_names,
        input_scale=tflite_metadata["input_scale"],
        input_zero_point=tflite_metadata["input_zero_point"],
        output_scale=tflite_metadata["output_scale"],
        output_zero_point=tflite_metadata["output_zero_point"],
        tensor_arena_size=args.tensor_arena_size,
    )

    print(json.dumps(selected_metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the mixed focal 6-channel M3 candidate for Arduino evidence collection.")
    parser.add_argument("--run-dir", "--source", dest="run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arduino-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2")
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--representative-samples", type=int, default=512)
    parser.add_argument("--tensor-arena-size", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
