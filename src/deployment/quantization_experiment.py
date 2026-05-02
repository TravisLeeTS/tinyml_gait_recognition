from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import OUTPUTS_DIR, SEED
from src.data.uci_har import load_uci_har_sequence_dataset
from src.deployment.convert_lightweight_to_tflite_micro import (
    benchmark_tflite_ms,
    run_tflite,
    write_model_header,
)
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import set_global_seed


def representative_dataset_from_indices(x_train: np.ndarray, indices: np.ndarray):
    def _generator():
        for idx in indices:
            yield [x_train[int(idx) : int(idx) + 1].astype(np.float32)]

    return _generator


def random_representative_indices(y_train: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(len(y_train), size=min(count, len(y_train)), replace=False)


def class_balanced_representative_indices(y_train: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    per_class = max(1, int(np.ceil(count / len(classes))))
    selected: list[np.ndarray] = []
    for cls in classes:
        candidates = np.flatnonzero(y_train == cls)
        take = min(per_class, len(candidates))
        selected.append(rng.choice(candidates, size=take, replace=False))
    indices = np.concatenate(selected)
    if len(indices) > count:
        indices = rng.choice(indices, size=count, replace=False)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def convert_int8_with_indices(model: keras.Model, x_train: np.ndarray, indices: np.ndarray) -> bytes:
    input_shape = [1, *x_train.shape[1:]]

    @tf.function
    def serve(inputs):
        return model(inputs, training=False)

    concrete_func = serve.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_from_indices(x_train, indices)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def static_confusion_total(outputs: dict) -> int:
    class_names = outputs["class_names"]
    cm = np.asarray(outputs["confusion_matrix"], dtype=int)
    sitting_idx = class_names.index("SITTING")
    standing_idx = class_names.index("STANDING")
    return int(cm[sitting_idx, standing_idx] + cm[standing_idx, sitting_idx])


def evaluate_variant(
    *,
    variant: str,
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    representative_count: int,
    seed: int,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict:
    if variant == "full_integer_int8_ptq":
        indices = random_representative_indices(y_train, representative_count, seed)
        representative_policy = "random_train_windows"
    elif variant == "class_balanced_full_integer_int8_ptq":
        indices = class_balanced_representative_indices(y_train, representative_count, seed)
        representative_policy = "class_balanced_train_windows"
    else:
        raise ValueError(f"Unknown quantization variant: {variant}")

    model_bytes = convert_int8_with_indices(model, x_train, indices)
    tflite_path = models_dir / f"lightweight_tiny_cnn_{variant}.tflite"
    tflite_path.write_bytes(model_bytes)

    probs, metadata = run_tflite(tflite_path, x_test)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y_test, y_pred, class_names, f"TinyDepthwiseSeparableCnn_{variant}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(
        np.asarray(outputs["confusion_matrix"], dtype=int),
        class_names,
    )
    outputs["model_info"] = {
        "quantization_variant": variant,
        "representative_policy": representative_policy,
        "representative_samples": int(len(indices)),
        "representative_class_counts": {
            class_names[int(cls)]: int((y_train[indices] == cls).sum()) for cls in np.unique(y_train)
        },
        "tflite_model_path": str(tflite_path),
        "tflite_size_bytes": tflite_path.stat().st_size,
        "input_shape": list(x_test.shape[1:]),
        "output_classes": class_names,
        "tflite_metadata": metadata,
        "host_tflite_latency": benchmark_tflite_ms(tflite_path, x_test[0], runs=50, warmup=5),
        "hardware_metrics_status": "pending Arduino measurement",
    }
    save_classification_outputs(outputs, metrics_dir, figures_dir, f"lightweight_tiny_cnn_{variant}")
    return outputs


def select_best_variant(rows: list[dict], qat_trigger_drop: float) -> tuple[dict, str]:
    materially_hurt = all(row["macro_f1_drop_vs_phase2"] < -qat_trigger_drop for row in rows)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["macro_f1"] < row["phase2_macro_f1"] - qat_trigger_drop,
            -row["macro_f1"],
            row["host_latency_mean_ms"],
            row["tflite_size_bytes"],
        ),
    )
    if materially_hurt:
        qat_status = (
            "INT8 QAT should be run next because all PTQ variants dropped macro F1 "
            f"by more than {qat_trigger_drop:.4f}."
        )
    else:
        qat_status = (
            "INT8 QAT not run in this pass because at least one PTQ variant stayed within "
            f"{qat_trigger_drop:.4f} macro F1 of the Phase 2 FP32 baseline."
        )
    return ranked[0], qat_status


def write_selected_deployment_artifacts(
    *,
    selected: dict,
    outputs_by_variant: dict[str, dict],
    model_bytes_path: Path,
    standardizer: SequenceStandardizer,
    class_names: list[str],
    output_dir: Path,
    header_path: Path,
    tensor_arena_size: int,
) -> None:
    deployment_model_path = output_dir / "models" / "lightweight_tiny_cnn_int8.tflite"
    deployment_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_bytes_path, deployment_model_path)

    selected_outputs = outputs_by_variant[selected["quantization_variant"]]
    metadata = selected_outputs["model_info"]["tflite_metadata"]
    write_model_header(
        model_bytes=deployment_model_path.read_bytes(),
        header_path=header_path,
        standardizer=standardizer,
        class_names=class_names,
        input_scale=metadata["input_scale"],
        input_zero_point=metadata["input_zero_point"],
        tensor_arena_size=tensor_arena_size,
    )


def write_selected_deployment_metrics(
    *,
    selected: dict,
    selected_outputs: dict,
    phase2: dict,
    fp32_tflite_path: Path,
    deployment_output_dir: Path,
) -> None:
    metrics_dir = deployment_output_dir / "metrics"
    figures_dir = deployment_output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected_outputs["model_info"] = selected_outputs["model_info"] | {
        "selected_quantization_variant": selected["quantization_variant"],
        "phase2_fp32_tflite_size_bytes": fp32_tflite_path.stat().st_size,
        "accuracy_drop_vs_phase2": selected["accuracy_drop_vs_phase2"],
        "macro_f1_drop_vs_phase2": selected["macro_f1_drop_vs_phase2"],
    }
    save_classification_outputs(selected_outputs, metrics_dir, figures_dir, "lightweight_tiny_cnn_int8")

    rows = [
        {
            "metric": "offline_accuracy",
            "phase2_lightweight_keras": phase2["accuracy"],
            "phase3_selected_int8_tflite": selected_outputs["accuracy"],
            "change": selected_outputs["accuracy"] - phase2["accuracy"],
        },
        {
            "metric": "offline_macro_f1",
            "phase2_lightweight_keras": phase2["macro_f1"],
            "phase3_selected_int8_tflite": selected_outputs["macro_f1"],
            "change": selected_outputs["macro_f1"] - phase2["macro_f1"],
        },
        {
            "metric": "model_size_bytes",
            "phase2_lightweight_keras": fp32_tflite_path.stat().st_size,
            "phase3_selected_int8_tflite": selected_outputs["model_info"]["tflite_size_bytes"],
            "change": selected_outputs["model_info"]["tflite_size_bytes"] - fp32_tflite_path.stat().st_size,
        },
    ]
    pd.DataFrame(rows).to_csv(metrics_dir / "m2_vs_m3_int8_comparison.csv", index=False)
    (metrics_dir / "m2_vs_m3_int8_comparison.json").write_text(
        json.dumps(
            {
                "baseline_metrics": phase2,
                "comparison_rows": rows,
                "selected_quantization_variant": selected["quantization_variant"],
                "int8_metrics": {
                    "accuracy": selected_outputs["accuracy"],
                    "macro_f1": selected_outputs["macro_f1"],
                    "weighted_f1": selected_outputs["weighted_f1"],
                    "tflite_size_bytes": selected_outputs["model_info"]["tflite_size_bytes"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare quantization strategies for the selected Phase 2 model.")
    parser.add_argument("--keras-model", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_tiny_cnn.keras")
    parser.add_argument("--standardizer", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_standardizer.json")
    parser.add_argument("--phase2-metrics", type=Path, default=OUTPUTS_DIR / "lightweight/metrics/lightweight_tiny_cnn_metrics.json")
    parser.add_argument("--fp32-tflite", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_tiny_cnn.tflite")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "deployment" / "quantization_experiments")
    parser.add_argument("--deployment-output-dir", type=Path, default=OUTPUTS_DIR / "deployment")
    parser.add_argument("--header", type=Path, default=Path("arduino/tinyml_har_m3/model_data.h"))
    parser.add_argument("--representative-samples", type=int, default=512)
    parser.add_argument("--qat-trigger-drop", type=float, default=0.01)
    parser.add_argument("--tensor-arena-size", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)
    phase2 = load_json(args.phase2_metrics)
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    standardizer = SequenceStandardizer.load(args.standardizer)
    x_train = standardizer.transform(data.train.x)
    x_test = standardizer.transform(data.test.x)
    model = keras.models.load_model(args.keras_model, compile=False)

    models_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    for path in [models_dir, metrics_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    variants = ["full_integer_int8_ptq", "class_balanced_full_integer_int8_ptq"]
    outputs_by_variant = {
        variant: evaluate_variant(
            variant=variant,
            model=model,
            x_train=x_train,
            y_train=data.train.y,
            x_test=x_test,
            y_test=data.test.y,
            class_names=data.class_names,
            representative_count=args.representative_samples,
            seed=args.seed,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        )
        for variant in variants
    }

    rows = []
    for variant, outputs in outputs_by_variant.items():
        latency = outputs["model_info"]["host_tflite_latency"]
        row = {
            "quantization_variant": variant,
            "phase2_model": str(args.keras_model),
            "phase2_accuracy": phase2["accuracy"],
            "phase2_macro_f1": phase2["macro_f1"],
            "phase2_fp32_tflite_size_bytes": args.fp32_tflite.stat().st_size,
            "accuracy": outputs["accuracy"],
            "macro_f1": outputs["macro_f1"],
            "weighted_f1": outputs["weighted_f1"],
            "accuracy_drop_vs_phase2": outputs["accuracy"] - phase2["accuracy"],
            "macro_f1_drop_vs_phase2": outputs["macro_f1"] - phase2["macro_f1"],
            "tflite_size_bytes": outputs["model_info"]["tflite_size_bytes"],
            "size_change_vs_phase2_fp32_tflite": outputs["model_info"]["tflite_size_bytes"]
            - args.fp32_tflite.stat().st_size,
            "host_latency_mean_ms": latency["mean_ms"],
            "host_latency_median_ms": latency["median_ms"],
            "host_latency_p95_ms": latency["p95_ms"],
            "static_sitting_standing_confusions": static_confusion_total(outputs),
            "representative_policy": outputs["model_info"]["representative_policy"],
            "representative_samples": outputs["model_info"]["representative_samples"],
            "model_path": outputs["model_info"]["tflite_model_path"],
        }
        rows.append(row)

    selected, qat_status = select_best_variant(rows, args.qat_trigger_drop)
    for row in rows:
        row["selected_for_deployment"] = row["quantization_variant"] == selected["quantization_variant"]
        row["qat_status"] = qat_status

    summary = {
        "phase2_selection": {
            "selected_model": "TinyDepthwiseSeparableCnn",
            "selection_source": str(args.phase2_metrics),
            "accuracy": phase2["accuracy"],
            "macro_f1": phase2["macro_f1"],
            "keras_parameters": phase2["model_info"]["keras_model_parameters"],
            "fp32_tflite_size_bytes": args.fp32_tflite.stat().st_size,
            "host_latency_note": phase2["model_info"]["host_latency"]["device_note"],
        },
        "qat_trigger_macro_f1_drop": args.qat_trigger_drop,
        "qat_status": qat_status,
        "selected_quantization_variant": selected,
        "quantization_rows": rows,
        "data_quality_note": (
            "Arduino V2 self-collected data is used as a small mixed-pocket adaptation "
            "training supplement, not as live validation. Baseline laptop replay on V2 "
            "still collapses to LAYING, so Table 4 live accuracy must come from separate "
            "right-pocket and left-pocket Arduino Serial trials."
        ),
    }

    pd.DataFrame(rows).to_csv(metrics_dir / "quantization_experiment_summary.csv", index=False)
    (metrics_dir / "quantization_experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    selected_model_path = Path(selected["model_path"])
    write_selected_deployment_artifacts(
        selected=selected,
        outputs_by_variant=outputs_by_variant,
        model_bytes_path=selected_model_path,
        standardizer=standardizer,
        class_names=data.class_names,
        output_dir=args.deployment_output_dir,
        header_path=args.header,
        tensor_arena_size=args.tensor_arena_size,
    )
    write_selected_deployment_metrics(
        selected=selected,
        selected_outputs=outputs_by_variant[selected["quantization_variant"]],
        phase2=phase2,
        fp32_tflite_path=args.fp32_tflite,
        deployment_output_dir=args.deployment_output_dir,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
