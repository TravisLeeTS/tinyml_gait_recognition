from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.arduino_collectdata import UCI_CLASS_NAMES, build_windowed_dataset
from src.deployment.convert_lightweight_to_tflite_micro import run_tflite
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer


def write_source_table(sources: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    rows = []
    for source, true_id, pred_id in zip(sources, y_true, y_pred, strict=True):
        rows.append(
            {
                "source": str(source),
                "true": UCI_CLASS_NAMES[int(true_id)],
                "predicted": UCI_CLASS_NAMES[int(pred_id)],
                "correct": bool(int(true_id) == int(pred_id)),
            }
        )
    table = pd.DataFrame(rows)
    summary = (
        table.assign(source_file=table["source"].str.replace(r"#segment\d+$", "", regex=True))
        .groupby(["source_file", "true"], as_index=False)
        .agg(windows=("correct", "count"), correct=("correct", "sum"))
    )
    summary["accuracy"] = summary["correct"] / summary["windows"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path.with_name(output_path.stem + "_window_predictions.csv"), index=False)
    summary.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay collected Arduino CSV windows through the INT8 TFLite model offline."
    )
    parser.add_argument("--root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v1")
    parser.add_argument(
        "--standardizer",
        type=Path,
        default=OUTPUTS_DIR / "lightweight/models/lightweight_standardizer.json",
    )
    parser.add_argument(
        "--tflite-model",
        type=Path,
        default=OUTPUTS_DIR / "deployment/models/lightweight_tiny_cnn_int8.tflite",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "arduino_collectdata")
    parser.add_argument(
        "--windowed-output",
        type=Path,
        default=PROCESSED_DATA_DIR / "arduino_collectdata_v1_windows_50hz.npz",
    )
    args = parser.parse_args()

    x_raw, y_true, sources = build_windowed_dataset(args.root, hz=50.0)
    scaler = SequenceStandardizer.load(args.standardizer)
    x = scaler.transform(x_raw)

    args.windowed_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.windowed_output,
        x=x_raw,
        x_standardized=x,
        y=y_true,
        sources=sources,
        class_names=np.asarray(UCI_CLASS_NAMES),
        sampling_hz=np.asarray([50.0], dtype=np.float32),
        window_size=np.asarray([128], dtype=np.int64),
        stride=np.asarray([64], dtype=np.int64),
        note=np.asarray(
            [
                "Offline replay dataset generated from real Arduino CSV files; "
                "not a live on-device accuracy measurement."
            ]
        ),
    )

    probs, metadata = run_tflite(args.tflite_model, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y_true, y_pred, UCI_CLASS_NAMES, "ArduinoCsvReplayInt8Tflite")
    outputs["model_info"] = {
        "evaluation_role": "offline replay of real Arduino-collected CSV test data",
        "important_limitation": (
            "This uses real Arduino sensor data, but inference runs on the laptop TFLite interpreter. "
            "It does not replace the M3 live on-device 20-trials/class confusion matrix."
        ),
        "source_dataset_root": str(args.root),
        "windowed_output": str(args.windowed_output),
        "sampling_note": "CSV timestamps were resampled to 50 Hz before 128-sample/64-stride windowing.",
        "normalization": "UCI-HAR training standardizer from lightweight_standardizer.json",
        "tflite_model": str(args.tflite_model),
        "tflite_metadata": metadata,
    }
    outputs["top_confusion_pairs"] = top_confusion_pairs(
        np.asarray(outputs["confusion_matrix"]),
        UCI_CLASS_NAMES,
        top_k=10,
    )

    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    save_classification_outputs(outputs, metrics_dir, figures_dir, "arduino_replay_int8")
    write_source_table(sources, y_true, y_pred, metrics_dir / "arduino_replay_source_accuracy.csv")

    print(
        json.dumps(
            {
                "windows": int(len(y_true)),
                "accuracy": outputs["accuracy"],
                "macro_f1": outputs["macro_f1"],
                "weighted_f1": outputs["weighted_f1"],
                "metrics": str(metrics_dir / "arduino_replay_int8_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
