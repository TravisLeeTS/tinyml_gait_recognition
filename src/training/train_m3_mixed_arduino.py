from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_DEVICE, OUTPUTS_DIR, RAW_DATA_DIR, SEED
from src.data.arduino_collectdata import UCI_CLASS_NAMES, build_windowed_dataset
from src.data.target_domain_features import GRAVITY_FEATURE_CHANNELS, TEN_CHANNEL_FEATURES, append_gravity_features
from src.data.uci_har import load_uci_har_sequence_dataset
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn, count_trainable_parameters
from src.training.tf_common import benchmark_latency_ms, predict_proba, save_tflite_model
from src.training.train_m3_second_improvements import (
    SparseCategoricalFocalLoss,
    _static_confusion_summary,
    _train_model,
)
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


BASELINE_METRICS_PATH = OUTPUTS_DIR / "lightweight" / "metrics" / "lightweight_tiny_cnn_metrics.json"
EXPERIMENT_ROOT = OUTPUTS_DIR / "lightweight" / "experiments" / "m3_arduino_v2_mixed_training"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _repeat_arduino(x: np.ndarray, y: np.ndarray, repeat: int) -> tuple[np.ndarray, np.ndarray]:
    if repeat <= 1:
        return x, y
    return np.concatenate([x] * repeat, axis=0), np.concatenate([y] * repeat, axis=0)


def _run_name(experiment: str) -> str:
    return {
        "baseline": "m3_mixed_v2_ds_cnn",
        "focal_loss": "m3_mixed_v2_focal_loss_tiny_cnn",
        "gravity_feature": "m3_mixed_v2_gravity_feature_tiny_cnn",
    }[experiment]


def _loss_for(experiment: str):
    if experiment == "focal_loss":
        return SparseCategoricalFocalLoss(gamma=2.0, name="focal_loss")
    return "sparse_categorical_crossentropy"


def _comparison_row(run_name: str, outputs: dict, baseline: dict | None) -> dict:
    row = {
        "run_name": run_name,
        "accuracy": outputs["accuracy"],
        "macro_f1": outputs["macro_f1"],
        "weighted_f1": outputs["weighted_f1"],
    }
    row.update(_static_confusion_summary(outputs))
    if baseline:
        baseline_static = _static_confusion_summary(baseline)
        row.update(
            {
                "phase2_baseline_accuracy": baseline["accuracy"],
                "phase2_baseline_macro_f1": baseline["macro_f1"],
                "phase2_baseline_weighted_f1": baseline["weighted_f1"],
                "phase2_baseline_static_confusion_total": baseline_static["static_confusion_total"],
                "accuracy_change_vs_phase2": outputs["accuracy"] - baseline["accuracy"],
                "macro_f1_change_vs_phase2": outputs["macro_f1"] - baseline["macro_f1"],
                "weighted_f1_change_vs_phase2": outputs["weighted_f1"] - baseline["weighted_f1"],
                "static_confusion_change_vs_phase2": row["static_confusion_total"]
                - baseline_static["static_confusion_total"],
            }
        )
    return row


def run_experiment(args: argparse.Namespace, experiment: str, baseline: dict | None) -> dict:
    set_global_seed(args.seed)
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    arduino_x_raw, arduino_y, arduino_sources = build_windowed_dataset(args.arduino_root, hz=50.0)
    if UCI_CLASS_NAMES != data.class_names:
        raise ValueError("Arduino class order does not match UCI class order.")

    arduino_x_raw, arduino_y = _repeat_arduino(arduino_x_raw, arduino_y, args.arduino_repeat)
    x_train_raw = np.concatenate([data.train.x, arduino_x_raw], axis=0).astype(np.float32)
    y_train = np.concatenate([data.train.y, arduino_y], axis=0).astype(np.int64)

    input_note = "128 x 6"
    feature_spec = None
    if experiment == "gravity_feature":
        x_train_raw = append_gravity_features(x_train_raw)
        val_raw = append_gravity_features(data.val.x)
        test_raw = append_gravity_features(data.test.x)
        input_note = "128 x 10"
        feature_spec = {
            "channels": TEN_CHANNEL_FEATURES,
            "added_channels": GRAVITY_FEATURE_CHANNELS,
            "source": "Computed from each raw window before standardization, then all 10 channels are standardized with training/adaptation statistics.",
            "deployment_note": "Requires Arduino-side feature generation and 10-channel normalization before this candidate can be deployed.",
        }
    else:
        val_raw = data.val.x
        test_raw = data.test.x

    scaler = SequenceStandardizer.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(val_raw)
    x_test = scaler.transform(test_raw)

    run_name = _run_name(experiment)
    run_dir = args.output_dir / run_name
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"
    for path in [model_dir, metrics_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / f"{run_name}.weights.h5"
    standardizer_path = model_dir / f"{run_name}_standardizer.json"
    scaler.save(standardizer_path)

    model = build_tiny_ds_cnn(input_shape=tuple(x_train.shape[1:]), num_classes=len(data.class_names))
    history, best_epoch, best_val_loss = _train_model(
        model,
        _loss_for(experiment),
        x_train,
        y_train,
        x_val,
        data.val.y,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        patience=args.patience,
        live_log_path=logs_dir / "training_history_live.csv",
    )
    history.to_csv(logs_dir / "training_history.csv", index=False)
    model.save_weights(weights_path)

    probs = predict_proba(model, x_test)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(data.test.y, y_pred, data.class_names, run_name)
    outputs["top_confusion_pairs"] = top_confusion_pairs(
        np.asarray(outputs["confusion_matrix"], dtype=int),
        data.class_names,
    )
    outputs["model_info"] = {
        "run_name": run_name,
        "experiment": experiment,
        "input_note": input_note,
        "training_role": "M3 deployment-adaptation experiment; not the paper reproduction baseline.",
        "train_sources": {
            "uci_har_train_windows": int(len(data.train.y)),
            "arduino_v2_unique_windows": int(len(arduino_sources)),
            "arduino_repeat": int(args.arduino_repeat),
            "arduino_v2_effective_training_windows": int(len(arduino_y)),
            "total_training_windows": int(len(y_train)),
            "arduino_root": str(args.arduino_root),
        },
        "validation_source": "UCI-HAR subject-aware validation split only",
        "test_source": "UCI-HAR official test split only",
        "normalization": "SequenceStandardizer fit on mixed UCI train + Arduino V2 training/adaptation windows only; UCI test and future live evidence are excluded",
        "standardizer_path": str(standardizer_path),
        "loss": "sparse focal loss" if experiment == "focal_loss" else "sparse categorical cross entropy",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.lr_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "early_stopping": {"monitor": "val_loss", "patience": args.patience},
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "trainable_parameters": count_trainable_parameters(model),
        "keras_model_parameters": int(model.count_params()),
        "weights_path": str(weights_path),
    }
    if feature_spec:
        outputs["feature_spec"] = feature_spec
        (run_dir / "feature_spec.json").write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")
    if args.export_tflite:
        outputs["model_info"]["tflite_size_bytes"] = save_tflite_model(model, model_dir / f"{run_name}.tflite")
        outputs["model_info"]["dynamic_range_tflite_size_bytes"] = save_tflite_model(
            model,
            model_dir / f"{run_name}_dynamic_range.tflite",
            quantize_dynamic_range=True,
        )
    if args.benchmark_host:
        outputs["model_info"]["host_latency"] = benchmark_latency_ms(model, x_test, runs=20, warmup=3)

    outputs["phase2_baseline_comparison"] = _comparison_row(run_name, outputs, baseline)
    save_classification_outputs(outputs, metrics_dir, figures_dir, run_name)
    pd.DataFrame([outputs["phase2_baseline_comparison"] | outputs["model_info"]]).to_csv(
        run_dir / "summary.csv",
        index=False,
    )
    print(json.dumps(outputs["phase2_baseline_comparison"], indent=2))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M3 DS-CNN candidates with UCI train + Arduino V2 windows.")
    parser.add_argument(
        "--experiment",
        choices=["baseline", "focal_loss", "gravity_feature", "all"],
        default="all",
    )
    parser.add_argument("--arduino-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2")
    parser.add_argument("--arduino-repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--benchmark-host", action="store_true")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    if args.arduino_repeat < 1:
        raise ValueError("--arduino-repeat must be >= 1")
    configure_tensorflow_device(args.device)
    baseline = _load_json(BASELINE_METRICS_PATH)
    selected = ["baseline", "focal_loss", "gravity_feature"] if args.experiment == "all" else [args.experiment]
    rows = []
    for experiment in selected:
        outputs = run_experiment(args, experiment, baseline)
        rows.append(outputs["phase2_baseline_comparison"] | outputs["model_info"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "m3_mixed_v2_training_summary.csv", index=False)


if __name__ == "__main__":
    main()
