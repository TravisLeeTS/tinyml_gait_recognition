from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import OUTPUTS_DIR, RAW_DATA_DIR, SEED
from src.data.arduino_collectdata import UCI_CLASS_NAMES, build_windowed_dataset
from src.data.uci_har import load_uci_har_sequence_dataset
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn, count_trainable_parameters
from src.training.tf_common import predict_proba, save_tflite_model
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "m4_strategy_a_experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded M4 experiment that keeps the new standardized dataset fully held out "
            "while optionally moving old M3 raw replay validation files into training."
        )
    )
    parser.add_argument("--standardized-input", required=True, type=Path, help="Standardized HAR zip or extracted folder.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--v2-root", default=RAW_DATA_DIR / "arduino_collectdata_v2", type=Path)
    parser.add_argument("--v2-1-root", default=RAW_DATA_DIR / "arduino_collectdata_v2_1_long_adaptation", type=Path)
    parser.add_argument("--right-root", default=RAW_DATA_DIR / "arduino_live_validation" / "right_60s", type=Path)
    parser.add_argument("--left-root", default=RAW_DATA_DIR / "arduino_live_validation" / "left_30s", type=Path)
    parser.add_argument(
        "--include-old-m3-raw-holdouts-in-train",
        action="store_true",
        help="Move old right_60s and left_30s raw replay validation windows into the new training set.",
    )
    parser.add_argument(
        "--include-uci-test-in-train",
        action="store_true",
        help="Deployment-only experiment option. If set, UCI test is no longer a reportable held-out test metric.",
    )
    parser.add_argument("--run-augmentation", action="store_true", help="Also train a training-only augmentation variant.")
    parser.add_argument("--augmentation-copies", default=1, type=int)
    parser.add_argument("--noise-std", default=0.02, type=float)
    parser.add_argument("--scale-min", default=0.95, type=float)
    parser.add_argument("--scale-max", default=1.05, type=float)
    parser.add_argument("--max-shift", default=4, type=int)
    parser.add_argument("--export-tflite", action="store_true")
    return parser.parse_args()


def extract_if_needed(input_path: Path, temp_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected .zip or folder for standardized dataset, got {input_path}")
    with zipfile.ZipFile(input_path) as archive:
        archive.extractall(temp_dir)
    children = [child for child in temp_dir.iterdir() if child.is_dir()]
    return children[0] if len(children) == 1 else temp_dir


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(UCI_CLASS_NAMES)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights[y].astype(np.float32)


def augment_training_windows(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    copies_x = [x.astype(np.float32)]
    copies_y = [y.astype(np.int64)]
    for _ in range(args.augmentation_copies):
        aug = x.astype(np.float32).copy()
        scale = rng.uniform(args.scale_min, args.scale_max, size=(len(x), 1, x.shape[2])).astype(np.float32)
        aug *= scale
        aug += rng.normal(0.0, args.noise_std, size=aug.shape).astype(np.float32)
        if args.max_shift > 0:
            shifts = rng.integers(-args.max_shift, args.max_shift + 1, size=len(x))
            for idx, shift in enumerate(shifts):
                aug[idx] = np.roll(aug[idx], shift=shift, axis=0)
        copies_x.append(aug)
        copies_y.append(y.astype(np.int64))
    return np.concatenate(copies_x, axis=0), np.concatenate(copies_y, axis=0)


def evaluate(model: keras.Model, x: np.ndarray, y: np.ndarray, run_name: str, split_name: str, output_dir: Path) -> dict:
    probs = predict_proba(model, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, UCI_CLASS_NAMES, f"{run_name}_{split_name}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), UCI_CLASS_NAMES)
    save_classification_outputs(outputs, output_dir / "metrics", output_dir / "figures", f"{run_name}_{split_name}")
    return outputs


def train_one(
    run_name: str,
    train_raw: np.ndarray,
    train_y: np.ndarray,
    internal_val_raw: np.ndarray,
    internal_val_y: np.ndarray,
    external_raw: np.ndarray,
    external_y: np.ndarray,
    args: argparse.Namespace,
    augment: bool,
) -> dict:
    run_dir = args.output_dir / run_name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    scaler = SequenceStandardizer.fit(train_raw)
    effective_train_raw = train_raw
    effective_train_y = train_y
    augmentation_meta = {"enabled": False}
    if augment:
        effective_train_raw, effective_train_y = augment_training_windows(train_raw, train_y, args)
        augmentation_meta = {
            "enabled": True,
            "copies": args.augmentation_copies,
            "noise_std": args.noise_std,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "max_shift": args.max_shift,
            "guardrail": "Only training windows were augmented. Internal validation and Strategy A external test were unchanged.",
        }

    x_train = scaler.transform(effective_train_raw)
    x_val = scaler.transform(internal_val_raw)
    x_external = scaler.transform(external_raw)

    model = build_tiny_ds_cnn(input_shape=tuple(x_train.shape[1:]), num_classes=len(UCI_CLASS_NAMES))
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, min_delta=1e-4),
        keras.callbacks.CSVLogger(str(run_dir / "logs" / "training_history.csv"), append=False),
    ]
    history = model.fit(
        x_train,
        effective_train_y,
        validation_data=(x_val, internal_val_y),
        sample_weight=balanced_weights(effective_train_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    pd.DataFrame(history.history).to_csv(run_dir / "logs" / "training_history.csv", index=False)
    model.save_weights(run_dir / "models" / f"{run_name}.weights.h5")
    scaler.save(run_dir / "models" / f"{run_name}_standardizer.json")
    tflite_size = None
    if args.export_tflite:
        tflite_size = save_tflite_model(model, run_dir / "models" / f"{run_name}.tflite")

    internal_outputs = evaluate(model, x_val, internal_val_y, run_name, "internal_val", run_dir)
    external_outputs = evaluate(model, x_external, external_y, run_name, "standardized_external", run_dir)
    summary = {
        "run_name": run_name,
        "strategy": "A: standardized dataset fully held out for external validation",
        "train_windows_before_augmentation": int(len(train_raw)),
        "train_windows_effective": int(len(effective_train_raw)),
        "internal_validation_windows": int(len(internal_val_raw)),
        "standardized_external_windows": int(len(external_raw)),
        "augmentation": augmentation_meta,
        "parameters": int(model.count_params()),
        "trainable_parameters": count_trainable_parameters(model),
        "tflite_size_bytes": tflite_size,
        "internal_val_accuracy": internal_outputs["accuracy"],
        "internal_val_macro_f1": internal_outputs["macro_f1"],
        "standardized_external_accuracy": external_outputs["accuracy"],
        "standardized_external_macro_f1": external_outputs["macro_f1"],
        "standardized_external_top_confusions": external_outputs["top_confusion_pairs"][:5],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    configure_tensorflow_device(args.device)

    data = load_uci_har_sequence_dataset()
    train_parts = [data.train.x]
    y_parts = [data.train.y]
    internal_val_x = data.val.x
    internal_val_y = data.val.y

    if args.include_uci_test_in_train:
        train_parts.append(data.test.x)
        y_parts.append(data.test.y)

    v2_x, v2_y, _ = build_windowed_dataset(args.v2_root, hz=50.0)
    v2_1_x, v2_1_y, _ = build_windowed_dataset(args.v2_1_root, hz=50.0)
    train_parts.extend([v2_x, v2_1_x])
    y_parts.extend([v2_y, v2_1_y])

    if args.include_old_m3_raw_holdouts_in_train:
        right_x, right_y, _ = build_windowed_dataset(args.right_root, hz=50.0)
        left_x, left_y, _ = build_windowed_dataset(args.left_root, hz=50.0)
        train_parts.extend([right_x, left_x])
        y_parts.extend([right_y, left_y])

    with tempfile.TemporaryDirectory(prefix="m4_standardized_") as temp_name:
        standardized_root = extract_if_needed(args.standardized_input.expanduser().resolve(), Path(temp_name))
        standardized_x, standardized_y, standardized_sources = build_windowed_dataset(standardized_root, hz=50.0)

    train_raw = np.concatenate(train_parts, axis=0).astype(np.float32)
    train_y = np.concatenate(y_parts, axis=0).astype(np.int64)
    summaries = []
    summaries.append(
        train_one(
            "m4_strategy_a_expanded_train_no_aug",
            train_raw,
            train_y,
            internal_val_x,
            internal_val_y,
            standardized_x,
            standardized_y,
            args,
            augment=False,
        )
    )
    if args.run_augmentation:
        summaries.append(
            train_one(
                "m4_strategy_a_expanded_train_aug",
                train_raw,
                train_y,
                internal_val_x,
                internal_val_y,
                standardized_x,
                standardized_y,
                args,
                augment=True,
            )
        )

    experiment = {
        "standardized_input_name": args.standardized_input.name,
        "policy": "Strategy A for M4 grading: standardized dataset is external validation only.",
        "old_m3_raw_holdouts_moved_to_training": bool(args.include_old_m3_raw_holdouts_in_train),
        "uci_test_moved_to_training": bool(args.include_uci_test_in_train),
        "uci_test_warning": (
            "UCI official test was included, so UCI test metrics are no longer held-out for this experiment."
            if args.include_uci_test_in_train
            else "UCI official test was not included and remains comparable."
        ),
        "never_used_for_training": ["returned live Serial logs", "standardized external dataset"],
        "standardized_external_windows": int(len(standardized_x)),
        "standardized_external_sources": int(len(set(standardized_sources.tolist()))),
        "runs": summaries,
    }
    (args.output_dir / "m4_strategy_a_experiment_summary.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(args.output_dir / "m4_strategy_a_experiment_summary.csv", index=False)
    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()
