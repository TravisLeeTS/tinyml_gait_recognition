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
from src.data.target_domain_features import TEN_CHANNEL_FEATURES, append_gravity_features
from src.data.uci_har import load_uci_har_sequence_dataset
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn, count_trainable_parameters
from src.training.train_m4_strategy_a_experiment import augment_training_windows, balanced_weights
from src.training.tf_common import predict_proba
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "m4_session_split_experiment"
SESSION_1 = "session_01_1person_allclass_30s"
SESSION_3 = "session_03_1person_4class_5min"
SESSION_4 = "session_04_2person_allclass_mixed_duration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train M4 session-aware experiments using standardized session 4, optionally "
            "standardized session 3, for training and standardized session 1 right/left "
            "pockets as separate robustness evaluations."
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
    parser.add_argument("--include-session3-train", action="store_true", help="Also include standardized session 3 in training.")
    parser.add_argument("--run-augmentation", action="store_true")
    parser.add_argument("--augmentation-copies", default=1, type=int)
    parser.add_argument("--noise-std", default=0.02, type=float)
    parser.add_argument("--scale-min", default=0.95, type=float)
    parser.add_argument("--scale-max", default=1.05, type=float)
    parser.add_argument("--max-shift", default=4, type=int)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["base_6ch", "gravity_aligned_6ch", "gravity_10ch"],
        choices=["base_6ch", "gravity_aligned_6ch", "gravity_10ch"],
    )
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


def source_mask(sources: np.ndarray, *, session: str | None = None, placement: str | None = None) -> np.ndarray:
    text = np.asarray([str(source) for source in sources])
    mask = np.ones(len(text), dtype=bool)
    if session:
        mask &= np.char.find(text, session) >= 0
    if placement == "right_pocket":
        mask &= np.char.find(text, "_right_") >= 0
    elif placement == "left_pocket":
        mask &= np.char.find(text, "_left_") >= 0
    elif placement:
        raise ValueError(f"Unsupported placement filter: {placement}")
    return mask


def gravity_align_windows(x_raw: np.ndarray) -> np.ndarray:
    """Rotate each window so its mean acceleration vector is aligned to +Z.

    This is an offline approximation of gravity-frame alignment. It is not yet
    implemented in the Arduino sketch, so any benefit must be treated as an
    algorithm experiment rather than a live deployment claim.
    """
    x = x_raw.astype(np.float32).copy()
    for idx in range(len(x)):
        g = x[idx, :, :3].mean(axis=0).astype(np.float64)
        norm = np.linalg.norm(g)
        if norm < 1e-6:
            continue
        a = g / norm
        b = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))
        if s < 1e-6:
            rotation = np.eye(3, dtype=np.float64) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.asarray(
                [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
                dtype=np.float64,
            )
            rotation = np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))
        x[idx, :, :3] = (rotation @ x[idx, :, :3].T).T.astype(np.float32)
        x[idx, :, 3:6] = (rotation @ x[idx, :, 3:6].T).T.astype(np.float32)
    return x


def apply_feature_mode(x_raw: np.ndarray, mode: str) -> np.ndarray:
    if mode == "base_6ch":
        return x_raw.astype(np.float32)
    if mode == "gravity_aligned_6ch":
        return gravity_align_windows(x_raw)
    if mode == "gravity_10ch":
        return append_gravity_features(x_raw)
    raise ValueError(f"Unsupported feature mode: {mode}")


def evaluate(model: keras.Model, x: np.ndarray, y: np.ndarray, run_name: str, split_name: str, output_dir: Path) -> dict:
    probs = predict_proba(model, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, UCI_CLASS_NAMES, f"{run_name}_{split_name}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), UCI_CLASS_NAMES)
    save_classification_outputs(outputs, output_dir / "metrics", output_dir / "figures", f"{run_name}_{split_name}")
    return outputs


def train_one(
    *,
    mode: str,
    train_raw_base: np.ndarray,
    train_y: np.ndarray,
    internal_val_raw_base: np.ndarray,
    internal_val_y: np.ndarray,
    eval_right_raw_base: np.ndarray,
    eval_right_y: np.ndarray,
    eval_left_raw_base: np.ndarray,
    eval_left_y: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    run_name = f"session4_train_session1_eval_{mode}"
    run_dir = args.output_dir / run_name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_raw = apply_feature_mode(train_raw_base, mode)
    internal_val_raw = apply_feature_mode(internal_val_raw_base, mode)
    eval_right_raw = apply_feature_mode(eval_right_raw_base, mode)
    eval_left_raw = apply_feature_mode(eval_left_raw_base, mode)

    scaler = SequenceStandardizer.fit(train_raw)
    effective_train_raw = train_raw
    effective_train_y = train_y
    augmentation_meta = {"enabled": False}
    if args.run_augmentation:
        effective_train_raw, effective_train_y = augment_training_windows(train_raw, train_y, args)
        augmentation_meta = {
            "enabled": True,
            "copies": args.augmentation_copies,
            "noise_std": args.noise_std,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "max_shift": args.max_shift,
            "guardrail": "Only training windows were augmented. Session 1 right/left evaluations were unchanged.",
        }

    x_train = scaler.transform(effective_train_raw)
    x_internal_val = scaler.transform(internal_val_raw)
    x_right = scaler.transform(eval_right_raw)
    x_left = scaler.transform(eval_left_raw)

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
        validation_data=(x_internal_val, internal_val_y),
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

    internal_outputs = evaluate(model, x_internal_val, internal_val_y, run_name, "uci_internal_val", run_dir)
    right_outputs = evaluate(model, x_right, eval_right_y, run_name, "session1_right_pocket", run_dir)
    left_outputs = evaluate(model, x_left, eval_left_y, run_name, "session1_left_pocket", run_dir)

    summary = {
        "run_name": run_name,
        "feature_mode": mode,
        "algorithm_mitigation": {
            "base_6ch": "Controlled baseline without algorithm-level orientation mitigation.",
            "gravity_aligned_6ch": "Offline gravity-frame alignment by rotating each window to align mean acceleration with +Z.",
            "gravity_10ch": "Orientation-aware repeated gravity direction and acceleration magnitude features.",
        }[mode],
        "calibration_pose_note": (
            "A real calibration-pose workflow is still a hardware protocol requirement. "
            "This experiment approximates calibration algorithmically from each window's mean acceleration."
            if mode == "gravity_aligned_6ch"
            else "No explicit calibration-pose samples are used in this run."
        ),
        "train_windows_before_augmentation": int(len(train_raw)),
        "train_windows_effective": int(len(effective_train_raw)),
        "internal_validation_windows": int(len(internal_val_raw)),
        "session1_right_windows": int(len(eval_right_raw)),
        "session1_left_windows": int(len(eval_left_raw)),
        "augmentation": augmentation_meta,
        "parameters": int(model.count_params()),
        "trainable_parameters": count_trainable_parameters(model),
        "uci_internal_val_accuracy": internal_outputs["accuracy"],
        "uci_internal_val_macro_f1": internal_outputs["macro_f1"],
        "session1_right_accuracy": right_outputs["accuracy"],
        "session1_right_macro_f1": right_outputs["macro_f1"],
        "session1_left_accuracy": left_outputs["accuracy"],
        "session1_left_macro_f1": left_outputs["macro_f1"],
        "session1_right_top_confusions": right_outputs["top_confusion_pairs"][:5],
        "session1_left_top_confusions": left_outputs["top_confusion_pairs"][:5],
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

    for root in [args.v2_root, args.v2_1_root, args.right_root, args.left_root]:
        x_part, y_part, _ = build_windowed_dataset(root, hz=50.0)
        train_parts.append(x_part)
        y_parts.append(y_part)

    with tempfile.TemporaryDirectory(prefix="m4_standardized_sessions_") as temp_name:
        standardized_root = extract_if_needed(args.standardized_input.expanduser().resolve(), Path(temp_name))
        standardized_x, standardized_y, standardized_sources = build_windowed_dataset(standardized_root, hz=50.0)

    train_mask = source_mask(standardized_sources, session=SESSION_4)
    if args.include_session3_train:
        train_mask |= source_mask(standardized_sources, session=SESSION_3)
    right_mask = source_mask(standardized_sources, session=SESSION_1, placement="right_pocket")
    left_mask = source_mask(standardized_sources, session=SESSION_1, placement="left_pocket")
    if not np.any(train_mask) or not np.any(right_mask) or not np.any(left_mask):
        raise ValueError("Session split produced an empty train/right/left set; inspect standardized dataset names.")

    train_parts.append(standardized_x[train_mask])
    y_parts.append(standardized_y[train_mask])
    train_raw = np.concatenate(train_parts, axis=0).astype(np.float32)
    train_y = np.concatenate(y_parts, axis=0).astype(np.int64)

    summaries = []
    for mode in args.feature_modes:
        summaries.append(
            train_one(
                mode=mode,
                train_raw_base=train_raw,
                train_y=train_y,
                internal_val_raw_base=data.val.x,
                internal_val_y=data.val.y,
                eval_right_raw_base=standardized_x[right_mask],
                eval_right_y=standardized_y[right_mask],
                eval_left_raw_base=standardized_x[left_mask],
                eval_left_y=standardized_y[left_mask],
                args=args,
            )
        )

    experiment = {
        "policy": "Session-aware M4 experiment: standardized session 4 moved into training; session 1 held out as right/left robustness evaluations.",
        "standardized_input_name": args.standardized_input.name,
        "standardized_train_sessions": [SESSION_4, *([SESSION_3] if args.include_session3_train else [])],
        "standardized_eval_session": SESSION_1,
        "leakage_guardrail": "No windows from standardized session 1 are used for training, validation, augmentation, or quantization in this experiment.",
        "live_serial_logs_used_for_training": False,
        "session4_train_windows": int(train_mask.sum()),
        "session1_right_windows": int(right_mask.sum()),
        "session1_left_windows": int(left_mask.sum()),
        "runs": summaries,
    }
    (args.output_dir / "m4_session_split_experiment_summary.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(args.output_dir / "m4_session_split_experiment_summary.csv", index=False)
    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()
