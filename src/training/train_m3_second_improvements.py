from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import DEFAULT_DEVICE, OUTPUTS_DIR, SEED
from src.data.target_domain_features import GRAVITY_FEATURE_CHANNELS, append_gravity_features
from src.data.uci_har import load_uci_har_sequence_dataset
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn, count_trainable_parameters
from src.training.tf_common import benchmark_latency_ms, predict_proba, save_tflite_model
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


BASELINE_METRICS_PATH = OUTPUTS_DIR / "lightweight" / "metrics" / "lightweight_tiny_cnn_metrics.json"
EXPERIMENT_ROOT = OUTPUTS_DIR / "lightweight" / "experiments"


@tf.keras.utils.register_keras_serializable(package="TinyML")
class SparseCategoricalFocalLoss(keras.losses.Loss):
    """Sparse focal loss for softmax probabilities."""

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        row_idx = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        pt = tf.gather_nd(y_pred, tf.stack([row_idx, y_true], axis=1))
        loss = tf.pow(1.0 - pt, self.gamma) * (-tf.math.log(pt))
        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=y_pred.dtype)
            loss *= tf.gather(alpha, y_true)
        return loss

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


def _append_gravity_features(x_standardized: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
    """Append per-window mean acceleration direction and magnitude as constant channels."""
    raw_with_gravity = append_gravity_features(x_raw)
    return np.concatenate([x_standardized.astype(np.float32), raw_with_gravity[:, :, 6:]], axis=-1)


def _train_model(
    model: keras.Model,
    loss: keras.losses.Loss | str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_decay: float,
    patience: int,
    live_log_path: Path,
) -> tuple[pd.DataFrame, int, float]:
    tf.keras.utils.set_random_seed(seed)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    def schedule(epoch: int, _lr: float) -> float:
        return float(learning_rate / (1.0 + lr_decay * epoch))

    callbacks = [
        keras.callbacks.LearningRateScheduler(schedule),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.CSVLogger(str(live_log_path), append=False),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    best_idx = int(hist["val_loss"].idxmin())
    return hist, int(hist.loc[best_idx, "epoch"]), float(hist.loc[best_idx, "val_loss"])


def _load_baseline_metrics() -> dict | None:
    if not BASELINE_METRICS_PATH.exists():
        return None
    return json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))


def _confusion_count(outputs: dict, true_label: str, pred_label: str) -> int:
    class_names = outputs["class_names"]
    cm = np.asarray(outputs["confusion_matrix"], dtype=int)
    return int(cm[class_names.index(true_label), class_names.index(pred_label)])


def _static_confusion_summary(outputs: dict) -> dict:
    sitting_to_standing = _confusion_count(outputs, "SITTING", "STANDING")
    standing_to_sitting = _confusion_count(outputs, "STANDING", "SITTING")
    return {
        "sitting_to_standing": sitting_to_standing,
        "standing_to_sitting": standing_to_sitting,
        "static_confusion_total": sitting_to_standing + standing_to_sitting,
        "sitting_f1": float(outputs["classification_report"]["SITTING"]["f1-score"]),
        "standing_f1": float(outputs["classification_report"]["STANDING"]["f1-score"]),
    }


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
                "baseline_accuracy": baseline["accuracy"],
                "baseline_macro_f1": baseline["macro_f1"],
                "baseline_static_confusion_total": baseline_static["static_confusion_total"],
                "macro_f1_change_vs_baseline": outputs["macro_f1"] - baseline["macro_f1"],
                "static_confusion_change_vs_baseline": row["static_confusion_total"]
                - baseline_static["static_confusion_total"],
            }
        )
    return row


def run_experiment(args: argparse.Namespace, experiment: str, baseline: dict | None) -> dict:
    set_global_seed(args.seed)
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    scaler = SequenceStandardizer.fit(data.train.x)
    x_train = scaler.transform(data.train.x)
    x_val = scaler.transform(data.val.x)
    x_test = scaler.transform(data.test.x)

    if experiment == "focal_loss":
        run_name = "m3_focal_loss_tiny_cnn"
        loss: keras.losses.Loss | str = SparseCategoricalFocalLoss(gamma=args.gamma, name="focal_loss")
        model_note = (
            "Same deployable 6-channel depthwise-separable CNN as M2/M3, "
            "trained with sparse focal loss."
        )
    elif experiment == "gravity_feature":
        run_name = "m3_gravity_feature_tiny_cnn"
        x_train = _append_gravity_features(x_train, data.train.x)
        x_val = _append_gravity_features(x_val, data.val.x)
        x_test = _append_gravity_features(x_test, data.test.x)
        loss = "sparse_categorical_crossentropy"
        model_note = (
            "Adds four constant channels per window: mean acceleration unit vector "
            "(gx, gy, gz) and mean acceleration magnitude."
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    run_dir = args.output_dir / run_name
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"
    for path in [model_dir, metrics_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / f"{run_name}.weights.h5"
    if args.evaluate_existing:
        if not weights_path.exists():
            raise FileNotFoundError(f"--evaluate-existing requested but weights are missing: {weights_path}")
        model = build_tiny_ds_cnn(input_shape=tuple(x_train.shape[1:]), num_classes=len(data.class_names))
        model.load_weights(weights_path)
        history_path = logs_dir / "training_history.csv"
        if not history_path.exists():
            history_path = logs_dir / "training_history_live.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        if not history.empty and "val_loss" in history:
            best_idx = int(history["val_loss"].idxmin())
            best_epoch = int(history.loc[best_idx, "epoch"])
            best_val_loss = float(history.loc[best_idx, "val_loss"])
        else:
            best_epoch = -1
            best_val_loss = float("nan")
    else:
        model = build_tiny_ds_cnn(input_shape=tuple(x_train.shape[1:]), num_classes=len(data.class_names))
        history, best_epoch, best_val_loss = _train_model(
            model,
            loss,
            x_train,
            data.train.y,
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
        "model_note": model_note,
        "input_shape": list(x_train.shape[1:]),
        "output_classes": data.class_names,
        "normalization": "per-channel standardization fit on training split only",
        "loss": "sparse focal loss" if experiment == "focal_loss" else "sparse categorical cross entropy",
        "focal_gamma": args.gamma if experiment == "focal_loss" else None,
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
        "model_artifact_type": "Keras weights checkpoint; reconstruct with build_tiny_ds_cnn and load_weights.",
        "weights_path": str(weights_path),
        "runtime_note": (
            "This is a Phase 2 loss/feature experiment. TFLite export and host latency are skipped "
            "by default; use src.deployment.quantization_experiment for deployable Phase 3 metrics."
        ),
    }
    if args.export_tflite:
        outputs["model_info"]["tflite_size_bytes"] = save_tflite_model(model, model_dir / f"{run_name}.tflite")
        outputs["model_info"]["dynamic_range_tflite_size_bytes"] = save_tflite_model(
            model,
            model_dir / f"{run_name}_dynamic_range.tflite",
            quantize_dynamic_range=True,
        )
    if args.benchmark_host:
        outputs["model_info"]["host_latency"] = benchmark_latency_ms(model, x_test, runs=20, warmup=3)
    if experiment == "gravity_feature":
        outputs["feature_spec"] = {
            "added_channels": GRAVITY_FEATURE_CHANNELS,
            "source": "Computed from raw total_acc_x/y/z before train-stat standardization.",
            "sequence_encoding": "Repeated across all 128 timesteps so the existing 1D CNN can consume it.",
        }
        (run_dir / "feature_spec.json").write_text(
            json.dumps(outputs["feature_spec"], indent=2),
            encoding="utf-8",
        )

    outputs["baseline_comparison"] = _comparison_row(run_name, outputs, baseline)
    save_classification_outputs(outputs, metrics_dir, figures_dir, run_name)
    pd.DataFrame([outputs["baseline_comparison"] | outputs["model_info"]]).to_csv(
        run_dir / "summary.csv",
        index=False,
    )
    print(json.dumps(outputs["baseline_comparison"], indent=2))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 second-improvement experiments.")
    parser.add_argument("--experiment", choices=["focal_loss", "gravity_feature", "all"], default="focal_loss")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--benchmark-host", action="store_true")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    configure_tensorflow_device(args.device)
    baseline = _load_baseline_metrics()
    selected = ["focal_loss", "gravity_feature"] if args.experiment == "all" else [args.experiment]
    rows = []
    for experiment in selected:
        outputs = run_experiment(args, experiment, baseline)
        rows.append(outputs["baseline_comparison"] | outputs["model_info"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "m3_second_improvements_summary.csv", index=False)


if __name__ == "__main__":
    main()
