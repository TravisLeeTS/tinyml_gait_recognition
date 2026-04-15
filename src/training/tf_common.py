from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


@dataclass
class TrainingResult:
    history: pd.DataFrame
    best_epoch: int
    best_val_loss: float


def compile_classifier(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_classifier(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    output_path: Path,
    seed: int,
    epochs: int = 64,
    batch_size: int = 50,
    learning_rate: float = 1e-3,
    lr_decay: float = 0.01,
    patience: int = 12,
) -> TrainingResult:
    tf.keras.utils.set_random_seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compile_classifier(model, learning_rate=learning_rate)

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
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_loss",
            save_best_only=True,
        ),
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
    return TrainingResult(
        history=hist,
        best_epoch=int(hist.loc[best_idx, "epoch"]),
        best_val_loss=float(hist.loc[best_idx, "val_loss"]),
    )


def predict_proba(model: keras.Model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    return model.predict(x, batch_size=batch_size, verbose=0)


def estimate_tflite_size_bytes(model: keras.Model, quantize_dynamic_range: bool = False) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        model.export(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if quantize_dynamic_range:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        return int(len(tflite_model))


def save_tflite_model(model: keras.Model, output_path: Path, quantize_dynamic_range: bool = False) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        model.export(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if quantize_dynamic_range:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return int(len(tflite_model))


def benchmark_latency_ms(
    model: keras.Model,
    x: np.ndarray,
    runs: int = 50,
    warmup: int = 5,
) -> dict:
    sample = x[:1]
    for _ in range(warmup):
        model.predict(sample, batch_size=1, verbose=0)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(sample, batch_size=1, verbose=0)
        timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "runs": int(runs),
        "device_note": "Host CPU Keras predict latency; not an Arduino latency claim.",
    }
