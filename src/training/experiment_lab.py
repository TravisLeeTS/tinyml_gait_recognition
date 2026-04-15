from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import OUTPUTS_DIR, SEED
from src.data.uci_har import class_count_table, load_uci_har_sequence_dataset
from src.training.tf_common import (
    benchmark_latency_ms,
    predict_proba,
    save_tflite_model,
    train_classifier,
)
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceMinMaxScaler, SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


def load_uci_har_experiment_data(normalization: str = "standard", seed: int = SEED) -> dict:
    data = load_uci_har_sequence_dataset(download=True, seed=seed)
    scaler_cls = {"standard": SequenceStandardizer, "minmax": SequenceMinMaxScaler}[normalization]
    scaler = scaler_cls.fit(data.train.x)
    return {
        "x_train": scaler.transform(data.train.x),
        "y_train": data.train.y,
        "x_val": scaler.transform(data.val.x),
        "y_val": data.val.y,
        "x_test": scaler.transform(data.test.x),
        "y_test": data.test.y,
        "class_names": data.class_names,
        "input_shape": tuple(data.train.x.shape[1:]),
        "class_counts": class_count_table(data),
        "normalizer": scaler,
    }


def run_keras_architecture(
    *,
    run_name: str,
    model_builder: Callable[[tuple[int, int], int], keras.Model],
    normalization: str = "standard",
    epochs: int = 40,
    batch_size: int = 50,
    learning_rate: float = 1e-3,
    lr_decay: float = 0.01,
    patience: int = 8,
    seed: int = SEED,
    device: str = "cpu",
    output_dir: Path = OUTPUTS_DIR / "lightweight" / "experiments",
) -> dict:
    set_global_seed(seed)
    configure_tensorflow_device(device)
    dataset = load_uci_har_experiment_data(normalization=normalization, seed=seed)
    run_dir = output_dir / run_name
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"
    for path in [model_dir, metrics_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    model = model_builder(dataset["input_shape"], len(dataset["class_names"]))
    history = train_classifier(
        model=model,
        x_train=dataset["x_train"],
        y_train=dataset["y_train"],
        x_val=dataset["x_val"],
        y_val=dataset["y_val"],
        output_path=model_dir / f"{run_name}.keras",
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        patience=patience,
    )
    history.history.to_csv(logs_dir / "training_history.csv", index=False)
    dataset["class_counts"].to_csv(metrics_dir / "uci_har_class_counts.csv", index=False)

    probs = predict_proba(model, dataset["x_test"])
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(dataset["y_test"], y_pred, dataset["class_names"], run_name)
    outputs["top_confusion_pairs"] = top_confusion_pairs(
        np.asarray(outputs["confusion_matrix"]),
        dataset["class_names"],
    )
    outputs["model_info"] = {
        "run_name": run_name,
        "input_shape": list(dataset["input_shape"]),
        "normalization": normalization,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "learning_rate_decay": lr_decay,
        "patience": patience,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "keras_params": int(model.count_params()),
        "keras_model_path": str(model_dir / f"{run_name}.keras"),
        "tflite_size_bytes": save_tflite_model(model, model_dir / f"{run_name}.tflite"),
        "dynamic_range_tflite_size_bytes": save_tflite_model(
            model,
            model_dir / f"{run_name}_dynamic_range.tflite",
            quantize_dynamic_range=True,
        ),
        "host_latency": benchmark_latency_ms(model, dataset["x_test"], runs=20, warmup=3),
    }
    save_classification_outputs(outputs, metrics_dir, figures_dir, run_name)
    pd.DataFrame([outputs["model_info"] | {
        "accuracy": outputs["accuracy"],
        "macro_f1": outputs["macro_f1"],
        "weighted_f1": outputs["weighted_f1"],
    }]).to_csv(run_dir / "summary.csv", index=False)
    return outputs
