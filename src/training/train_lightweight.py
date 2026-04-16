from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import DEFAULT_DEVICE, OUTPUTS_DIR, SEED
from src.data.uci_har import class_count_table, load_uci_har_sequence_dataset
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn, count_trainable_parameters
from src.training.tf_common import (
    benchmark_latency_ms,
    predict_proba,
    save_tflite_model,
    train_classifier,
)
from src.utils.metrics import (
    classification_outputs,
    print_classification_summary,
    save_classification_outputs,
    top_confusion_pairs,
)
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight TinyML-oriented baseline.")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "lightweight")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = configure_tensorflow_device(args.device)
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)

    scaler = SequenceStandardizer.fit(data.train.x)
    x_train = scaler.transform(data.train.x)
    x_val = scaler.transform(data.val.x)
    x_test = scaler.transform(data.test.x)

    model = build_tiny_ds_cnn(input_shape=tuple(x_train.shape[1:]), num_classes=len(data.class_names))
    model_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    logs_dir = args.output_dir / "logs"
    for path in [model_dir, metrics_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    scaler.save(model_dir / "lightweight_standardizer.json")
    (metrics_dir / "uci_har_class_counts.csv").write_text(
        class_count_table(data).to_csv(index=False),
        encoding="utf-8",
    )
    model_info = {
        "model_name": "TinyDepthwiseSeparableCnn",
        "baseline_role": "Lightweight TinyML-Oriented Baseline",
        "input_shape": list(x_train.shape[1:]),
        "output_classes": data.class_names,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.lr_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "early_stopping": {"monitor": "val_loss", "patience": args.patience},
        "loss": "cross_entropy",
        "normalization": "per-channel standardization fit on training split only",
        "trainable_parameters": count_trainable_parameters(model),
        "keras_model_parameters": int(model.count_params()),
        "device": device,
    }
    (model_dir / "lightweight_model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    history = train_classifier(
        model=model,
        x_train=x_train,
        y_train=data.train.y,
        x_val=x_val,
        y_val=data.val.y,
        output_path=model_dir / "lightweight_tiny_cnn.keras",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        patience=args.patience,
    )
    history.history.to_csv(logs_dir / "lightweight_training_history.csv", index=False)

    probs = predict_proba(model, x_test)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(data.test.y, y_pred, data.class_names, "TinyDepthwiseSeparableCnn")
    tflite_size = save_tflite_model(model, model_dir / "lightweight_tiny_cnn.tflite")
    quant_tflite_size = save_tflite_model(
        model,
        model_dir / "lightweight_tiny_cnn_dynamic_range.tflite",
        quantize_dynamic_range=True,
    )
    latency = benchmark_latency_ms(model, x_test, runs=20, warmup=3)
    outputs["model_info"] = model_info | {
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "keras_model_path": str(model_dir / "lightweight_tiny_cnn.keras"),
        "tflite_size_bytes": tflite_size,
        "dynamic_range_tflite_size_bytes": quant_tflite_size,
        "host_latency": latency,
    }
    outputs["top_confusion_pairs"] = top_confusion_pairs(
        cm=np.asarray(outputs["confusion_matrix"]),
        class_names=data.class_names,
    )
    save_classification_outputs(outputs, metrics_dir, figures_dir, "lightweight_tiny_cnn")
    print_classification_summary(outputs, "Lightweight TinyCNN held-out metrics")


if __name__ == "__main__":
    main()
