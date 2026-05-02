from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import DEFAULT_DEVICE, OUTPUTS_DIR, RAW_DATA_DIR, SEED
from src.data.arduino_collectdata import UCI_CLASS_NAMES, build_windowed_dataset
from src.data.target_domain_features import TEN_CHANNEL_FEATURES, append_gravity_features, orientation_summary
from src.data.uci_har import load_uci_har_sequence_dataset
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn, count_trainable_parameters
from src.training.tf_common import benchmark_latency_ms, predict_proba, save_tflite_model
from src.training.train_m3_second_improvements import SparseCategoricalFocalLoss
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


EXPERIMENT_ROOT = OUTPUTS_DIR / "lightweight" / "experiments" / "m3_target_domain"


def _feature_mode_for(experiment: str) -> str:
    return "gravity_10ch" if experiment in {"mixed_gravity_10ch"} else "base_6ch"


def _apply_features(x_raw: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode == "base_6ch":
        return x_raw.astype(np.float32)
    if feature_mode == "gravity_10ch":
        return append_gravity_features(x_raw)
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def _split_arduino(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray([str(source).split("#segment", maxsplit=1)[0] for source in sources])
    rng = np.random.default_rng(seed)
    val_groups_by_class = []
    train_groups_by_class = []
    for class_id in sorted(set(y.tolist())):
        class_groups = np.asarray(
            sorted({groups[index] for index in np.flatnonzero(y == class_id)}),
            dtype=object,
        )
        rng.shuffle(class_groups)
        if len(class_groups) < 2:
            raise ValueError(
                f"Arduino V2 class {UCI_CLASS_NAMES[class_id]} has fewer than two source groups; "
                "cannot create leakage-safe train/validation split."
            )
        val_count = max(1, int(round(len(class_groups) * val_fraction)))
        val_count = min(val_count, len(class_groups) - 1)
        val_groups_by_class.extend(class_groups[:val_count].tolist())
        train_groups_by_class.extend(class_groups[val_count:].tolist())
    train_groups = np.asarray(sorted(train_groups_by_class), dtype=object)
    val_groups = np.asarray(sorted(val_groups_by_class), dtype=object)
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)
    return (
        x[train_mask],
        x[val_mask],
        y[train_mask],
        y[val_mask],
        sources[train_mask],
        sources[val_mask],
    )


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(UCI_CLASS_NAMES)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights[y].astype(np.float32)


def _fit_model(
    model: keras.Model,
    *,
    loss,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    log_path: Path,
) -> pd.DataFrame:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.CSVLogger(str(log_path), append=False),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    return hist


def _evaluate(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    run_name: str,
    split_name: str,
    output_dir: Path,
) -> dict:
    probs = predict_proba(model, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, class_names, f"{run_name}_{split_name}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(
        np.asarray(outputs["confusion_matrix"], dtype=int),
        class_names,
    )
    save_classification_outputs(outputs, output_dir / "metrics", output_dir / "figures", f"{run_name}_{split_name}")
    return outputs


def _per_class_f1(outputs: dict) -> dict:
    return {
        f"arduino_v2_val_f1_{class_name.lower()}": float(
            outputs["classification_report"][class_name]["f1-score"]
        )
        for class_name in outputs["class_names"]
    }


def _set_trainable(model: keras.Model, trainable_layer_prefixes: tuple[str, ...] | None) -> None:
    if trainable_layer_prefixes is None:
        for layer in model.layers:
            layer.trainable = True
        return
    for layer in model.layers:
        layer.trainable = any(layer.name.startswith(prefix) for prefix in trainable_layer_prefixes)


def run_experiment(args: argparse.Namespace, experiment: str) -> dict:
    set_global_seed(args.seed)
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    arduino_x, arduino_y, arduino_sources = build_windowed_dataset(args.arduino_root, hz=50.0)
    if data.class_names != UCI_CLASS_NAMES:
        raise ValueError("UCI-HAR and Arduino class orders differ.")

    arduino_train_x, arduino_val_x, arduino_train_y, arduino_val_y, train_sources, val_sources = _split_arduino(
        arduino_x,
        arduino_y,
        arduino_sources,
        val_fraction=args.target_val_fraction,
        seed=args.seed,
    )
    feature_mode = _feature_mode_for(experiment)
    run_name = f"m3_target_{experiment}"
    run_dir = args.output_dir / run_name
    model_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    uci_train_raw = _apply_features(data.train.x, feature_mode)
    uci_val_raw = _apply_features(data.val.x, feature_mode)
    uci_test_raw = _apply_features(data.test.x, feature_mode)
    arduino_train_raw = _apply_features(arduino_train_x, feature_mode)
    arduino_val_raw = _apply_features(arduino_val_x, feature_mode)

    if experiment == "uci_baseline":
        scaler_fit_raw = uci_train_raw
    elif experiment == "arduino_only":
        scaler_fit_raw = arduino_train_raw
    else:
        scaler_fit_raw = np.concatenate([uci_train_raw, arduino_train_raw], axis=0)

    scaler = SequenceStandardizer.fit(scaler_fit_raw)
    scaler_path = model_dir / f"{run_name}_standardizer.json"
    scaler.save(scaler_path)

    x_uci_train = scaler.transform(uci_train_raw)
    x_uci_val = scaler.transform(uci_val_raw)
    x_uci_test = scaler.transform(uci_test_raw)
    x_arduino_train = scaler.transform(arduino_train_raw)
    x_arduino_val = scaler.transform(arduino_val_raw)

    loss = SparseCategoricalFocalLoss(gamma=2.0, name="focal_loss") if experiment == "mixed_focal_6ch" else "sparse_categorical_crossentropy"
    model = build_tiny_ds_cnn(input_shape=tuple(x_uci_train.shape[1:]), num_classes=len(data.class_names))
    histories: dict[str, str] = {}

    if experiment == "uci_baseline":
        weights = _balanced_weights(data.train.y) if args.class_balanced else None
        hist = _fit_model(
            model,
            loss=loss,
            x_train=x_uci_train,
            y_train=data.train.y,
            x_val=x_uci_val,
            y_val=data.val.y,
            sample_weight=weights,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            log_path=logs_dir / "uci_baseline_history.csv",
        )
        hist.to_csv(logs_dir / "uci_baseline_history.csv", index=False)
        histories["uci_baseline"] = str(logs_dir / "uci_baseline_history.csv")
    elif experiment == "arduino_only":
        weights = _balanced_weights(arduino_train_y) if args.class_balanced else None
        hist = _fit_model(
            model,
            loss=loss,
            x_train=x_arduino_train,
            y_train=arduino_train_y,
            x_val=x_arduino_val,
            y_val=arduino_val_y,
            sample_weight=weights,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            log_path=logs_dir / "arduino_only_history.csv",
        )
        hist.to_csv(logs_dir / "arduino_only_history.csv", index=False)
        histories["arduino_only"] = str(logs_dir / "arduino_only_history.csv")
    elif experiment == "finetune":
        pretrain_hist = _fit_model(
            model,
            loss=loss,
            x_train=x_uci_train,
            y_train=data.train.y,
            x_val=x_uci_val,
            y_val=data.val.y,
            sample_weight=_balanced_weights(data.train.y) if args.class_balanced else None,
            learning_rate=args.learning_rate,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            log_path=logs_dir / "finetune_pretrain_history.csv",
        )
        pretrain_hist.to_csv(logs_dir / "finetune_pretrain_history.csv", index=False)
        histories["pretrain"] = str(logs_dir / "finetune_pretrain_history.csv")

        _set_trainable(model, ("activity",))
        head_hist = _fit_model(
            model,
            loss=loss,
            x_train=x_arduino_train,
            y_train=arduino_train_y,
            x_val=x_arduino_val,
            y_val=arduino_val_y,
            sample_weight=_balanced_weights(arduino_train_y) if args.class_balanced else None,
            learning_rate=args.finetune_learning_rate,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            log_path=logs_dir / "finetune_head_history.csv",
        )
        head_hist.to_csv(logs_dir / "finetune_head_history.csv", index=False)
        histories["head"] = str(logs_dir / "finetune_head_history.csv")

        _set_trainable(model, ("ds_conv_2", "ds_bn_2", "ds_relu_2", "global_avg_pool", "activity"))
        block_hist = _fit_model(
            model,
            loss=loss,
            x_train=x_arduino_train,
            y_train=arduino_train_y,
            x_val=x_arduino_val,
            y_val=arduino_val_y,
            sample_weight=_balanced_weights(arduino_train_y) if args.class_balanced else None,
            learning_rate=args.finetune_learning_rate * 0.5,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            log_path=logs_dir / "finetune_last_block_history.csv",
        )
        block_hist.to_csv(logs_dir / "finetune_last_block_history.csv", index=False)
        histories["last_block"] = str(logs_dir / "finetune_last_block_history.csv")
        _set_trainable(model, None)
    else:
        x_train = np.concatenate([x_uci_train, x_arduino_train], axis=0)
        y_train = np.concatenate([data.train.y, arduino_train_y], axis=0)
        uci_weights = _balanced_weights(data.train.y) * np.float32(args.uci_loss_weight)
        arduino_weights = _balanced_weights(arduino_train_y)
        sample_weight = np.concatenate([uci_weights, arduino_weights], axis=0) if args.class_balanced else np.concatenate(
            [
                np.full(len(data.train.y), args.uci_loss_weight, dtype=np.float32),
                np.ones(len(arduino_train_y), dtype=np.float32),
            ]
        )
        hist = _fit_model(
            model,
            loss=loss,
            x_train=x_train,
            y_train=y_train,
            x_val=x_arduino_val,
            y_val=arduino_val_y,
            sample_weight=sample_weight,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            log_path=logs_dir / "mixed_history.csv",
        )
        hist.to_csv(logs_dir / "mixed_history.csv", index=False)
        histories["mixed"] = str(logs_dir / "mixed_history.csv")

    weights_path = model_dir / f"{run_name}.weights.h5"
    model.save_weights(weights_path)
    if args.export_tflite:
        tflite_size_bytes = save_tflite_model(model, model_dir / f"{run_name}.tflite")
    else:
        tflite_size_bytes = None

    evaluations = {
        "uci_test": _evaluate(model, x_uci_test, data.test.y, data.class_names, run_name, "uci_test", run_dir),
        "arduino_v2_val": _evaluate(model, x_arduino_val, arduino_val_y, data.class_names, run_name, "arduino_v2_val", run_dir),
    }

    summary = {
        "run_name": run_name,
        "experiment": experiment,
        "feature_mode": feature_mode,
        "input_shape": list(x_uci_train.shape[1:]),
        "channels": TEN_CHANNEL_FEATURES if feature_mode == "gravity_10ch" else ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "selection_rule": "Prioritize held-out Arduino target-domain validation first; use UCI-HAR official test as source-domain retention second.",
        "normalization": {
            "fit_scope": "training/adaptation data only",
            "standardizer_path": str(scaler_path),
            "excluded": ["UCI-HAR official test", "Arduino V2 validation split", "future right-pocket live evidence", "future left-pocket live evidence"],
        },
        "data_sources": {
            "uci_train_windows": int(len(data.train.y)),
            "uci_val_windows": int(len(data.val.y)),
            "uci_test_windows": int(len(data.test.y)),
            "arduino_v2_train_windows": int(len(arduino_train_y)),
            "arduino_v2_val_windows": int(len(arduino_val_y)),
            "arduino_root": str(args.arduino_root),
            "target_val_fraction": args.target_val_fraction,
            "uci_loss_weight": args.uci_loss_weight,
            "arduino_split": "group-aware by source capture file; overlapping windows from the same file stay in the same split",
            "arduino_v2_train_groups": sorted(set(str(source).split("#segment", maxsplit=1)[0] for source in train_sources)),
            "arduino_v2_val_groups": sorted(set(str(source).split("#segment", maxsplit=1)[0] for source in val_sources)),
        },
        "artifacts": {
            "weights_path": str(weights_path),
            "histories": histories,
        },
        "model": {
            "parameters": int(model.count_params()),
            "trainable_parameters": count_trainable_parameters(model),
            "loss": "focal" if experiment == "mixed_focal_6ch" else "sparse_categorical_crossentropy",
            "fp32_tflite_size_bytes": tflite_size_bytes,
        },
        "metrics": {
            "uci_test_accuracy": evaluations["uci_test"]["accuracy"],
            "uci_test_macro_f1": evaluations["uci_test"]["macro_f1"],
            "arduino_v2_val_accuracy": evaluations["arduino_v2_val"]["accuracy"],
            "arduino_v2_val_macro_f1": evaluations["arduino_v2_val"]["macro_f1"],
            **_per_class_f1(evaluations["arduino_v2_val"]),
        },
    }
    if args.benchmark_host:
        summary["model"]["host_latency_proxy"] = benchmark_latency_ms(model, x_arduino_val, runs=20, warmup=3)
    if feature_mode == "gravity_10ch":
        summary["feature_spec"] = {
            "raw_feature_channels": TEN_CHANNEL_FEATURES,
            "arduino_reproducibility": (
                "For each 128-sample window, compute mean acc_x/y/z, normalize that vector to "
                "gravity_dir_x/y/z, compute acc_magnitude, repeat those four values for all timesteps, "
                "then apply the saved 10-channel standardizer."
            ),
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_row = {
        "run_name": run_name,
        "experiment": experiment,
        "feature_mode": feature_mode,
        **summary["metrics"],
        "fp32_tflite_size_bytes": tflite_size_bytes,
        "host_latency_mean_ms": summary["model"].get("host_latency_proxy", {}).get("mean_ms"),
        "host_latency_note": summary["model"].get("host_latency_proxy", {}).get("device_note"),
    }
    pd.DataFrame([summary_row]).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(orientation_summary(arduino_train_x, arduino_train_y, data.class_names)).to_csv(
        run_dir / "arduino_v2_train_orientation_summary.csv",
        index=False,
    )
    pd.DataFrame(orientation_summary(arduino_val_x, arduino_val_y, data.class_names)).to_csv(
        run_dir / "arduino_v2_val_orientation_summary.csv",
        index=False,
    )
    print(json.dumps(summary["metrics"] | {"run_name": run_name}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 target-domain adaptation experiments.")
    parser.add_argument(
        "--experiment",
        choices=["uci_baseline", "arduino_only", "finetune", "mixed_6ch", "mixed_focal_6ch", "mixed_gravity_10ch", "all"],
        default="all",
    )
    parser.add_argument("--arduino-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2")
    parser.add_argument("--target-val-fraction", type=float, default=0.25)
    parser.add_argument("--uci-loss-weight", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class-balanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--benchmark-host", action="store_true")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    if not 0.05 <= args.target_val_fraction <= 0.5:
        raise ValueError("--target-val-fraction should be between 0.05 and 0.5")
    configure_tensorflow_device(args.device)
    selected = (
        ["uci_baseline", "arduino_only", "finetune", "mixed_6ch", "mixed_focal_6ch", "mixed_gravity_10ch"]
        if args.experiment == "all"
        else [args.experiment]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment in selected:
        summary = run_experiment(args, experiment)
        rows.append(
            {
                "run_name": summary["run_name"],
                "experiment": summary["experiment"],
                "feature_mode": summary["feature_mode"],
                **summary["metrics"],
                "fp32_tflite_size_bytes": summary["model"].get("fp32_tflite_size_bytes"),
                "host_latency_mean_ms": summary["model"].get("host_latency_proxy", {}).get("mean_ms"),
                "host_latency_note": summary["model"].get("host_latency_proxy", {}).get("device_note"),
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "m3_target_domain_summary.csv", index=False)


if __name__ == "__main__":
    main()
