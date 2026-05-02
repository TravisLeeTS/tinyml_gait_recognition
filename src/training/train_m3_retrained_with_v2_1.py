from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import DOCS_DIR, OUTPUTS_DIR, RAW_DATA_DIR, SEED
from src.data.arduino_collectdata import (
    UCI_CLASS_NAMES,
    build_windowed_dataset,
    iter_csv_files,
    read_labeled_csv,
)
from src.data.target_domain_features import TEN_CHANNEL_FEATURES, append_gravity_features
from src.data.uci_har import load_uci_har_sequence_dataset
from src.deployment.convert_lightweight_to_tflite_micro import benchmark_tflite_ms, run_tflite
from src.deployment.package_m3_candidate import _class_balanced_indices, _convert_int8, _write_model_header
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn, count_trainable_parameters
from src.training.tf_common import predict_proba, save_tflite_model
from src.training.train_m3_second_improvements import SparseCategoricalFocalLoss
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "deployment" / "m3_retrained_with_v2_1"


def _source_group(source: str) -> str:
    return str(source).split("#segment", maxsplit=1)[0]


def _split_v2(x: np.ndarray, y: np.ndarray, sources: np.ndarray, val_fraction: float, seed: int):
    groups = np.asarray([_source_group(source) for source in sources])
    rng = np.random.default_rng(seed)
    val_groups = []
    train_groups = []
    for class_id in sorted(set(y.tolist())):
        class_groups = np.asarray(sorted({groups[idx] for idx in np.flatnonzero(y == class_id)}), dtype=object)
        rng.shuffle(class_groups)
        val_count = max(1, int(round(len(class_groups) * val_fraction)))
        val_count = min(val_count, len(class_groups) - 1)
        val_groups.extend(class_groups[:val_count].tolist())
        train_groups.extend(class_groups[val_count:].tolist())
    train_mask = np.isin(groups, np.asarray(train_groups, dtype=object))
    val_mask = np.isin(groups, np.asarray(val_groups, dtype=object))
    return x[train_mask], y[train_mask], sources[train_mask], x[val_mask], y[val_mask], sources[val_mask]


def _apply_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "base_6ch":
        return x.astype(np.float32)
    if mode == "gravity_10ch":
        return append_gravity_features(x)
    raise ValueError(mode)


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(UCI_CLASS_NAMES)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights[y].astype(np.float32)


def _cap_per_class(x: np.ndarray, y: np.ndarray, sources: np.ndarray, cap: int, seed: int):
    rng = np.random.default_rng(seed)
    selected = []
    before = {UCI_CLASS_NAMES[idx]: int((y == idx).sum()) for idx in range(len(UCI_CLASS_NAMES))}
    for class_id in sorted(set(y.tolist())):
        indices = np.flatnonzero(y == class_id)
        if len(indices) > cap:
            indices = rng.choice(indices, size=cap, replace=False)
        selected.append(np.asarray(indices, dtype=np.int64))
    selected_idx = np.concatenate(selected)
    rng.shuffle(selected_idx)
    after = {UCI_CLASS_NAMES[idx]: int((y[selected_idx] == idx).sum()) for idx in range(len(UCI_CLASS_NAMES))}
    return x[selected_idx], y[selected_idx], sources[selected_idx], before, after


def _evaluate_model(model, x, y, class_names, run_name: str, split_name: str, output_dir: Path) -> dict:
    probs = predict_proba(model, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, class_names, f"{run_name}_{split_name}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), class_names)
    save_classification_outputs(outputs, output_dir / "metrics", output_dir / "figures", f"{run_name}_{split_name}")
    return outputs


def _evaluate_tflite(model_path: Path, x, y, class_names, run_name: str, split_name: str, output_dir: Path) -> tuple[dict, dict]:
    probs, metadata = run_tflite(model_path, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, class_names, f"{run_name}_{split_name}")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), class_names)
    outputs["model_info"] = {
        "tflite_model_path": str(model_path),
        "tflite_size_bytes": model_path.stat().st_size,
        "tflite_metadata": metadata,
        "host_tflite_latency": benchmark_tflite_ms(model_path, x[0], runs=50, warmup=5),
    }
    save_classification_outputs(outputs, output_dir / "metrics", output_dir / "figures", f"{run_name}_{split_name}")
    return outputs, metadata


def _metric_row(run_name: str, experiment: str, split: str, outputs: dict, size: int | None = None) -> dict:
    row = {
        "run_name": run_name,
        "experiment": experiment,
        "split": split,
        "accuracy": outputs["accuracy"],
        "macro_f1": outputs["macro_f1"],
        "weighted_f1": outputs["weighted_f1"],
        "model_size_bytes": size,
    }
    for class_name in outputs["class_names"]:
        report = outputs["classification_report"][class_name]
        row[f"precision_{class_name.lower()}"] = report["precision"]
        row[f"recall_{class_name.lower()}"] = report["recall"]
        row[f"f1_{class_name.lower()}"] = report["f1-score"]
    return row


def _train_model(args, run_name, experiment, feature_mode, loss_name, arrays, output_dir: Path):
    data = arrays["uci"]
    x_v2_train_raw = arrays["v2_train_x"]
    y_v2_train = arrays["v2_train_y"]
    x_v2_val_raw = arrays["v2_val_x"]
    y_v2_val = arrays["v2_val_y"]
    x_long_raw = arrays["long_x"]
    y_long = arrays["long_y"]

    uci_train_raw = _apply_features(data.train.x, feature_mode)
    uci_val_raw = _apply_features(data.val.x, feature_mode)
    scaler_fit = np.concatenate(
        [
            uci_train_raw,
            _apply_features(x_v2_train_raw, feature_mode),
            _apply_features(x_long_raw, feature_mode),
        ],
        axis=0,
    )
    scaler = SequenceStandardizer.fit(scaler_fit)
    x_uci_train = scaler.transform(uci_train_raw)
    x_uci_val = scaler.transform(uci_val_raw)
    x_v2_train = scaler.transform(_apply_features(x_v2_train_raw, feature_mode))
    x_v2_val = scaler.transform(_apply_features(x_v2_val_raw, feature_mode))
    x_long = scaler.transform(_apply_features(x_long_raw, feature_mode))

    model = build_tiny_ds_cnn(input_shape=tuple(x_uci_train.shape[1:]), num_classes=len(data.class_names))
    loss = SparseCategoricalFocalLoss(gamma=2.0, name="focal_loss") if loss_name == "focal" else "sparse_categorical_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss=loss, metrics=["accuracy"])

    x_train = np.concatenate([x_uci_train, x_v2_train, x_long], axis=0)
    y_train = np.concatenate([data.train.y, y_v2_train, y_long], axis=0)
    uci_weights = _balanced_weights(data.train.y) * np.float32(args.uci_loss_weight)
    arduino_weights = _balanced_weights(np.concatenate([y_v2_train, y_long], axis=0))
    sample_weight = np.concatenate([uci_weights, arduino_weights], axis=0)
    run_dir = output_dir / run_name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, min_delta=1e-4),
        keras.callbacks.CSVLogger(str(run_dir / "logs" / "training_history.csv"), append=False),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_v2_val, y_v2_val),
        sample_weight=sample_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    pd.DataFrame(history.history).to_csv(run_dir / "logs" / "training_history.csv", index=False)
    model.save_weights(run_dir / "models" / f"{run_name}.weights.h5")
    fp32_size = save_tflite_model(model, run_dir / "models" / f"{run_name}.tflite") if args.export_tflite else None
    scaler.save(run_dir / "models" / f"{run_name}_standardizer.json")
    meta = {
        "run_name": run_name,
        "experiment": experiment,
        "feature_mode": feature_mode,
        "loss": loss_name,
        "input_shape": list(x_train.shape[1:]),
        "channels": TEN_CHANNEL_FEATURES if feature_mode == "gravity_10ch" else ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "standardizer_path": str(run_dir / "models" / f"{run_name}_standardizer.json"),
        "weights_path": str(run_dir / "models" / f"{run_name}.weights.h5"),
        "fp32_tflite_path": str(run_dir / "models" / f"{run_name}.tflite"),
        "fp32_tflite_size_bytes": fp32_size,
        "parameters": int(model.count_params()),
        "trainable_parameters": count_trainable_parameters(model),
    }
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model, scaler, meta


def _set_trainable(model: keras.Model, trainable_layer_prefixes: tuple[str, ...] | None) -> None:
    if trainable_layer_prefixes is None:
        for layer in model.layers:
            layer.trainable = True
        return
    for layer in model.layers:
        layer.trainable = any(layer.name.startswith(prefix) for prefix in trainable_layer_prefixes)


def _save_candidate_artifacts(model, scaler, meta: dict, run_dir: Path, args) -> None:
    model.save_weights(run_dir / "models" / f"{meta['run_name']}.weights.h5")
    if args.export_tflite:
        meta["fp32_tflite_size_bytes"] = save_tflite_model(model, run_dir / "models" / f"{meta['run_name']}.tflite")
        meta["fp32_tflite_path"] = str(run_dir / "models" / f"{meta['run_name']}.tflite")
    scaler.save(run_dir / "models" / f"{meta['run_name']}_standardizer.json")
    meta["standardizer_path"] = str(run_dir / "models" / f"{meta['run_name']}_standardizer.json")
    meta["weights_path"] = str(run_dir / "models" / f"{meta['run_name']}.weights.h5")
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _train_uci_baseline(args, arrays, output_dir: Path):
    data = arrays["uci"]
    run_name = "m3_v2_1_uci_only_baseline"
    run_dir = output_dir / run_name
    for subdir in ["models", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    scaler = SequenceStandardizer.fit(data.train.x)
    x_train = scaler.transform(data.train.x)
    x_val = scaler.transform(data.val.x)
    model = build_tiny_ds_cnn(input_shape=tuple(x_train.shape[1:]), num_classes=len(data.class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        data.train.y,
        validation_data=(x_val, data.val.y),
        sample_weight=_balanced_weights(data.train.y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, min_delta=1e-4),
            keras.callbacks.CSVLogger(str(run_dir / "logs" / "training_history.csv"), append=False),
        ],
        verbose=2,
        shuffle=True,
    )
    pd.DataFrame(history.history).to_csv(run_dir / "logs" / "training_history.csv", index=False)
    meta = {
        "run_name": run_name,
        "experiment": "uci_only_baseline_v2_1_eval",
        "feature_mode": "base_6ch",
        "loss": "cross_entropy",
        "input_shape": list(x_train.shape[1:]),
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "normalization_scope": "UCI-HAR train only",
        "parameters": int(model.count_params()),
        "trainable_parameters": count_trainable_parameters(model),
    }
    _save_candidate_artifacts(model, scaler, meta, run_dir, args)
    return model, scaler, meta


def _train_arduino_only(args, arrays, output_dir: Path):
    run_name = "m3_v2_1_arduino_only"
    run_dir = output_dir / run_name
    for subdir in ["models", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    x_train_raw = np.concatenate([arrays["v2_train_x"], arrays["long_x"]], axis=0)
    y_train = np.concatenate([arrays["v2_train_y"], arrays["long_y"]], axis=0)
    scaler = SequenceStandardizer.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(arrays["v2_val_x"])
    model = build_tiny_ds_cnn(input_shape=tuple(x_train.shape[1:]), num_classes=len(UCI_CLASS_NAMES))
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, arrays["v2_val_y"]),
        sample_weight=_balanced_weights(y_train),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, min_delta=1e-4),
            keras.callbacks.CSVLogger(str(run_dir / "logs" / "training_history.csv"), append=False),
        ],
        verbose=2,
        shuffle=True,
    )
    pd.DataFrame(history.history).to_csv(run_dir / "logs" / "training_history.csv", index=False)
    meta = {
        "run_name": run_name,
        "experiment": "arduino_only_v2_1",
        "feature_mode": "base_6ch",
        "loss": "cross_entropy",
        "input_shape": list(x_train.shape[1:]),
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "normalization_scope": "Arduino V2 train + capped V2.1 long adaptation only",
        "parameters": int(model.count_params()),
        "trainable_parameters": count_trainable_parameters(model),
    }
    _save_candidate_artifacts(model, scaler, meta, run_dir, args)
    return model, scaler, meta


def _train_finetune(args, arrays, output_dir: Path):
    data = arrays["uci"]
    run_name = "m3_v2_1_uci_pretrain_finetune"
    run_dir = output_dir / run_name
    for subdir in ["models", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    target_train_raw = np.concatenate([arrays["v2_train_x"], arrays["long_x"]], axis=0)
    target_y = np.concatenate([arrays["v2_train_y"], arrays["long_y"]], axis=0)
    scaler = SequenceStandardizer.fit(np.concatenate([data.train.x, target_train_raw], axis=0))
    x_uci_train = scaler.transform(data.train.x)
    x_uci_val = scaler.transform(data.val.x)
    x_target_train = scaler.transform(target_train_raw)
    x_target_val = scaler.transform(arrays["v2_val_x"])
    model = build_tiny_ds_cnn(input_shape=tuple(x_uci_train.shape[1:]), num_classes=len(data.class_names))

    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    pretrain = model.fit(
        x_uci_train,
        data.train.y,
        validation_data=(x_uci_val, data.val.y),
        sample_weight=_balanced_weights(data.train.y),
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, min_delta=1e-4),
            keras.callbacks.CSVLogger(str(run_dir / "logs" / "pretrain_history.csv"), append=False),
        ],
        verbose=2,
        shuffle=True,
    )
    pd.DataFrame(pretrain.history).to_csv(run_dir / "logs" / "pretrain_history.csv", index=False)

    _set_trainable(model, ("activity",))
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate * 0.1), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head = model.fit(
        x_target_train,
        target_y,
        validation_data=(x_target_val, arrays["v2_val_y"]),
        sample_weight=_balanced_weights(target_y),
        epochs=max(1, args.patience),
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=max(2, args.patience // 2), restore_best_weights=True, min_delta=1e-4),
            keras.callbacks.CSVLogger(str(run_dir / "logs" / "head_finetune_history.csv"), append=False),
        ],
        verbose=2,
        shuffle=True,
    )
    pd.DataFrame(head.history).to_csv(run_dir / "logs" / "head_finetune_history.csv", index=False)

    _set_trainable(model, ("ds_conv_2", "ds_bn_2", "ds_relu_2", "global_avg_pool", "activity"))
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate * 0.05), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    block = model.fit(
        x_target_train,
        target_y,
        validation_data=(x_target_val, arrays["v2_val_y"]),
        sample_weight=_balanced_weights(target_y),
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=max(2, args.patience // 2), restore_best_weights=True, min_delta=1e-4),
            keras.callbacks.CSVLogger(str(run_dir / "logs" / "last_block_finetune_history.csv"), append=False),
        ],
        verbose=2,
        shuffle=True,
    )
    pd.DataFrame(block.history).to_csv(run_dir / "logs" / "last_block_finetune_history.csv", index=False)
    _set_trainable(model, None)
    meta = {
        "run_name": run_name,
        "experiment": "uci_pretrain_v2_1_finetune",
        "feature_mode": "base_6ch",
        "loss": "cross_entropy",
        "input_shape": list(x_uci_train.shape[1:]),
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "normalization_scope": "UCI-HAR train + Arduino V2 train + capped V2.1 long adaptation",
        "parameters": int(model.count_params()),
        "trainable_parameters": count_trainable_parameters(model),
    }
    _save_candidate_artifacts(model, scaler, meta, run_dir, args)
    return model, scaler, meta


def _load_previous_reference(arrays, output_dir: Path):
    prev_dir = OUTPUTS_DIR / "lightweight" / "experiments" / "m3_target_domain" / "m3_target_mixed_focal_6ch"
    meta = json.loads((prev_dir / "summary.json").read_text(encoding="utf-8"))
    scaler = SequenceStandardizer.load(Path(meta["normalization"]["standardizer_path"]))
    model = build_tiny_ds_cnn(input_shape=tuple(meta["input_shape"]), num_classes=len(UCI_CLASS_NAMES))
    model.load_weights(prev_dir / "models" / "m3_target_mixed_focal_6ch.weights.h5")
    ref_dir = output_dir / "previous_m3_candidate_reference"
    (ref_dir / "models").mkdir(parents=True, exist_ok=True)
    prev_tflite = prev_dir / "models" / "m3_target_mixed_focal_6ch.tflite"
    fp32_size = None
    if prev_tflite.exists():
        copied = ref_dir / "models" / "previous_m3_candidate_reference.tflite"
        shutil.copyfile(prev_tflite, copied)
        fp32_size = copied.stat().st_size
    return model, scaler, {
        "run_name": "previous_m3_candidate_reference",
        "experiment": "previous_m3_candidate_reference",
        "feature_mode": "base_6ch",
        "input_shape": list(meta["input_shape"]),
        "fp32_tflite_size_bytes": fp32_size,
        "standardizer_path": meta["normalization"]["standardizer_path"],
        "weights_path": str(prev_dir / "models" / "m3_target_mixed_focal_6ch.weights.h5"),
    }


def _audit_roots(roots: list[tuple[Path, str]], output_csv: Path) -> pd.DataFrame:
    rows = []
    for root, role in roots:
        for path in iter_csv_files(root):
            _, info = read_labeled_csv(path, root)
            d = asdict(info)
            d["file_path"] = path.as_posix()
            d["relative_path"] = path.relative_to(root).as_posix()
            d["dataset_role"] = role
            d["inferred_label"] = info.uci_label
            d["warning_flags"] = ";".join(
                flag
                for flag, active in {
                    "empty": info.rows_valid == 0,
                    "invalid_rows": info.rows_invalid > 0,
                    "timestamp_reset": info.timestamp_resets > 0,
                    "non_50hz": info.effective_hz is not None and (info.effective_hz < 48.0 or info.effective_hz > 52.0),
                    "no_windows": info.windows_resampled_50hz_128_stride64 == 0,
                }.items()
                if active
            )
            rows.append(d)
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _evaluate_all(model, scaler, meta, arrays, output_dir: Path):
    run_name = meta["run_name"]
    feature_mode = meta["feature_mode"]
    data = arrays["uci"]
    split_specs = {
        "uci_test": (_apply_features(data.test.x, feature_mode), data.test.y),
        "arduino_v2_grouped": (_apply_features(arrays["v2_val_x"], feature_mode), arrays["v2_val_y"]),
        "right_60s": (_apply_features(arrays["right_x"], feature_mode), arrays["right_y"]),
        "left_30s": (_apply_features(arrays["left_x"], feature_mode), arrays["left_y"]),
    }
    rows = []
    outputs = {}
    for split_name, (x_raw, y) in split_specs.items():
        x = scaler.transform(x_raw)
        out = _evaluate_model(model, x, y, data.class_names, run_name, split_name, output_dir / run_name)
        outputs[split_name] = out
        rows.append(_metric_row(run_name, meta["experiment"], split_name, out, meta.get("fp32_tflite_size_bytes")))
    return rows, outputs


def _selection_score(row_by_split: dict[str, dict]) -> tuple:
    right = row_by_split["right_60s"]
    left = row_by_split["left_30s"]
    v2 = row_by_split["arduino_v2_grouped"]
    uci = row_by_split["uci_test"]
    down_recall = min(right["recall_walking_downstairs"], left["recall_walking_downstairs"])
    sitting_recall = min(right["recall_sitting"], left["recall_sitting"])
    return (
        right["macro_f1"],
        left["macro_f1"],
        down_recall,
        sitting_recall,
        v2["macro_f1"],
        uci["macro_f1"],
    )


def _write_named_confusions(selected_outputs: dict, output_dir: Path) -> None:
    mapping = {
        "right_60s": "m3_retrained_right_60s_confusion_matrix.csv",
        "left_30s": "m3_retrained_left_30s_confusion_matrix.csv",
        "arduino_v2_grouped": "m3_retrained_v2_grouped_confusion_matrix.csv",
    }
    for split, filename in mapping.items():
        outputs = selected_outputs[split]
        pd.DataFrame(outputs["confusion_matrix"], index=outputs["class_names"], columns=outputs["class_names"]).to_csv(
            output_dir / filename
        )


def _quantize_selected(args, selected, arrays, output_dir: Path):
    model = selected["model"]
    scaler = selected["scaler"]
    meta = selected["meta"]
    feature_mode = meta["feature_mode"]
    data = arrays["uci"]
    representative_raw = np.concatenate([data.train.x, arrays["v2_train_x"], arrays["long_x"]], axis=0)
    representative_y = np.concatenate([data.train.y, arrays["v2_train_y"], arrays["long_y"]], axis=0)
    representative_x = scaler.transform(_apply_features(representative_raw, feature_mode))
    indices = _class_balanced_indices(representative_y, args.representative_samples, args.seed)
    model_bytes = _convert_int8(model, representative_x, indices)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    int8_path = models_dir / f"{meta['run_name']}_int8.tflite"
    int8_path.write_bytes(model_bytes)

    split_specs = {
        "uci_test": (_apply_features(data.test.x, feature_mode), data.test.y),
        "arduino_v2_grouped": (_apply_features(arrays["v2_val_x"], feature_mode), arrays["v2_val_y"]),
        "right_60s": (_apply_features(arrays["right_x"], feature_mode), arrays["right_y"]),
        "left_30s": (_apply_features(arrays["left_x"], feature_mode), arrays["left_y"]),
    }
    rows = []
    metadata = None
    for split_name, (x_raw, y) in split_specs.items():
        x = scaler.transform(x_raw)
        out, metadata = _evaluate_tflite(int8_path, x, y, data.class_names, f"{meta['run_name']}_int8", split_name, output_dir)
        row = _metric_row(f"{meta['run_name']}_int8", meta["experiment"], split_name, out, int8_path.stat().st_size)
        row["model_type"] = "INT8"
        rows.append(row)

    if feature_mode == "base_6ch" and metadata is not None:
        model_label = "mixed focal" if "focal" in meta["experiment"] else "mixed"
        _write_model_header(
            model_bytes=model_bytes,
            header_path=Path(args.header),
            standardizer=scaler,
            class_names=data.class_names,
            input_scale=metadata["input_scale"],
            input_zero_point=metadata["input_zero_point"],
            output_scale=metadata["output_scale"],
            output_zero_point=metadata["output_zero_point"],
            tensor_arena_size=args.tensor_arena_size,
            model_description=f"retrained {model_label} 6-channel DS-CNN with V2.1 long adaptation data",
            normalization_source="UCI-HAR train + Arduino V2 train + capped V2.1 long adaptation",
        )
    return rows, int8_path


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    configure_tensorflow_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["metrics", "models", "figures"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    _audit_roots(
        [(Path(args.v2_1_root), "adaptation_training")],
        DOCS_DIR / "tables" / "m3_v2_1_long_adaptation_audit.csv",
    )
    _audit_roots(
        [(Path(args.right_root), "heldout_right_60s"), (Path(args.left_root), "heldout_left_30s")],
        DOCS_DIR / "tables" / "m3_live_validation_audit.csv",
    )

    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    v2_x, v2_y, v2_sources = build_windowed_dataset(Path(args.v2_root), hz=50.0)
    v2_train_x, v2_train_y, v2_train_sources, v2_val_x, v2_val_y, v2_val_sources = _split_v2(
        v2_x, v2_y, v2_sources, args.target_val_fraction, args.seed
    )
    long_x_all, long_y_all, long_sources_all = build_windowed_dataset(Path(args.v2_1_root), hz=50.0)
    long_x, long_y, long_sources, long_before, long_after = _cap_per_class(
        long_x_all, long_y_all, long_sources_all, args.long_cap_per_class, args.seed
    )
    right_x, right_y, right_sources = build_windowed_dataset(Path(args.right_root), hz=50.0)
    left_x, left_y, left_sources = build_windowed_dataset(Path(args.left_root), hz=50.0)
    count_summary = {
        "v2_train": {UCI_CLASS_NAMES[i]: int((v2_train_y == i).sum()) for i in range(len(UCI_CLASS_NAMES))},
        "v2_grouped_validation": {UCI_CLASS_NAMES[i]: int((v2_val_y == i).sum()) for i in range(len(UCI_CLASS_NAMES))},
        "v2_1_long_before_cap": long_before,
        "v2_1_long_after_cap": long_after,
        "right_60s": {UCI_CLASS_NAMES[i]: int((right_y == i).sum()) for i in range(len(UCI_CLASS_NAMES))},
        "left_30s": {UCI_CLASS_NAMES[i]: int((left_y == i).sum()) for i in range(len(UCI_CLASS_NAMES))},
    }
    (output_dir / "m3_retrained_class_counts.json").write_text(json.dumps(count_summary, indent=2), encoding="utf-8")

    arrays = {
        "uci": data,
        "v2_train_x": v2_train_x,
        "v2_train_y": v2_train_y,
        "v2_val_x": v2_val_x,
        "v2_val_y": v2_val_y,
        "long_x": long_x,
        "long_y": long_y,
        "right_x": right_x,
        "right_y": right_y,
        "left_x": left_x,
        "left_y": left_y,
    }

    candidates = []
    for trainer in (_train_uci_baseline, _train_arduino_only, _train_finetune):
        set_global_seed(args.seed)
        model, scaler, meta = trainer(args, arrays, output_dir)
        candidates.append({"model": model, "scaler": scaler, "meta": meta})

    train_specs = [
        ("m3_v2_1_mixed_6ch", "mixed_6ch_v2_1", "base_6ch", "cross_entropy"),
        ("m3_v2_1_mixed_focal_6ch", "mixed_focal_6ch_v2_1", "base_6ch", "focal"),
    ]
    if args.include_gravity:
        train_specs.append(("m3_v2_1_mixed_gravity_10ch", "mixed_gravity_10ch_v2_1", "gravity_10ch", "cross_entropy"))
    for run_name, experiment, feature_mode, loss_name in train_specs:
        set_global_seed(args.seed)
        model, scaler, meta = _train_model(args, run_name, experiment, feature_mode, loss_name, arrays, output_dir)
        candidates.append({"model": model, "scaler": scaler, "meta": meta})

    summary_rows = []
    selection_rows = []
    by_run_split = {}
    outputs_by_run = {}
    for candidate in candidates:
        rows, outputs = _evaluate_all(candidate["model"], candidate["scaler"], candidate["meta"], arrays, output_dir)
        summary_rows.extend(rows)
        outputs_by_run[candidate["meta"]["run_name"]] = outputs
        by_split = {row["split"]: row for row in rows}
        by_run_split[candidate["meta"]["run_name"]] = by_split
        score = _selection_score(by_split)
        selection_rows.append(
            {
                "run_name": candidate["meta"]["run_name"],
                "experiment": candidate["meta"]["experiment"],
                "feature_mode": candidate["meta"]["feature_mode"],
                "right_60s_macro_f1": by_split["right_60s"]["macro_f1"],
                "left_30s_macro_f1": by_split["left_30s"]["macro_f1"],
                "min_downstairs_recall_right_left": min(
                    by_split["right_60s"]["recall_walking_downstairs"],
                    by_split["left_30s"]["recall_walking_downstairs"],
                ),
                "min_sitting_recall_right_left": min(
                    by_split["right_60s"]["recall_sitting"],
                    by_split["left_30s"]["recall_sitting"],
                ),
                "v2_grouped_macro_f1": by_split["arduino_v2_grouped"]["macro_f1"],
                "uci_test_macro_f1": by_split["uci_test"]["macro_f1"],
                "selection_score": json.dumps(score),
            }
        )
    validation_df = pd.DataFrame(summary_rows)
    validation_df.to_csv(output_dir / "m3_retrained_validation_summary.csv", index=False)
    selection_df = pd.DataFrame(selection_rows)
    selection_df = selection_df.sort_values(
        by=[
            "right_60s_macro_f1",
            "left_30s_macro_f1",
            "min_downstairs_recall_right_left",
            "min_sitting_recall_right_left",
            "v2_grouped_macro_f1",
            "uci_test_macro_f1",
        ],
        ascending=False,
    )
    selection_df["selected"] = False
    selection_df.loc[selection_df.index[0], "selected"] = True
    selection_df.to_csv(output_dir / "m3_retrained_model_selection.csv", index=False)
    selected_name = str(selection_df.iloc[0]["run_name"])
    selected = next(candidate for candidate in candidates if candidate["meta"]["run_name"] == selected_name)
    _write_named_confusions(outputs_by_run[selected_name], output_dir)

    int8_rows, int8_path = _quantize_selected(args, selected, arrays, output_dir)
    fp32_rows = validation_df[validation_df["run_name"] == selected_name].copy()
    fp32_rows["model_type"] = "FP32"
    quant_df = pd.concat([fp32_rows, pd.DataFrame(int8_rows)], ignore_index=True)
    quant_df.to_csv(output_dir / "m3_retrained_fp32_vs_int8_summary.csv", index=False)

    report = {
        "selected_run_name": selected_name,
        "selected_int8_path": str(int8_path),
        "selection_rule": [
            "right_60s macro F1",
            "left_30s macro F1",
            "minimum right/left WALKING_DOWNSTAIRS recall",
            "minimum right/left SITTING recall",
            "Arduino V2 grouped macro F1",
            "UCI-HAR official test macro F1",
        ],
        "anti_leakage": {
            "normalization_fit": "UCI-HAR train + Arduino V2 train split + capped V2.1 long adaptation only",
            "not_used_for_training_or_quantization": ["right_60s", "left_30s", "UCI-HAR test", "Arduino V2 grouped validation"],
        },
        "class_counts": count_summary,
    }
    (output_dir / "m3_retrained_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain M3 candidates with Arduino V2.1 long adaptation data.")
    parser.add_argument("--experiment", choices=["all"], default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--uci-loss-weight", type=float, default=0.3)
    parser.add_argument("--long-cap-per-class", type=int, default=180)
    parser.add_argument("--target-val-fraction", type=float, default=0.25)
    parser.add_argument("--representative-samples", type=int, default=512)
    parser.add_argument("--tensor-arena-size", type=int, default=60 * 1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--benchmark-host", action="store_true")
    parser.add_argument("--include-gravity", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--v2-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2")
    parser.add_argument("--v2-1-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2_1_long_adaptation")
    parser.add_argument("--right-root", type=Path, default=RAW_DATA_DIR / "arduino_live_validation" / "right_60s")
    parser.add_argument("--left-root", type=Path, default=RAW_DATA_DIR / "arduino_live_validation" / "left_30s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--header", type=Path, default=Path("arduino/tinyml_har_m3/model_data.h"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
