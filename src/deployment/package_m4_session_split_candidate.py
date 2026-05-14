from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import OUTPUTS_DIR, RAW_DATA_DIR, SEED
from src.data.arduino_collectdata import UCI_CLASS_NAMES, build_windowed_dataset
from src.data.uci_har import load_uci_har_sequence_dataset
from src.deployment.convert_lightweight_to_tflite_micro import benchmark_tflite_ms, run_tflite
from src.deployment.package_m3_candidate import _class_balanced_indices, _convert_int8, _write_model_header
from src.models.lightweight.tiny_cnn import build_tiny_ds_cnn
from src.training.train_m4_session_split_experiment import (
    SESSION_1,
    SESSION_3,
    SESSION_4,
    apply_feature_mode,
    extract_if_needed,
    source_mask,
)
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import set_global_seed


DEFAULT_RUN_DIR = (
    OUTPUTS_DIR
    / "m4_session_split_with_session3_experiment"
    / "session4_train_session1_eval_gravity_10ch"
)
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "m4_live_handoff_candidate"
DEFAULT_HEADER = Path("arduino/tinyml_har_m3/model_data.h")


def repo_display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the selected M4 session3+4 model as an INT8 Arduino live-test "
            "candidate. Standardized session 1 remains held out for reported "
            "right/left robustness metrics and is excluded from quantization."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--standardized-input", type=Path, required=True, help="Standardized HAR v2 zip or extracted folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--feature-mode", default="gravity_10ch", choices=["base_6ch", "gravity_aligned_6ch", "gravity_10ch"])
    parser.add_argument("--v2-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2")
    parser.add_argument("--v2-1-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2_1_long_adaptation")
    parser.add_argument("--right-root", type=Path, default=RAW_DATA_DIR / "arduino_live_validation" / "right_60s")
    parser.add_argument("--left-root", type=Path, default=RAW_DATA_DIR / "arduino_live_validation" / "left_30s")
    parser.add_argument("--representative-samples", type=int, default=512)
    parser.add_argument("--tensor-arena-size", type=int, default=72 * 1024)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _load_standardized(input_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="m4_package_sessions_") as temp_name:
        root = extract_if_needed(input_path.expanduser().resolve(), Path(temp_name))
        return build_windowed_dataset(root, hz=50.0)


def _training_raw_with_sessions(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    train_parts = [data.train.x]
    y_parts = [data.train.y]

    for root in [args.v2_root, args.v2_1_root, args.right_root, args.left_root]:
        x_part, y_part, _ = build_windowed_dataset(root, hz=50.0)
        train_parts.append(x_part)
        y_parts.append(y_part)

    standardized_x, standardized_y, standardized_sources = _load_standardized(args.standardized_input)
    train_mask = source_mask(standardized_sources, session=SESSION_4) | source_mask(standardized_sources, session=SESSION_3)
    right_mask = source_mask(standardized_sources, session=SESSION_1, placement="right_pocket")
    left_mask = source_mask(standardized_sources, session=SESSION_1, placement="left_pocket")
    if not np.any(train_mask) or not np.any(right_mask) or not np.any(left_mask):
        raise ValueError("Expected non-empty session 3/4 train and session 1 right/left holdout windows.")

    train_parts.append(standardized_x[train_mask])
    y_parts.append(standardized_y[train_mask])
    train_raw = np.concatenate(train_parts, axis=0).astype(np.float32)
    train_y = np.concatenate(y_parts, axis=0).astype(np.int64)
    return train_raw, train_y, standardized_x[right_mask], standardized_y[right_mask], standardized_x[left_mask], standardized_y[left_mask]


def _evaluate_tflite_split(model_path: Path, x: np.ndarray, y: np.ndarray, output_dir: Path, run_name: str, split_name: str) -> dict:
    probs, metadata = run_tflite(model_path, x)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y, y_pred, UCI_CLASS_NAMES, f"{run_name}_{split_name}_int8")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), UCI_CLASS_NAMES)
    outputs["model_info"] = {
        "tflite_path": repo_display_path(model_path),
        "tflite_size_bytes": model_path.stat().st_size,
        "tflite_metadata": metadata,
        "host_tflite_latency": benchmark_tflite_ms(model_path, x[0], runs=50, warmup=5),
        "device_note": "Host TFLite latency is not an Arduino latency claim.",
    }
    save_classification_outputs(outputs, output_dir / "metrics", output_dir / "figures", f"{run_name}_{split_name}_int8")
    return outputs


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    run_dir = args.run_dir
    summary_path = run_dir / "summary.json"
    weights_path = run_dir / "models" / f"{run_dir.name}.weights.h5"
    standardizer_path = run_dir / "models" / f"{run_dir.name}_standardizer.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    if not standardizer_path.exists():
        raise FileNotFoundError(f"Missing standardizer: {standardizer_path}")

    output_dir = args.output_dir
    models_dir = output_dir / "models"
    for path in [models_dir, output_dir / "metrics", output_dir / "figures"]:
        path.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("feature_mode") != args.feature_mode:
        raise ValueError(
            f"Run feature mode {source_summary.get('feature_mode')} does not match --feature-mode {args.feature_mode}."
        )

    standardizer = SequenceStandardizer.load(standardizer_path)
    train_raw_base, train_y, right_raw_base, right_y, left_raw_base, left_y = _training_raw_with_sessions(args)
    train_raw = apply_feature_mode(train_raw_base, args.feature_mode)
    right_raw = apply_feature_mode(right_raw_base, args.feature_mode)
    left_raw = apply_feature_mode(left_raw_base, args.feature_mode)

    representative_x = standardizer.transform(train_raw)
    representative_indices = _class_balanced_indices(train_y, args.representative_samples, args.seed)

    model = build_tiny_ds_cnn(input_shape=tuple(representative_x.shape[1:]), num_classes=len(UCI_CLASS_NAMES))
    model.load_weights(weights_path)
    int8_bytes = _convert_int8(model, representative_x, representative_indices)

    candidate_name = f"m4_session3_4_{args.feature_mode}_int8"
    int8_path = models_dir / f"{candidate_name}.tflite"
    int8_path.write_bytes(int8_bytes)

    right_x = standardizer.transform(right_raw)
    left_x = standardizer.transform(left_raw)
    right_outputs = _evaluate_tflite_split(int8_path, right_x, right_y, output_dir, candidate_name, "session1_right_pocket")
    left_outputs = _evaluate_tflite_split(int8_path, left_x, left_y, output_dir, candidate_name, "session1_left_pocket")
    _, tflite_metadata = run_tflite(int8_path, right_x[:1])

    _write_model_header(
        model_bytes=int8_bytes,
        header_path=args.header,
        standardizer=standardizer,
        class_names=UCI_CLASS_NAMES,
        input_scale=tflite_metadata["input_scale"],
        input_zero_point=tflite_metadata["input_zero_point"],
        output_scale=tflite_metadata["output_scale"],
        output_zero_point=tflite_metadata["output_zero_point"],
        tensor_arena_size=args.tensor_arena_size,
        channel_count=int(representative_x.shape[-1]),
        model_description="M4 session3+4 gravity/orientation 10-channel DS-CNN, INT8 PTQ live-test candidate",
        normalization_source=(
            "UCI-HAR train + Arduino V2/V2.1 + old M3 raw replay validation + "
            "standardized sessions 3 and 4; standardized session 1 and live Serial logs excluded"
        ),
    )

    metadata = {
        "candidate_name": candidate_name,
        "feature_mode": args.feature_mode,
        "candidate_status": "Arduino live-test candidate; not a new live claim until flashed and rescored.",
        "selection_reason": (
            "Best average offline macro-F1 across session-1 right and left pocket holdouts among "
            "the session3+4 experiments."
        ),
        "source_run_dir": repo_display_path(run_dir),
        "source_weights": repo_display_path(weights_path),
        "selected_int8_tflite": repo_display_path(int8_path),
        "arduino_header": repo_display_path(args.header),
        "tflite_size_bytes": int8_path.stat().st_size,
        "channel_count": int(representative_x.shape[-1]),
        "standardized_training_sessions": [SESSION_3, SESSION_4],
        "standardized_holdout_session": SESSION_1,
        "leakage_guardrail": (
            "No standardized session 1 windows are used for training, validation, augmentation, "
            "representative quantization, or model selection inside this packager."
        ),
        "important_caveat": (
            "The holdout is session-level, not a strict unseen-person result. High offline scores "
            "must be confirmed with new live Arduino logs."
        ),
        "session1_right_int8_accuracy": right_outputs["accuracy"],
        "session1_right_int8_macro_f1": right_outputs["macro_f1"],
        "session1_left_int8_accuracy": left_outputs["accuracy"],
        "session1_left_int8_macro_f1": left_outputs["macro_f1"],
        "session1_right_top_confusions": right_outputs["top_confusion_pairs"][:5],
        "session1_left_top_confusions": left_outputs["top_confusion_pairs"][:5],
        "representative_dataset": {
            "policy": "class-balanced samples from training-side windows only",
            "samples": int(len(representative_indices)),
            "excluded": ["standardized session 1 holdout", "returned live Serial logs", "UCI-HAR official test"],
        },
        "source_fp32_summary": source_summary,
        "tflite_metadata": tflite_metadata,
    }
    (output_dir / "m4_live_candidate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
