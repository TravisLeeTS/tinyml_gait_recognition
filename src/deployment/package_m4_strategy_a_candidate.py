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
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import set_global_seed


DEFAULT_RUN_DIR = OUTPUTS_DIR / "m4_strategy_a_experiment" / "m4_strategy_a_expanded_train_aug"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "m4_strategy_a_deployment_candidate"
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
            "Package the M4 Strategy A augmented offline candidate as an INT8 Arduino header. "
            "The standardized dataset remains external validation only and is never used for "
            "representative quantization."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--standardized-input", type=Path, required=True, help="Standardized HAR v2 zip or extracted folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--v2-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2")
    parser.add_argument("--v2-1-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2_1_long_adaptation")
    parser.add_argument("--right-root", type=Path, default=RAW_DATA_DIR / "arduino_live_validation" / "right_60s")
    parser.add_argument("--left-root", type=Path, default=RAW_DATA_DIR / "arduino_live_validation" / "left_30s")
    parser.add_argument("--representative-samples", type=int, default=512)
    parser.add_argument("--tensor-arena-size", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def extract_if_needed(input_path: Path, temp_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file or extracted folder, got {input_path}")
    with zipfile.ZipFile(input_path) as archive:
        archive.extractall(temp_dir)
    children = [child for child in temp_dir.iterdir() if child.is_dir()]
    return children[0] if len(children) == 1 else temp_dir


def _load_training_representative_raw(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    parts_x = [data.train.x]
    parts_y = [data.train.y]

    for root in [args.v2_root, args.v2_1_root, args.right_root, args.left_root]:
        x_part, y_part, _ = build_windowed_dataset(root, hz=50.0)
        parts_x.append(x_part)
        parts_y.append(y_part)

    return np.concatenate(parts_x, axis=0).astype(np.float32), np.concatenate(parts_y, axis=0).astype(np.int64)


def _evaluate_tflite_external(
    *,
    model_path: Path,
    x_external: np.ndarray,
    y_external: np.ndarray,
    output_dir: Path,
    run_name: str,
) -> tuple[dict, dict]:
    probs, metadata = run_tflite(model_path, x_external)
    y_pred = probs.argmax(axis=1)
    outputs = classification_outputs(y_external, y_pred, UCI_CLASS_NAMES, f"{run_name}_standardized_external_int8")
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"], dtype=int), UCI_CLASS_NAMES)
    outputs["model_info"] = {
        "tflite_path": repo_display_path(model_path),
        "tflite_size_bytes": model_path.stat().st_size,
        "tflite_metadata": metadata,
        "host_tflite_latency": benchmark_tflite_ms(model_path, x_external[0], runs=50, warmup=5),
        "device_note": "This is host TFLite latency. Arduino latency must be remeasured after flashing.",
    }
    save_classification_outputs(outputs, output_dir / "metrics", output_dir / "figures", f"{run_name}_standardized_external_int8")
    return outputs, metadata


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    run_dir = args.run_dir
    summary_path = run_dir / "summary.json"
    weights_path = run_dir / "models" / f"{run_dir.name}.weights.h5"
    standardizer_path = run_dir / "models" / f"{run_dir.name}_standardizer.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing M4 run summary: {summary_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing M4 model weights: {weights_path}")
    if not standardizer_path.exists():
        raise FileNotFoundError(f"Missing M4 standardizer: {standardizer_path}")

    output_dir = args.output_dir
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    for path in [models_dir, metrics_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    standardizer = SequenceStandardizer.load(standardizer_path)
    train_raw, train_y = _load_training_representative_raw(args)
    representative_x = standardizer.transform(train_raw)
    representative_indices = _class_balanced_indices(train_y, args.representative_samples, args.seed)

    model = build_tiny_ds_cnn(input_shape=(128, 6), num_classes=len(UCI_CLASS_NAMES))
    model.load_weights(weights_path)
    int8_bytes = _convert_int8(model, representative_x, representative_indices)
    int8_path = models_dir / "m4_strategy_a_aug_int8.tflite"
    int8_path.write_bytes(int8_bytes)

    with tempfile.TemporaryDirectory(prefix="m4_standardized_eval_") as temp_name:
        standardized_root = extract_if_needed(args.standardized_input.expanduser().resolve(), Path(temp_name))
        external_raw, external_y, external_sources = build_windowed_dataset(standardized_root, hz=50.0)

    external_x = standardizer.transform(external_raw)
    external_outputs, tflite_metadata = _evaluate_tflite_external(
        model_path=int8_path,
        x_external=external_x,
        y_external=external_y,
        output_dir=output_dir,
        run_name="m4_strategy_a_aug",
    )

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
        model_description="M4 Strategy A augmented 6-channel DS-CNN, INT8 PTQ packaged for Arduino re-test",
        normalization_source=(
            "UCI-HAR train + Arduino V2/V2.1 + old M3 raw replay validation; "
            "standardized HAR v2 and live Serial logs excluded"
        ),
    )

    metadata = {
        "candidate_name": "m4_strategy_a_aug_int8",
        "candidate_status": (
            "Latest packaged Arduino candidate. It improves offline standardized external validation, "
            "but it is not a new live-result claim until flashed and rescored on hardware."
        ),
        "source_run_dir": repo_display_path(run_dir),
        "source_weights": repo_display_path(weights_path),
        "selected_int8_tflite": repo_display_path(int8_path),
        "arduino_header": repo_display_path(args.header),
        "standardized_input_name": args.standardized_input.name,
        "standardized_external_sources": int(len(set(external_sources))),
        "standardized_external_windows": int(len(external_raw)),
        "standardized_external_int8_accuracy": external_outputs["accuracy"],
        "standardized_external_int8_macro_f1": external_outputs["macro_f1"],
        "standardized_external_top_confusions": external_outputs["top_confusion_pairs"][:5],
        "tflite_size_bytes": int8_path.stat().st_size,
        "representative_dataset": {
            "policy": "class-balanced samples from training-side windows only",
            "samples": int(len(representative_indices)),
            "excluded": ["standardized HAR v2 external validation", "returned live Serial logs", "UCI-HAR official test"],
        },
        "source_offline_summary": {
            "standardized_external_accuracy": summary.get("standardized_external_accuracy"),
            "standardized_external_macro_f1": summary.get("standardized_external_macro_f1"),
            "augmentation": summary.get("augmentation"),
        },
        "tflite_metadata": tflite_metadata,
    }
    (output_dir / "m4_strategy_a_candidate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
