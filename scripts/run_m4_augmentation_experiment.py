from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a controlled training-only augmentation dataset. This script does not train a model; "
            "use its output with the same train/validation/test split to compare augmentation fairly."
        )
    )
    parser.add_argument("--input-npz", required=True, type=Path, help="NPZ containing x_train, y_train, x_val, y_val, x_test, y_test.")
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--copies", default=1, type=int, help="Augmented copies per original training window.")
    parser.add_argument("--noise-std", default=0.02, type=float)
    parser.add_argument("--scale-min", default=0.95, type=float)
    parser.add_argument("--scale-max", default=1.05, type=float)
    parser.add_argument("--max-shift", default=4, type=int, help="Maximum circular time shift in samples.")
    return parser.parse_args()


def augment_windows(x: np.ndarray, rng: np.random.Generator, args: argparse.Namespace) -> np.ndarray:
    augmented = x.astype(np.float32).copy()
    scale = rng.uniform(args.scale_min, args.scale_max, size=(len(x), 1, x.shape[2])).astype(np.float32)
    augmented *= scale
    augmented += rng.normal(0.0, args.noise_std, size=augmented.shape).astype(np.float32)
    if args.max_shift > 0:
        shifts = rng.integers(-args.max_shift, args.max_shift + 1, size=len(x))
        for idx, shift in enumerate(shifts):
            augmented[idx] = np.roll(augmented[idx], shift=shift, axis=0)
    return augmented


def main() -> None:
    args = parse_args()
    if not args.input_npz.exists():
        raise FileNotFoundError(f"Input NPZ does not exist: {args.input_npz}")
    data = np.load(args.input_npz)
    required = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Input NPZ missing arrays: {missing}")

    x_train = data["x_train"].astype(np.float32)
    y_train = data["y_train"]
    rng = np.random.default_rng(args.seed)
    augmented_x = [x_train]
    augmented_y = [y_train]
    for _ in range(args.copies):
        augmented_x.append(augment_windows(x_train, rng, args))
        augmented_y.append(y_train.copy())

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        x_train=np.concatenate(augmented_x, axis=0),
        y_train=np.concatenate(augmented_y, axis=0),
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
    )
    metadata = {
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "guardrail": "Only x_train/y_train were augmented. Validation and test arrays were copied unchanged.",
        "copies": args.copies,
        "noise_std": args.noise_std,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "max_shift": args.max_shift,
        "original_train_windows": int(len(x_train)),
        "augmented_train_windows": int(len(x_train) * (args.copies + 1)),
    }
    args.output_npz.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
