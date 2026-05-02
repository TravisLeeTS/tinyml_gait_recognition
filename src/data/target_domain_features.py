from __future__ import annotations

import numpy as np


BASE_IMU_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
GRAVITY_FEATURE_CHANNELS = ["gravity_dir_x", "gravity_dir_y", "gravity_dir_z", "acc_magnitude"]
TEN_CHANNEL_FEATURES = [*BASE_IMU_CHANNELS, *GRAVITY_FEATURE_CHANNELS]


def gravity_feature_matrix(x_raw: np.ndarray) -> np.ndarray:
    """Return per-window mean acceleration direction and magnitude.

    Input is expected as [windows, timesteps, 6] with acceleration in the first
    three channels. The four returned features are constant per window so they
    can be repeated across all timesteps for Conv1D models and Arduino TFLM.
    """
    if x_raw.ndim != 3 or x_raw.shape[-1] < 3:
        raise ValueError(f"Expected [windows, timesteps, channels>=3], got {x_raw.shape}")
    mean_acc = x_raw[:, :, :3].mean(axis=1).astype(np.float32)
    magnitude = np.linalg.norm(mean_acc, axis=1, keepdims=True).astype(np.float32)
    direction = mean_acc / np.maximum(magnitude, np.float32(1e-6))
    return np.concatenate([direction, magnitude], axis=1).astype(np.float32)


def append_gravity_features(x_raw: np.ndarray) -> np.ndarray:
    """Append repeated gravity direction/magnitude features to raw IMU windows."""
    features = gravity_feature_matrix(x_raw)
    repeated = np.repeat(features[:, None, :], x_raw.shape[1], axis=1)
    return np.concatenate([x_raw.astype(np.float32), repeated.astype(np.float32)], axis=-1)


def orientation_summary(x_raw: np.ndarray, y: np.ndarray, class_names: list[str]) -> list[dict]:
    """Summarize per-class mean acceleration direction for domain diagnostics."""
    features = gravity_feature_matrix(x_raw)
    rows = []
    for class_id, class_name in enumerate(class_names):
        mask = y == class_id
        if not np.any(mask):
            continue
        class_features = features[mask]
        rows.append(
            {
                "class": class_name,
                "windows": int(mask.sum()),
                "gravity_dir_x_mean": float(class_features[:, 0].mean()),
                "gravity_dir_y_mean": float(class_features[:, 1].mean()),
                "gravity_dir_z_mean": float(class_features[:, 2].mean()),
                "acc_magnitude_mean": float(class_features[:, 3].mean()),
                "acc_magnitude_std": float(class_features[:, 3].std()),
            }
        )
    return rows
