from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    RAW_DATA_DIR,
    SEED,
    WISDM_CLASSIC_CLASSES,
    WISDM_CLASSIC_DIR,
    WISDM_CLASSIC_SAMPLING_HZ,
    WISDM_CLASSIC_TGZ,
    WISDM_CLASSIC_URL,
)
from src.utils.download import download_first_available, extract_archive


@dataclass
class WisdmWindows:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    class_names: list[str]


def ensure_wisdm_classic(download: bool = True) -> Path:
    if WISDM_CLASSIC_DIR.exists():
        return WISDM_CLASSIC_DIR
    if not download:
        raise FileNotFoundError(f"WISDM classic dataset not found at {WISDM_CLASSIC_DIR}")
    used_url = download_first_available([WISDM_CLASSIC_URL], WISDM_CLASSIC_TGZ)
    extract_archive(WISDM_CLASSIC_TGZ, RAW_DATA_DIR)
    if not WISDM_CLASSIC_DIR.exists():
        raise FileNotFoundError(f"Archive extracted but {WISDM_CLASSIC_DIR} was not created")
    (RAW_DATA_DIR / "wisdm_classic_download_source.txt").write_text(used_url, encoding="utf-8")
    return WISDM_CLASSIC_DIR


def parse_wisdm_classic_raw(root: Path | None = None) -> pd.DataFrame:
    root = root or ensure_wisdm_classic(download=True)
    raw_path = root / "WISDM_ar_v1.1_raw.txt"
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    records: list[list[str]] = []
    bad_records: list[str] = []
    for record in text.split(";"):
        record = record.strip()
        if not record:
            continue
        if record.endswith(","):
            record = record[:-1]
        parts = [part.strip() for part in record.split(",")]
        if len(parts) != 6:
            bad_records.append(record[:120])
            continue
        records.append(parts)
    df = pd.DataFrame(records, columns=["subject", "activity", "timestamp", "x", "y", "z"])
    df["subject"] = pd.to_numeric(df["subject"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in ["x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().copy()
    df["subject"] = df["subject"].astype(int)
    # WISDM about file: a value of 10 equals 1g = 9.81 m/s^2.
    for col in ["x", "y", "z"]:
        df[f"{col}_mps2"] = df[col] * (9.80665 / 10.0)
    df.attrs["bad_record_count"] = len(bad_records)
    df.attrs["bad_record_examples"] = bad_records[:5]
    return df


def timestamp_audit(df: pd.DataFrame) -> dict:
    sorted_df = df.sort_values(["subject", "activity", "timestamp"])
    dt = sorted_df.groupby(["subject", "activity"])["timestamp"].diff() / 1e9
    valid = dt[(dt > 0) & (dt < 10)]
    rounded_ms = (valid * 1000).round().value_counts().head(20)
    if valid.empty:
        return {"valid_dt_count": 0}
    q = valid.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    median = float(valid.median())
    iqr = float(q.loc[0.75] - q.loc[0.25])
    return {
        "valid_dt_count": int(valid.shape[0]),
        "dt_seconds_mean": float(valid.mean()),
        "dt_seconds_std": float(valid.std()),
        "dt_seconds_min": float(valid.min()),
        "dt_seconds_quantiles": {str(k): float(v) for k, v in q.items()},
        "dt_seconds_max": float(valid.max()),
        "median_effective_hz": float(1.0 / median),
        "outlier_threshold_seconds_median_plus_5iqr": float(median + 5.0 * iqr),
        "outlier_count": int((valid > median + 5.0 * iqr).sum()),
        "top_rounded_dt_ms_counts": {str(float(k)): int(v) for k, v in rounded_ms.items()},
    }


def inspect_wisdm_classic(root: Path | None = None) -> dict:
    df = parse_wisdm_classic_raw(root)
    return {
        "dataset": "WISDM Activity Prediction v1.1",
        "root": str(root or WISDM_CLASSIC_DIR),
        "variant": "Fordham WISDM classic accelerometer-only raw time series",
        "classes": WISDM_CLASSIC_CLASSES,
        "rows": int(df.shape[0]),
        "subjects": int(df["subject"].nunique()),
        "subject_ids": sorted(int(s) for s in df["subject"].unique().tolist()),
        "activity_counts": {k: int(v) for k, v in df["activity"].value_counts().items()},
        "sensor_axes": ["x", "y", "z"],
        "nominal_sampling_rate_hz": WISDM_CLASSIC_SAMPLING_HZ,
        "bad_record_count": int(df.attrs.get("bad_record_count", 0)),
        "bad_record_examples": df.attrs.get("bad_record_examples", []),
        "timestamp_audit": timestamp_audit(df),
        "license_note": (
            "Fordham WISDM page requests citation and inclusion of readme.txt when "
            "redistributing; no OSI/Creative Commons license text was found in the downloaded files."
        ),
    }


def regularize_group_to_hz(group: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    group = group.sort_values("timestamp")
    t = group["timestamp"].to_numpy(dtype=np.float64) / 1e9
    if len(t) < 2:
        return pd.DataFrame()
    t = t - t[0]
    unique_idx = np.r_[True, np.diff(t) > 0]
    group = group.iloc[unique_idx]
    t = t[unique_idx]
    if len(t) < 2:
        return pd.DataFrame()
    grid = np.arange(0.0, t[-1], 1.0 / target_hz)
    if len(grid) < 2:
        return pd.DataFrame()
    out = {
        "subject": int(group["subject"].iloc[0]),
        "activity": group["activity"].iloc[0],
        "t_seconds": grid,
    }
    for col in ["x_mps2", "y_mps2", "z_mps2"]:
        out[col] = np.interp(grid, t, group[col].to_numpy(dtype=np.float64))
    return pd.DataFrame(out)


def build_wisdm_windows(
    df: pd.DataFrame,
    target_hz: int = WISDM_CLASSIC_SAMPLING_HZ,
    window_seconds: float = 4.0,
    overlap: float = 0.50,
) -> WisdmWindows:
    class_to_idx = {name: idx for idx, name in enumerate(WISDM_CLASSIC_CLASSES)}
    window_size = int(round(target_hz * window_seconds))
    stride = max(1, int(round(window_size * (1.0 - overlap))))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    for (_, _), group in df.groupby(["subject", "activity"], sort=True):
        regular = regularize_group_to_hz(group, target_hz=target_hz)
        if regular.empty:
            continue
        values = regular[["x_mps2", "y_mps2", "z_mps2"]].to_numpy(dtype=np.float32)
        activity = regular["activity"].iloc[0]
        if activity not in class_to_idx:
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            xs.append(values[start : start + window_size])
            ys.append(class_to_idx[activity])
            subjects.append(int(regular["subject"].iloc[0]))
    if not xs:
        raise ValueError("No WISDM windows were created")
    return WisdmWindows(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=np.int64),
        class_names=WISDM_CLASSIC_CLASSES,
    )


def split_wisdm_windows(
    windows: WisdmWindows,
    test_size: float = 0.20,
    seed: int = SEED,
) -> tuple[WisdmWindows, WisdmWindows]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(windows.x, windows.y, groups=windows.subjects))
    return (
        WisdmWindows(windows.x[train_idx], windows.y[train_idx], windows.subjects[train_idx], windows.class_names),
        WisdmWindows(windows.x[test_idx], windows.y[test_idx], windows.subjects[test_idx], windows.class_names),
    )


def save_wisdm_inspection(output_path: Path) -> dict:
    inspection = inspect_wisdm_classic()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    return inspection
