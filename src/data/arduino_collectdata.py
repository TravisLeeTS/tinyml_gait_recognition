from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DOCS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, OUTPUTS_DIR


EXPECTED_COLUMNS = ["time_ms", "ax", "ay", "az", "gx", "gy", "gz", "label"]
SENSOR_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
NUMERIC_COLUMNS = ["time_ms", *SENSOR_COLUMNS]

FOLDER_TO_UCI_CLASS = {
    "walking": "WALKING",
    "walk_up": "WALKING_UPSTAIRS",
    "walk_down": "WALKING_DOWNSTAIRS",
    "sitting": "SITTING",
    "standing": "STANDING",
    "laying": "LAYING",
}
UCI_CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
UCI_CLASS_TO_ID = {name: idx for idx, name in enumerate(UCI_CLASS_NAMES)}


@dataclass
class ArduinoCsvFile:
    path: Path
    folder_label: str
    uci_label: str
    pocket: str
    rows_raw: int
    rows_valid: int
    rows_invalid: int
    timestamp_resets: int
    segments: int
    duration_seconds: float
    median_dt_ms: float | None
    effective_hz: float | None
    windows_raw_128_stride64: int
    windows_resampled_50hz_128_stride64: int


def iter_csv_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def infer_pocket(path: Path) -> str:
    match = re.search(r"-\s*([lr])\s*-", path.stem.lower())
    if not match:
        return "unknown"
    return {"l": "left", "r": "right"}[match.group(1)]


def _valid_numeric_frame(path: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    df = raw[EXPECTED_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    valid_mask = df[NUMERIC_COLUMNS].notna().all(axis=1) & df["label"].ne("")
    invalid_rows = int((~valid_mask).sum())
    df = df.loc[valid_mask].copy()
    df["time_ms"] = df["time_ms"].astype(float)
    for column in SENSOR_COLUMNS:
        df[column] = df[column].astype(np.float32)
    return df, invalid_rows


def read_labeled_csv(path: Path, root: Path) -> tuple[pd.DataFrame, ArduinoCsvFile]:
    folder_label = path.relative_to(root).parts[0].lower()
    if folder_label not in FOLDER_TO_UCI_CLASS:
        raise ValueError(f"Unknown Arduino label folder {folder_label!r} in {path}")

    df, invalid_rows = _valid_numeric_frame(path)
    uci_label = FOLDER_TO_UCI_CLASS[folder_label]
    df["folder_label"] = folder_label
    df["uci_label"] = uci_label
    df["class_id"] = UCI_CLASS_TO_ID[uci_label]
    df["source_file"] = path.relative_to(root).as_posix()
    df["pocket"] = infer_pocket(path)

    dt = df["time_ms"].diff()
    reset_mask = dt <= 0
    timestamp_resets = int(reset_mask.sum())
    df["segment_id"] = reset_mask.cumsum().astype(int)

    segment_groups = df.groupby("segment_id", sort=True)
    durations = []
    median_dts = []
    windows_raw = 0
    windows_resampled = 0
    for _, segment in segment_groups:
        if len(segment) < 2:
            continue
        duration = float((segment["time_ms"].iloc[-1] - segment["time_ms"].iloc[0]) / 1000.0)
        if duration > 0:
            durations.append(duration)
        diffs = segment["time_ms"].diff().dropna()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            median_dts.append(float(diffs.median()))
        windows_raw += count_windows(len(segment), window_size=128, stride=64)
        resampled_len = estimate_resampled_length(duration, hz=50.0)
        windows_resampled += count_windows(resampled_len, window_size=128, stride=64)

    median_dt = float(np.median(median_dts)) if median_dts else None
    effective_hz = float(1000.0 / median_dt) if median_dt else None
    file_info = ArduinoCsvFile(
        path=path,
        folder_label=folder_label,
        uci_label=uci_label,
        pocket=infer_pocket(path),
        rows_raw=int(len(df) + invalid_rows),
        rows_valid=int(len(df)),
        rows_invalid=invalid_rows,
        timestamp_resets=timestamp_resets,
        segments=int(df["segment_id"].nunique()) if not df.empty else 0,
        duration_seconds=float(sum(durations)),
        median_dt_ms=median_dt,
        effective_hz=effective_hz,
        windows_raw_128_stride64=int(windows_raw),
        windows_resampled_50hz_128_stride64=int(windows_resampled),
    )
    return df, file_info


def count_windows(sample_count: int, window_size: int, stride: int) -> int:
    if sample_count < window_size:
        return 0
    return int(((sample_count - window_size) // stride) + 1)


def estimate_resampled_length(duration_seconds: float, hz: float) -> int:
    if duration_seconds <= 0:
        return 0
    return int(np.floor(duration_seconds * hz)) + 1


def resample_segment(segment: pd.DataFrame, hz: float = 50.0) -> np.ndarray:
    if len(segment) < 2:
        return np.empty((0, len(SENSOR_COLUMNS)), dtype=np.float32)

    t = segment["time_ms"].to_numpy(dtype=np.float64)
    t = (t - t[0]) / 1000.0
    duration = float(t[-1])
    if duration <= 0:
        return np.empty((0, len(SENSOR_COLUMNS)), dtype=np.float32)

    target_t = np.arange(0.0, duration + 1e-9, 1.0 / hz, dtype=np.float64)
    channels = []
    for column in SENSOR_COLUMNS:
        channels.append(np.interp(target_t, t, segment[column].to_numpy(dtype=np.float64)))
    values = np.stack(channels, axis=-1).astype(np.float32)
    values[:, 3:6] *= np.float32(np.pi / 180.0)
    return values


def make_windows(values: np.ndarray, window_size: int = 128, stride: int = 64) -> np.ndarray:
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[-1]), dtype=np.float32)
    starts = range(0, len(values) - window_size + 1, stride)
    return np.stack([values[start : start + window_size] for start in starts]).astype(np.float32)


def build_windowed_dataset(root: Path, hz: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    for csv_path in iter_csv_files(root):
        frame, _ = read_labeled_csv(csv_path, root)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No Arduino CSV files found under {root}")

    all_rows = pd.concat(frames, ignore_index=True)
    xs = []
    ys = []
    sources = []
    group_cols = ["source_file", "segment_id", "class_id"]
    for (source_file, segment_id, class_id), segment in all_rows.groupby(group_cols, sort=True):
        values = resample_segment(segment.sort_values("time_ms"), hz=hz)
        windows = make_windows(values)
        if windows.size == 0:
            continue
        xs.append(windows)
        ys.append(np.full((len(windows),), int(class_id), dtype=np.int64))
        sources.extend([f"{source_file}#segment{segment_id}"] * len(windows))

    if not xs:
        raise ValueError("Arduino data produced zero windows; check sampling and duration.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.asarray(sources)


def summarize(root: Path, output_dir: Path, docs_tables_dir: Path) -> dict:
    frames = []
    file_infos = []
    for csv_path in iter_csv_files(root):
        frame, file_info = read_labeled_csv(csv_path, root)
        frames.append(frame)
        file_infos.append(file_info)

    if not file_infos:
        raise FileNotFoundError(f"No Arduino CSV files found under {root}")

    file_table = pd.DataFrame(
        [
            {
                **asdict(info),
                "path": info.path.relative_to(root).as_posix(),
            }
            for info in file_infos
        ]
    )

    class_table = (
        file_table.groupby(["folder_label", "uci_label"], as_index=False)
        .agg(
            files=("path", "count"),
            valid_rows=("rows_valid", "sum"),
            invalid_rows=("rows_invalid", "sum"),
            segments=("segments", "sum"),
            duration_seconds=("duration_seconds", "sum"),
            median_dt_ms=("median_dt_ms", "median"),
            effective_hz=("effective_hz", "median"),
            windows_raw_128_stride64=("windows_raw_128_stride64", "sum"),
            windows_resampled_50hz_128_stride64=("windows_resampled_50hz_128_stride64", "sum"),
        )
        .sort_values("uci_label")
    )
    all_rows = pd.concat(frames, ignore_index=True)
    channel_stats = (
        all_rows.groupby(["folder_label", "uci_label"], as_index=False)[SENSOR_COLUMNS]
        .agg(["mean", "std", "min", "max"])
    )
    channel_stats.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column for column in channel_stats.columns
    ]

    warnings = []
    median_hz = float(file_table["effective_hz"].dropna().median())
    if median_hz < 48.0 or median_hz > 52.0:
        warnings.append(
            "Observed Arduino CSV cadence is not 50 Hz; median effective rate is "
            f"{median_hz:.2f} Hz. Use the M3 sketch timing or resample before offline evaluation."
        )
    low_window_rows = class_table[class_table["windows_resampled_50hz_128_stride64"] < 100]
    if not low_window_rows.empty:
        labels = ", ".join(low_window_rows["uci_label"].tolist())
        warnings.append(
            "Current Arduino data has fewer than 100 resampled 128-sample/50%-overlap windows "
            f"for: {labels}."
        )
    bad_files = file_table[file_table["rows_invalid"] > 0]
    if not bad_files.empty:
        warnings.append(
            "Some CSV rows were skipped because numeric fields were invalid; inspect "
            + ", ".join(bad_files["path"].tolist())
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_tables_dir.mkdir(parents=True, exist_ok=True)
    file_table.to_csv(docs_tables_dir / "arduino_collectdata_file_quality.csv", index=False)
    class_table.to_csv(docs_tables_dir / "arduino_collectdata_class_counts.csv", index=False)
    channel_stats.to_csv(docs_tables_dir / "arduino_collectdata_channel_stats.csv", index=False)

    summary = {
        "dataset": "TinyML_arduino_collectdata_v1",
        "root": str(root),
        "csv_files": int(len(file_infos)),
        "expected_columns": EXPECTED_COLUMNS,
        "labels": FOLDER_TO_UCI_CLASS,
        "total_valid_rows": int(file_table["rows_valid"].sum()),
        "total_invalid_rows": int(file_table["rows_invalid"].sum()),
        "median_effective_hz": median_hz,
        "class_summary": class_table.to_dict(orient="records"),
        "channel_stats_csv": str(docs_tables_dir / "arduino_collectdata_channel_stats.csv"),
        "warnings": warnings,
        "notes": [
            "Only CSV files are used; raw .txt files from the Arduino are intentionally ignored.",
            "Gyroscope channels are converted from deg/s to rad/s when writing the windowed NPZ.",
            "Offline Arduino windows are resampled to 50 Hz to match the UCI HAR model input length.",
        ],
    }
    (output_dir / "arduino_collectdata_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and window the Arduino HAR CSV collection.")
    parser.add_argument("--root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v1")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "arduino_collectdata")
    parser.add_argument("--docs-tables-dir", type=Path, default=DOCS_DIR / "tables")
    parser.add_argument(
        "--windowed-output",
        type=Path,
        default=PROCESSED_DATA_DIR / "arduino_collectdata_v1_windows_50hz.npz",
    )
    parser.add_argument("--skip-windowed", action="store_true")
    args = parser.parse_args()

    summary = summarize(args.root, args.output_dir, args.docs_tables_dir)
    if not args.skip_windowed:
        x, y, sources = build_windowed_dataset(args.root, hz=50.0)
        args.windowed_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.windowed_output,
            x=x,
            y=y,
            sources=sources,
            class_names=np.asarray(UCI_CLASS_NAMES),
            channels=np.asarray(SENSOR_COLUMNS),
            sampling_hz=np.asarray([50.0], dtype=np.float32),
            window_size=np.asarray([128], dtype=np.int64),
            stride=np.asarray([64], dtype=np.int64),
        )
        summary["windowed_output"] = str(args.windowed_output)
        summary["windowed_shape"] = list(x.shape)
        summary["windowed_class_counts"] = {
            UCI_CLASS_NAMES[idx]: int((y == idx).sum()) for idx in range(len(UCI_CLASS_NAMES))
        }
        (args.output_dir / "arduino_collectdata_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({k: summary[k] for k in ["csv_files", "total_valid_rows", "median_effective_hz", "warnings"]}, indent=2))


if __name__ == "__main__":
    main()
