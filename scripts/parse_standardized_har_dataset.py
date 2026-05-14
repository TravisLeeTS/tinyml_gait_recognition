from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["time_ms", "ax", "ay", "az", "gx", "gy", "gz"]
SENSOR_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
CLASS_MAP = {
    "lay": "LAYING",
    "laying": "LAYING",
    "sit": "SITTING",
    "sitting": "SITTING",
    "stand": "STANDING",
    "standing": "STANDING",
    "walk": "WALKING",
    "walking": "WALKING",
    "walkup": "WALKING_UPSTAIRS",
    "walk_up": "WALKING_UPSTAIRS",
    "walking_upstairs": "WALKING_UPSTAIRS",
    "upstairs": "WALKING_UPSTAIRS",
    "walkdown": "WALKING_DOWNSTAIRS",
    "walk_down": "WALKING_DOWNSTAIRS",
    "walking_downstairs": "WALKING_DOWNSTAIRS",
    "downstairs": "WALKING_DOWNSTAIRS",
}


@dataclass
class FileSummary:
    source_file: str
    session: str
    person: str
    activity_folder: str
    uci_class: str
    placement: str
    trial: str
    rows_raw: int
    rows_valid: int
    rows_invalid: int
    duration_seconds: float
    median_dt_ms: float | None
    effective_hz: float | None
    windows_128_stride64: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse the standardized Arduino HAR dataset zip or extracted folder."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to TINYML_HAR_DATA_STANDARDIZED_v2.zip or an extracted dataset folder.",
    )
    parser.add_argument(
        "--output",
        default=Path("results/standardized_dataset_summary"),
        type=Path,
        help="Output directory for CSV and JSON summaries.",
    )
    parser.add_argument("--window-size", default=128, type=int)
    parser.add_argument("--stride", default=64, type=int)
    return parser.parse_args()


def count_windows(sample_count: int, window_size: int, stride: int) -> int:
    if sample_count < window_size:
        return 0
    return ((sample_count - window_size) // stride) + 1


def extract_if_needed(input_path: Path, temp_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file or folder, got: {input_path}")
    with zipfile.ZipFile(input_path) as archive:
        archive.extractall(temp_dir)
    children = [child for child in temp_dir.iterdir() if child.is_dir()]
    if len(children) == 1:
        return children[0]
    return temp_dir


def infer_session(parts: tuple[str, ...]) -> str:
    for part in parts:
        if part.lower().startswith("session_"):
            return part
    return "unknown"


def infer_person(parts: tuple[str, ...]) -> str:
    for part in parts:
        if re.fullmatch(r"person[_-]?[A-Za-z0-9]+", part, flags=re.IGNORECASE):
            return part
    return "unknown"


def infer_activity(parts: tuple[str, ...], stem: str) -> tuple[str, str]:
    candidates = [part.lower() for part in parts] + [stem.lower()]
    for token in candidates:
        normalized = token.replace("-", "_")
        if normalized in CLASS_MAP:
            return normalized, CLASS_MAP[normalized]
    joined = "_".join(candidates).lower()
    if "walkdown" in joined or "walk_down" in joined or "downstairs" in joined:
        return "walkdown", CLASS_MAP["walkdown"]
    if "walkup" in joined or "walk_up" in joined or "upstairs" in joined:
        return "walkup", CLASS_MAP["walkup"]
    for key in ("walk", "stand", "sit", "lay"):
        if re.search(rf"(^|[_\W]){key}([_\W]|$)", joined):
            return key, CLASS_MAP[key]
    return "unknown", "UNKNOWN"


def infer_placement(stem: str, parts: tuple[str, ...]) -> str:
    text = "/".join([*parts, stem]).lower()
    if re.search(r"(^|[_\W])left([_\W]|$)", text):
        return "left_pocket"
    if re.search(r"(^|[_\W])right([_\W]|$)", text):
        return "right_pocket"
    if re.search(r"(^|[_\W])l([_\W]|$)", text):
        return "left_pocket"
    if re.search(r"(^|[_\W])r([_\W]|$)", text):
        return "right_pocket"
    return "unknown"


def infer_trial(stem: str) -> str:
    match = re.search(r"trial\s*0*(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return f"trial{int(match.group(1)):02d}"
    match = re.search(r"(?:^|[_\W])0*(\d+)$", stem)
    if match:
        return f"trial{int(match.group(1)):02d}"
    return "unknown"


def read_sensor_file(path: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(path, sep=";", engine="python")
    raw.columns = [str(column).strip() for column in raw.columns]
    if not set(EXPECTED_COLUMNS).issubset(raw.columns):
        raw = pd.read_csv(path, sep=";", names=EXPECTED_COLUMNS, header=None, engine="python")
        raw.columns = [str(column).strip() for column in raw.columns]
    missing = [column for column in EXPECTED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = raw[EXPECTED_COLUMNS].copy()
    for column in EXPECTED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    valid = df[EXPECTED_COLUMNS].notna().all(axis=1)
    invalid_rows = int((~valid).sum())
    return df.loc[valid].copy(), invalid_rows


def summarize_file(path: Path, root: Path, window_size: int, stride: int) -> FileSummary:
    rel = path.relative_to(root)
    parts = rel.parts
    session = infer_session(parts)
    person = infer_person(parts)
    activity, uci_class = infer_activity(parts[:-1], path.stem)
    placement = infer_placement(path.stem, parts[:-1])
    trial = infer_trial(path.stem)

    if path.stat().st_size == 0:
        return FileSummary(
            rel.as_posix(),
            session,
            person,
            activity,
            uci_class,
            placement,
            trial,
            0,
            0,
            0,
            0.0,
            None,
            None,
            0,
        )

    frame, invalid_rows = read_sensor_file(path)
    valid_rows = len(frame)
    raw_rows = valid_rows + invalid_rows
    if valid_rows >= 2:
        duration_seconds = float((frame["time_ms"].iloc[-1] - frame["time_ms"].iloc[0]) / 1000.0)
        diffs = frame["time_ms"].diff().dropna()
        diffs = diffs[diffs > 0]
        median_dt = float(diffs.median()) if not diffs.empty else None
    else:
        duration_seconds = 0.0
        median_dt = None
    effective_hz = float(1000.0 / median_dt) if median_dt else None
    return FileSummary(
        rel.as_posix(),
        session,
        person,
        activity,
        uci_class,
        placement,
        trial,
        raw_rows,
        valid_rows,
        invalid_rows,
        max(duration_seconds, 0.0),
        median_dt,
        effective_hz,
        count_windows(valid_rows, window_size, stride),
    )


def grouped_summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = (
        frame.groupby(by, dropna=False)
        .agg(
            files=("source_file", "count"),
            rows_valid=("rows_valid", "sum"),
            duration_seconds=("duration_seconds", "sum"),
            windows_128_stride64=("windows_128_stride64", "sum"),
        )
        .reset_index()
        .sort_values(by)
    )
    return rows


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset input does not exist: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="standardized_har_") as temp_name:
        root = extract_if_needed(input_path, Path(temp_name))
        files = sorted(path for path in root.rglob("*.txt") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No .txt sensor files found under {root}")

        summaries = [summarize_file(path, root, args.window_size, args.stride) for path in files]

    file_frame = pd.DataFrame([asdict(row) for row in summaries])
    file_frame.to_csv(args.output / "file_summary.csv", index=False)

    outputs = {
        "class_distribution.csv": grouped_summary(file_frame, ["uci_class"]),
        "session_summary.csv": grouped_summary(file_frame, ["session"]),
        "person_summary.csv": grouped_summary(file_frame, ["person"]),
        "placement_summary.csv": grouped_summary(file_frame, ["placement"]),
        "session_person_class_summary.csv": grouped_summary(file_frame, ["session", "person", "uci_class"]),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output / filename, index=False)

    metadata = {
        "input_name": input_path.name,
        "input_kind": "zip" if input_path.suffix.lower() == ".zip" else "folder",
        "sensor_columns": EXPECTED_COLUMNS,
        "format": "semicolon-separated text files with header time_ms; ax; ay; az; gx; gy; gz",
        "files": int(len(file_frame)),
        "sessions": sorted(file_frame["session"].unique().tolist()),
        "persons": sorted(file_frame["person"].unique().tolist()),
        "placements": sorted(file_frame["placement"].unique().tolist()),
        "classes": sorted(file_frame["uci_class"].unique().tolist()),
        "rows_valid": int(file_frame["rows_valid"].sum()),
        "duration_seconds": float(file_frame["duration_seconds"].sum()),
        "windows_128_stride64": int(file_frame["windows_128_stride64"].sum()),
        "invalid_rows": int(file_frame["rows_invalid"].sum()),
        "median_effective_hz": float(np.nanmedian(file_frame["effective_hz"])) if file_frame["effective_hz"].notna().any() else None,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Standardized HAR dataset summary")
    print(f"  files: {metadata['files']}")
    print(f"  sessions: {', '.join(metadata['sessions'])}")
    print(f"  persons: {', '.join(metadata['persons'])}")
    print(f"  placements: {', '.join(metadata['placements'])}")
    print(f"  valid rows: {metadata['rows_valid']}")
    print(f"  estimated 128/64 windows: {metadata['windows_128_stride64']}")
    print(f"  output: {args.output}")
    print("\nClass distribution:")
    print(outputs["class_distribution.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
