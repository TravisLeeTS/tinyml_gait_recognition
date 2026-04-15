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
    UCI_HAR_CLASSES,
    UCI_HAR_DIR,
    UCI_HAR_SEQUENCE_SIGNALS,
    UCI_HAR_URLS,
    UCI_HAR_ZIP,
)
from src.utils.download import download_first_available, extract_archive


@dataclass
class SequenceSplit:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


@dataclass
class UciHarData:
    train: SequenceSplit
    val: SequenceSplit
    test: SequenceSplit
    class_names: list[str]
    metadata: dict


def ensure_uci_har(download: bool = True) -> Path:
    if UCI_HAR_DIR.exists():
        return UCI_HAR_DIR
    if not download:
        raise FileNotFoundError(f"UCI HAR dataset not found at {UCI_HAR_DIR}")
    used_url = download_first_available(UCI_HAR_URLS, UCI_HAR_ZIP)
    extract_archive(UCI_HAR_ZIP, RAW_DATA_DIR)
    if not UCI_HAR_DIR.exists():
        raise FileNotFoundError(f"Archive extracted but {UCI_HAR_DIR} was not created")
    (RAW_DATA_DIR / "uci_har_download_source.txt").write_text(used_url, encoding="utf-8")
    return UCI_HAR_DIR


def _load_signal_file(root: Path, split: str, signal: str) -> np.ndarray:
    path = root / split / "Inertial Signals" / f"{signal}_{split}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.loadtxt(path, dtype=np.float32)


def load_sequence_split(root: Path, split: str) -> SequenceSplit:
    signals = [_load_signal_file(root, split, signal) for signal in UCI_HAR_SEQUENCE_SIGNALS]
    x = np.stack(signals, axis=-1).astype(np.float32)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    return SequenceSplit(x=x, y=y, subjects=subjects)


def split_train_val_by_subject(
    full_train: SequenceSplit,
    val_size: float = 0.20,
    seed: int = SEED,
) -> tuple[SequenceSplit, SequenceSplit]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(full_train.x, full_train.y, groups=full_train.subjects))
    return (
        SequenceSplit(full_train.x[train_idx], full_train.y[train_idx], full_train.subjects[train_idx]),
        SequenceSplit(full_train.x[val_idx], full_train.y[val_idx], full_train.subjects[val_idx]),
    )


def load_uci_har_sequence_dataset(
    download: bool = True,
    val_size: float = 0.20,
    seed: int = SEED,
) -> UciHarData:
    root = ensure_uci_har(download=download)
    full_train = load_sequence_split(root, "train")
    train, val = split_train_val_by_subject(full_train, val_size=val_size, seed=seed)
    test = load_sequence_split(root, "test")
    metadata = inspect_uci_har(root, val_subjects=sorted(set(val.subjects.tolist())))
    return UciHarData(train=train, val=val, test=test, class_names=UCI_HAR_CLASSES, metadata=metadata)


def inspect_uci_har(root: Path | None = None, val_subjects: list[int] | None = None) -> dict:
    root = root or ensure_uci_har(download=True)
    labels = pd.read_csv(root / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
    label_map = dict(zip(labels["id"].astype(int), labels["activity"]))
    out: dict = {
        "dataset": "UCI HAR Dataset",
        "version": "1.0",
        "root": str(root),
        "classes": [label_map[i] for i in sorted(label_map)],
        "sequence_input_shape": [128, len(UCI_HAR_SEQUENCE_SIGNALS)],
        "sampling_rate_hz": 50,
        "window_seconds": 2.56,
        "overlap": 0.50,
        "signals_used": UCI_HAR_SEQUENCE_SIGNALS,
        "source_note": "Standard UCI HAR release with prewindowed inertial-signal rows.",
    }
    counts = []
    for split in ["train", "test"]:
        y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64)
        subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
        out[f"{split}_samples"] = int(len(y))
        out[f"{split}_subjects"] = sorted(int(s) for s in set(subjects.tolist()))
        for cls_id, cls_name in label_map.items():
            counts.append({"split": split, "class": cls_name, "count": int((y == cls_id).sum())})
    out["official_counts"] = counts
    if val_subjects:
        out["validation_subjects"] = val_subjects
    return out


def class_count_table(data: UciHarData) -> pd.DataFrame:
    rows = []
    for split_name, split in [("train", data.train), ("val", data.val), ("test", data.test)]:
        for idx, name in enumerate(data.class_names):
            rows.append({"split": split_name, "class": name, "count": int((split.y == idx).sum())})
    table = pd.DataFrame(rows).pivot(index="class", columns="split", values="count").reset_index()
    table["class"] = pd.Categorical(table["class"], categories=data.class_names, ordered=True)
    table = table.sort_values("class")
    table["class"] = table["class"].astype(str)
    for col in ["train", "val", "test"]:
        if col not in table:
            table[col] = 0
    table["total"] = table[["train", "val", "test"]].sum(axis=1)
    return table[["class", "train", "val", "test", "total"]]


def save_uci_inspection(output_path: Path) -> dict:
    data = load_uci_har_sequence_dataset(download=True)
    inspection = data.metadata
    inspection["class_counts_train_val_test"] = class_count_table(data).to_dict(orient="records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    return inspection
