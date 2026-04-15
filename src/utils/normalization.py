from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class SequenceStandardizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, x: np.ndarray) -> "SequenceStandardizer":
        if x.ndim != 3:
            raise ValueError(f"Expected [samples, timesteps, channels], got shape {x.shape}")
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        std = x.reshape(-1, x.shape[-1]).std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return ((x.astype(np.float32) - mean) / std).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SequenceStandardizer":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class SequenceMinMaxScaler:
    min_value: list[float]
    max_value: list[float]

    @classmethod
    def fit(cls, x: np.ndarray) -> "SequenceMinMaxScaler":
        if x.ndim != 3:
            raise ValueError(f"Expected [samples, timesteps, channels], got shape {x.shape}")
        flat = x.reshape(-1, x.shape[-1])
        min_value = flat.min(axis=0)
        max_value = flat.max(axis=0)
        max_value = np.where((max_value - min_value) < 1e-8, min_value + 1.0, max_value)
        return cls(min_value=min_value.astype(float).tolist(), max_value=max_value.astype(float).tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        min_value = np.asarray(self.min_value, dtype=np.float32)
        max_value = np.asarray(self.max_value, dtype=np.float32)
        return ((x.astype(np.float32) - min_value) / (max_value - min_value)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SequenceMinMaxScaler":
        return cls(**json.loads(path.read_text(encoding="utf-8")))
