from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DOCS_DIR, OUTPUTS_DIR, RAW_DATA_DIR, SEED
from src.data.arduino_collectdata import build_windowed_dataset
from src.data.target_domain_features import orientation_summary
from src.data.uci_har import load_uci_har_sequence_dataset
from src.utils.normalization import SequenceStandardizer


def _prefix_rows(rows: list[dict], domain: str) -> list[dict]:
    return [{"domain": domain, **row} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UCI-HAR vs Arduino target-domain orientation shift.")
    parser.add_argument("--arduino-root", type=Path, default=RAW_DATA_DIR / "arduino_collectdata_v2")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "arduino_domain_audit")
    parser.add_argument("--docs-tables-dir", type=Path, default=DOCS_DIR / "tables")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    uci = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    arduino_x, arduino_y, _sources = build_windowed_dataset(args.arduino_root, hz=50.0)
    scaler = SequenceStandardizer.fit(uci.train.x)
    arduino_z = scaler.transform(arduino_x)
    uci_test_z = scaler.transform(uci.test.x)

    orientation_rows = [
        *_prefix_rows(orientation_summary(uci.test.x, uci.test.y, uci.class_names), "uci_har_test"),
        *_prefix_rows(orientation_summary(arduino_x, arduino_y, uci.class_names), "arduino_v2"),
    ]
    orientation_table = pd.DataFrame(orientation_rows)

    standardized_shift = {
        "standardizer_fit_scope": "UCI-HAR official training split only",
        "arduino_root": str(args.arduino_root),
        "uci_test_standardized_mean": uci_test_z.reshape(-1, uci_test_z.shape[-1]).mean(axis=0).astype(float).tolist(),
        "uci_test_standardized_std": uci_test_z.reshape(-1, uci_test_z.shape[-1]).std(axis=0).astype(float).tolist(),
        "arduino_v2_standardized_mean": arduino_z.reshape(-1, arduino_z.shape[-1]).mean(axis=0).astype(float).tolist(),
        "arduino_v2_standardized_std": arduino_z.reshape(-1, arduino_z.shape[-1]).std(axis=0).astype(float).tolist(),
        "channel_order": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "interpretation": (
            "Large standardized offsets, especially on acceleration axes, indicate that the "
            "Arduino target domain should not use UCI-only normalization for final deployment."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_tables_dir.mkdir(parents=True, exist_ok=True)
    orientation_csv = args.docs_tables_dir / "m3_domain_orientation_comparison.csv"
    shift_json = args.output_dir / "m3_domain_standardized_shift.json"
    orientation_table.to_csv(orientation_csv, index=False)
    shift_json.write_text(json.dumps(standardized_shift, indent=2), encoding="utf-8")

    result = {
        "orientation_comparison_csv": str(orientation_csv),
        "standardized_shift_json": str(shift_json),
        "arduino_v2_acc_x_standardized_mean": float(np.asarray(standardized_shift["arduino_v2_standardized_mean"])[0]),
        "arduino_v2_windows": int(len(arduino_y)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
