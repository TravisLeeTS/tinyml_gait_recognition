from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import DOCS_DIR, OUTPUTS_DIR
from src.data.uci_har import class_count_table, load_uci_har_sequence_dataset
from src.data.wisdm import inspect_wisdm_classic


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect UCI HAR and WISDM datasets.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "datacards")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    uci = load_uci_har_sequence_dataset(download=True)
    uci_counts = class_count_table(uci)
    uci_counts.to_csv(DOCS_DIR / "tables" / "uci_har_class_counts.csv", index=False)
    uci_summary = dict(uci.metadata)
    uci_summary["class_counts_train_val_test"] = uci_counts.to_dict(orient="records")
    (args.output_dir / "uci_har_inspection.json").write_text(
        json.dumps(uci_summary, indent=2),
        encoding="utf-8",
    )

    wisdm_summary = inspect_wisdm_classic()
    pd.DataFrame(
        [{"class": k, "count": v} for k, v in wisdm_summary["activity_counts"].items()]
    ).to_csv(DOCS_DIR / "tables" / "wisdm_classic_raw_counts.csv", index=False)
    (args.output_dir / "wisdm_classic_inspection.json").write_text(
        json.dumps(wisdm_summary, indent=2),
        encoding="utf-8",
    )

    combined = {"uci_har": uci_summary, "wisdm_classic": wisdm_summary}
    (args.output_dir / "dataset_inspection_summary.json").write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(combined, indent=2)[:4000])


if __name__ == "__main__":
    main()
