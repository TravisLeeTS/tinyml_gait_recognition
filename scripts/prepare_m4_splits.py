from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe split metadata for the standardized HAR dataset."
    )
    parser.add_argument(
        "--file-summary",
        type=Path,
        default=Path("results/standardized_dataset_summary/file_summary.csv"),
        help="CSV produced by scripts/parse_standardized_har_dataset.py.",
    )
    parser.add_argument(
        "--strategy",
        choices=["A", "B", "C"],
        default="A",
        help="A=fully held-out external validation, B=partial train plus clean holdout, C=leave-one-person/session out.",
    )
    parser.add_argument("--holdout-person", default=None)
    parser.add_argument("--holdout-session", default=None)
    parser.add_argument("--holdout-placement", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/standardized_dataset_summary/m4_split_plan.csv"),
    )
    return parser.parse_args()


def assign_strategy(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = "external_holdout"
    out["split_reason"] = "Strategy A keeps every standardized recording out of training."

    if args.strategy == "A":
        return out

    holdout_mask = pd.Series(False, index=out.index)
    reasons = []
    if args.holdout_person:
        holdout_mask |= out["person"].eq(args.holdout_person)
        reasons.append(f"person={args.holdout_person}")
    if args.holdout_session:
        holdout_mask |= out["session"].eq(args.holdout_session)
        reasons.append(f"session={args.holdout_session}")
    if args.holdout_placement:
        holdout_mask |= out["placement"].eq(args.holdout_placement)
        reasons.append(f"placement={args.holdout_placement}")

    if not reasons:
        if args.strategy == "B":
            if "left_pocket" in set(out["placement"]):
                holdout_mask = out["placement"].eq("left_pocket")
                reasons.append("placement=left_pocket default holdout")
            else:
                last_session = sorted(out["session"].unique())[-1]
                holdout_mask = out["session"].eq(last_session)
                reasons.append(f"session={last_session} default holdout")
        else:
            persons = sorted(out["person"].unique())
            if len(persons) > 1:
                holdout_person = persons[-1]
                holdout_mask = out["person"].eq(holdout_person)
                reasons.append(f"person={holdout_person} default holdout")
            else:
                last_session = sorted(out["session"].unique())[-1]
                holdout_mask = out["session"].eq(last_session)
                reasons.append(f"session={last_session} default holdout")

    out.loc[~holdout_mask, "split"] = "candidate_train"
    out.loc[~holdout_mask, "split_reason"] = (
        f"Strategy {args.strategy} candidate training file; no windows from this recording may enter holdout."
    )
    out.loc[holdout_mask, "split"] = "final_holdout"
    out.loc[holdout_mask, "split_reason"] = (
        f"Strategy {args.strategy} final holdout selected by {', '.join(reasons)}."
    )
    return out


def main() -> None:
    args = parse_args()
    if not args.file_summary.exists():
        raise FileNotFoundError(
            f"Missing file summary: {args.file_summary}. Run scripts/parse_standardized_har_dataset.py first."
        )
    frame = pd.read_csv(args.file_summary)
    required = {"source_file", "session", "person", "placement", "uci_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"File summary is missing required columns: {sorted(missing)}")

    split = assign_strategy(frame, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output, index=False)

    metadata = {
        "strategy": args.strategy,
        "leakage_rule": "The atomic unit is a raw recording file/session path; overlapping windows must be generated after this split and must not cross splits.",
        "holdout_person": args.holdout_person,
        "holdout_session": args.holdout_session,
        "holdout_placement": args.holdout_placement,
        "counts_by_split": split.groupby("split")["source_file"].count().to_dict(),
        "windows_by_split": split.groupby("split")["windows_128_stride64"].sum().astype(int).to_dict()
        if "windows_128_stride64" in split.columns
        else {},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote split plan: {args.output}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
