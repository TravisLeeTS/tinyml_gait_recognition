from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support


CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def _normalize_label(value: object) -> str:
    text = str(value).strip().upper()
    text = text.replace(" ", "_").replace("-", "_")
    aliases = {
        "WALK": "WALKING",
        "WALKING_UP": "WALKING_UPSTAIRS",
        "WALK_UP": "WALKING_UPSTAIRS",
        "WALK_UPSTAIRS": "WALKING_UPSTAIRS",
        "UPSTAIRS": "WALKING_UPSTAIRS",
        "WALKING_UPSTAIR": "WALKING_UPSTAIRS",
        "WALKING_UPSTAIRSS": "WALKING_UPSTAIRS",
        "WALKING_DOWN": "WALKING_DOWNSTAIRS",
        "WALK_DOWN": "WALKING_DOWNSTAIRS",
        "WALK_DOWNSTAIRS": "WALKING_DOWNSTAIRS",
        "DOWNSTAIRS": "WALKING_DOWNSTAIRS",
        "SIT": "SITTING",
        "STAND": "STANDING",
        "LAY": "LAYING",
    }
    text = aliases.get(text, text)
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(CLASS_NAMES):
            return CLASS_NAMES[idx]
    return text


def _label_from_filename(path: Path) -> str:
    return _normalize_label(path.stem)


def _is_prediction_row(parts: list[str]) -> bool:
    if len(parts) < 6:
        return False
    try:
        int(parts[0].strip())
        int(parts[1].strip())
        int(parts[2].strip())
    except ValueError:
        return False
    return _normalize_label(parts[3]) in CLASS_NAMES


def _read_serial_trial_file(path: Path, condition: str) -> pd.DataFrame:
    true_label = _label_from_filename(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = next(csv.reader([line]))
        parts = [part.strip() for part in parts]
        if not _is_prediction_row(parts):
            continue
        rows.append(
            {
                "condition": condition,
                "trial_id": f"{path.stem}_{len(rows) + 1}",
                "source_file": str(path),
                "source_line": line_number,
                "true_label": true_label,
                # Use the displayed label, not prediction_id, for scoring Arduino output.
                "predicted_label": _normalize_label(parts[3]),
                "timestamp_ms": int(parts[0]),
                "window_id": int(parts[1]),
                "prediction_id": int(parts[2]),
                "confidence": float(parts[4]),
                "latency_us": float(parts[5]),
                "top1_score": float(parts[6]) if len(parts) > 6 and parts[6] else pd.NA,
                "top2_label": _normalize_label(parts[7]) if len(parts) > 7 and parts[7] else pd.NA,
                "top2_score": float(parts[8]) if len(parts) > 8 and parts[8] else pd.NA,
                "avg_latency_us": float(parts[9]) if len(parts) > 9 and parts[9] else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _read_serial_trial_dir(path: Path, condition: str) -> pd.DataFrame:
    frames = [_read_serial_trial_file(file_path, condition) for file_path in sorted(path.glob("*.txt"))]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"No prediction rows found in {path}")
    return pd.concat(frames, ignore_index=True)


def _read_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    aliases = {
        "prediction_label": "predicted_label",
        "pred": "predicted_label",
        "label": "predicted_label",
        "actual_label": "true_label",
        "truth": "true_label",
        "timestamp": "timestamp_ms",
        "window": "window_id",
    }
    df = df.rename(columns={col: aliases.get(col, col) for col in df.columns})
    required = {"condition", "true_label", "predicted_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "trial_id" not in df.columns:
        df["trial_id"] = range(1, len(df) + 1)
    if "timestamp_ms" not in df.columns:
        df["timestamp_ms"] = pd.NA
    if "confidence" not in df.columns:
        df["confidence"] = pd.NA
    if "latency_us" not in df.columns:
        df["latency_us"] = pd.NA
    df["true_label"] = df["true_label"].map(_normalize_label)
    df["predicted_label"] = df["predicted_label"].map(_normalize_label)
    df["condition"] = df["condition"].astype(str).str.strip()
    return df


def _condition_key(condition: str) -> str:
    lower = condition.lower()
    if "right" in lower or "controlled" in lower:
        return "right_pocket_controlled"
    if "left" in lower or "robust" in lower or "new" in lower:
        return "left_pocket_robustness"
    return lower.replace(" ", "_")


def _write_confusion(y_true: list[str], y_pred: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(path)


def _per_class_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        zero_division=0,
    )
    rows = []
    for idx, label in enumerate(CLASS_NAMES):
        class_df = df[df["true_label"] == label]
        correct = int((class_df["predicted_label"] == label).sum())
        mistakes = class_df[class_df["predicted_label"] != label]["predicted_label"]
        dominant = ""
        dominant_count = 0
        if not mistakes.empty:
            counts = mistakes.value_counts()
            dominant = str(counts.index[0])
            dominant_count = int(counts.iloc[0])
        rows.append(
            {
                "class": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "correct": correct,
                "dominant_confusion_target": dominant,
                "dominant_confusion_count": dominant_count,
            }
        )
    return pd.DataFrame(rows)


def _metrics_for(df: pd.DataFrame) -> dict:
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    latency = pd.to_numeric(df["latency_us"], errors="coerce")
    confidence = pd.to_numeric(df["confidence"], errors="coerce")
    return {
        "n_predictions": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_NAMES, average="weighted", zero_division=0)),
        "per_class": report,
        "mean_confidence": None if confidence.dropna().empty else float(confidence.mean()),
        "mean_latency_us": None if latency.dropna().empty else float(latency.mean()),
        "median_latency_us": None if latency.dropna().empty else float(latency.median()),
        "source": "Arduino Serial predictions; not laptop replay.",
    }


def _write_summary_from_metrics(output_dir: Path) -> None:
    rows = []
    main_failures = {
        "right_pocket_controlled": "WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS",
        "left_pocket_robustness": "WALKING_UPSTAIRS -> WALKING_DOWNSTAIRS",
    }
    display_names = {
        "right_pocket_controlled": "Right pocket controlled",
        "left_pocket_robustness": "Left pocket robustness",
    }
    for metrics_path in sorted(output_dir.glob("*_metrics.json")):
        if metrics_path.name == "summary_metrics.json":
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "n_predictions" not in metrics:
            continue
        condition = metrics_path.name.removesuffix("_metrics.json")
        rows.append(
            {
                "Condition": display_names.get(condition, condition),
                "Source": "Live Arduino Serial",
                "Rows": metrics["n_predictions"],
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Main Failure": main_failures.get(condition, ""),
            }
        )
    if rows:
        order = {
            "right_pocket_controlled": 0,
            "Right pocket controlled": 0,
            "left_pocket_robustness": 1,
            "Left pocket robustness": 1,
        }
        summary = pd.DataFrame(rows).sort_values(by="Condition", key=lambda col: col.map(lambda x: order.get(x, 99)))
        summary.to_csv(output_dir / "live_controlled_vs_robustness_summary.csv", index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.inputs or ([args.input] if args.input else [])
    if not inputs:
        raise ValueError("Provide at least one CSV input or --input pointing to a Serial log directory/file.")
    frames = []
    for path in inputs:
        path = Path(path)
        if path.is_dir():
            if not args.condition:
                raise ValueError("--condition is required when --input/inputs include a Serial log directory.")
            frames.append(_read_serial_trial_dir(path, args.condition))
        elif path.suffix.lower() == ".txt":
            if not args.condition:
                raise ValueError("--condition is required when scoring a raw Serial .txt file.")
            frames.append(_read_serial_trial_file(path, args.condition))
        else:
            frames.append(_read_trials(path))
    all_df = pd.concat(frames, ignore_index=True)
    all_df["condition_key"] = all_df["condition"].map(_condition_key)

    summary_rows = []
    for condition_key, condition_df in all_df.groupby("condition_key", sort=True):
        metrics = _metrics_for(condition_df)
        metrics_path = output_dir / f"{condition_key}_metrics.json"
        cm_path = output_dir / f"{condition_key}_confusion_matrix.csv"
        per_class_path = output_dir / f"{condition_key}_per_class_metrics.csv"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_confusion(condition_df["true_label"].tolist(), condition_df["predicted_label"].tolist(), cm_path)
        _per_class_diagnostics(condition_df).to_csv(per_class_path, index=False)
        condition_df.to_csv(output_dir / f"{condition_key}_predictions_normalized.csv", index=False)
        summary_rows.append(
            {
                "condition": condition_key,
                "n_predictions": metrics["n_predictions"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mean_latency_us": metrics["mean_latency_us"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "live_controlled_vs_robustness_summary.csv", index=False)
    all_df.to_csv(output_dir / "live_serial_predictions_normalized.csv", index=False)
    _write_summary_from_metrics(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "conditions": summary_rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score M3 live Arduino Serial trial predictions.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=None,
        type=Path,
        help=(
            "CSV files with at least condition,true_label,predicted_label, or raw Serial .txt files. "
            "Optional when --input is used."
        ),
    )
    parser.add_argument("--input", type=Path, help="A raw Serial log directory/file or normalized CSV file.")
    parser.add_argument("--condition", help="Condition label for raw Serial logs, e.g. right_pocket_controlled.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/live_evidence"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
