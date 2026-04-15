from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_samples": int(len(y_true)),
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def save_classification_outputs(
    outputs: dict,
    metrics_dir: Path,
    figures_dir: Path,
    prefix: str,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )

    class_names = outputs["class_names"]
    report = outputs["classification_report"]
    rows = []
    for label in class_names + ["macro avg", "weighted avg"]:
        if label in report:
            row = {"class": label}
            row.update(report[label])
            rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_dir / f"{prefix}_per_class_metrics.csv", index=False)

    cm = np.asarray(outputs["confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        metrics_dir / f"{prefix}_confusion_matrix.csv"
    )
    plot_confusion_matrix(cm, class_names, figures_dir / f"{prefix}_confusion_matrix.png")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    width = max(8, len(class_names) * 1.25)
    fig, ax = plt.subplots(figsize=(width, width * 0.85))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def top_confusion_pairs(cm: np.ndarray, class_names: list[str], top_k: int = 5) -> list[dict]:
    pairs = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            count = int(cm[i, j])
            if count:
                pairs.append({"true": true_name, "predicted": pred_name, "count": count})
    return sorted(pairs, key=lambda x: x["count"], reverse=True)[:top_k]
