from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.config import DEFAULT_DEVICE, OUTPUTS_DIR, SEED
from src.data.uci_har import load_uci_har_sequence_dataset
from src.training.tf_common import predict_proba
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


STATIC_CLASS_NAMES = ["SITTING", "STANDING", "LAYING"]


def gravity_features(x_raw: np.ndarray) -> np.ndarray:
    mean_acc = x_raw[:, :, :3].mean(axis=1).astype(np.float32)
    magnitude = np.linalg.norm(mean_acc, axis=1, keepdims=True).astype(np.float32)
    direction = mean_acc / np.maximum(magnitude, np.float32(1e-6))
    return np.concatenate([mean_acc, direction, magnitude], axis=1).astype(np.float32)


def remap_static_labels(y: np.ndarray, static_ids: list[int]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(static_ids)}
    return np.asarray([mapping[int(label)] for label in y], dtype=np.int64)


def static_confusion_total(outputs: dict) -> int:
    class_names = outputs["class_names"]
    cm = np.asarray(outputs["confusion_matrix"], dtype=int)
    sitting_idx = class_names.index("SITTING")
    standing_idx = class_names.index("STANDING")
    return int(cm[sitting_idx, standing_idx] + cm[standing_idx, sitting_idx])


def apply_gravity_rescue(
    baseline_pred: np.ndarray,
    static_pred_original: np.ndarray,
    static_confidence: np.ndarray,
    static_ids: list[int],
    threshold: float,
) -> np.ndarray:
    rescue_pred = baseline_pred.copy()
    rescue_mask = np.isin(baseline_pred, static_ids) & (static_confidence >= threshold)
    rescue_pred[rescue_mask] = static_pred_original[rescue_mask]
    return rescue_pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a gravity-direction posture feature for SITTING/STANDING confusion."
    )
    parser.add_argument("--keras-model", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_tiny_cnn.keras")
    parser.add_argument("--standardizer", type=Path, default=OUTPUTS_DIR / "lightweight/models/lightweight_standardizer.json")
    parser.add_argument("--baseline-metrics", type=Path, default=OUTPUTS_DIR / "lightweight/metrics/lightweight_tiny_cnn_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "lightweight/experiments/m3_gravity_feature_probe")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)
    configure_tensorflow_device(args.device)

    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    sequence_standardizer = SequenceStandardizer.load(args.standardizer)
    x_val = sequence_standardizer.transform(data.val.x)
    x_test = sequence_standardizer.transform(data.test.x)

    static_ids = [data.class_names.index(name) for name in STATIC_CLASS_NAMES]
    train_static_mask = np.isin(data.train.y, static_ids)
    test_static_mask = np.isin(data.test.y, static_ids)

    feature_scaler = StandardScaler()
    train_features = gravity_features(data.train.x)
    val_features = gravity_features(data.val.x)
    test_features = gravity_features(data.test.x)
    train_features_scaled = feature_scaler.fit_transform(train_features)
    val_features_scaled = feature_scaler.transform(val_features)
    test_features_scaled = feature_scaler.transform(test_features)

    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(
        train_features_scaled[train_static_mask],
        remap_static_labels(data.train.y[train_static_mask], static_ids),
    )

    static_pred_mapped = clf.predict(test_features_scaled[test_static_mask])
    static_true_mapped = remap_static_labels(data.test.y[test_static_mask], static_ids)
    static_outputs = classification_outputs(
        static_true_mapped,
        static_pred_mapped,
        STATIC_CLASS_NAMES,
        "GravityDirectionStaticPostureProbe",
    )
    static_outputs["model_info"] = {
        "experiment_role": "Phase 2 posture-feature probe, not the deployable Phase 3 quantized model",
        "features": [
            "mean_total_acc_x",
            "mean_total_acc_y",
            "mean_total_acc_z",
            "mean_acc_unit_x",
            "mean_acc_unit_y",
            "mean_acc_unit_z",
            "mean_acc_magnitude",
        ],
        "classifier": "LogisticRegression trained only on UCI-HAR train split static windows",
        "static_classes": STATIC_CLASS_NAMES,
    }

    model = keras.models.load_model(args.keras_model, compile=False)
    baseline_val_probs = predict_proba(model, x_val)
    baseline_val_pred = baseline_val_probs.argmax(axis=1)
    baseline_probs = predict_proba(model, x_test)
    baseline_pred = baseline_probs.argmax(axis=1)

    val_static_probs = clf.predict_proba(val_features_scaled)
    val_static_pred_mapped = val_static_probs.argmax(axis=1)
    val_static_pred_original = np.asarray([static_ids[int(idx)] for idx in val_static_pred_mapped], dtype=np.int64)
    val_static_confidence = val_static_probs.max(axis=1)

    all_static_probs = clf.predict_proba(test_features_scaled)
    all_static_pred_mapped = all_static_probs.argmax(axis=1)
    all_static_pred_original = np.asarray([static_ids[int(idx)] for idx in all_static_pred_mapped], dtype=np.int64)
    all_static_confidence = all_static_probs.max(axis=1)

    threshold_rows = []
    for threshold in np.linspace(0.50, 0.99, 50):
        val_rescue_pred = apply_gravity_rescue(
            baseline_val_pred,
            val_static_pred_original,
            val_static_confidence,
            static_ids,
            float(threshold),
        )
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "val_accuracy": float(accuracy_score(data.val.y, val_rescue_pred)),
                "val_macro_f1": float(f1_score(data.val.y, val_rescue_pred, average="macro", zero_division=0)),
                "val_replaced_predictions": int(
                    (np.isin(baseline_val_pred, static_ids) & (val_static_confidence >= threshold)).sum()
                ),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    best_threshold_row = threshold_table.sort_values(
        ["val_macro_f1", "val_accuracy", "val_replaced_predictions"],
        ascending=[False, False, True],
    ).iloc[0]
    best_threshold = float(best_threshold_row["threshold"])

    rescue_pred = apply_gravity_rescue(
        baseline_pred,
        all_static_pred_original,
        all_static_confidence,
        static_ids,
        threshold=0.0,
    )
    tuned_rescue_pred = apply_gravity_rescue(
        baseline_pred,
        all_static_pred_original,
        all_static_confidence,
        static_ids,
        threshold=best_threshold,
    )

    rescue_outputs = classification_outputs(
        data.test.y,
        rescue_pred,
        data.class_names,
        "TinyDepthwiseSeparableCnnWithGravityPostureRescue",
    )
    rescue_outputs["top_confusion_pairs"] = top_confusion_pairs(
        np.asarray(rescue_outputs["confusion_matrix"], dtype=int),
        data.class_names,
    )
    rescue_outputs["model_info"] = {
        "experiment_role": "Phase 2 posture-feature probe, not the deployable Phase 3 quantized model",
        "base_model": str(args.keras_model),
        "posture_rule": "If the base DS-CNN predicts SITTING, STANDING, or LAYING, replace that static prediction with the gravity-feature static classifier prediction.",
        "feature_scaler": "StandardScaler fit on UCI-HAR train split gravity features only",
        "static_classifier": "LogisticRegression",
    }
    tuned_rescue_outputs = classification_outputs(
        data.test.y,
        tuned_rescue_pred,
        data.class_names,
        "TinyDepthwiseSeparableCnnWithThresholdedGravityPostureRescue",
    )
    tuned_rescue_outputs["top_confusion_pairs"] = top_confusion_pairs(
        np.asarray(tuned_rescue_outputs["confusion_matrix"], dtype=int),
        data.class_names,
    )
    tuned_rescue_outputs["model_info"] = rescue_outputs["model_info"] | {
        "threshold_selection": "Threshold chosen on subject-aware validation split to maximize macro F1.",
        "selected_threshold": best_threshold,
        "validation_selection_row": best_threshold_row.to_dict(),
    }

    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    comparison_rows = [
        {
            "method": "phase2_tiny_ds_cnn_baseline",
            "accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
            "static_sitting_standing_confusions": static_confusion_total(baseline),
            "notes": "Original Phase 2 selected model.",
        },
        {
            "method": "gravity_static_classifier_static_subset",
            "accuracy": static_outputs["accuracy"],
            "macro_f1": static_outputs["macro_f1"],
            "static_sitting_standing_confusions": int(
                np.asarray(static_outputs["confusion_matrix"], dtype=int)[0, 1]
                + np.asarray(static_outputs["confusion_matrix"], dtype=int)[1, 0]
            ),
            "notes": "Only SITTING/STANDING/LAYING test windows.",
        },
        {
            "method": "phase2_tiny_ds_cnn_plus_gravity_static_rescue",
            "accuracy": rescue_outputs["accuracy"],
            "macro_f1": rescue_outputs["macro_f1"],
            "static_sitting_standing_confusions": static_confusion_total(rescue_outputs),
            "notes": "Full six-class test set with static predictions replaced by gravity-feature classifier.",
        },
        {
            "method": "phase2_tiny_ds_cnn_plus_thresholded_gravity_rescue",
            "accuracy": tuned_rescue_outputs["accuracy"],
            "macro_f1": tuned_rescue_outputs["macro_f1"],
            "static_sitting_standing_confusions": static_confusion_total(tuned_rescue_outputs),
            "notes": f"Same rescue rule, but only when gravity classifier confidence >= {best_threshold:.2f}.",
        },
    ]

    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    save_classification_outputs(static_outputs, metrics_dir, figures_dir, "gravity_static_classifier")
    save_classification_outputs(rescue_outputs, metrics_dir, figures_dir, "gravity_posture_rescue")
    save_classification_outputs(tuned_rescue_outputs, metrics_dir, figures_dir, "gravity_posture_rescue_thresholded")
    threshold_table.to_csv(metrics_dir / "gravity_rescue_threshold_sweep.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(metrics_dir / "gravity_feature_probe_comparison.csv", index=False)
    (metrics_dir / "gravity_feature_probe_comparison.json").write_text(
        json.dumps(
            {
                "comparison_rows": comparison_rows,
                "interpretation": (
                    "The gravity-direction probe tests whether accelerometer mean direction can help "
                    "separate static posture classes that were confused in M2."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"comparison_rows": comparison_rows}, indent=2))


if __name__ == "__main__":
    main()
