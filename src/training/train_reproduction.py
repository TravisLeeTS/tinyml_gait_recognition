from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import DEFAULT_DEVICE, OUTPUTS_DIR, SEED
from src.data.uci_har import class_count_table, load_uci_har_sequence_dataset
from src.models.reproduction.keras_models import REPRODUCTION_MODEL_NAMES, build_reproduction_model
from src.training.tf_common import predict_proba, train_classifier
from src.utils.metrics import classification_outputs, save_classification_outputs, top_confusion_pairs
from src.utils.normalization import SequenceMinMaxScaler, SequenceStandardizer
from src.utils.reproducibility import configure_tensorflow_device, set_global_seed


def _normalizer(kind: str):
    if kind == "minmax":
        return SequenceMinMaxScaler
    if kind == "standard":
        return SequenceStandardizer
    raise ValueError(f"Unsupported normalization: {kind}")


def _xgb_grid(seed: int, fast: bool) -> GridSearchCV:
    params = {
        "n_estimators": [100] if fast else [100, 200],
        "max_depth": [2] if fast else [2, 3],
        "learning_rate": [0.1] if fast else [0.05, 0.1],
        "gamma": [0.0] if fast else [0.0, 0.1],
    }
    estimator = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=6,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper reproduction stacking baseline.")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--normalization", choices=["minmax", "standard"], default="minmax")
    parser.add_argument("--models", nargs="*", default=REPRODUCTION_MODEL_NAMES)
    parser.add_argument("--subsequences", type=int, default=4)
    parser.add_argument("--fast-xgb-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "reproduction")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = configure_tensorflow_device(args.device)
    data = load_uci_har_sequence_dataset(download=True, seed=args.seed)
    scaler_cls = _normalizer(args.normalization)
    scaler = scaler_cls.fit(data.train.x)
    x_train = scaler.transform(data.train.x)
    x_val = scaler.transform(data.val.x)
    x_test = scaler.transform(data.test.x)

    model_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    logs_dir = args.output_dir / "logs"
    for path in [model_dir, metrics_dir, figures_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    scaler.save(model_dir / f"reproduction_{args.normalization}_normalizer.json")
    (metrics_dir / "uci_har_class_counts.csv").write_text(
        class_count_table(data).to_csv(index=False),
        encoding="utf-8",
    )

    run_info = {
        "baseline_role": "Paper Reproduction Baseline",
        "baseline_description": "Offline reference baseline; not intended for direct Arduino deployment.",
        "framework": "TensorFlow/Keras implementation of paper-specified model families",
        "paper_framework": "TensorFlow 2.9 reported by paper",
        "input_shape": list(x_train.shape[1:]),
        "output_classes": data.class_names,
        "models": args.models,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.lr_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "early_stopping": {"monitor": "val_loss", "patience": args.patience},
        "loss": "cross_entropy",
        "normalization": args.normalization,
        "subsequences": args.subsequences,
        "xgboost_meta_learner": "GridSearchCV with 5-fold CV on validation-stack predictions",
        "device": device,
    }
    (model_dir / "reproduction_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    val_stack = []
    test_stack = []
    base_results = []

    for model_idx, model_name in enumerate(args.models):
        print(f"\n=== Training {model_name} ===")
        model = build_reproduction_model(
            model_name,
            input_shape=tuple(x_train.shape[1:]),
            num_classes=len(data.class_names),
            subsequences=args.subsequences,
        )
        model_path = model_dir / f"{model_name.lower().replace('-', '_')}.keras"
        history = train_classifier(
            model=model,
            x_train=x_train,
            y_train=data.train.y,
            x_val=x_val,
            y_val=data.val.y,
            output_path=model_path,
            seed=args.seed + model_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_decay=args.lr_decay,
            patience=args.patience,
        )
        history.history.to_csv(
            logs_dir / f"{model_name.lower().replace('-', '_')}_training_history.csv",
            index=False,
        )
        val_probs = predict_proba(model, x_val)
        test_probs = predict_proba(model, x_test)
        val_stack.append(val_probs)
        test_stack.append(test_probs)

        base_pred = test_probs.argmax(axis=1)
        base_output = classification_outputs(data.test.y, base_pred, data.class_names, model_name)
        base_output["model_info"] = {
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "paper_role": "level-0 learner",
            "keras_model_path": str(model_path),
            "keras_model_parameters": int(model.count_params()),
        }
        base_output["top_confusion_pairs"] = top_confusion_pairs(
            np.asarray(base_output["confusion_matrix"]),
            data.class_names,
        )
        save_classification_outputs(
            base_output,
            metrics_dir,
            figures_dir,
            f"{model_name.lower().replace('-', '_')}",
        )
        base_results.append({k: base_output[k] for k in ["model_name", "accuracy", "macro_f1", "weighted_f1"]})

    val_features = np.concatenate(val_stack, axis=1)
    test_features = np.concatenate(test_stack, axis=1)
    np.save(metrics_dir / "stacked_val_predictions.npy", val_features)
    np.save(metrics_dir / "stacked_test_predictions.npy", test_features)

    print("\n=== Training XGBoost meta-learner ===")
    grid = _xgb_grid(args.seed, fast=args.fast_xgb_grid)
    grid.fit(val_features, data.val.y)
    meta = grid.best_estimator_
    joblib.dump(meta, model_dir / "xgboost_meta_learner.joblib")
    xgb_info = {
        "best_params": grid.best_params_,
        "best_cv_macro_f1": float(grid.best_score_),
        "cv_results_csv": str(logs_dir / "xgboost_grid_search_results.csv"),
    }
    pd.DataFrame(grid.cv_results_).to_csv(logs_dir / "xgboost_grid_search_results.csv", index=False)
    (model_dir / "xgboost_meta_learner_info.json").write_text(json.dumps(xgb_info, indent=2), encoding="utf-8")

    y_pred = meta.predict(test_features)
    outputs = classification_outputs(data.test.y, y_pred, data.class_names, "XGBoost Stacking Meta-Learner")
    outputs["model_info"] = run_info | {"xgboost": xgb_info, "base_model_results": base_results}
    outputs["top_confusion_pairs"] = top_confusion_pairs(np.asarray(outputs["confusion_matrix"]), data.class_names)
    save_classification_outputs(outputs, metrics_dir, figures_dir, "paper_reproduction_stacking")
    print(json.dumps({k: outputs[k] for k in ["accuracy", "macro_f1", "weighted_f1", "test_samples"]}, indent=2))


if __name__ == "__main__":
    main()
