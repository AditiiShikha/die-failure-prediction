"""Model A baseline: Logistic Regression on die-level + raw coordinate features.

Deliberately simple — no spatial/neighborhood features yet (that's Phase 6).
This establishes how well plain die-level parametric features separate
newly-failed from passing dies, before any spatial engineering.

Design choices (see Phase 4 write-up for full rationale):
- StandardScaler + LogisticRegression: features span very different
  scales (config: base_mean log-uniform over [0.1, 5000]), so scaling
  is required for a linear model.
- class_weight="balanced": reweights the loss for the ~4.23% positive
  class. No resampling, no data duplication -> safe on every fold.
- GroupKFold by wafer_id: no wafer's dies ever span train/val within a
  fold, preventing within-wafer leakage during model selection.
- test.csv is evaluated exactly once, after the pipeline is fixed, and
  is never used to choose between alternatives.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_config, load_test, load_train
from src.evaluate import evaluate_predictions, format_report
from src.target import build_model_a_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5  # not specified anywhere in company materials; placeholder to revisit


def build_baseline_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])


def wafer_grouped_cv(X: pd.DataFrame, y: pd.Series, wafer_id: pd.Series,
                      n_splits: int = 5, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=wafer_id)):
        train_wafers = set(wafer_id.iloc[train_idx])
        val_wafers = set(wafer_id.iloc[val_idx])
        assert train_wafers.isdisjoint(val_wafers), f"wafer leakage detected in fold {fold_idx}"

        pipe = build_baseline_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

        proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        pred = (proba >= threshold).astype(int)

        metrics = evaluate_predictions(y.iloc[val_idx], pred, proba)
        metrics["fold"] = fold_idx
        metrics["n_train_wafers"] = len(train_wafers)
        metrics["n_val_wafers"] = len(val_wafers)
        fold_metrics.append(metrics)
    return fold_metrics


def main():
    config = load_config()

    print("Loading train.csv / test.csv (block_readings excluded at read time)...")
    train_raw = load_train(config)
    test_raw = load_test(config)

    train_ds = build_model_a_dataset(train_raw, config)
    test_ds = build_model_a_dataset(test_raw, config)
    del train_raw, test_raw

    print(f"Train (eligible): X={train_ds.X.shape}, positive rate={train_ds.y.mean():.4%}")
    print(f"Test  (eligible): X={test_ds.X.shape}, positive rate={test_ds.y.mean():.4%}")

    print("\n--- Wafer-grouped 5-fold CV on train.csv (model-selection signal) ---")
    fold_metrics = wafer_grouped_cv(train_ds.X, train_ds.y, train_ds.ids["wafer_id"], n_splits=5)
    for m in fold_metrics:
        print(f"Fold {m['fold']} (train_wafers={m['n_train_wafers']}, val_wafers={m['n_val_wafers']}): "
              f"Fail F1={m['fail_f1']:.4f}  Fail Recall={m['fail_accuracy_recall']:.4f}  "
              f"Fail Precision={m['fail_precision']:.4f}  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

    cv_df = pd.DataFrame(fold_metrics)
    print("\nCV mean +/- std (across 5 wafer-grouped folds):")
    for col in ["overall_accuracy", "pass_accuracy_recall", "fail_accuracy_recall",
                "pass_precision", "fail_precision", "pass_f1", "fail_f1", "roc_auc", "pr_auc"]:
        print(f"  {col:24s} {cv_df[col].mean():.4f} +/- {cv_df[col].std():.4f}")

    print("\n--- Fit on full train.csv, evaluate once on test.csv (not used for selection) ---")
    final_pipe = build_baseline_pipeline()
    final_pipe.fit(train_ds.X, train_ds.y)

    test_proba = final_pipe.predict_proba(test_ds.X)[:, 1]
    test_pred = (test_proba >= DEFAULT_THRESHOLD).astype(int)
    test_metrics = evaluate_predictions(test_ds.y, test_pred, test_proba)
    print(format_report(test_metrics, title="Baseline (Logistic Regression) on test.csv:"))

    models_dir = _REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model_a_baseline_logreg.joblib"
    joblib.dump(final_pipe, model_path)
    print(f"\nSaved baseline pipeline to {model_path}")

    return cv_df, test_metrics


if __name__ == "__main__":
    main()
