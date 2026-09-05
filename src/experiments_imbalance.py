"""Phase 5: imbalance-handling experiments for Model A.

Reference: Phase 4 baseline = StandardScaler + LogisticRegression(class_weight="balanced"),
threshold=0.5, wafer-grouped 5-fold CV (src/train.py).

All experiments here reuse the identical feature set (die_row, die_col,
feature_1..feature_500) and the identical GroupKFold(n_splits=5) wafer
split. test.csv is evaluated exactly once per experiment, after the
model/threshold are already fixed from train.csv-only data - never used
for selection.

Threshold tuning (experiments B, C) selects the threshold that maximizes
Fail F1 on pooled out-of-fold CV probabilities from train.csv only.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_config, load_test, load_train
from src.evaluate import evaluate_predictions, format_report, select_best_threshold
from src.target import build_model_a_dataset

RANDOM_STATE = 42
N_SPLITS = 5


def build_pipeline(class_weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def wafer_grouped_oof_proba(X, y, wafer_id, class_weight, n_splits=N_SPLITS):
    """Out-of-fold probabilities for every row of X, train.csv only."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.full(len(X), np.nan)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=wafer_id)):
        train_wafers = set(wafer_id.iloc[train_idx])
        val_wafers = set(wafer_id.iloc[val_idx])
        assert train_wafers.isdisjoint(val_wafers), f"wafer leakage in fold {fold_idx}"
        pipe = build_pipeline(class_weight)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    assert not np.isnan(oof_proba).any(), "every row should get an out-of-fold prediction exactly once"
    return oof_proba


def run_experiment(name, X_train, y_train, wafer_id_train, X_test, y_test,
                    class_weight, tune_threshold, fixed_threshold=0.5):
    oof_proba = wafer_grouped_oof_proba(X_train, y_train, wafer_id_train, class_weight)

    if tune_threshold:
        threshold, _, _ = select_best_threshold(y_train, oof_proba, metric="fail_f1")
    else:
        threshold = fixed_threshold

    cv_pred = (oof_proba >= threshold).astype(int)
    cv_metrics = evaluate_predictions(y_train, cv_pred, oof_proba)

    final_pipe = build_pipeline(class_weight)
    final_pipe.fit(X_train, y_train)
    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = evaluate_predictions(y_test, test_pred, test_proba)

    return {
        "experiment": name,
        "class_weight": str(class_weight),
        "threshold": threshold,
        "cv_fail_f1": cv_metrics["fail_f1"],
        "cv_fail_recall": cv_metrics["fail_accuracy_recall"],
        "cv_fail_precision": cv_metrics["fail_precision"],
        "cv_pr_auc": cv_metrics["pr_auc"],
        "cv_roc_auc": cv_metrics["roc_auc"],
        "test_fail_f1": test_metrics["fail_f1"],
        "test_fail_recall": test_metrics["fail_accuracy_recall"],
        "test_fail_precision": test_metrics["fail_precision"],
        "test_pass_f1": test_metrics["pass_f1"],
        "test_overall_accuracy": test_metrics["overall_accuracy"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "_test_metrics_full": test_metrics,
    }


def main():
    config = load_config()
    train_raw = load_train(config)
    test_raw = load_test(config)
    train_ds = build_model_a_dataset(train_raw, config)
    test_ds = build_model_a_dataset(test_raw, config)
    del train_raw, test_raw

    wafer_id_train = train_ds.ids["wafer_id"]
    results = []

    print("Running: baseline (balanced, t=0.5) [Phase 4 reference, re-run here for a like-for-like table]")
    results.append(run_experiment(
        "baseline (balanced, t=0.5)", train_ds.X, train_ds.y, wafer_id_train,
        test_ds.X, test_ds.y, class_weight="balanced", tune_threshold=False, fixed_threshold=0.5))

    print("Running: A - unweighted, t=0.5 (imbalance ablation)")
    results.append(run_experiment(
        "A: unweighted, t=0.5", train_ds.X, train_ds.y, wafer_id_train,
        test_ds.X, test_ds.y, class_weight=None, tune_threshold=False, fixed_threshold=0.5))

    print("Running: B - unweighted, CV-tuned threshold")
    results.append(run_experiment(
        "B: unweighted, tuned t", train_ds.X, train_ds.y, wafer_id_train,
        test_ds.X, test_ds.y, class_weight=None, tune_threshold=True))

    print("Running: C - balanced, CV-tuned threshold")
    results.append(run_experiment(
        "C: balanced, tuned t", train_ds.X, train_ds.y, wafer_id_train,
        test_ds.X, test_ds.y, class_weight="balanced", tune_threshold=True))

    df = pd.DataFrame(results).drop(columns=["_test_metrics_full"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print("PHASE 5 COMPARISON TABLE (test.csv, evaluated once per experiment)")
    print("=" * 100)
    print(df.to_string(index=False))

    for r in results:
        print("\n" + format_report(r["_test_metrics_full"], title=f"{r['experiment']} on test.csv:"))

    df.to_csv("outputs/phase5_imbalance_experiments.csv", index=False)
    print("\nSaved comparison table to outputs/phase5_imbalance_experiments.csv")

    return df


if __name__ == "__main__":
    main()
