"""Phase 10: final, one-shot independent evaluation of Model A on test.csv.

Frozen configuration (Phases 4-9, no changes made here):
  - 502 non-spatial features: die_row, die_col, feature_1..feature_500.
  - StandardScaler -> LogisticRegression(class_weight=None, max_iter=2000,
    random_state=42).
  - Threshold = 0.42, selected in Phase 5 on train.csv pooled OOF
    predictions only, reused unchanged through Phases 7-9.
  - Spatial features (3x3 / 5x5 / combined): evaluated, not adopted.
  - LightGBM / ExtraTrees: evaluated, rejected (Phase 8).

This script does not perform any CV, threshold tuning, or model/feature
selection. It fits the already-frozen pipeline once on the full eligible
train.csv, then evaluates test.csv exactly once. The test result is a
report, not a further selection input - nothing here changes based on
what test.csv shows.

Per company_provided/README.md, evaluation is restricted to eligible dies
(old_label==0); already-failed dies (old_label==1) are excluded because
they trivially remain failed. build_model_a_dataset() enforces this same
filter used everywhere else in this project.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_config, load_test, load_train
from src.evaluate import evaluate_predictions, format_report
from src.target import build_model_a_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
FROZEN_THRESHOLD = 0.42


def build_final_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight=None, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def main():
    config = load_config()

    print("Loading train.csv, fitting frozen pipeline on full eligible training data...")
    train_raw = load_train(config)
    train_ds = build_model_a_dataset(train_raw, config)
    print(f"  Train eligible rows: {len(train_ds.X)}, features: {train_ds.X.shape[1]}")

    pipe = build_final_pipeline()
    pipe.fit(train_ds.X, train_ds.y)

    print("\nLoading test.csv for the one-shot final evaluation...")
    test_raw = load_test(config)
    n_test_total = len(test_raw)
    test_ds = build_model_a_dataset(test_raw, config)
    n_eligible = len(test_ds.X)
    n_excluded_old_fail = n_test_total - n_eligible
    print(f"  Total test.csv rows: {n_test_total}")
    print(f"  Eligible (old_label==0) rows: {n_eligible}")
    print(f"  Excluded (old_label==1) rows: {n_excluded_old_fail}")

    test_proba = pipe.predict_proba(test_ds.X)[:, 1]
    test_pred = (test_proba >= FROZEN_THRESHOLD).astype(int)

    metrics = evaluate_predictions(test_ds.y, test_pred, test_proba)

    confusion_sum = metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"]
    sanity_ok = confusion_sum == n_eligible == metrics["n_eligible"]
    assert sanity_ok, (
        f"Confusion-matrix sanity check FAILED: tp+fp+fn+tn={confusion_sum}, "
        f"n_eligible={n_eligible}, metrics['n_eligible']={metrics['n_eligible']}"
    )
    print(f"\nSanity check: TP+FP+TN+FN ({confusion_sum}) == eligible test rows ({n_eligible}) -> PASS")

    print("\n" + format_report(
        metrics,
        title="PHASE 10 - FINAL test.csv evaluation (old_label==0 eligible dies only, evaluated once):"
    ))

    models_dir = _REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model_a_final_lr.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nSaved final fitted pipeline to {model_path}")

    result_row = {
        "n_features": train_ds.X.shape[1],
        "threshold": FROZEN_THRESHOLD,
        "random_state": RANDOM_STATE,
        "n_train_eligible": len(train_ds.X),
        "n_test_total_rows": n_test_total,
        "n_test_eligible_old_label_0": n_eligible,
        "n_test_excluded_old_label_1": n_excluded_old_fail,
        "confusion_sum_tp_fp_fn_tn": confusion_sum,
        "sanity_check_passed": sanity_ok,
        "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"], "tn": metrics["tn"],
        "overall_accuracy": metrics["overall_accuracy"],
        "pass_accuracy_recall": metrics["pass_accuracy_recall"],
        "fail_accuracy_recall": metrics["fail_accuracy_recall"],
        "pass_precision": metrics["pass_precision"],
        "fail_precision": metrics["fail_precision"],
        "pass_f1": metrics["pass_f1"],
        "fail_f1": metrics["fail_f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }
    outputs_dir = _REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / "phase10_final_test_evaluation.csv"
    pd.DataFrame([result_row]).to_csv(out_path, index=False)
    print(f"Saved final evaluation metrics to {out_path}")

    return metrics, result_row


if __name__ == "__main__":
    main()
