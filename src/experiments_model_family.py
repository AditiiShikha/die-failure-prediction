"""Phase 8: model-family comparison for Model A (LR vs LightGBM vs ExtraTrees).

Isolates model family only. Everything else is frozen:
  - Feature set: the 502 non-spatial features (die_row, die_col,
    feature_1..feature_500). No spatial/neighborhood features (Phase 6/7).
  - Imbalance handling: unweighted training for every model. LightGBM
    keeps is_unbalance=False and scale_pos_weight=1.0 explicitly (both
    already the library defaults) so it gets no implicit rebalancing that
    LR/ExtraTrees don't also get.
  - CV folds: all three models reuse one shared GroupKFold(n_splits=5)
    split (computed once from the single shared X/y/wafer_id derived from
    the 502-feature dataset), so fold membership is identical by
    construction - not re-verified per-model the way Phase 7's spatial
    variants needed to be, since there's only one dataset here.
  - No hyperparameter search: every model uses its library's own default
    settings (documented at each constructor) plus a fixed random_state.

Missing values: verified 0 NaN / 0 inf across all 502 features on the
eligible population (see printed check below) - no imputer added.

Threshold: pooled out-of-fold probabilities per model, tuned once via
src.evaluate.select_best_threshold (same np.arange(0.01, 1.00, 0.01) grid
used in Phase 5/7), maximizing Fail F1. Applied only after CV is complete.

test.csv is not loaded or touched anywhere in this script.
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_config, load_train
from src.evaluate import evaluate_predictions, select_best_threshold
from src.target import build_model_a_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
N_SPLITS = 5
EXTRATREES_FOLD_TIME_LIMIT_SEC = 20 * 60  # runtime guard: skip ExtraTrees if fold 0 alone exceeds this


def build_lr_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight=None, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_lightgbm():
    # Library defaults, passed explicitly for documentation - no tuning.
    return LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        is_unbalance=False,
        scale_pos_weight=1.0,
        n_jobs=-1,
        verbosity=-1,
    )


def build_extratrees():
    # Library defaults, passed explicitly for documentation - no tuning.
    return ExtraTreesClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run_model_cv(name, build_model, X, y, wafer_id, folds, fold_time_limit=None):
    """Pooled OOF probabilities + per-fold timing. Returns (oof_proba, fold_times, abort_reason)."""
    oof_proba = np.full(len(X), np.nan)
    fold_times = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_wafers = set(wafer_id.iloc[train_idx])
        val_wafers = set(wafer_id.iloc[val_idx])
        assert train_wafers.isdisjoint(val_wafers), f"wafer leakage in fold {fold_idx}"

        model = build_model()
        t0 = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        t_fit = time.perf_counter() - t0

        if fold_idx == 0 and fold_time_limit is not None and t_fit > fold_time_limit:
            reason = (f"{name}: fold 0 fit took {t_fit:.1f}s, exceeding the "
                      f"{fold_time_limit}s guard - aborting per Phase 8 runtime constraint")
            return None, fold_times, reason

        t0 = time.perf_counter()
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        t_pred = time.perf_counter() - t0

        model_bytes = len(pickle.dumps(model)) if fold_idx == 0 else None

        oof_proba[val_idx] = proba
        fold_times.append({
            "fold": fold_idx, "n_train": len(train_idx), "n_val": len(val_idx),
            "fit_sec": t_fit, "predict_sec": t_pred, "model_pickle_bytes": model_bytes,
        })
        size_note = f" model_size~{model_bytes / 1e6:.1f}MB" if model_bytes else ""
        print(f"  [{name}] fold {fold_idx}: fit={t_fit:.1f}s predict={t_pred:.2f}s "
              f"(train={len(train_idx)} val={len(val_idx)}){size_note}")

    assert not np.isnan(oof_proba).any(), f"{name}: every row should get an OOF prediction exactly once"
    return oof_proba, fold_times, None


def summarize(name, y, wafer_id, folds, oof_proba, threshold):
    pooled_pred = (oof_proba >= threshold).astype(int)
    pooled = evaluate_predictions(y, pooled_pred, oof_proba)

    fold_f1 = []
    for fold_idx, (_, val_idx) in enumerate(folds):
        pred = (oof_proba[val_idx] >= threshold).astype(int)
        m = evaluate_predictions(y.iloc[val_idx], pred, oof_proba[val_idx])
        fold_f1.append(m["fail_f1"])

    return {
        "model": name,
        "threshold": threshold,
        "pooled_fail_f1": pooled["fail_f1"],
        "pooled_fail_precision": pooled["fail_precision"],
        "pooled_fail_recall": pooled["fail_accuracy_recall"],
        "pooled_pr_auc": pooled["pr_auc"],
        "pooled_roc_auc": pooled["roc_auc"],
        "pooled_overall_accuracy": pooled["overall_accuracy"],
        "fold_fail_f1": fold_f1,
        "fold_fail_f1_mean": float(np.mean(fold_f1)),
        "fold_fail_f1_std": float(np.std(fold_f1, ddof=1)),
    }


def main():
    config = load_config()
    train_raw = load_train(config)
    ds = build_model_a_dataset(train_raw, config)
    X, y, wafer_id = ds.X, ds.y, ds.ids["wafer_id"]

    na_total = int(X.isna().sum().sum())
    inf_total = int(np.isinf(X.to_numpy(dtype=float)).sum())
    print(f"Missing-value check on 502-feature X: {na_total} NaN cells, {inf_total} inf cells "
          f"-> {'no imputation needed' if na_total == 0 and inf_total == 0 else 'IMPUTATION REQUIRED'}")
    if na_total > 0 or inf_total > 0:
        raise RuntimeError("Missing/inf values found in X; Phase 8 plan requires resolving imputation before proceeding.")

    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups=wafer_id))
    print(f"Shared GroupKFold({N_SPLITS}) folds (n_train, n_val) per fold: "
          f"{[(len(tr), len(va)) for tr, va in folds]}")
    print("All three models below reuse these exact same fold splits (same X/y/wafer_id).\n")

    results = []
    runtime_log = {}

    print("=== Logistic Regression (reference: unweighted, StandardScaler, random_state=42) ===")
    t0 = time.perf_counter()
    lr_oof, lr_times, lr_abort = run_model_cv("LR", build_lr_pipeline, X, y, wafer_id, folds)
    lr_total = time.perf_counter() - t0
    lr_thr, _, _ = select_best_threshold(y, lr_oof, metric="fail_f1")
    results.append(summarize("Logistic Regression", y, wafer_id, folds, lr_oof, lr_thr))
    runtime_log["Logistic Regression"] = {"total_cv_sec": lr_total, "fold_times": lr_times}
    print(f"  total CV time: {lr_total:.1f}s, tuned threshold={lr_thr:.2f}\n")

    print("=== LightGBM (unweighted, is_unbalance=False, scale_pos_weight=1.0, random_state=42) ===")
    t0 = time.perf_counter()
    lgbm_oof, lgbm_times, lgbm_abort = run_model_cv("LightGBM", build_lightgbm, X, y, wafer_id, folds)
    lgbm_total = time.perf_counter() - t0
    lgbm_thr, _, _ = select_best_threshold(y, lgbm_oof, metric="fail_f1")
    results.append(summarize("LightGBM", y, wafer_id, folds, lgbm_oof, lgbm_thr))
    runtime_log["LightGBM"] = {"total_cv_sec": lgbm_total, "fold_times": lgbm_times}
    print(f"  total CV time: {lgbm_total:.1f}s, tuned threshold={lgbm_thr:.2f}\n")

    print("=== ExtraTrees (unweighted, n_estimators=100, random_state=42) ===")
    t0 = time.perf_counter()
    et_oof, et_times, et_abort = run_model_cv("ExtraTrees", build_extratrees, X, y, wafer_id, folds,
                                               fold_time_limit=EXTRATREES_FOLD_TIME_LIMIT_SEC)
    et_total = time.perf_counter() - t0
    if et_abort:
        print(f"  ABORTED: {et_abort}\n")
        runtime_log["ExtraTrees"] = {"aborted": True, "reason": et_abort, "fold_times": et_times}
    else:
        et_thr, _, _ = select_best_threshold(y, et_oof, metric="fail_f1")
        results.append(summarize("ExtraTrees", y, wafer_id, folds, et_oof, et_thr))
        runtime_log["ExtraTrees"] = {"total_cv_sec": et_total, "fold_times": et_times}
        print(f"  total CV time: {et_total:.1f}s, tuned threshold={et_thr:.2f}\n")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "fold_fail_f1"} for r in results])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=" * 110)
    print("PHASE 8 POOLED-OOF COMPARISON (train.csv only, single global threshold per model)")
    print("=" * 110)
    print(df.to_string(index=False))

    fold_df = pd.DataFrame({r["model"]: r["fold_fail_f1"] for r in results})
    fold_df.index.name = "fold"
    print("\nPer-fold Fail F1 (each model's own tuned threshold applied within each fold's OOF slice):")
    print(fold_df.to_string())
    print("\nFold mean/std (ddof=1):")
    for r in results:
        print(f"  {r['model']:22s} mean={r['fold_fail_f1_mean']:.6f}  std={r['fold_fail_f1_std']:.6f}")

    print("\nRuntime summary:")
    for name, log in runtime_log.items():
        if log.get("aborted"):
            print(f"  {name}: ABORTED - {log['reason']}")
            continue
        fit_secs = [t["fit_sec"] for t in log["fold_times"]]
        pred_secs = [t["predict_sec"] for t in log["fold_times"]]
        size0 = log["fold_times"][0]["model_pickle_bytes"]
        print(f"  {name}: total_cv={log['total_cv_sec']:.1f}s  "
              f"avg_fold_fit={np.mean(fit_secs):.2f}s  avg_fold_predict={np.mean(pred_secs):.3f}s  "
              f"fold0_model_pickle_size={size0 / 1e6:.2f}MB")
    print("  (psutil not installed - no process RSS measured; wall-clock time and "
          "pickled fold-0 model size are used as rough complexity/memory proxies.)")

    outputs_dir = _REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    df.to_csv(outputs_dir / "phase8_model_family_experiments.csv", index=False)
    fold_df.to_csv(outputs_dir / "phase8_per_fold_fail_f1.csv")
    print(f"\nSaved pooled comparison to {outputs_dir / 'phase8_model_family_experiments.csv'}")
    print(f"Saved per-fold Fail F1 to {outputs_dir / 'phase8_per_fold_fail_f1.csv'}")

    return df, fold_df, runtime_log


if __name__ == "__main__":
    main()
