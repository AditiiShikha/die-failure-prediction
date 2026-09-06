"""Phase 7: spatial-feature experiments for Model A.

Reference: Phase 5 winner = StandardScaler + LogisticRegression(class_weight=None),
threshold=0.42, wafer-grouped 5-fold CV, 502 features (die_row, die_col,
feature_1..feature_500). That threshold was already selected in Phase 5 and
is kept fixed here rather than re-tuned.

Spatial features (src/features.py, Phase 6) are computed on the FULL wafer
population (old_label 0 and 1) before eligibility filtering - see
src/features.py module docstring for the leakage rationale. Three variants
are compared, each unweighted LR with its own threshold tuned on pooled
train.csv out-of-fold CV probabilities (never test.csv):
  - 3x3-only:    502 + dist_from_center, edge_proximity, neigh_*_m3   = 507 features
  - 5x5-only:    502 + dist_from_center, edge_proximity, neigh_*_m5   = 507 features
  - 3x3+5x5:     502 + dist_from_center, edge_proximity, neigh_*_m3/5 = 510 features

The best spatial variant is selected by CV Fail F1 among the three spatial
variants (the reference is a comparison point, not a selection candidate).
test.csv is touched exactly once, after that selection is already fixed.
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
from src.evaluate import evaluate_predictions, format_report, select_best_threshold
from src.features import add_spatial_features, spatial_feature_names
from src.target import build_model_a_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
N_SPLITS = 5
REFERENCE_THRESHOLD = 0.42  # Phase 5 CV-tuned threshold for unweighted LR, fixed here


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight=None, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def wafer_grouped_oof_proba(X, y, wafer_id, n_splits=N_SPLITS):
    """Out-of-fold probabilities for every row of X, train.csv only."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.full(len(X), np.nan)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=wafer_id)):
        train_wafers = set(wafer_id.iloc[train_idx])
        val_wafers = set(wafer_id.iloc[val_idx])
        assert train_wafers.isdisjoint(val_wafers), f"wafer leakage in fold {fold_idx}"
        pipe = build_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    assert not np.isnan(oof_proba).any(), "every row should get an out-of-fold prediction exactly once"
    return oof_proba


def cv_only(name, X, y, wafer_id, tune_threshold, fixed_threshold=None):
    oof_proba = wafer_grouped_oof_proba(X, y, wafer_id)
    if tune_threshold:
        threshold, _, _ = select_best_threshold(y, oof_proba, metric="fail_f1")
    else:
        threshold = fixed_threshold
    pred = (oof_proba >= threshold).astype(int)
    metrics = evaluate_predictions(y, pred, oof_proba)
    return {
        "experiment": name,
        "n_features": X.shape[1],
        "threshold": threshold,
        "cv_fail_f1": metrics["fail_f1"],
        "cv_fail_precision": metrics["fail_precision"],
        "cv_fail_recall": metrics["fail_accuracy_recall"],
        "cv_pr_auc": metrics["pr_auc"],
        "cv_roc_auc": metrics["roc_auc"],
    }


def main():
    config = load_config()
    train_raw = load_train(config)
    test_raw = load_test(config)

    ref_train_ds = build_model_a_dataset(train_raw, config)
    wafer_id_train = ref_train_ds.ids["wafer_id"]
    ref_ids = ref_train_ds.ids.reset_index(drop=True)

    results = []
    print("Running: reference (502 features, unweighted, t=0.42 fixed from Phase 5)")
    results.append(cv_only("reference (502, t=0.42)", ref_train_ds.X, ref_train_ds.y, wafer_id_train,
                           tune_threshold=False, fixed_threshold=REFERENCE_THRESHOLD))

    variants = {
        "3x3-only (507)": (3,),
        "5x5-only (507)": (5,),
        "3x3+5x5 (510)": (3, 5),
    }
    variant_datasets = {}
    for name, ws in variants.items():
        print(f"Running: {name}")
        extra_cols = spatial_feature_names(ws)
        train_spatial = add_spatial_features(train_raw, window_sizes=ws)
        ds = build_model_a_dataset(train_spatial, config, extra_feature_columns=extra_cols)

        ids = ds.ids.reset_index(drop=True)
        assert (ids[["wafer_id", "die_row", "die_col"]] == ref_ids[["wafer_id", "die_row", "die_col"]]).all().all(), \
            f"{name}: eligible row order differs from reference - wafer_id grouping would be misaligned"

        variant_datasets[name] = (ds, ws, extra_cols)
        results.append(cv_only(name, ds.X, ds.y, wafer_id_train, tune_threshold=True))

    df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print("PHASE 7 CV COMPARISON (pooled out-of-fold, train.csv only)")
    print("=" * 100)
    print(df.to_string(index=False))

    spatial_rows = df[df["experiment"] != "reference (502, t=0.42)"]
    best_idx = spatial_rows["cv_fail_f1"].idxmax()
    best_name = spatial_rows.loc[best_idx, "experiment"]
    best_threshold = spatial_rows.loc[best_idx, "threshold"]
    print(f"\nSelected best spatial variant by CV Fail F1: {best_name} (threshold={best_threshold:.2f})")

    print(f"\n--- Fitting {best_name} on full train.csv, evaluating once on test.csv ---")
    ds, ws, extra_cols = variant_datasets[best_name]
    test_spatial = add_spatial_features(test_raw, window_sizes=ws)
    test_ds = build_model_a_dataset(test_spatial, config, extra_feature_columns=extra_cols)

    final_pipe = build_pipeline()
    final_pipe.fit(ds.X, ds.y)
    test_proba = final_pipe.predict_proba(test_ds.X)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    test_metrics = evaluate_predictions(test_ds.y, test_pred, test_proba)
    print("\n" + format_report(test_metrics, title=f"{best_name} on test.csv (evaluated once):"))

    outputs_dir = _REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    df.to_csv(outputs_dir / "phase7_spatial_experiments.csv", index=False)
    print(f"\nSaved CV comparison table to {outputs_dir / 'phase7_spatial_experiments.csv'}")

    models_dir = _REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model_a_spatial_best.joblib"
    joblib.dump(final_pipe, model_path)
    print(f"Saved selected model to {model_path}")

    return df, best_name, best_threshold, test_metrics


if __name__ == "__main__":
    main()
