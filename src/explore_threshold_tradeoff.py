"""EXPLORATORY threshold trade-off analysis for the frozen Model A.

This script is NOT part of the numbered Phase 4-10 pipeline and does NOT
change, retune, or replace anything about the frozen Model A. It exists
only to look at the precision/recall trade-off around the existing frozen
threshold (0.42) using train.csv pooled out-of-fold (OOF) predictions.

Data used: train.csv ONLY, via the pooled out-of-fold probabilities
produced by the existing, unmodified `wafer_grouped_oof_proba` function in
src/experiments_imbalance.py, called with class_weight=None (the exact
call that produced the frozen threshold 0.42 in Phase 5, experiment
"B: unweighted, tuned t"). Same 502 features (die_row, die_col,
feature_1..feature_500), same GroupKFold(5) grouped by wafer_id, same
StandardScaler -> LogisticRegression(class_weight=None, max_iter=2000,
random_state=42) pipeline, same evaluate_predictions() metric
implementation from src/evaluate.py.

test.csv and validation.csv are NEVER loaded anywhere in this script.

No raw OOF probability array was previously saved anywhere in the repo
(only pooled summary metrics in outputs/phase5_imbalance_experiments.csv
etc.), so the OOF probabilities are regenerated here by calling the
existing function unchanged - no new training logic, no hyperparameter,
CV, feature, or preprocessing changes.

Outputs:
  outputs/explore_threshold_tradeoff.csv   - full threshold grid, 0.10-0.90
  outputs/explore_precision_recall_curve.png

The frozen threshold (0.42) is NOT changed by this script. No new
threshold or model is selected here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.data_loading import load_config, load_train
from src.evaluate import evaluate_predictions
from src.experiments_imbalance import wafer_grouped_oof_proba
from src.target import build_model_a_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_THRESHOLD = 0.42
ANNOTATED_THRESHOLDS = [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55]
RECALL_FLOORS = [0.40, 0.45, 0.50, 0.55]


def regenerate_train_oof():
    """Train.csv-only pooled OOF probabilities via the existing, unmodified
    wafer_grouped_oof_proba (src/experiments_imbalance.py), class_weight=None.
    Does not load test.csv or validation.csv.
    """
    config = load_config()
    train_raw = load_train(config)
    train_ds = build_model_a_dataset(train_raw, config)
    X, y, wafer_id = train_ds.X, train_ds.y, train_ds.ids["wafer_id"]

    oof_proba = wafer_grouped_oof_proba(X, y, wafer_id, class_weight=None)
    return y, oof_proba


def build_threshold_grid(y, oof_proba, thresholds):
    rows = []
    for t in thresholds:
        pred = (oof_proba >= t).astype(int)
        m = evaluate_predictions(y, pred)
        rows.append({
            "threshold": round(float(t), 2),
            "is_frozen_threshold": bool(np.isclose(t, FROZEN_THRESHOLD)),
            "fail_precision": m["fail_precision"],
            "fail_recall": m["fail_accuracy_recall"],
            "fail_f1": m["fail_f1"],
            "pass_precision": m["pass_precision"],
            "pass_recall": m["pass_accuracy_recall"],
            "pass_f1": m["pass_f1"],
            "overall_accuracy": m["overall_accuracy"],
            "predicted_fail_count": m["tp"] + m["fp"],
            "predicted_pass_count": m["tn"] + m["fn"],
        })
    return pd.DataFrame(rows)


def recall_floor_analysis(grid: pd.DataFrame, floors: list[float]) -> pd.DataFrame:
    rows = []
    for floor in floors:
        candidates = grid[grid["fail_recall"] >= floor]
        if candidates.empty:
            rows.append({
                "recall_floor": floor, "threshold": None, "fail_precision": None,
                "fail_recall": None, "fail_f1": None, "predicted_fail_count": None,
            })
            continue
        best = candidates.loc[candidates["fail_precision"].idxmax()]
        rows.append({
            "recall_floor": floor,
            "threshold": best["threshold"],
            "fail_precision": best["fail_precision"],
            "fail_recall": best["fail_recall"],
            "fail_f1": best["fail_f1"],
            "predicted_fail_count": best["predicted_fail_count"],
        })
    return pd.DataFrame(rows)


def plot_pr_curve(y, oof_proba, grid: pd.DataFrame, out_path: Path):
    precision, recall, pr_thresholds = precision_recall_curve(y, oof_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="steelblue", lw=1.5,
            label="Precision-Recall curve (train.csv pooled OOF)")

    for t in ANNOTATED_THRESHOLDS:
        row = grid.loc[np.isclose(grid["threshold"], t)]
        if row.empty:
            continue
        r = row["fail_recall"].values[0]
        p = row["fail_precision"].values[0]
        is_frozen = bool(row["is_frozen_threshold"].values[0])
        color = "crimson" if is_frozen else "darkorange"
        marker_size = 90 if is_frozen else 50
        ax.scatter([r], [p], color=color, s=marker_size, zorder=5,
                    edgecolor="black", linewidth=0.6)
        label = f"{t:.2f}" + (" (frozen)" if is_frozen else "")
        ax.annotate(label, (r, p), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Fail Recall")
    ax.set_ylabel("Fail Precision")
    ax.set_title("Model A - Precision/Recall Trade-off (train.csv pooled OOF, exploratory)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 100)
    print("EXPLORATORY threshold trade-off analysis - NOT part of the numbered Phase 4-10 pipeline")
    print("Frozen Model A is NOT modified. Frozen threshold 0.42 is NOT changed. train.csv OOF only.")
    print("=" * 100)

    print("\nRegenerating train.csv pooled OOF probabilities via wafer_grouped_oof_proba(class_weight=None) ...")
    y, oof_proba = regenerate_train_oof()
    print(f"  OOF probabilities: {len(oof_proba)} rows, {np.isnan(oof_proba).sum()} NaN (should be 0)")

    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    grid = build_threshold_grid(y, oof_proba, thresholds)

    outputs_dir = _REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    csv_path = outputs_dir / "explore_threshold_tradeoff.csv"
    grid.to_csv(csv_path, index=False)
    print(f"\nSaved full threshold grid ({len(grid)} rows) to {csv_path}")

    png_path = outputs_dir / "explore_precision_recall_curve.png"
    plot_pr_curve(y, oof_proba, grid, png_path)
    print(f"Saved precision-recall curve to {png_path}")

    frozen_row = grid.loc[grid["is_frozen_threshold"]].iloc[0]
    print("\n" + "-" * 100)
    print("A. FROZEN OPERATING POINT (threshold = 0.42)")
    print("-" * 100)
    print(frozen_row.to_string())

    floor_df = recall_floor_analysis(grid, RECALL_FLOORS)
    print("\n" + "-" * 100)
    print("B. RECALL-FLOOR ANALYSIS (max Fail Precision subject to Fail Recall >= floor)")
    print("-" * 100)
    print(floor_df.to_string(index=False))

    print("\n" + "-" * 100)
    print("Nearby operating points (0.30-0.55) for reference")
    print("-" * 100)
    nearby = grid.loc[grid["threshold"].isin(ANNOTATED_THRESHOLDS)]
    print(nearby[["threshold", "is_frozen_threshold", "fail_precision", "fail_recall",
                   "fail_f1", "predicted_fail_count"]].to_string(index=False))

    print("\nSTOP: exploratory analysis complete. No threshold changed, no model selected.")
    return grid, floor_df


if __name__ == "__main__":
    main()
