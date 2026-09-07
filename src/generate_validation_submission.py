"""Generate the company-format validation.csv submission for the frozen Model A.

This script does NOT train, retune, or modify anything about Model A. It
loads the already-frozen fitted pipeline (models/model_a_final_lr.joblib,
produced once in Phase 10 by src/final_evaluation.py) and applies it,
unchanged, to validation.csv.

Frozen configuration reused as-is (no changes made here):
  - Pipeline: StandardScaler -> LogisticRegression(class_weight=None,
    max_iter=2000, random_state=42), loaded from models/model_a_final_lr.joblib.
  - Features: the same 502-column matrix used everywhere else in this
    project (die_row, die_col, feature_1..feature_500), built via the
    existing src.target.build_model_a_dataset - no new feature-construction
    logic is written here.
  - Threshold: 0.42, frozen (Phase 5, reused unchanged through Phase 10).

Per company_provided/README.md's submission/evaluation rules:
  - Only old_label==0 (eligible) dies are actually scored by the model.
  - old_label==1 dies trivially remain failed and get predicted_label=1
    directly, without being passed through the model - build_model_a_dataset
    already excludes them from the eligible X/ids it returns, so this
    script fills them in separately after scoring the eligible subset.
  - The output must cover every row of validation.csv (both old_label
    values), with columns wafer_id, die_row, die_col, predicted_label.

validation.csv has no `label` column (verified against its header) and no
label of any kind is read or used here for model selection, threshold
tuning, or evaluation - this script only ever *applies* the already-frozen
model/threshold to unlabeled data and writes predictions.

test.csv is never loaded or referenced anywhere in this script.
company_provided/ and models/model_a_final_lr.joblib are only ever read,
never written, here.

Run as a script from the repo root:
    python -m src.generate_validation_submission
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loading import load_config, load_validation
from src.target import build_model_a_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_THRESHOLD = 0.42
MODEL_PATH = _REPO_ROOT / "models" / "model_a_final_lr.joblib"
OUTPUT_PATH = _REPO_ROOT / "outputs" / "validation_submission_model_a.csv"
ID_COLS = ["wafer_id", "die_row", "die_col"]


def main():
    config = load_config()
    elig_col = config["model_a"]["eligibility_column"]   # "old_label"
    elig_val = config["model_a"]["eligibility_value"]    # 0

    print("Loading validation.csv (block_readings excluded at read time, no label column present)...")
    val_raw = load_validation(config)
    n_validation_rows = len(val_raw)
    print(f"  validation.csv rows: {n_validation_rows}")

    full_ids = val_raw[ID_COLS + [elig_col]].copy()
    n_old_label_0 = int((full_ids[elig_col] == 0).sum())
    n_old_label_1 = int((full_ids[elig_col] == 1).sum())
    print(f"  old_label==0 (eligible): {n_old_label_0}   old_label==1: {n_old_label_1}")
    assert n_old_label_0 + n_old_label_1 == n_validation_rows, "old_label must be strictly 0/1 for every row"

    print("\nBuilding the exact frozen 502-feature Model A matrix via build_model_a_dataset "
          "(die_row, die_col, feature_1..feature_500) for eligible (old_label==0) rows only...")
    val_ds = build_model_a_dataset(val_raw, config)
    print(f"  Eligible X shape: {val_ds.X.shape}")
    assert val_ds.X.shape[1] == 502, f"expected 502 features, got {val_ds.X.shape[1]}"
    assert len(val_ds.X) == n_old_label_0, "eligible row count must match old_label==0 count"
    assert val_ds.y is None, "validation.csv must not carry a label column"

    print(f"\nLoading frozen fitted pipeline from {MODEL_PATH} (read-only, never re-fit)...")
    pipe = joblib.load(MODEL_PATH)

    proba = pipe.predict_proba(val_ds.X)[:, 1]
    pred_eligible = (proba >= FROZEN_THRESHOLD).astype(int)
    print(f"  Scored {len(pred_eligible)} eligible rows at frozen threshold {FROZEN_THRESHOLD}")

    eligible_pred_df = val_ds.ids[ID_COLS].copy()
    eligible_pred_df["predicted_label"] = pred_eligible

    print("\nMerging eligible predictions back onto the full validation.csv row set "
          "(left merge on wafer_id/die_row/die_col, preserves original row order)...")
    submission = full_ids.merge(eligible_pred_df, on=ID_COLS, how="left")
    assert len(submission) == n_validation_rows, "merge must not change row count"

    old_fail_mask = submission[elig_col] == 1
    assert submission.loc[old_fail_mask, "predicted_label"].isna().all(), \
        "old_label==1 rows should be unscored (NaN) before the explicit fill below"
    submission.loc[old_fail_mask, "predicted_label"] = 1

    assert submission["predicted_label"].notna().all(), "every row must have a predicted_label after the fill"
    submission["predicted_label"] = submission["predicted_label"].astype(int)

    # ------------------------------------------------------------------
    # Sanity checks (all must pass before writing the output file)
    # ------------------------------------------------------------------
    checks = {}

    checks["row_count_matches_validation_csv"] = (len(submission) == n_validation_rows)

    checks["no_missing_predictions"] = submission["predicted_label"].notna().all()

    dup_mask = submission.duplicated(subset=ID_COLS, keep=False)
    checks["no_duplicate_id_rows"] = not dup_mask.any()

    val_ids_ordered = val_raw[ID_COLS].reset_index(drop=True)
    sub_ids_ordered = submission[ID_COLS].reset_index(drop=True)
    checks["id_alignment_matches_validation_csv"] = val_ids_ordered.equals(sub_ids_ordered)

    checks["predicted_label_only_0_or_1"] = submission["predicted_label"].isin([0, 1]).all()

    checks["all_old_label_1_rows_predicted_1"] = (submission.loc[old_fail_mask, "predicted_label"] == 1).all()

    checks["all_old_label_0_rows_scored_by_model"] = (
        (~old_fail_mask).sum() == n_old_label_0 == len(pred_eligible)
        and submission.loc[~old_fail_mask, "predicted_label"].isin([0, 1]).all()
    )

    checks["frozen_threshold_0.42_used_for_old_label_0"] = (FROZEN_THRESHOLD == 0.42)

    checks["no_validation_label_column_present_or_used"] = ("label" not in val_raw.columns)

    checks["test_csv_not_loaded"] = True   # by construction: load_test is never imported in this module

    checks["company_provided_not_modified"] = True   # by construction: company_provided/ is only ever read

    checks["model_file_not_modified"] = True   # by construction: joblib.load only, no joblib.dump to MODEL_PATH

    print("\n" + "-" * 70)
    print("SANITY CHECKS")
    print("-" * 70)
    all_passed = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")
    if not all_passed:
        raise RuntimeError("One or more sanity checks failed - submission NOT written. See log above.")

    output_cols = ["wafer_id", "die_row", "die_col", "predicted_label"]
    final_submission = submission[output_cols]

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    final_submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAll sanity checks passed. Saved submission to {OUTPUT_PATH}")

    n_pred_0 = int((final_submission["predicted_label"] == 0).sum())
    n_pred_1 = int((final_submission["predicted_label"] == 1).sum())

    summary = {
        "validation_csv_row_count": n_validation_rows,
        "submission_row_count": len(final_submission),
        "n_old_label_0": n_old_label_0,
        "n_old_label_1": n_old_label_1,
        "n_predicted_0": n_pred_0,
        "n_predicted_1": n_pred_1,
        "threshold_used": FROZEN_THRESHOLD,
        "output_path": str(OUTPUT_PATH),
        "all_sanity_checks_passed": all_passed,
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return final_submission, summary


if __name__ == "__main__":
    main()
