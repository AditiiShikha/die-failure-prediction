"""Model A dataset assembly: eligibility filter + X/y construction.

Verified target mechanics (Phase 2, checked row-by-row on train.csv and
test.csv, zero violations):
    label >= old_label for every row (old fails never "recover")
    sum(label) == sum(old_label) + sum(newly_failed), disjoint by construction

Model A population: old_label == 0 (dies not already failed pre-test).
On that population, label is exactly the "newly failed" indicator.
old_label itself is dropped from X: it's available pre-test (not a
leakage risk) but constant (0) across this population, so it carries
no signal for the classifier — it's only used here as the filter.
"""
from dataclasses import dataclass

import pandas as pd

from src.data_loading import load_config


def feature_columns(config: dict | None = None) -> list[str]:
    config = config or load_config()
    ma = config["model_a"]
    return [f"{ma['feature_prefix']}{i}" for i in range(1, ma["num_features"] + 1)]


@dataclass
class ModelADataset:
    X: pd.DataFrame            # spatial columns + feature_1..feature_500
    y: pd.Series | None        # target ("newly failed"); None for validation (unlabeled)
    ids: pd.DataFrame           # wafer_id, die_row, die_col — grouping & output reconstruction


def build_model_a_dataset(df: pd.DataFrame, config: dict | None = None,
                           extra_feature_columns: list[str] | None = None) -> ModelADataset:
    """Filter to eligible dies (old_label==0) and assemble X/y for Model A.

    extra_feature_columns: additional columns already present in df to
    include in X (e.g. spatial features from src.features.add_spatial_features,
    which must be computed on the FULL wafer population before this filter
    is applied - see src/features.py docstring). None by default, so
    existing callers (Phase 3-5) are unaffected.
    """
    config = config or load_config()
    ma = config["model_a"]
    elig_col = ma["eligibility_column"]
    elig_val = ma["eligibility_value"]
    target_col = ma["target_column"]
    spatial_cols = ma["spatial_columns"]
    id_cols = ma["id_columns"]
    feat_cols = feature_columns(config)
    extra_cols = extra_feature_columns or []

    if elig_col not in df.columns:
        raise ValueError(f"Missing eligibility column '{elig_col}'")

    required = set(feat_cols) | set(spatial_cols) | set(id_cols) | set(extra_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    eligible = df[df[elig_col] == elig_val].reset_index(drop=True)

    X = eligible[spatial_cols + feat_cols + extra_cols].copy()
    y = eligible[target_col].copy() if target_col in eligible.columns else None
    ids = eligible[id_cols].copy()

    return ModelADataset(X=X, y=y, ids=ids)
