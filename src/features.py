"""Model A spatial feature engineering.

Every feature here is computed from `wafer_id` (grouping only), `die_row`,
`die_col`, and `old_label` - all legitimately available before the
post-test measurement being predicted. Nothing here ever reads `label`
(the functions don't require the column to exist at all, so they work
unchanged on validation.csv, which has no `label` column).

Must be run on the FULL per-wafer die population (old_label 0 AND 1),
not the eligible (old_label==0) subset - an eligible die's neighbors can
include already-failed dies, and that neighboring pre-test-failure
signal is the entire point of the neighborhood features. Filtering to
old_label==0 must happen AFTER these features are computed
(src.target.build_model_a_dataset does that filtering).

Wafers are circular within their bounding box (~75-80% occupancy per
Phase 6 inspection), not solid rectangles - missing neighbors (both
off-grid and off-wafer-circle) are handled uniformly by checking actual
die existence per wafer, never by padding/imputing.
"""
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import correlate

COORD_FEATURE_NAMES = ["dist_from_center", "edge_proximity"]


def neighbor_feature_names(m: int) -> list[str]:
    return [f"neigh_old_fail_count_m{m}", f"neigh_valid_count_m{m}", f"neigh_old_fail_density_m{m}"]


def spatial_feature_names(window_sizes: Iterable[int] = (3, 5)) -> list[str]:
    names = list(COORD_FEATURE_NAMES)
    for m in window_sizes:
        names += neighbor_feature_names(m)
    return names


def _wafer_coord_features(die_row: np.ndarray, die_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dist_from_center and edge_proximity for one wafer's dies, using its own bounding box."""
    rows = int(die_row.max()) + 1
    cols = int(die_col.max()) + 1
    cy, cx = rows / 2.0, cols / 2.0

    radial = np.sqrt((die_row - cy) ** 2 + (die_col - cx) ** 2)
    max_r = radial.max()
    dist_from_center = radial / max_r if max_r > 0 else np.zeros_like(radial, dtype=float)

    edge_dist = np.minimum.reduce([die_row, rows - 1 - die_row, die_col, cols - 1 - die_col]).astype(float)
    max_edge = edge_dist.max()
    edge_proximity = 1.0 - (edge_dist / max_edge if max_edge > 0 else np.zeros_like(edge_dist))

    return dist_from_center, edge_proximity


def _wafer_neighbor_features(die_row: np.ndarray, die_col: np.ndarray, old_label: np.ndarray,
                              window_sizes: Iterable[int]) -> dict:
    """Neighbor old_label count/density for one wafer's dies, per window size, self excluded."""
    rows = int(die_row.max()) + 1
    cols = int(die_col.max()) + 1

    occupancy = np.zeros((rows, cols), dtype=float)
    old_fail_grid = np.zeros((rows, cols), dtype=float)
    occupancy[die_row, die_col] = 1.0
    old_fail_grid[die_row, die_col] = old_label.astype(float)

    out = {}
    for m in window_sizes:
        kernel = np.ones((m, m), dtype=float)
        fail_sum_incl_self = correlate(old_fail_grid, kernel, mode="constant", cval=0.0)
        valid_sum_incl_self = correlate(occupancy, kernel, mode="constant", cval=0.0)

        fail_sum = fail_sum_incl_self[die_row, die_col] - old_fail_grid[die_row, die_col]
        valid_sum = valid_sum_incl_self[die_row, die_col] - occupancy[die_row, die_col]
        density = np.where(valid_sum > 0, fail_sum / valid_sum, 0.0)

        out[f"neigh_old_fail_count_m{m}"] = fail_sum
        out[f"neigh_valid_count_m{m}"] = valid_sum
        out[f"neigh_old_fail_density_m{m}"] = density
    return out


def add_spatial_features(df: pd.DataFrame, window_sizes: Iterable[int] = (3, 5)) -> pd.DataFrame:
    """Return a copy of df with spatial feature columns added (see module docstring for scope)."""
    required = {"wafer_id", "die_row", "die_col", "old_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_spatial_features requires columns {missing}")

    frames = []
    for _, g in df.groupby("wafer_id", sort=False):
        die_row = g["die_row"].to_numpy()
        die_col = g["die_col"].to_numpy()
        old_label = g["old_label"].to_numpy()

        dist_from_center, edge_proximity = _wafer_coord_features(die_row, die_col)
        neighbor_feats = _wafer_neighbor_features(die_row, die_col, old_label, window_sizes)

        frames.append(pd.DataFrame({
            "dist_from_center": dist_from_center,
            "edge_proximity": edge_proximity,
            **neighbor_feats,
        }, index=g.index))

    all_feats = pd.concat(frames).sort_index()
    return df.join(all_feats)
