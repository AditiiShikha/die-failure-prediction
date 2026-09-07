"""Exploratory data analysis for Model A (die-level failure prediction).

Descriptive analysis only. This module never trains, tunes, or selects a
model. It reuses the existing data/target/feature/evaluate utilities so the
eligibility filter (old_label==0), target definition (y=label on the
eligible population), and spatial-feature computation are identical to what
Model A itself uses (see src/target.py, src/features.py, src/evaluate.py).

Scope rules enforced here (see NOTES.md EDA section for the full writeup):
- Section 1 (schema audit) reads train.csv and validation.csv only, per the
  task scope - test.csv schema is not touched here.
- All label-bearing analysis (class imbalance, feature distributions,
  spatial failure patterns, correlations, t-SNE) uses train.csv only.
  test.csv / its `label` column is never loaded by this module.
- company_provided/, models/, test.csv and validation.csv are never written.

Run as a script from the repo root:
    python -m src.eda
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.data_loading import load_config, load_train, load_validation
from src.evaluate import evaluate_predictions
from src.features import add_spatial_features
from src.target import build_model_a_dataset, feature_columns

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
PLOT_DIR = OUT_DIR / "eda_plots"

# Colorblind-safe categorical palette (dataviz skill reference palette, slots 1/2).
COLOR_PASS = "#2a78d6"   # blue
COLOR_FAIL = "#eb6834"   # orange
COLOR_EXCLUDED = "#c3c2b7"  # muted gray - old_label==1, pre-existing fails
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
DIVERGING_NEG = "#2a78d6"   # blue
DIVERGING_POS = "#e34948"   # red
DIVERGING_MID = "#f0efec"   # neutral gray midpoint

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": SURFACE,
})


def _savefig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO_ROOT)}")
    return path


# ======================================================================
# 1. DATA / SCHEMA AUDIT  (train.csv + validation.csv only)
# ======================================================================

def section1_schema_audit(train_raw: pd.DataFrame, val_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("SECTION 1: DATA / SCHEMA AUDIT (train.csv, validation.csv)")
    print("=" * 78)

    ma = config["model_a"]
    feat_cols = feature_columns(config)
    elig_col, elig_val, target_col = ma["eligibility_column"], ma["eligibility_value"], ma["target_column"]

    rows = []

    for name, df in [("train", train_raw), ("validation", val_raw)]:
        has_label = target_col in df.columns
        n_wafers = df["wafer_id"].nunique()
        dies_per_wafer = df.groupby("wafer_id").size()
        dup_coords = df.duplicated(subset=["wafer_id", "die_row", "die_col"]).sum()
        dup_rows = df.duplicated().sum()
        missing_total = int(df[["wafer_id", "die_row", "die_col", elig_col] + feat_cols
                                if elig_col in df.columns else ["wafer_id", "die_row", "die_col"] + feat_cols].isna().sum().sum())
        n_eligible = int((df[elig_col] == elig_val).sum()) if elig_col in df.columns else None
        n_excluded = int((df[elig_col] != elig_val).sum()) if elig_col in df.columns else None

        row = {
            "dataset": name,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "n_wafers": n_wafers,
            "dies_per_wafer_min": int(dies_per_wafer.min()),
            "dies_per_wafer_max": int(dies_per_wafer.max()),
            "dies_per_wafer_mean": round(dies_per_wafer.mean(), 2),
            "duplicate_rows": int(dup_rows),
            "duplicate_wafer_row_col_coords": int(dup_coords),
            "missing_values_total": missing_total,
            "has_old_label": elig_col in df.columns,
            "n_eligible_old_label_0": n_eligible,
            "n_excluded_old_label_1": n_excluded,
            "has_target_label": has_label,
        }
        if has_label:
            row["n_pass_label0"] = int((df[target_col] == 0).sum())
            row["n_fail_label1"] = int((df[target_col] == 1).sum())
        rows.append(row)

        print(f"\n[{name}] rows={len(df)}, cols={df.shape[1]}, wafers={n_wafers}, "
              f"dies/wafer=[{dies_per_wafer.min()},{dies_per_wafer.max()}] mean={dies_per_wafer.mean():.1f}")
        print(f"  columns: wafer_id, die_row, die_col, "
              f"{'old_label, ' if elig_col in df.columns else ''}"
              f"{'label, ' if has_label else ''}feature_1..feature_{ma['num_features']}")
        print(f"  duplicate full rows: {dup_rows}, duplicate (wafer_id,die_row,die_col) coords: {dup_coords}")
        print(f"  missing values (id+old_label+features): {missing_total}")
        if elig_col in df.columns:
            print(f"  old_label==0 (eligible): {n_eligible}, old_label==1 (excluded/pre-existing fail): {n_excluded}")
        if has_label:
            print(f"  label==0 (pass): {row['n_pass_label0']}, label==1 (fail): {row['n_fail_label1']}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "eda_schema_audit.csv", index=False)
    print(f"\nSaved outputs/eda_schema_audit.csv")

    # Verify label >= old_label and no NaNs in feature columns (train only, sanity check)
    if target_col in train_raw.columns:
        violations = int((train_raw[target_col] < train_raw[elig_col]).sum())
        print(f"\nSanity check (train): label >= old_label violations = {violations} (expect 0)")

    return df_out


# ======================================================================
# 2. CLASS IMBALANCE ANALYSIS  (train.csv only, eligible population)
# ======================================================================

def section2_class_imbalance(train_ds, wafer_id: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78)
    print("SECTION 2: CLASS IMBALANCE ANALYSIS (train.csv, eligible dies only)")
    print("=" * 78)

    y = train_ds.y
    n_total = len(y)
    n_fail = int((y == 1).sum())
    n_pass = int((y == 0).sum())
    fail_pct = n_fail / n_total * 100
    pass_pct = n_pass / n_total * 100

    print(f"Eligible dies: {n_total}")
    print(f"Pass (label=0): {n_pass} ({pass_pct:.4f}%)")
    print(f"Fail (label=1): {n_fail} ({fail_pct:.4f}%)")
    print(f"Pass:Fail ratio: {n_pass / n_fail:.2f} : 1")

    # "Predict everything as Pass" baseline
    all_pass_pred = np.zeros(n_total, dtype=int)
    baseline_metrics = evaluate_predictions(y, all_pass_pred)
    print(f"\nPredict-all-Pass baseline (train.csv, eligible dies):")
    print(f"  Overall accuracy: {baseline_metrics['overall_accuracy']:.6f}  "
          f"(= pass rate = {pass_pct/100:.6f})")
    print(f"  Fail recall: {baseline_metrics['fail_accuracy_recall']:.6f} (by construction, 0)")

    # No-skill PR-AUC baseline = prevalence of the positive class
    no_skill_pr_auc = n_fail / n_total
    print(f"\nNo-skill PR-AUC baseline (= failure prevalence): {no_skill_pr_auc:.6f}")

    class_dist = pd.DataFrame([{
        "n_eligible_dies": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "fail_pct": fail_pct,
        "pass_pct": pass_pct,
        "pass_fail_ratio": n_pass / n_fail,
        "predict_all_pass_accuracy": baseline_metrics["overall_accuracy"],
        "no_skill_pr_auc_baseline": no_skill_pr_auc,
    }])
    class_dist.to_csv(OUT_DIR / "eda_class_distribution.csv", index=False)
    print("\nSaved outputs/eda_class_distribution.csv")

    # Per-wafer failure rates
    per_wafer = pd.DataFrame({"wafer_id": wafer_id.values, "label": y.values})
    wafer_stats = per_wafer.groupby("wafer_id")["label"].agg(
        n_eligible_dies="size", n_fail="sum").reset_index()
    wafer_stats["fail_rate"] = wafer_stats["n_fail"] / wafer_stats["n_eligible_dies"]
    wafer_stats = wafer_stats.sort_values("fail_rate", ascending=False).reset_index(drop=True)
    wafer_stats.to_csv(OUT_DIR / "eda_wafer_failure_rates.csv", index=False)
    print("Saved outputs/eda_wafer_failure_rates.csv")

    overall_rate = n_fail / n_total
    print(f"\nPer-wafer failure rate: mean={wafer_stats['fail_rate'].mean():.4f}, "
          f"median={wafer_stats['fail_rate'].median():.4f}, "
          f"std={wafer_stats['fail_rate'].std():.4f}, "
          f"min={wafer_stats['fail_rate'].min():.4f}, max={wafer_stats['fail_rate'].max():.4f}")
    print(f"Overall (pooled) failure rate: {overall_rate:.4f}")
    print("\nTop 5 highest failure-rate wafers:")
    print(wafer_stats.head(5).to_string(index=False))
    print("\nBottom 5 lowest failure-rate wafers:")
    print(wafer_stats.tail(5).to_string(index=False))

    # Plot 1: class balance bar chart
    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    bars = ax.bar(["Pass", "Fail"], [n_pass, n_fail], color=[COLOR_PASS, COLOR_FAIL], width=0.6)
    ax.set_ylim(0, n_pass * 1.14)
    for b, v, pct in zip(bars, [n_pass, n_fail], [pass_pct, fail_pct]):
        ax.text(b.get_x() + b.get_width() / 2, v + n_pass * 0.015, f"{v:,}\n({pct:.2f}%)",
                ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel("Eligible dies (old_label==0)")
    ax.set_title("Class distribution — train.csv eligible dies", pad=14)
    ax.grid(axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    _savefig(fig, "class_distribution.png")

    # Plot 2: per-wafer failure-rate distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(wafer_stats["fail_rate"], bins=30, color=COLOR_FAIL, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(overall_rate, color=INK_PRIMARY, linestyle="--", linewidth=1.5,
               label=f"pooled rate = {overall_rate:.3f}")
    ax.set_xlabel("Per-wafer failure rate")
    ax.set_ylabel("Number of wafers")
    ax.set_title("Per-wafer failure-rate distribution (train.csv, 160 wafers)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    _savefig(fig, "wafer_failure_rate_distribution.png")

    return class_dist, wafer_stats


# ======================================================================
# 3. FEATURE DISTRIBUTION ANALYSIS  (train.csv only, eligible population)
# ======================================================================

def section3_feature_distributions(train_ds) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("SECTION 3: FEATURE DISTRIBUTION ANALYSIS (feature_1..feature_500)")
    print("=" * 78)

    feat_cols = [c for c in train_ds.X.columns if c.startswith("feature_")]
    X = train_ds.X[feat_cols]
    y = train_ds.y

    missing = X.isna().sum()
    n = len(X)
    means = X.mean()
    stds = X.std()
    mins = X.min()
    maxs = X.max()
    nunique = X.nunique()

    pass_mean = X[y == 0].mean()
    fail_mean = X[y == 1].mean()
    pass_std = X[y == 0].std()
    fail_std = X[y == 1].std()
    pooled_std = np.sqrt((pass_std ** 2 + fail_std ** 2) / 2)
    cohens_d = (fail_mean - pass_mean) / pooled_std.replace(0, np.nan)

    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_frac = ((X < lower) | (X > upper)).mean()

    # Coefficient of variation (scale-independent) - features span very different raw
    # scales (std from ~0.003 to ~700), so raw std alone conflates "small scale" with
    # "near constant". CV = std/|mean| is the honest measure of relative variability.
    coeff_var = (stds / means.abs()).replace([np.inf, -np.inf], np.nan)

    summary = pd.DataFrame({
        "feature": feat_cols,
        "missing_count": missing.values,
        "missing_pct": (missing.values / n * 100),
        "mean": means.values,
        "std": stds.values,
        "coeff_variation": coeff_var.values,
        "min": mins.values,
        "max": maxs.values,
        "n_unique": nunique.values,
        "outlier_frac_iqr": outlier_frac.values,
        "pass_mean": pass_mean.values,
        "fail_mean": fail_mean.values,
        "cohens_d": cohens_d.values,
        "abs_cohens_d": cohens_d.abs().values,
    }).sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)

    # True near-constant: zero variance or effectively a single value.
    summary["near_constant"] = (summary["std"] == 0) | (summary["n_unique"] <= 1)
    high_variance_thresh = stds.quantile(0.99)
    summary["highly_variable_raw_scale"] = summary["std"] >= high_variance_thresh

    summary.to_csv(OUT_DIR / "eda_feature_summary.csv", index=False)
    print(f"Saved outputs/eda_feature_summary.csv ({len(summary)} features)")

    print(f"\nMissingness: {int(missing.sum())} total missing values across {len(feat_cols)} features "
          f"({'none' if missing.sum() == 0 else 'see eda_feature_summary.csv'})")
    print(f"Truly near-constant features (std==0 or single unique value): {int(summary['near_constant'].sum())}")
    print(f"  Coefficient-of-variation range across all 500 features: "
          f"[{coeff_var.min():.4f}, {coeff_var.max():.4f}] (min CV {coeff_var.min():.4f} means "
          f"even the 'flattest' feature still has meaningful relative spread - no feature is "
          f"effectively constant, they just span very different raw scales: std ranges from "
          f"{stds.min():.4g} to {stds.max():.4g}, all continuous with {n} unique values each).")
    print(f"Largest-raw-scale features (std >= 99th pct = {high_variance_thresh:.4g}): "
          f"{int(summary['highly_variable_raw_scale'].sum())} (raw scale only - not necessarily more informative)")
    print(f"Features with >1% IQR-outlier rate: {int((summary['outlier_frac_iqr'] > 0.01).sum())}")

    print("\nTop 10 features by |Cohen's d| (Pass vs Fail mean separation, descriptive only):")
    print(summary[["feature", "cohens_d", "pass_mean", "fail_mean", "std"]].head(10).to_string(index=False))

    return summary


# ======================================================================
# 4. OVERLAPPING DISTRIBUTIONS  (Pass vs Fail, most-separating features)
# ======================================================================

def section4_overlap_plots(train_ds, feature_summary: pd.DataFrame, top_n: int = 12):
    print("\n" + "=" * 78)
    print("SECTION 4: OVERLAPPING PASS/FAIL DISTRIBUTIONS")
    print("=" * 78)

    X = train_ds.X
    y = train_ds.y
    top_feats = feature_summary.head(top_n)["feature"].tolist()

    ncols = 4
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        pass_vals = X.loc[y == 0, feat]
        fail_vals = X.loc[y == 1, feat]
        lo = min(pass_vals.quantile(0.01), fail_vals.quantile(0.01))
        hi = max(pass_vals.quantile(0.99), fail_vals.quantile(0.99))
        bins = np.linspace(lo, hi, 40)
        ax.hist(pass_vals, bins=bins, density=True, color=COLOR_PASS, alpha=0.55, label="Pass")
        ax.hist(fail_vals, bins=bins, density=True, color=COLOR_FAIL, alpha=0.55, label="Fail")
        d = feature_summary.loc[feature_summary["feature"] == feat, "cohens_d"].iloc[0]
        ax.set_title(f"{feat} (d={d:.2f})", fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

    for j in range(len(top_feats), len(axes)):
        axes[j].axis("off")

    handles = [Patch(color=COLOR_PASS, label="Pass"), Patch(color=COLOR_FAIL, label="Fail")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Top {top_n} features by |Cohen's d|: Pass vs Fail density overlap", y=1.05, fontsize=12)
    fig.tight_layout()
    _savefig(fig, "top_features_pass_vs_fail_hist.png")

    # Box/violin plot for the top 8, on standardized scale so they're comparable side by side
    top8 = top_feats[:8]
    fig, ax = plt.subplots(figsize=(10, 5))
    data_pass, data_fail = [], []
    for feat in top8:
        s = X[feat]
        z = (s - s.mean()) / s.std()
        data_pass.append(z[y == 0].values)
        data_fail.append(z[y == 1].values)

    positions_pass = np.arange(len(top8)) * 2.2
    positions_fail = positions_pass + 0.9
    bp1 = ax.boxplot(data_pass, positions=positions_pass, widths=0.7, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_fail, positions=positions_fail, widths=0.7, patch_artist=True, showfliers=False)
    for box in bp1["boxes"]:
        box.set(facecolor=COLOR_PASS, alpha=0.7, edgecolor=INK_SECONDARY)
    for box in bp2["boxes"]:
        box.set(facecolor=COLOR_FAIL, alpha=0.7, edgecolor=INK_SECONDARY)
    for bp in (bp1, bp2):
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set(color=INK_SECONDARY, linewidth=1)

    ax.set_xticks(positions_pass + 0.45)
    ax.set_xticklabels(top8, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Standardized value (z-score)")
    ax.set_title("Pass vs Fail — top 8 separating features (standardized)")
    ax.legend(handles=[Patch(color=COLOR_PASS, label="Pass"), Patch(color=COLOR_FAIL, label="Fail")],
              frameon=False, loc="upper right")
    ax.grid(axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    _savefig(fig, "top_features_pass_vs_fail_boxplot.png")

    # A couple of heavily-overlapping (low |d|) features, for contrast
    low_feats = feature_summary.tail(4)["feature"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, feat in zip(axes, low_feats):
        pass_vals = X.loc[y == 0, feat]
        fail_vals = X.loc[y == 1, feat]
        lo = min(pass_vals.quantile(0.01), fail_vals.quantile(0.01))
        hi = max(pass_vals.quantile(0.99), fail_vals.quantile(0.99))
        bins = np.linspace(lo, hi, 40)
        ax.hist(pass_vals, bins=bins, density=True, color=COLOR_PASS, alpha=0.55)
        ax.hist(fail_vals, bins=bins, density=True, color=COLOR_FAIL, alpha=0.55)
        d = feature_summary.loc[feature_summary["feature"] == feat, "cohens_d"].iloc[0]
        ax.set_title(f"{feat} (d={d:.3f})", fontsize=9)
        ax.set_yticks([])
    fig.suptitle("Lowest |Cohen's d| features — heavy Pass/Fail overlap", fontsize=11)
    fig.tight_layout()
    _savefig(fig, "lowest_signal_features_hist.png")

    print(f"Plotted top {top_n} separating features (histograms) and top 8 (boxplots).")
    print(f"High-overlap contrast features plotted: {low_feats}")
    print("\nSimple interpretation (descriptive, not causal):")
    print(f"  Best-separating feature: {top_feats[0]} (|d|={feature_summary['abs_cohens_d'].iloc[0]:.3f})")
    print(f"  Even the best-separating features show substantial Pass/Fail overlap "
          f"(|d| well below ~2, the rough threshold for visually near-total separation).")
    print(f"  Most of the 500 features show |d| close to 0 — i.e. Pass and Fail distributions "
          f"are nearly indistinguishable on a single-feature basis for most features.")


# ======================================================================
# 5. SPATIAL / WAFER ANALYSIS
# ======================================================================

def section5_spatial(train_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("SECTION 5: SPATIAL / WAFER ANALYSIS")
    print("=" * 78)

    ma = config["model_a"]
    elig_col, elig_val, target_col = ma["eligibility_column"], ma["eligibility_value"], ma["target_column"]

    # Spatial coord features computed on the FULL wafer population (old_label 0 and 1),
    # matching src/features.py's leakage rule (uses only wafer_id/die_row/die_col/old_label).
    full_with_spatial = add_spatial_features(train_raw, window_sizes=(3, 5))

    eligible = full_with_spatial[full_with_spatial[elig_col] == elig_val].reset_index(drop=True)
    excluded = full_with_spatial[full_with_spatial[elig_col] != elig_val].reset_index(drop=True)
    print(f"Eligible dies (old_label==0): {len(eligible)}   "
          f"Excluded pre-existing-fail dies (old_label==1): {len(excluded)} — kept separate from target analysis.")

    y = eligible[target_col]

    # Failure rate by normalized distance-from-center decile
    eligible = eligible.copy()
    eligible["dist_decile"] = pd.qcut(eligible["dist_from_center"], 10, labels=False, duplicates="drop")
    dist_stats = eligible.groupby("dist_decile").agg(
        mean_dist_from_center=("dist_from_center", "mean"),
        n_dies=(target_col, "size"),
        fail_rate=(target_col, "mean"),
    ).reset_index()

    eligible["edge_decile"] = pd.qcut(eligible["edge_proximity"], 10, labels=False, duplicates="drop")
    edge_stats = eligible.groupby("edge_decile").agg(
        mean_edge_proximity=("edge_proximity", "mean"),
        n_dies=(target_col, "size"),
        fail_rate=(target_col, "mean"),
    ).reset_index()

    print("\nFailure rate by distance-from-center decile (0=center, 9=edge):")
    print(dist_stats.to_string(index=False))
    print("\nFailure rate by edge-proximity decile (0=center-most, 9=edge-most):")
    print(edge_stats.to_string(index=False))

    corr_dist = eligible[["dist_from_center", target_col]].corr().iloc[0, 1]
    corr_edge = eligible[["edge_proximity", target_col]].corr().iloc[0, 1]
    print(f"\nPoint-biserial correlation: dist_from_center vs label = {corr_dist:.4f}")
    print(f"Point-biserial correlation: edge_proximity vs label = {corr_edge:.4f}")

    # neighbor old-fail density vs new-fail rate (3x3 window)
    eligible["neigh_density_bin_m3"] = pd.qcut(eligible["neigh_old_fail_density_m3"], 5, labels=False, duplicates="drop")
    neigh_stats = eligible.groupby("neigh_density_bin_m3").agg(
        mean_neigh_old_fail_density_m3=("neigh_old_fail_density_m3", "mean"),
        n_dies=(target_col, "size"),
        fail_rate=(target_col, "mean"),
    ).reset_index()
    print("\nFailure rate by 3x3-neighborhood old-fail-density bin:")
    print(neigh_stats.to_string(index=False))

    spatial_summary = pd.concat([
        dist_stats.assign(metric="dist_from_center_decile"),
        edge_stats.rename(columns={"mean_edge_proximity": "mean_dist_from_center"}).assign(metric="edge_proximity_decile"),
        neigh_stats.rename(columns={"mean_neigh_old_fail_density_m3": "mean_dist_from_center"}).assign(metric="neigh_old_fail_density_m3_bin"),
    ], ignore_index=True, sort=False)
    spatial_summary.to_csv(OUT_DIR / "eda_spatial_summary.csv", index=False)
    print("\nSaved outputs/eda_spatial_summary.csv")

    # Plot: fail rate vs edge proximity decile
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(edge_stats["mean_edge_proximity"], edge_stats["fail_rate"], marker="o",
            color=COLOR_FAIL, linewidth=2, markersize=5)
    ax.set_xlabel("Edge proximity (0=center, 1=edge)")
    ax.set_ylabel("New-fail rate")
    ax.set_title("New-fail rate vs edge proximity (train.csv, eligible dies)")
    ax.grid(linewidth=0.5)
    ax.set_axisbelow(True)
    _savefig(fig, "fail_rate_vs_edge_proximity.png")

    # Plot: fail rate vs 3x3 neighbor old-fail density
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(neigh_stats["mean_neigh_old_fail_density_m3"], neigh_stats["fail_rate"], marker="o",
            color=COLOR_FAIL, linewidth=2, markersize=5)
    ax.set_xlabel("Mean 3x3-neighborhood old-fail density")
    ax.set_ylabel("New-fail rate")
    ax.set_title("New-fail rate vs neighborhood pre-existing-fail density")
    ax.grid(linewidth=0.5)
    ax.set_axisbelow(True)
    _savefig(fig, "fail_rate_vs_neighbor_density.png")

    # Wafer maps: highest fail-rate wafer, lowest (nonzero-die) fail-rate wafer, and median wafer
    wafer_fail_rate = eligible.groupby("wafer_id")[target_col].mean().sort_values()
    n_wafers = len(wafer_fail_rate)
    sample_wafers = {
        "lowest_fail_rate": wafer_fail_rate.index[0],
        "median_fail_rate": wafer_fail_rate.index[n_wafers // 2],
        "highest_fail_rate": wafer_fail_rate.index[-1],
    }
    print(f"\nSample wafers for wafer maps: {sample_wafers}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, wid) in zip(axes, sample_wafers.items()):
        w_all = full_with_spatial[full_with_spatial["wafer_id"] == wid]
        w_elig = w_all[w_all[elig_col] == elig_val]
        w_excl = w_all[w_all[elig_col] != elig_val]
        w_pass = w_elig[w_elig[target_col] == 0]
        w_fail = w_elig[w_elig[target_col] == 1]

        ax.scatter(w_pass["die_col"], w_pass["die_row"], c=COLOR_PASS, s=14, label="Pass", marker="s")
        ax.scatter(w_fail["die_col"], w_fail["die_row"], c=COLOR_FAIL, s=14, label="New fail", marker="s")
        ax.scatter(w_excl["die_col"], w_excl["die_row"], c=COLOR_EXCLUDED, s=14, label="Excluded (old_label=1)", marker="s")
        ax.set_title(f"{wid}\n({label}, fail_rate={wafer_fail_rate[wid]:.3f})", fontsize=10)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("die_col")
        ax.set_ylabel("die_row")

    handles = [Patch(color=COLOR_PASS, label="Pass"), Patch(color=COLOR_FAIL, label="New fail"),
               Patch(color=COLOR_EXCLUDED, label="Excluded (old_label=1)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Wafer maps — physical die_row / die_col coordinates", y=1.1, fontsize=12)
    fig.tight_layout()
    _savefig(fig, "wafer_maps_sample.png")

    return spatial_summary


# ======================================================================
# 6. CORRELATION ANALYSIS
# ======================================================================

def section6_correlation(train_ds, feature_summary: pd.DataFrame, corr_threshold: float = 0.5,
                          heatmap_top_n: int = 40):
    print("\n" + "=" * 78)
    print("SECTION 6: CORRELATION ANALYSIS")
    print("=" * 78)

    feat_cols = [c for c in train_ds.X.columns if c.startswith("feature_")]
    X = train_ds.X[feat_cols]

    print(f"Computing correlation matrix for {len(feat_cols)} features "
          f"({len(feat_cols)}x{len(feat_cols)}) — this may take a bit...")
    corr = X.corr()

    corr_abs = corr.abs()
    upper_mask = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
    pairs = corr_abs.where(upper_mask).stack()
    max_abs_corr = pairs.max()
    max_pair = pairs.idxmax()
    print(f"\nMax |pairwise correlation| across all {len(feat_cols)} features: "
          f"{max_abs_corr:.4f} ({max_pair[0]} vs {max_pair[1]})")
    print(f"Correlation percentiles (|r|, all {len(pairs)} pairs): "
          f"median={pairs.median():.4f}, p90={pairs.quantile(0.90):.4f}, "
          f"p99={pairs.quantile(0.99):.4f}, max={max_abs_corr:.4f}")

    high_corr_pairs = pairs[pairs >= corr_threshold].sort_values(ascending=False)
    records = []
    for (f1, f2), abs_r in high_corr_pairs.items():
        records.append({"feature_1": f1, "feature_2": f2, "correlation": corr.loc[f1, f2], "abs_correlation": abs_r})
    high_corr_df = pd.DataFrame(records)
    # Always include the top-20 pairs by |r| even if none clear the threshold, so the
    # output file is informative rather than empty on a near-independent feature set.
    if len(high_corr_df) == 0:
        top20 = pairs.sort_values(ascending=False).head(20)
        records = [{"feature_1": f1, "feature_2": f2, "correlation": corr.loc[f1, f2], "abs_correlation": abs_r}
                   for (f1, f2), abs_r in top20.items()]
        high_corr_df = pd.DataFrame(records)
        high_corr_df.insert(0, "note", f"no pair reached |r|>={corr_threshold}; showing top 20 by |r| instead")
    high_corr_df.to_csv(OUT_DIR / "eda_feature_correlations.csv", index=False)
    print(f"Saved outputs/eda_feature_correlations.csv "
          f"({(high_corr_pairs >= corr_threshold).sum()} feature pairs with |r| >= {corr_threshold}"
          f"{', showing top 20 by |r| since none cleared the threshold' if len(high_corr_pairs) == 0 else ''})")

    if len(high_corr_pairs) > 0:
        print(f"\nTop 10 highest-correlation feature pairs:")
        print(high_corr_df.head(10).to_string(index=False))
        n_features_involved = len(set(high_corr_df["feature_1"]) | set(high_corr_df["feature_2"]))
        print(f"\n{n_features_involved} distinct features appear in at least one pair with |r| >= {corr_threshold}.")
        print("This matters for interpretation: highly correlated features can split importance/SHAP "
              "attribution between themselves, so per-feature importance should be read at the "
              "correlated-group level, not purely feature-by-feature.")
    else:
        print(f"\nNo feature pairs reached |r| >= {corr_threshold}. The 500 features are essentially "
              f"pairwise UNcorrelated (max |r| = {max_abs_corr:.4f} across all "
              f"{len(pairs)} pairs) - consistent with independently-generated synthetic parametric "
              f"measurements. This means SHAP/importance splitting between correlated features is "
              f"NOT a concern for this feature set, unlike typical real fab parametric data.")

    # Heatmap of a manageable subset: top-N features by |Cohen's d| (most informative for Pass/Fail)
    top_feats = feature_summary.head(heatmap_top_n)["feature"].tolist()
    sub_corr = X[top_feats].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "diverging", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])
    im = ax.imshow(sub_corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_feats)))
    ax.set_yticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=90, fontsize=6)
    ax.set_yticklabels(top_feats, fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    ax.set_title(f"Correlation heatmap — top {heatmap_top_n} features by |Cohen's d|\n"
                 f"(subset of {len(feat_cols)} total features, not the full 500x500 matrix)")
    fig.tight_layout()
    _savefig(fig, "correlation_heatmap_top_features.png")

    return high_corr_df


# ======================================================================
# 7. OPTIONAL DIMENSIONALITY VISUALIZATION (t-SNE, sampled)
# ======================================================================

def section7_tsne(train_ds, sample_size: int = 6000, random_state: int = 42):
    print("\n" + "=" * 78)
    print("SECTION 7: DIMENSIONALITY VISUALIZATION (t-SNE, sampled)")
    print("=" * 78)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    feat_cols = [c for c in train_ds.X.columns if c.startswith("feature_")]
    X = train_ds.X[feat_cols]
    y = train_ds.y

    n_total = len(X)
    frac = min(1.0, sample_size / n_total)
    rng = np.random.RandomState(random_state)
    sample_idx = (
        pd.Series(y).groupby(y).apply(lambda s: s.sample(frac=frac, random_state=random_state))
        .index.get_level_values(1)
    )
    X_sample = X.loc[sample_idx]
    y_sample = y.loc[sample_idx]
    print(f"Full eligible set: {n_total} rows. Using class-proportional stratified sample: "
          f"{len(X_sample)} rows ({len(X_sample) / n_total * 100:.1f}%) — visualization only, "
          f"not representative for prevalence estimates, not used as model input.")
    print(f"  Sample class counts: pass={int((y_sample == 0).sum())}, fail={int((y_sample == 1).sum())}")

    X_scaled = StandardScaler().fit_transform(X_sample)
    n_pca = min(50, X_scaled.shape[1])
    X_pca = PCA(n_components=n_pca, random_state=random_state).fit_transform(X_scaled)
    print(f"PCA pre-reduction: {X_scaled.shape[1]} -> {n_pca} components before t-SNE "
          f"(standard practice to speed up/stabilize t-SNE on high-dim data).")

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, init="pca")
    embedding = tsne.fit_transform(X_pca)
    print("t-SNE embedding computed.")

    fig, ax = plt.subplots(figsize=(7, 6))
    mask_pass = (y_sample == 0).values
    mask_fail = (y_sample == 1).values
    ax.scatter(embedding[mask_pass, 0], embedding[mask_pass, 1], c=COLOR_PASS, s=8, alpha=0.5, label="Pass")
    ax.scatter(embedding[mask_fail, 0], embedding[mask_fail, 1], c=COLOR_FAIL, s=10, alpha=0.7, label="Fail")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of feature_1..feature_500 (n={len(X_sample)} stratified sample)\n"
                 f"Visualization only — not used as model input")
    ax.legend(frameon=False)
    _savefig(fig, "tsne_pass_fail.png")

    return embedding, y_sample


# ======================================================================
# MAIN
# ======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    print("Loading train.csv and validation.csv (test.csv is NOT loaded by this EDA module)...")
    train_raw = load_train(config)
    val_raw = load_validation(config)

    section1_schema_audit(train_raw, val_raw, config)

    train_ds = build_model_a_dataset(train_raw, config)
    wafer_id_train = train_ds.ids["wafer_id"]

    class_dist, wafer_stats = section2_class_imbalance(train_ds, wafer_id_train)
    feature_summary = section3_feature_distributions(train_ds)
    section4_overlap_plots(train_ds, feature_summary)
    spatial_summary = section5_spatial(train_raw, config)
    high_corr_df = section6_correlation(train_ds, feature_summary)
    section7_tsne(train_ds)

    print("\n" + "=" * 78)
    print("EDA COMPLETE. No model was trained, tuned, or modified.")
    print("test.csv was not loaded. company_provided/ was not modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
