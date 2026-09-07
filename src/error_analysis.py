"""Post-hoc error analysis for the FROZEN final Model A on test.csv.

This module explains WHERE the already-frozen `models/model_a_final_lr.joblib`
gets test.csv wrong. It is purely descriptive/post-hoc: nothing here selects,
tunes, or changes the model, its 502-feature contract, or its 0.42 threshold.
`.fit()` is never called anywhere in this module.

Why test.csv is loaded here at all (and why that's not "rerunning test.csv"
in the prohibited sense): Phase 10 (`src/final_evaluation.py`) evaluated
test.csv exactly once and saved only the AGGREGATE confusion matrix/metrics
to `outputs/phase10_final_test_evaluation.csv` - it never saved per-die
predicted probabilities or predicted labels anywhere. Sections 2-7 of this
error analysis (probability distributions, FN/FP patterns, wafer/spatial
concentration, feature comparisons) are fundamentally per-die questions
that cannot be answered from the aggregate Phase 10 CSV alone. Since no
per-die artifact exists anywhere in the repo, this module performs exactly
one forward pass of the frozen pipeline over test.csv
(`pipe.predict_proba` only - no `.fit()`, no threshold search, no
re-evaluation loop) to produce that missing per-die artifact, then
immediately verifies the resulting confusion matrix reproduces the
already-documented Phase 10 numbers EXACTLY before any analysis proceeds.
Section 1's headline numbers are read directly from the existing
`outputs/phase10_final_test_evaluation.csv`, not recomputed independently.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.data_loading import load_config, load_test
from src.evaluate import evaluate_predictions
from src.features import add_spatial_features
from src.target import build_model_a_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
MODEL_PATH = REPO_ROOT / "models" / "model_a_final_lr.joblib"
PHASE10_CSV = OUT_DIR / "phase10_final_test_evaluation.csv"
FROZEN_THRESHOLD = 0.42

# Documented frozen Phase 10 result (outputs/phase10_final_test_evaluation.csv) - used as the
# integrity check that this module's one-time inference pass reproduces exactly.
EXPECTED = {"tp": 509, "fn": 871, "fp": 24, "tn": 31194, "n_eligible": 32598, "n_excluded": 6753}

COLOR_TP = "#2a78d6"   # blue   - correct
COLOR_FN = "#eb6834"   # orange - missed fail (error)
COLOR_TN = "#2a78d6"   # blue   - correct
COLOR_FP = "#e34948"   # red    - false alarm (error, distinct from FN for the 2-panel plot)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY, "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED, "grid.color": GRIDLINE, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": SURFACE,
})


def _savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved outputs/{name}")
    return path


# ======================================================================
# 0. LOAD FROZEN MODEL, GENERATE THE MISSING PER-DIE TEST ARTIFACT
# ======================================================================

def generate_per_die_test_artifact(config):
    print("=" * 78)
    print("STEP 0: LOAD FROZEN MODEL, GENERATE PER-DIE TEST PREDICTIONS")
    print("=" * 78)
    print("(no per-die test artifact exists in the repo - Phase 10 saved aggregate metrics only)")

    pipe = joblib.load(MODEL_PATH)
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    assert [n for n, _ in pipe.steps] == ["scaler", "clf"]
    feature_names = list(pipe.feature_names_in_)
    print(f"Loaded frozen pipeline: {[n for n,_ in pipe.steps]}, {len(feature_names)} features "
          f"(read-only - .fit() is never called in this module)")

    test_raw = load_test(config)
    ds = build_model_a_dataset(test_raw, config)
    assert list(ds.X.columns) == feature_names, "feature order mismatch vs frozen pipeline"
    print(f"test.csv: {len(test_raw)} total rows, {len(ds.X)} eligible (old_label==0), "
          f"{len(test_raw) - len(ds.X)} excluded (old_label==1)")

    proba = pipe.predict_proba(ds.X)[:, 1]   # inference only - no .fit()
    pred = (proba >= FROZEN_THRESHOLD).astype(int)
    y = ds.y.to_numpy()

    metrics = evaluate_predictions(y, pred, proba)
    got = {"tp": metrics["tp"], "fn": metrics["fn"], "fp": metrics["fp"], "tn": metrics["tn"],
           "n_eligible": metrics["n_eligible"],
           "n_excluded": len(test_raw) - len(ds.X)}
    print(f"\nIntegrity check vs documented Phase 10 result: {got}")
    mismatches = {k: (got[k], EXPECTED[k]) for k in EXPECTED if got[k] != EXPECTED[k]}
    assert not mismatches, f"Per-die inference does NOT reproduce the frozen Phase 10 result: {mismatches}"
    print("MATCH - per-die inference exactly reproduces the documented frozen Phase 10 confusion matrix.")

    error_type = np.select(
        [(y == 1) & (pred == 1), (y == 1) & (pred == 0), (y == 0) & (pred == 1), (y == 0) & (pred == 0)],
        ["TP", "FN", "FP", "TN"], default="")

    per_die = ds.ids.reset_index(drop=True).copy()
    per_die["true_label"] = y
    per_die["predicted_proba"] = proba
    per_die["predicted_label"] = pred
    per_die["error_type"] = error_type

    out_path = OUT_DIR / "model_a_test_predictions.csv"
    per_die.to_csv(out_path, index=False)
    print(f"Saved {out_path.relative_to(REPO_ROOT)} ({len(per_die)} rows: wafer_id, die_row, die_col, "
          f"true_label, predicted_proba, predicted_label, error_type) - the per-die artifact "
          f"Phase 10 did not save, generated here via one frozen-pipeline inference pass.")

    return per_die, ds.X, test_raw, feature_names


# ======================================================================
# 1. CONFUSION-MATRIX BREAKDOWN (numbers sourced from the EXISTING Phase 10 CSV)
# ======================================================================

def confusion_matrix_breakdown():
    print("\n" + "=" * 78)
    print("SECTION 1: CONFUSION-MATRIX BREAKDOWN (from existing outputs/phase10_final_test_evaluation.csv)")
    print("=" * 78)

    phase10 = pd.read_csv(PHASE10_CSV).iloc[0]
    tp, fn, fp, tn = int(phase10["tp"]), int(phase10["fn"]), int(phase10["fp"]), int(phase10["tn"])
    precision = phase10["fail_precision"]
    recall = phase10["fail_accuracy_recall"]
    f1 = phase10["fail_f1"]
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"Precision (Fail): {precision:.6f}   Recall (Fail): {recall:.6f}   Fail F1: {f1:.6f}")
    print(f"False-negative rate FN/(FN+TP): {fnr:.6f}   False-positive rate FP/(FP+TN): {fpr:.6f}")

    summary = pd.DataFrame([{
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision_fail": precision, "recall_fail": recall, "f1_fail": f1,
        "false_negative_rate": fnr, "false_positive_rate": fpr,
        "n_eligible": int(phase10["n_test_eligible_old_label_0"]),
        "n_excluded_old_label_1": int(phase10["n_test_excluded_old_label_1"]),
        "threshold": phase10["threshold"],
        "source": "outputs/phase10_final_test_evaluation.csv (existing, not recomputed)",
    }])
    out_path = OUT_DIR / "model_a_error_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved {out_path.relative_to(REPO_ROOT)}")
    return summary


# ======================================================================
# 2. PREDICTED-PROBABILITY ANALYSIS
# ======================================================================

def probability_analysis(per_die: pd.DataFrame):
    print("\n" + "=" * 78)
    print("SECTION 2: PREDICTED-PROBABILITY ANALYSIS (post-hoc, test.csv)")
    print("=" * 78)

    tp = per_die[per_die["error_type"] == "TP"]["predicted_proba"]
    fn = per_die[per_die["error_type"] == "FN"]["predicted_proba"]
    fp = per_die[per_die["error_type"] == "FP"]["predicted_proba"]
    tn = per_die[per_die["error_type"] == "TN"]["predicted_proba"]

    print(f"TP (n={len(tp)}): proba mean={tp.mean():.4f}, median={tp.median():.4f}, "
          f"min={tp.min():.4f}, max={tp.max():.4f}")
    print(f"FN (n={len(fn)}): proba mean={fn.mean():.4f}, median={fn.median():.4f}, "
          f"min={fn.min():.4f}, max={fn.max():.4f}")
    print(f"FP (n={len(fp)}): proba mean={fp.mean():.4f}, median={fp.median():.4f}, "
          f"min={fp.min():.4f}, max={fp.max():.4f}")
    print(f"TN (n={len(tn)}): proba mean={tn.mean():.4f}, median={tn.median():.4f}, "
          f"min={tn.min():.4f}, max={tn.max():.4f}")

    fn_gap = FROZEN_THRESHOLD - fn   # how far below threshold (larger = more confidently missed)
    fp_gap = fp - FROZEN_THRESHOLD   # how far above threshold

    fn_bins = pd.cut(fn, bins=[0, 0.05, 0.15, 0.30, 0.42], right=False,
                      labels=["[0.00,0.05) high-confidence miss", "[0.05,0.15)", "[0.15,0.30)", "[0.30,0.42) near-miss"])
    fp_bins = pd.cut(fp, bins=[0.42, 0.55, 0.70, 0.85, 1.01], right=False,
                      labels=["[0.42,0.55) near-miss", "[0.55,0.70)", "[0.70,0.85)", "[0.85,1.00] high-confidence false alarm"])

    fn_bin_counts = fn_bins.value_counts().sort_index()
    fp_bin_counts = fp_bins.value_counts().sort_index()
    print(f"\nFN distance-from-threshold (0.42 - proba): mean={fn_gap.mean():.4f}, median={fn_gap.median():.4f}")
    print("FN probability bins:")
    print(fn_bin_counts.to_string())
    print(f"\nFP distance-from-threshold (proba - 0.42): mean={fp_gap.mean():.4f}, median={fp_gap.median():.4f}")
    print("FP probability bins:")
    print(fp_bin_counts.to_string())

    n_high_conf_fn = int((fn < 0.05).sum())
    n_high_conf_fp = int((fp >= 0.85).sum())
    print(f"\nHigh-confidence false negatives (proba<0.05, i.e. model was almost certain 'Pass' "
          f"despite being a true Fail): {n_high_conf_fn} of {len(fn)} ({n_high_conf_fn/len(fn)*100:.1f}%)")
    print(f"High-confidence false positives (proba>=0.85): {n_high_conf_fp} of {len(fp)} "
          f"({n_high_conf_fp/len(fp)*100:.1f}%)" if len(fp) else "n/a")

    bins_df = pd.concat([
        fn_bin_counts.rename("count").reset_index().rename(columns={"index": "bin"}).assign(group="FN"),
        fp_bin_counts.rename("count").reset_index().rename(columns={"index": "bin"}).assign(group="FP"),
    ], ignore_index=True)
    bins_out = OUT_DIR / "model_a_error_probability_bins.csv"
    bins_df.to_csv(bins_out, index=False)
    print(f"Saved {bins_out.relative_to(REPO_ROOT)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins_true_fail = np.linspace(0, 1, 41)
    axes[0].hist(tp, bins=bins_true_fail, color=COLOR_TP, alpha=0.6, label=f"TP (n={len(tp)})")
    axes[0].hist(fn, bins=bins_true_fail, color=COLOR_FN, alpha=0.7, label=f"FN (n={len(fn)})")
    axes[0].axvline(FROZEN_THRESHOLD, color=INK_PRIMARY, linestyle="--", linewidth=1.5, label="threshold=0.42")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("True Fail dies: TP vs FN")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(tn, bins=bins_true_fail, color=COLOR_TN, alpha=0.35, label=f"TN (n={len(tn)})")
    axes[1].hist(fp, bins=bins_true_fail, color=COLOR_FP, alpha=0.8, label=f"FP (n={len(fp)})")
    axes[1].axvline(FROZEN_THRESHOLD, color=INK_PRIMARY, linestyle="--", linewidth=1.5, label="threshold=0.42")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_yscale("log")
    axes[1].set_title("True Pass dies: TN vs FP (log-scale y, TN >> FP)")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(axis="y", linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Model A test.csv — predicted-probability distribution by outcome (post-hoc)", y=1.02)
    _savefig(fig, "model_a_error_probability.png")

    return {"fn_gap_mean": fn_gap.mean(), "fp_gap_mean": fp_gap.mean(),
            "n_high_conf_fn": n_high_conf_fn, "n_high_conf_fp": n_high_conf_fp}


# ======================================================================
# 5. WAFER-LEVEL ERROR CONCENTRATION (computed early - reused by sections 3/4/6)
# ======================================================================

def wafer_level_errors(per_die: pd.DataFrame, min_reliable_n: int = 5):
    print("\n" + "=" * 78)
    print("SECTION 5: WAFER-LEVEL ERROR CONCENTRATION (post-hoc, test.csv)")
    print("=" * 78)

    g = per_die.groupby("wafer_id")
    wafer_stats = g.apply(lambda d: pd.Series({
        "n_eligible": len(d),
        "n_true_fail": int((d["true_label"] == 1).sum()),
        "n_true_pass": int((d["true_label"] == 0).sum()),
        "tp": int((d["error_type"] == "TP").sum()),
        "fn": int((d["error_type"] == "FN").sum()),
        "fp": int((d["error_type"] == "FP").sum()),
        "tn": int((d["error_type"] == "TN").sum()),
    }), include_groups=False).reset_index()

    wafer_stats["fn_rate"] = np.where(wafer_stats["n_true_fail"] > 0,
                                       wafer_stats["fn"] / wafer_stats["n_true_fail"], np.nan)
    wafer_stats["fp_rate"] = np.where(wafer_stats["n_true_pass"] > 0,
                                       wafer_stats["fp"] / wafer_stats["n_true_pass"], np.nan)
    wafer_stats["fn_rate_reliable"] = wafer_stats["n_true_fail"] >= min_reliable_n
    wafer_stats["fp_rate_reliable"] = wafer_stats["n_true_pass"] >= min_reliable_n

    out_path = OUT_DIR / "model_a_error_by_wafer.csv"
    wafer_stats.sort_values("fn_rate", ascending=False, na_position="last").to_csv(out_path, index=False)
    print(f"Saved {out_path.relative_to(REPO_ROOT)} ({len(wafer_stats)} wafers)")

    n_wafers_with_fn = int((wafer_stats["fn"] > 0).sum())
    n_wafers_with_fp = int((wafer_stats["fp"] > 0).sum())
    print(f"Wafers with >=1 FN: {n_wafers_with_fn} of {len(wafer_stats)}")
    print(f"Wafers with >=1 FP: {n_wafers_with_fp} of {len(wafer_stats)}")

    reliable_fn = wafer_stats[wafer_stats["fn_rate_reliable"]].sort_values("fn_rate", ascending=False)
    print(f"\nTop 5 wafers by FN rate (among wafers with >={min_reliable_n} true-fail dies, "
          f"n={len(reliable_fn)} such wafers):")
    print(reliable_fn.head(5)[["wafer_id", "n_true_fail", "fn", "fn_rate"]].to_string(index=False))

    reliable_fp = wafer_stats[wafer_stats["fp_rate_reliable"]].sort_values("fp_rate", ascending=False)
    print(f"\nWafers with >={min_reliable_n} true-pass dies AND at least one FP, by FP rate "
          f"(n={len(reliable_fp[reliable_fp['fp']>0])} such wafers - FP is rare so most wafers have fp_rate=0):")
    nonzero_fp = reliable_fp[reliable_fp["fp"] > 0]
    print(nonzero_fp[["wafer_id", "n_true_pass", "fp", "fp_rate"]].to_string(index=False)
          if len(nonzero_fp) else "  (none - every reliable wafer has 0 FP)")

    small_sample_flagged = int((~wafer_stats["fn_rate_reliable"] & (wafer_stats["n_true_fail"] > 0)).sum())
    print(f"\n{small_sample_flagged} wafers have 1-{min_reliable_n-1} true-fail dies - their fn_rate is "
          f"in the CSV but should NOT be over-interpreted (e.g. 1/1=100% from a single die).")

    print(f"\nConcentration check: FN spread across {n_wafers_with_fn} of {len(wafer_stats)} wafers "
          f"({'broadly distributed' if n_wafers_with_fn > len(wafer_stats)*0.5 else 'concentrated in a subset'}) "
          f"- not isolated to one or two wafers." if n_wafers_with_fn else "")

    return wafer_stats


# ======================================================================
# 6. SPATIAL ERROR ANALYSIS (reuses EDA's dist_from_center / edge_proximity)
# ======================================================================

def spatial_error_analysis(per_die: pd.DataFrame, test_raw: pd.DataFrame, wafer_stats: pd.DataFrame):
    print("\n" + "=" * 78)
    print("SECTION 6: SPATIAL ERROR ANALYSIS (post-hoc, reusing EDA's spatial feature definitions)")
    print("=" * 78)

    # add_spatial_features needs the FULL per-wafer population (old_label 0 and 1), same rule
    # as the EDA phase and src/features.py's own leakage rule.
    full_with_spatial = add_spatial_features(test_raw, window_sizes=(3,))
    spatial_cols = full_with_spatial[["wafer_id", "die_row", "die_col", "dist_from_center", "edge_proximity"]]
    merged = per_die.merge(spatial_cols, on=["wafer_id", "die_row", "die_col"], how="left")
    assert merged["dist_from_center"].isna().sum() == 0

    true_fail = merged[merged["true_label"] == 1].copy()
    true_pass = merged[merged["true_label"] == 0].copy()
    true_fail["edge_decile"] = pd.qcut(true_fail["edge_proximity"], 10, labels=False, duplicates="drop")
    fn_by_decile = true_fail.groupby("edge_decile").apply(
        lambda d: pd.Series({"mean_edge_proximity": d["edge_proximity"].mean(),
                              "n_true_fail": len(d), "fn_rate": (d["error_type"] == "FN").mean()}),
        include_groups=False).reset_index()
    print("FN rate (among true-Fail dies) by edge-proximity decile (0=center, 9=edge):")
    print(fn_by_decile.to_string(index=False))

    corr_edge_fn = np.corrcoef(true_fail["edge_proximity"], (true_fail["error_type"] == "FN").astype(int))[0, 1]
    print(f"\nPoint-biserial correlation: edge_proximity vs is_FN (among true-Fail dies) = {corr_edge_fn:.4f}")

    n_fp = int((true_pass["error_type"] == "FP").sum())
    print(f"\nFP spatial pattern: only {n_fp} FP dies total - too few for a reliable decile breakdown; "
          f"reporting raw edge_proximity values instead (see FP section).")

    spatial_summary = fn_by_decile.assign(metric="fn_rate_by_edge_proximity_decile")
    out_path = OUT_DIR / "model_a_error_spatial_summary.csv"
    spatial_summary.to_csv(out_path, index=False)
    print(f"Saved {out_path.relative_to(REPO_ROOT)}")

    # Plot 1: FN rate vs edge proximity (line) ; Plot 2: wafer maps for the highest-FN-rate wafers
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fn_by_decile["mean_edge_proximity"], fn_by_decile["fn_rate"], marker="o",
                 color=COLOR_FN, linewidth=2)
    axes[0].set_xlabel("Edge proximity (0=center, 1=edge)")
    axes[0].set_ylabel("FN rate (among true-Fail dies)")
    axes[0].set_title("Miss rate vs edge proximity")
    axes[0].grid(linewidth=0.5)
    axes[0].set_axisbelow(True)

    fp_rows = merged[merged["error_type"] == "FP"]
    axes[1].scatter(true_pass["edge_proximity"], np.random.RandomState(42).normal(0, 0.02, len(true_pass)),
                     s=4, color=COLOR_TN, alpha=0.15, label=f"TN (n={len(true_pass)-n_fp})")
    axes[1].scatter(fp_rows["edge_proximity"], np.zeros(len(fp_rows)), s=60, color=COLOR_FP,
                     marker="x", linewidths=2, label=f"FP (n={n_fp})")
    axes[1].set_xlabel("Edge proximity (0=center, 1=edge)")
    axes[1].set_yticks([])
    axes[1].set_title("FP locations (edge proximity) vs TN jittered scatter")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Model A test.csv — spatial error pattern (post-hoc, descriptive only)", y=1.02)
    _savefig(fig, "model_a_error_spatial.png")

    # Wafer-map style plot for the 2 wafers with the most FN and 1 wafer with an FP, if any
    top_fn_wafers = wafer_stats[wafer_stats["fn"] > 0].sort_values("fn", ascending=False).head(2)["wafer_id"].tolist()
    fp_wafers = per_die[per_die["error_type"] == "FP"]["wafer_id"].unique()
    sample_wafers = list(dict.fromkeys(top_fn_wafers + ([fp_wafers[0]] if len(fp_wafers) else [])))[:3]

    if sample_wafers:
        fig, axes = plt.subplots(1, len(sample_wafers), figsize=(5.5 * len(sample_wafers), 5))
        axes = np.atleast_1d(axes)
        for ax, wid in zip(axes, sample_wafers):
            w = merged[merged["wafer_id"] == wid]
            for etype, color, marker in [("TN", "#c3c2b7", "s"), ("TP", COLOR_TP, "s"),
                                          ("FN", COLOR_FN, "s"), ("FP", COLOR_FP, "s")]:
                sub = w[w["error_type"] == etype]
                ax.scatter(sub["die_col"], sub["die_row"], c=color, s=16, marker=marker, label=etype)
            n_fn_w = int((w["error_type"] == "FN").sum())
            n_fp_w = int((w["error_type"] == "FP").sum())
            ax.set_title(f"{wid}\n(FN={n_fn_w}, FP={n_fp_w})", fontsize=10)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xlabel("die_col")
            ax.set_ylabel("die_row")
        handles = [Patch(color="#c3c2b7", label="TN"), Patch(color=COLOR_TP, label="TP"),
                   Patch(color=COLOR_FN, label="FN"), Patch(color=COLOR_FP, label="FP")]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
        fig.suptitle("Sample wafer maps — error locations (post-hoc)", y=1.15)
        fig.tight_layout()
        _savefig(fig, "model_a_error_wafer_maps.png")

    return merged


# ======================================================================
# 3+4+7. FN / FP FEATURE-SPACE ANALYSIS (TP vs FN, TN vs FP)
# ======================================================================

def feature_space_error_analysis(per_die_spatial: pd.DataFrame, X: pd.DataFrame, feature_names):
    print("\n" + "=" * 78)
    print("SECTION 3/4/7: FEATURE-SPACE ERROR ANALYSIS (TP vs FN, TN vs FP) - post-hoc, test.csv")
    print("=" * 78)

    feat_cols = [c for c in feature_names if c.startswith("feature_")]
    X_ = X.reset_index(drop=True)
    error_type = per_die_spatial["error_type"].reset_index(drop=True)

    def group_summary(group_a_label, group_b_label, out_name):
        mask_a = (error_type == group_a_label).to_numpy()
        mask_b = (error_type == group_b_label).to_numpy()
        Xa, Xb = X_.loc[mask_a, feat_cols], X_.loc[mask_b, feat_cols]
        mean_a, mean_b = Xa.mean(), Xb.mean()
        std_a, std_b = Xa.std(), Xb.std()
        pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
        cohens_d = (mean_b - mean_a) / pooled_std.replace(0, np.nan)
        df = pd.DataFrame({
            "feature": feat_cols, f"{group_a_label.lower()}_mean": mean_a.values,
            f"{group_b_label.lower()}_mean": mean_b.values, "cohens_d": cohens_d.values,
            "abs_cohens_d": cohens_d.abs().values,
        }).sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        out_path = OUT_DIR / out_name
        df.to_csv(out_path, index=False)
        print(f"\n{group_a_label} (n={mask_a.sum()}) vs {group_b_label} (n={mask_b.sum()}): "
              f"saved {out_path.relative_to(REPO_ROOT)}")
        print(f"Top 8 features by |Cohen's d| ({group_a_label} vs {group_b_label}):")
        print(df.head(8)[["feature", "cohens_d"]].to_string(index=False))
        print(f"Max |Cohen's d|: {df['abs_cohens_d'].max():.4f} "
              f"({'small effect - features do not clearly separate these groups' if df['abs_cohens_d'].max() < 0.5 else 'notable difference found'})")
        return df, mask_a, mask_b

    fn_df, tp_mask, fn_mask = group_summary("TP", "FN", "model_a_fn_feature_summary.csv")
    fp_df, tn_mask, fp_mask = group_summary("TN", "FP", "model_a_fp_feature_summary.csv")
    print(f"\nNote on FP comparison: only {fp_mask.sum()} FP observations - Cohen's d estimates here "
          f"are high-variance and illustrative only, not a reliable statistical signal (see limitations).")

    # Plot: top 6 FN-vs-TP distinguishing features, histogram overlay (n is large enough: 871 vs 509)
    top6_fn = fn_df.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feat in zip(axes.ravel(), top6_fn):
        tp_vals, fn_vals = X_.loc[tp_mask, feat], X_.loc[fn_mask, feat]
        lo = min(tp_vals.quantile(0.01), fn_vals.quantile(0.01))
        hi = max(tp_vals.quantile(0.99), fn_vals.quantile(0.99))
        bins = np.linspace(lo, hi, 30)
        ax.hist(tp_vals, bins=bins, density=True, color=COLOR_TP, alpha=0.55, label="TP")
        ax.hist(fn_vals, bins=bins, density=True, color=COLOR_FN, alpha=0.6, label="FN")
        d = fn_df.loc[fn_df["feature"] == feat, "cohens_d"].iloc[0]
        ax.set_title(f"{feat} (d={d:.2f})", fontsize=9)
        ax.set_yticks([])
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("TP vs FN — top 6 distinguishing features among true-Fail dies (post-hoc, test.csv)", y=1.02)
    fig.tight_layout()
    _savefig(fig, "model_a_fn_vs_tp_top_features.png")

    # Plot: top 6 FP-vs-TN distinguishing features, BOXPLOT (n=24 FP too small for histograms)
    top6_fp = fp_df.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feat in zip(axes.ravel(), top6_fp):
        tn_vals = X_.loc[tn_mask, feat].sample(min(2000, int(tn_mask.sum())), random_state=42)
        fp_vals = X_.loc[fp_mask, feat]
        bp = ax.boxplot([tn_vals, fp_vals], tick_labels=["TN\n(n=2000 sample)", f"FP\n(n={len(fp_vals)})"],
                         patch_artist=True, showfliers=False, widths=0.5)
        bp["boxes"][0].set(facecolor=COLOR_TN, alpha=0.6)
        bp["boxes"][1].set(facecolor=COLOR_FP, alpha=0.7)
        d = fp_df.loc[fp_df["feature"] == feat, "cohens_d"].iloc[0]
        ax.set_title(f"{feat} (d={d:.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("TN vs FP — top 6 distinguishing features (post-hoc, test.csv) — CAUTION: n=24 FP only",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _savefig(fig, "model_a_fp_vs_tn_top_features.png")

    return fn_df, fp_df


# ======================================================================
# MAIN
# ======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    per_die, X, test_raw, feature_names = generate_per_die_test_artifact(config)
    confusion_matrix_breakdown()
    probability_analysis(per_die)
    wafer_stats = wafer_level_errors(per_die)
    per_die_spatial = spatial_error_analysis(per_die, test_raw, wafer_stats)
    feature_space_error_analysis(per_die_spatial, X, feature_names)

    print("\n" + "=" * 78)
    print("POST-HOC ERROR ANALYSIS COMPLETE.")
    print("Frozen Model A was not retrained, refit, or modified. Threshold 0.42 unchanged.")
    print("This was a ONE-TIME inference pass to generate the missing per-die artifact; the")
    print("resulting confusion matrix was verified to exactly match the documented Phase 10 result.")
    print("company_provided/ was not modified. train.csv/test.csv/validation.csv were not modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
