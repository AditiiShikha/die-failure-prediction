"""Interpretability analysis for the FROZEN final Model A pipeline.

This module explains the already-fitted `models/model_a_final_lr.joblib`
pipeline. It never calls `.fit()` on the pipeline, the scaler, or the
classifier - only `.transform()`, `.decision_function()`, `.predict_proba()`
and direct inspection of `.coef_`/`.intercept_`/`.mean_`/`.scale_`. The
pipeline, its 502-feature contract, and the frozen threshold (0.42) are
treated as fixed inputs, not something this script can change.

Two complementary interpretability methods are produced:

1. Standardized logistic-regression coefficients - exact and free, since
   the model IS StandardScaler -> LogisticRegression. Coefficients are
   already in standardized-feature space (mean 0, std 1 per feature on the
   training data the scaler was fit on), so |coefficient| is directly
   comparable across all 502 features without further normalization.

2. SHAP, via `shap.LinearExplainer` on the fitted `LogisticRegression`
   step, applied to the StandardScaler-transformed features (i.e. exactly
   the input space the classifier actually sees). This is the SHAP method
   built specifically for linear models - it is exact (not a sampling
   approximation) given a background distribution, and its outputs are in
   log-odds units (the model's decision_function), which is verified
   below by reconstructing decision_function from the SHAP values exactly.

   The masker is `shap.maskers.Independent` (features perturbed
   independently of one another). This is an assumption, and it is
   justified here rather than merely default: the EDA phase (see NOTES.md,
   "EXPLORATORY DATA ANALYSIS" section) found a max pairwise |correlation|
   of 0.0131 across all 500 feature_* columns on train.csv - i.e. the
   features are empirically almost perfectly uncorrelated, so treating
   them as independent for SHAP's perturbation is a reasonable match to
   the actual data, not a simplification that ignores real structure.

Data used: train.csv only (via the existing src.data_loading /
src.target utilities, identical to how Model A itself was trained/
evaluated). test.csv is never loaded by this module.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.data_loading import load_config, load_train
from src.target import build_model_a_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
MODEL_PATH = REPO_ROOT / "models" / "model_a_final_lr.joblib"
FROZEN_THRESHOLD = 0.42
RANDOM_STATE = 42

COLOR_POS = "#eb6834"  # orange - associated with higher predicted fail-risk
COLOR_NEG = "#2a78d6"  # blue   - associated with lower predicted fail-risk
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

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
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved outputs/{name}")
    return path


# ======================================================================
# 0. LOAD FROZEN MODEL + RECONSTRUCT FEATURES EXACTLY
# ======================================================================

def load_frozen_pipeline():
    print("=" * 78)
    print("LOADING FROZEN MODEL A PIPELINE (read-only - no .fit() calls in this module)")
    print("=" * 78)
    pipe = joblib.load(MODEL_PATH)
    print(f"Loaded {MODEL_PATH.relative_to(REPO_ROOT)}")
    print(f"Pipeline steps: {[name for name, _ in pipe.steps]}")

    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    assert [n for n, _ in pipe.steps] == ["scaler", "clf"], "unexpected pipeline structure"
    assert type(scaler).__name__ == "StandardScaler"
    assert type(clf).__name__ == "LogisticRegression"
    params = clf.get_params()
    assert params["class_weight"] is None, f"expected class_weight=None, got {params['class_weight']}"
    assert params["max_iter"] == 2000, f"expected max_iter=2000, got {params['max_iter']}"
    assert params["random_state"] == 42, f"expected random_state=42, got {params['random_state']}"
    assert clf.coef_.shape == (1, 502), f"expected coef_ shape (1, 502), got {clf.coef_.shape}"
    assert scaler.n_features_in_ == 502

    feature_names = list(pipe.feature_names_in_)
    print(f"Confirmed: StandardScaler -> LogisticRegression(class_weight=None, max_iter=2000, "
          f"random_state=42), {len(feature_names)} features, intercept={clf.intercept_[0]:.6f}")
    return pipe, scaler, clf, feature_names


def load_reconstructed_train_X(config, feature_names):
    print("\nLoading train.csv and reconstructing X via src.target.build_model_a_dataset "
          "(same eligibility filter + feature order Model A itself uses)...")
    train_raw = load_train(config)
    ds = build_model_a_dataset(train_raw, config)
    assert list(ds.X.columns) == feature_names, (
        "Reconstructed feature order does not match the frozen pipeline's feature_names_in_ - "
        "refusing to proceed, this would silently mis-attribute coefficients/SHAP values."
    )
    print(f"  Reconstructed X: {ds.X.shape} (eligible dies, old_label==0), column order verified "
          f"identical to pipe.feature_names_in_.")
    return ds


# ======================================================================
# 1. LOGISTIC REGRESSION COEFFICIENT ANALYSIS (standardized-feature space)
# ======================================================================

def coefficient_analysis(clf, feature_names):
    print("\n" + "=" * 78)
    print("SECTION 1: STANDARDIZED COEFFICIENT ANALYSIS")
    print("=" * 78)

    coefs = clf.coef_.ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "absolute_coefficient": np.abs(coefs),
        "direction": np.where(coefs >= 0, "positive", "negative"),
    }).sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    out_path = OUT_DIR / "model_a_lr_coefficient_importance.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path.relative_to(REPO_ROOT)} ({len(df)} features, ranked by |coefficient|)")

    top_pos = df[df["direction"] == "positive"].sort_values("coefficient", ascending=False).head(15)
    top_neg = df[df["direction"] == "negative"].sort_values("coefficient", ascending=True).head(15)

    print("\nTop 15 POSITIVE coefficients (associated with HIGHER predicted fail-probability):")
    print(top_pos[["feature", "coefficient", "rank"]].to_string(index=False))
    print("\nTop 15 NEGATIVE coefficients (associated with LOWER predicted fail-probability):")
    print(top_neg[["feature", "coefficient", "rank"]].to_string(index=False))

    # Global importance plot: top 20 by |coefficient|, colored by direction
    top20 = df.head(20).iloc[::-1]  # reverse so rank 1 is at the top of the barh
    colors = [COLOR_POS if d == "positive" else COLOR_NEG for d in top20["direction"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top20["feature"], top20["coefficient"], color=colors)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.set_title("Model A — top 20 features by |standardized coefficient|")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", linewidth=0.5)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLOR_POS, label="Positive (higher fail-risk)"),
                        Patch(color=COLOR_NEG, label="Negative (lower fail-risk)")],
              loc="lower right", frameon=False, fontsize=8)
    _savefig(fig, "model_a_lr_global_importance.png")

    return df


# ======================================================================
# 2. SHAP (LinearExplainer on the fitted classifier, in scaled-feature space)
# ======================================================================

def shap_analysis(scaler, clf, X: pd.DataFrame, feature_names, n_background=200, n_beeswarm_sample=5000):
    print("\n" + "=" * 78)
    print("SECTION 2: SHAP ANALYSIS (shap.LinearExplainer, frozen LogisticRegression)")
    print("=" * 78)

    X_scaled = scaler.transform(X)  # scaler.transform only - never fit here

    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_scaled), size=n_background, replace=False)
    background = X_scaled[bg_idx]
    masker = shap.maskers.Independent(background, max_samples=n_background)
    explainer = shap.LinearExplainer(clf, masker)
    print(f"Background: {n_background} rows sampled from train.csv eligible dies "
          f"(random_state={RANDOM_STATE}), shap.maskers.Independent (see module docstring for "
          f"why independence is a reasonable assumption here).")

    print(f"Computing SHAP values for the full eligible train population ({len(X_scaled)} rows) - "
          f"closed-form for a linear model + independent masker, so no sampling approximation needed.")
    shap_values_full = explainer.shap_values(X_scaled)
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(expected_value.ravel()[0])

    # Sanity check: SHAP values must reconstruct the model's actual decision_function exactly.
    decision = clf.decision_function(X_scaled)
    reconstructed = shap_values_full.sum(axis=1) + expected_value
    max_diff = np.max(np.abs(decision - reconstructed))
    print(f"Sanity check: max|decision_function - (sum(SHAP) + expected_value)| = {max_diff:.2e} "
          f"(should be ~0 / floating-point noise)")
    assert max_diff < 1e-6, "SHAP values do not reconstruct the frozen model's decision_function"

    mean_abs_shap = np.abs(shap_values_full).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.insert(0, "rank", np.arange(1, len(shap_df) + 1))
    shap_out_path = OUT_DIR / "model_a_shap_global_importance.csv"
    shap_df.to_csv(shap_out_path, index=False)
    print(f"Saved {shap_out_path.relative_to(REPO_ROOT)} (global SHAP importance, "
          f"full {len(X_scaled)}-row eligible train population, log-odds units)")

    print("\nTop 15 features by mean |SHAP value| (log-odds units):")
    print(shap_df.head(15).to_string(index=False))

    # Global bar plot. Built directly from shap_df rather than shap.plots.bar(): the library
    # default groups every feature past max_display into one "sum of N other features" bar,
    # which for 502 features dwarfs and visually drowns out the real top-20 bars. A plain
    # matplotlib bar of the top 20 by mean|SHAP| (styled like the coefficient plot, colored by
    # each feature's coefficient sign - agreement confirmed in Section 5 below) is clearer.
    top20_shap = shap_df.head(20).merge(
        pd.DataFrame({"feature": feature_names, "coefficient": clf.coef_.ravel()}), on="feature"
    ).iloc[::-1]
    colors = [COLOR_POS if c >= 0 else COLOR_NEG for c in top20_shap["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top20_shap["feature"], top20_shap["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Model A — top 20 features by global SHAP importance")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", linewidth=0.5)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLOR_POS, label="Positive coefficient (higher fail-risk)"),
                        Patch(color=COLOR_NEG, label="Negative coefficient (lower fail-risk)")],
              loc="lower right", frameon=False, fontsize=8)
    _savefig(fig, "model_a_shap_bar.png")

    # Beeswarm: readability requires a bounded number of points, so use a documented,
    # deterministic, class-proportional stratified sample (same approach as the EDA t-SNE plot).
    frac = min(1.0, n_beeswarm_sample / len(X))
    y_full = None
    # X passed in already eligible-only; class labels are attached by caller for the sample draw.
    return shap_values_full, expected_value, shap_df, X_scaled


def shap_beeswarm_plot(shap_values_full, expected_value, X_scaled, feature_names, y, n_sample=5000):
    rng_idx = pd.Series(y).groupby(y).apply(
        lambda s: s.sample(frac=min(1.0, n_sample / len(y)), random_state=RANDOM_STATE)
    ).index.get_level_values(1)
    sample_pos = np.asarray(rng_idx)
    print(f"\nBeeswarm plot sample: {len(sample_pos)} rows, class-proportional stratified "
          f"(random_state={RANDOM_STATE}) - visualization readability only, global bar/CSV "
          f"importance above already uses the full population.")

    explanation_sample = shap.Explanation(
        values=shap_values_full[sample_pos],
        base_values=np.full(len(sample_pos), expected_value),
        data=X_scaled[sample_pos],
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(9, 8))
    # group_remaining_features=False: with 502 features, the default "sum of N other features"
    # row is a wide, uninformative catch-all that crowds out the real top features - show only
    # genuine individual features up to max_display instead.
    shap.plots.beeswarm(explanation_sample, max_display=20, group_remaining_features=False, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Model A — SHAP summary (beeswarm), n={len(sample_pos)} stratified sample", y=1.02)
    _savefig(fig, "model_a_shap_summary.png")


# ======================================================================
# 3. LOCAL / INDIVIDUAL EXPLANATIONS (deterministic selection rule)
# ======================================================================

def local_explanations(clf, shap_values_full, expected_value, X, X_scaled, feature_names, ds):
    print("\n" + "=" * 78)
    print("SECTION 3: LOCAL EXPLANATIONS (deterministic selection, train.csv)")
    print("=" * 78)

    proba = clf.predict_proba(X_scaled)[:, 1]
    pred = (proba >= FROZEN_THRESHOLD).astype(int)
    y = ds.y.to_numpy()
    ids = ds.ids.reset_index(drop=True)

    def pick(mask, argfn):
        idx_pool = np.flatnonzero(mask)
        if len(idx_pool) == 0:
            return None
        local_arg = argfn(proba[idx_pool])
        return int(idx_pool[local_arg])

    rules = {
        "most_confident_correct_fail": (
            "argmax(predicted_proba) among rows with true label=Fail AND predicted=Fail",
            pick((y == 1) & (pred == 1), np.argmax)),
        "most_confident_correct_pass": (
            "argmin(predicted_proba) among rows with true label=Pass AND predicted=Pass",
            pick((y == 0) & (pred == 0), np.argmin)),
        "borderline_near_threshold": (
            f"argmin(|predicted_proba - {FROZEN_THRESHOLD}|) over all eligible rows",
            int(np.argmin(np.abs(proba - FROZEN_THRESHOLD)))),
        "worst_false_negative": (
            "argmin(predicted_proba) among rows with true label=Fail AND predicted=Pass (missed fail)",
            pick((y == 1) & (pred == 0), np.argmin)),
        "worst_false_positive": (
            "argmax(predicted_proba) among rows with true label=Pass AND predicted=Fail (false alarm)",
            pick((y == 0) & (pred == 1), np.argmax)),
    }

    records = []
    for rule_name, (rule_desc, row_idx) in rules.items():
        if row_idx is None:
            print(f"  [{rule_name}] no rows matched this rule - skipped.")
            continue
        print(f"\n[{rule_name}] rule: {rule_desc}")
        print(f"  row_idx={row_idx}, wafer_id={ids.loc[row_idx, 'wafer_id']}, "
              f"die_row={ids.loc[row_idx, 'die_row']}, die_col={ids.loc[row_idx, 'die_col']}, "
              f"true_label={y[row_idx]}, predicted_label={pred[row_idx]}, "
              f"predicted_proba={proba[row_idx]:.4f}")

        exp = shap.Explanation(
            values=shap_values_full[row_idx], base_values=expected_value,
            data=X_scaled[row_idx], feature_names=feature_names,
        )
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(exp, max_display=12, show=False)
        fig = plt.gcf()
        fig.suptitle(f"{rule_name}\nwafer={ids.loc[row_idx,'wafer_id']} "
                     f"die=({ids.loc[row_idx,'die_row']},{ids.loc[row_idx,'die_col']}) "
                     f"true={y[row_idx]} pred={pred[row_idx]} proba={proba[row_idx]:.3f}", fontsize=9)
        _savefig(fig, f"model_a_shap_local_{rule_name}.png")

        row_shap = shap_values_full[row_idx]
        top10_idx = np.argsort(-np.abs(row_shap))[:10]
        for rank, fidx in enumerate(top10_idx, start=1):
            records.append({
                "rule": rule_name, "rule_description": rule_desc, "row_idx": row_idx,
                "wafer_id": ids.loc[row_idx, "wafer_id"], "die_row": ids.loc[row_idx, "die_row"],
                "die_col": ids.loc[row_idx, "die_col"], "true_label": int(y[row_idx]),
                "predicted_label": int(pred[row_idx]), "predicted_proba": float(proba[row_idx]),
                "contributor_rank": rank, "feature": feature_names[fidx],
                "feature_value_raw": float(X.iloc[row_idx, fidx]),
                "feature_value_scaled": float(X_scaled[row_idx, fidx]),
                "shap_value": float(row_shap[fidx]),
            })

    local_df = pd.DataFrame(records)
    out_path = OUT_DIR / "model_a_shap_local_examples.csv"
    local_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path.relative_to(REPO_ROOT)} "
          f"({local_df['rule'].nunique()} examples x top-10 contributing features each)")
    return local_df


# ======================================================================
# 4. COORDINATE (die_row / die_col) REPORTING
# ======================================================================

def coordinate_reporting(coef_df, shap_df):
    print("\n" + "=" * 78)
    print("SECTION 4: COORDINATE FEATURE (die_row, die_col) SENSITIVITY")
    print("=" * 78)
    print("Reported as model sensitivity/association only - NOT a causal claim that die "
          "position causes failure.")

    for coord in ["die_row", "die_col"]:
        c = coef_df[coef_df["feature"] == coord].iloc[0]
        s = shap_df[shap_df["feature"] == coord].iloc[0]
        print(f"\n{coord}:")
        print(f"  Standardized coefficient: {c['coefficient']:+.4f} "
              f"(rank {int(c['rank'])} of {len(coef_df)} by |coefficient|, direction={c['direction']})")
        print(f"  Mean |SHAP|: {s['mean_abs_shap']:.4f} (rank {int(s['rank'])} of {len(shap_df)})")


# ======================================================================
# 5. CROSS-CHECK: COEFFICIENT RANKING vs SHAP RANKING
# ======================================================================

def cross_check(coef_df, shap_df, top_n=20):
    print("\n" + "=" * 78)
    print("SECTION 5: COEFFICIENT vs SHAP CROSS-CHECK")
    print("=" * 78)

    merged = coef_df[["feature", "coefficient", "absolute_coefficient", "rank"]].rename(
        columns={"rank": "coef_rank"}).merge(
        shap_df[["feature", "mean_abs_shap", "rank"]].rename(columns={"rank": "shap_rank"}),
        on="feature")
    merged["rank_diff"] = (merged["coef_rank"] - merged["shap_rank"]).abs()

    rho, pval = spearmanr(merged["coef_rank"], merged["shap_rank"])
    top_coef = set(coef_df.head(top_n)["feature"])
    top_shap = set(shap_df.head(top_n)["feature"])
    overlap = len(top_coef & top_shap)

    print(f"Spearman rank correlation between |coefficient| rank and mean|SHAP| rank "
          f"(all 502 features): rho={rho:.4f} (p={pval:.2e})")
    print(f"Top-{top_n} overlap: {overlap} of {top_n} features appear in both the top-{top_n} "
          f"|coefficient| list and the top-{top_n} mean|SHAP| list.")
    print(f"Mean |rank difference| across all 502 features: {merged['rank_diff'].mean():.2f}")

    out_path = OUT_DIR / "model_a_coef_vs_shap_comparison.csv"
    merged.sort_values("coef_rank").to_csv(out_path, index=False)
    print(f"Saved {out_path.relative_to(REPO_ROOT)}")

    print("\nCollinearity note: the EDA phase found a max pairwise |correlation| of 0.0131 across "
          "all feature_1..feature_500 columns on train.csv (see NOTES.md, EDA section) - i.e. "
          "essentially no collinearity. This matters here because collinear features would let "
          "coefficient magnitude and SHAP attribution disagree (importance gets arbitrarily split "
          "between correlated siblings). With near-zero correlation, no such splitting is expected, "
          "and the high rank agreement above is consistent with that.")

    return merged, rho, overlap


# ======================================================================
# MAIN
# ======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    pipe, scaler, clf, feature_names = load_frozen_pipeline()
    ds = load_reconstructed_train_X(config, feature_names)

    coef_df = coefficient_analysis(clf, feature_names)
    shap_values_full, expected_value, shap_df, X_scaled = shap_analysis(scaler, clf, ds.X, feature_names)
    shap_beeswarm_plot(shap_values_full, expected_value, X_scaled, feature_names, ds.y.to_numpy())
    local_explanations(clf, shap_values_full, expected_value, ds.X, X_scaled, feature_names, ds)
    coordinate_reporting(coef_df, shap_df)
    cross_check(coef_df, shap_df)

    print("\n" + "=" * 78)
    print("INTERPRETABILITY ANALYSIS COMPLETE.")
    print("Frozen Model A was not retrained, refit, or modified.")
    print("test.csv was not loaded. company_provided/ was not modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
