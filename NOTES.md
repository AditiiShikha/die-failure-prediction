# Notes
# Die Failure Prediction - Project Notes

## Current Branch
- Person A branch: `aditi`
- Branch created before Model A implementation.
- Company-provided files/configuration are kept separate from Person A's modeling code.

---

## Data Generation - COMPLETE

### Source
- Dataset: WM-811K (`LSWMD.pkl`)
- Location:
  `company_provided/data/LSWMD.pkl`
- Size: ~2.1 GB

### Company Generator
- Script: `company_provided/generate_data.py`
- Generator config: `company_provided/config.yaml`
- Company generator/config was NOT modified.
- Generator was run from inside `company_provided/`.

### Generated Files
- `company_provided/input/train.csv`
- `company_provided/input/test.csv`
- `company_provided/input/validation.csv`

### Generated Dataset
| Dataset | Wafers | Rows |
|---|---:|---:|
| train | 160 | 173,099 |
| test | 40 | 39,351 |
| validation | 40 | 39,351 |

### Important
- `train.csv` and `test.csv` contain labels.
- `validation.csv` does not contain `label`.
- `validation.csv` has the same 40 wafers / rows as `test.csv`, with the label column removed, according to `generate_data.py`.
- Do NOT use this overlap to tune the model unless the hackathon organizers confirm how validation is actually graded.
- `block_readings` is a ~2000-value column and is excluded from Model A.

---

# Target Definition - VERIFIED

Verified row-by-row on both `train.csv` and `test.csv`.

### Model A population
Only dies satisfying:

`old_label == 0`

are eligible for prediction.

### Target
For eligible dies:

`y = label`

Since `old_label == 0`, this is equivalent to predicting whether the die **newly fails**.

### Verified properties
- `old_label` and `label` are strictly `{0,1}`.
- `label >= old_label` for every row.
- No old-failed die changes back from `1 -> 0`.
- Target construction has zero violations.
- `sum(label) == sum(old_label) + newly_failed` exactly.

### Eligible population

Train:
- Eligible dies: 154,037
- Newly failed: 6,519
- Positive rate: 4.2321%

Test:
- Eligible dies: 32,598
- Newly failed: 1,380
- Positive rate: 4.2334%

The train/test newly-failed rate is therefore consistent at ~4.23%.

---

# Model A Feature Contract - VERIFIED

## Included

### Die-level measurements
- `feature_1` ... `feature_500`

These are the die-level parametric test measurements specified for Model A.

### Position
- `die_row`
- `die_col`

These are retained as raw positional features and will also be used to construct spatial features later.

## Excluded from model features

### `wafer_id`
- Used only as an identifier/grouping key.
- Used for wafer-level CV splitting.
- Used later to reconstruct predictions.
- Never fed into the model.

### `old_label`
- Used to define the eligible population.
- Not included in X because it is always `0` after filtering.

### `label`
- Target only.
- Never included in X.

### `block_readings`
- Explicitly excluded from Model A.
- Model A scope is die-level features + spatial context.
- `block_readings` belongs to Model B.

### Current Model A feature matrix
Before spatial engineering:

`die_row + die_col + feature_1...feature_500`

Total: 502 features.

---

# Phase 3 - Dataset Construction - COMPLETE

Files created:

- `config.yaml`
- `src/__init__.py`
- `src/data_loading.py`
- `src/target.py`

### Data loading
`src/data_loading.py`

- Reads only required columns.
- `block_readings` is excluded at the CSV-reading stage.
- It is NOT loaded and then discarded.

### Dataset assembly
`src/target.py`

Creates:
- `X`
- `y`
- `ids`

For train:
- X shape: `(154037, 502)`
- y positive rate: `4.2321%`

For test:
- X shape: `(32598, 502)`
- y positive rate: `4.2334%`

For validation:
- X shape: `(32598, 502)`
- y is `None` because validation has no label column.

---

# Phase 4 - Baseline Model - COMPLETE

## Model

Baseline:

`StandardScaler -> LogisticRegression`

Configuration:
- `class_weight="balanced"`
- `max_iter=2000`
- `random_state=42`
- threshold = `0.5`

No spatial/neighborhood features are included yet.

### Why Logistic Regression?
It provides a simple, interpretable baseline before testing stronger models or spatial features.

The baseline should answer:

> How well can die-level measurements alone predict newly failed dies?

---

# Cross-Validation Strategy

- 5-fold `GroupKFold`
- Group = `wafer_id`
- Performed on `train.csv`
- No wafer appears in both training and validation within a fold.
- 128 train wafers / 32 validation wafers per fold.

`test.csv` is kept separate from model selection and evaluated once after the baseline pipeline was fixed.

---

# Baseline Results

## 5-fold wafer-grouped CV

| Metric | Mean ± Std |
|---|---:|
| Fail Accuracy / Recall | 0.6901 ± 0.0172 |
| Fail Precision | 0.1468 ± 0.0376 |
| Fail F1 | 0.2400 ± 0.0486 |
| Pass Accuracy / Recall | 0.8233 ± 0.0085 |
| Overall Accuracy | 0.8175 ± 0.0069 |
| ROC-AUC | 0.8426 ± 0.0053 |
| PR-AUC | 0.5110 ± 0.0194 |

## Independent test.csv evaluation

Confusion matrix on eligible dies:

- Actual Fail / Pred Fail: 990
- Actual Fail / Pred Pass: 390
- Actual Pass / Pred Fail: 5,569
- Actual Pass / Pred Pass: 25,649

Metrics:

| Metric | Value |
|---|---:|
| Fail Accuracy / Recall | 0.717391 |
| Fail Precision | 0.150938 |
| Fail F1 | 0.249402 |
| Pass Accuracy / Recall | 0.821609 |
| Overall Accuracy | 0.817197 |
| ROC-AUC | 0.855580 |
| PR-AUC | 0.523890 |

---

# Baseline Interpretation

- Die-level measurements contain meaningful predictive signal.
- PR-AUC is ~0.51 despite the eligible positive rate being only ~4.23%.
- At threshold 0.5 with balanced class weights, recall is relatively high but precision is low.
- Test Fail F1 = ~0.249.
- The baseline is NOT considered the final model.
- Threshold tuning and alternative imbalance strategies are still open for experimentation.

---

# Important Open Decisions

## 1. Spatial window size (`m`)
The company specifies an `m × m` neighborhood window but does NOT specify the value of `m`.

Do not assume `m=5` is the final choice.

Treat window size as an experiment.

Possible values should be tested during the spatial-feature phase.

## 2. Decision threshold
The company materials do not specify the threshold for converting predicted probabilities into:

`predicted_label`

Baseline currently uses:

`threshold = 0.5`

This is a baseline placeholder and should be evaluated/tuned later.

## 3. Interpretability method
Company requires per-die feature importance and spatial contribution, but does not mandate a specific interpretability method.

SHAP is not currently a company requirement.

## 4. Validation-set grading
`validation.csv` contains the same 40 wafers/rows as `test.csv` with the label column removed according to the generator.

Need to clarify with hackathon organizers whether the external grader uses this exact generated validation set or regenerates/uses a separate hidden validation set.

## 5. Model B ownership
The rubric requires Model A → Model B delta analysis.

Current team split does not explicitly assign implementation of Model B.

This needs to be resolved at the team level.

## Phase 5 - Imbalance Experiments

- Tested: baseline balanced LR, unweighted LR, unweighted + tuned threshold, balanced + tuned threshold
- Same 5-fold wafer-grouped CV and same 502 features for all experiments
- Threshold tuning used train CV only, never test

| Model | Threshold | CV Fail F1 | CV PR-AUC | Test Fail F1 | Test Precision | Test Recall |
|---|---:|---:|---:|---:|---:|---:|
| Baseline balanced | 0.50 | 0.2417 | 0.5100 | 0.2494 | 0.1509 | 0.7174 |
| A: Unweighted | 0.50 | 0.5250 | 0.5108 | 0.5301 | 0.9767 | 0.3638 |
| B: Unweighted + tuned | 0.42 | 0.5267 | 0.5108 | 0.5321 | 0.9550 | 0.3688 |
| C: Balanced + tuned | 0.93 | 0.5261 | 0.5100 | 0.5307 | 0.9410 | 0.3696 |

### Key findings
- `class_weight="balanced"` did NOT improve ranking: ROC-AUC/PR-AUC stayed almost identical.
- Threshold choice had a much larger effect on Fail F1 than class weighting.
- Tuned threshold increased test Fail F1 from `0.2494` → ~`0.53`.
- F1-optimal models trade recall for very high precision.
- Experiment A disproved the original hypothesis that unweighted LR would collapse to predicting almost all passes.
- Candidate reference going into Phase 6: **unweighted LR + CV-tuned threshold (0.42)**, pending final confirmation.
- Do NOT start Phase 6 automatically.

## PHASE 6 — SPATIAL FEATURES

Wafer structure:
- Wafers are circular within their bounding boxes.
- Occupancy ≈ 74–80%.
- Missing grid positions are treated as missing dies, not imputed.

Spatial features added: 8
- dist_from_center
- edge_proximity
- neigh_old_fail_count_m3
- neigh_valid_count_m3
- neigh_old_fail_density_m3
- neigh_old_fail_count_m5
- neigh_valid_count_m5
- neigh_old_fail_density_m5

Feature count:
- Before: 502
- After: 510

Window sizes:
- Both 3×3 and 5×5 are computed.
- No window size chosen yet.
- Comparison will happen in Phase 7.

Important leakage rule:
- Spatial features use only die_row, die_col and old_label.
- Never use post-test label.
- Features are computed on the FULL wafer before filtering old_label==0.

Why neighborhood density matters:
- Generator itself uses pre-test neighborhood fail density when calculating new-fail probability.

Edge handling:
- A neighbor counts only if that die actually exists on that wafer.
- Missing/circular-corner positions are excluded from the denominator.

Phase 6 status:
- Spatial feature engineering implemented and sanity-checked.
- No spatial model trained yet.
- Next: Phase 7, train/evaluate and compare 3×3 vs 5×5.


### Phase 7 feature-count verification

- Recalculated feature counts directly against the actual code output before running experiments.
- Base feature set = 502: die_row + die_col + feature_1…feature_500.
- 3×3-only = 507 features: base 502 + 5 spatial features.
- 5×5-only = 507 features: base 502 + 5 spatial features.
- 3×3 + 5×5 = 510 features: base 502 + 8 spatial features.
- The two coordinate features (die_row, die_col) are always included, so single-window experiments add 5 features, not 3.
- Verified actual X shapes: 507 / 507 / 510.
- Previous note claiming 505 features for single-window experiments was incorrect and is superseded by this verified count.
- No duplicate or silently dropped features detected.

### Phase 7 conclusion

- All three spatial variants were compared against the Phase 5 non-spatial reference using the same 5-fold wafer-grouped CV protocol.
- Spatial variants were effectively tied on CV Fail F1:
  - 3×3-only: 0.5265
  - 5×5-only: 0.5261
  - 3×3 + 5×5: 0.5265
  - Phase 5 reference: 0.5267
- The Phase 5 non-spatial reference remained marginally best on CV Fail F1.
- Among spatial variants, 3×3-only was selected by the predefined rule.
- 3×3 test result: Fail F1 = 0.5286, Precision = 0.9618, Recall = 0.3645, ROC-AUC = 0.8624, PR-AUC = 0.5304.
- Conclusion: spatial features show some improvement in ranking metrics, but do not provide a clear improvement in Fail F1. Spatial contribution is therefore considered inconclusive rather than a confirmed improvement.
- Phase 5 502-feature model remains the simpler reference; 3×3 is retained as the best spatial variant for further controlled experimentation.

## Exploratory Recall Improvement — Threshold Trade-off

- Frozen threshold 0.42 reproduced exactly from train-only pooled OOF predictions.
- At 0.42:
  - Fail Precision = 0.9515
  - Fail Recall = 0.3642
  - Fail F1 = 0.5267
- Lowering threshold improves recall but causes a steep precision drop.
- Recall ~0.40 requires threshold ~0.22:
  - Precision = 0.6871
  - Recall = 0.4014
  - F1 = 0.5068
- Recall ~0.45 requires threshold ~0.14:
  - Precision = 0.4533
  - Recall = 0.4553
  - F1 = 0.4543
- Maximum recall in the 0.10–0.90 sweep was ~0.5064.
- Conclusion: recall limitation is not mainly a threshold problem; improving recall likely requires a different training strategy or better features.
- Frozen threshold 0.42 remains unchanged.

---

# EXPLORATORY DATA ANALYSIS (EDA)

Date: 2026-09-07

Script: `src/eda.py` (run as `python -m src.eda` from repo root).

**This is a separate, purely descriptive analysis pass. It does not modify,
retrain, or tune Model A in any way**, and is independent of whatever else
appears above this section. It reuses the existing Model A utilities
(`src/data_loading.py`, `src/target.py`, `src/features.py`,
`src/evaluate.py`) rather than re-implementing data loading or the
eligibility/target logic, so its numbers are computed the same way Model A
computes them.

**Scope / data-use rules followed:**
- Section 1 (schema audit) reads `train.csv` and `validation.csv` only.
- All label-bearing analysis (class imbalance, feature distributions,
  spatial failure patterns, correlations, t-SNE) uses `train.csv` only.
- `test.csv` and its `label` column were **not loaded anywhere** in this
  module - `src/eda.py` never imports `load_test`.
- `company_provided/`, `models/`, `test.csv`, `validation.csv` were not
  modified. No model was trained, tuned, or selected. No thresholds were
  chosen. No features were selected for a future model - all
  feature-ranking here (Cohen's d, correlation) is descriptive, to aid
  understanding/interpretation only.

Outputs: `outputs/eda_*.csv` (schema audit, class distribution, per-wafer
failure rates, feature summary, feature correlations, spatial summary) and
`outputs/eda_plots/*.png`. Existing `outputs/phase*.csv` files were not
touched.

## Dataset size / schema

Verified directly from the CSVs (not assumed):

| Dataset | Rows | Cols | Wafers | Dies/wafer (min/mean/max) | old_label==0 (eligible) | old_label==1 (excluded) |
|---|---:|---:|---:|---|---:|---:|
| train.csv | 173,099 | 505 | 160 | 498 / 1,081.9 / 5,340 | 154,037 | 19,062 |
| validation.csv | 39,351 | 504 | 40 | 505 / 983.8 / 4,096 | 32,598 | 6,753 |

- Columns: `wafer_id`, `die_row`, `die_col`, `old_label` (train+validation),
  `label` (train only - validation has no target column, confirmed),
  `feature_1`..`feature_500`. `block_readings` exists in the raw CSVs but is
  excluded at read time by `src/data_loading.py` (not loaded here either).
- `old_label`: pre-test status, 0=pass/1=fail. `label`: post-test status
  (old fails + new fails). Eligibility = `old_label==0`; on that population
  `label` is exactly "newly failed" (per `config.yaml` / `src/target.py`,
  cross-checked against `company_provided/README.md`).
- 0 missing values across id/old_label/feature columns in both files.
- 0 duplicate full rows and 0 duplicate `(wafer_id, die_row, die_col)`
  coordinate pairs in either file.
- Wafer bounding boxes vary substantially across wafers (48 distinct
  `(row_max, col_max)` combinations across 160 train wafers, from ~24x26 up
  to 80x80) - raw `die_row`/`die_col` are therefore **not comparable
  across wafers** without normalization; this is why Section 5 below uses
  the already-normalized `dist_from_center`/`edge_proximity` spatial
  features (from `src/features.py`) rather than raw coordinates for any
  cross-wafer aggregation.
- Train `label`: 147,518 pass (label=0) / 25,581 fail (label=1) over all
  173,099 rows (includes old-fails); restricted to the 154,037 eligible
  rows this becomes the class distribution below.
- `label >= old_label` sanity check: 0 violations on train.csv (matches the
  Phase 2 verification already documented above).

## Class imbalance (train.csv, eligible dies only)

| Metric | Value |
|---|---:|
| Eligible dies | 154,037 |
| Pass (label=0) | 147,518 (95.7679%) |
| Fail (label=1) | 6,519 (4.2321%) |
| Pass:Fail ratio | 22.63 : 1 |
| **Predict-all-Pass baseline accuracy** | **0.957679** |
| **No-skill PR-AUC baseline (= fail prevalence)** | **0.042321** |

- Predict-all-Pass reproduces the pass rate exactly (0.957679), as
  expected, with fail recall = 0 by construction - this is the "free"
  accuracy any model must beat to be useful, and it's why overall accuracy
  alone is a poor headline metric for this problem (Fail F1 / PR-AUC are
  more informative, consistent with why the project already tracks those).
- Per-wafer failure rate (160 wafers): mean=0.0414, median=0.0274,
  std=0.0529, min=0.0000, max=0.5000. Pooled (overall) rate=0.0423 - the
  mean-of-wafer-rates (0.0414) is close to but not identical to the pooled
  rate, and the wide spread (std=0.053, i.e. larger than the mean itself)
  shows failure rate is very unevenly distributed across wafers.
- Highest failure-rate wafers: `W_F_0007` (2 eligible dies, 1 fail = 0.50 -
  tiny sample, not reliable), `W_N_0148` (1,443 dies, 25.1%), `W_F_0039`
  (988 dies, 17.6%), `W_F_0029` (1,014 dies, 16.5%), `W_F_0049` (1,417
  dies, 15.9%).
- Lowest failure-rate wafers: several wafers (`W_F_0020`, `W_N_0007`,
  `W_N_0027`, `W_N_0033`, `W_N_0034`, ...) have exactly 0 new fails despite
  459-1,440 eligible dies each.
- Full per-wafer table: `outputs/eda_wafer_failure_rates.csv`.

Plots: `outputs/eda_plots/class_distribution.png`,
`outputs/eda_plots/wafer_failure_rate_distribution.png`.

## Feature distributions (feature_1..feature_500, train.csv eligible dies)

- **0 missing values** across all 500 features.
- **0 truly near-constant features** (std==0 or a single unique value) -
  every feature is fully continuous with all 154,037 values unique.
- Raw `std` spans a huge range (0.0035 to ~702) purely because features are
  on very different physical scales, which makes raw std misleading for
  "how variable is this feature" - **coefficient of variation (std/|mean|)
  is the honest measure** and it's tightly bounded: **0.0201 to 0.1502**
  across all 500 features. No feature is even close to constant in
  relative terms.
- 5 features sit at/above the 99th percentile of raw std (large absolute
  scale only, not necessarily more informative - flagged in
  `eda_feature_summary.csv` as `highly_variable_raw_scale`).
- 0 features exceed a 1% IQR-outlier rate - no feature shows unusual
  outlier behavior by the standard 1.5xIQR rule.
- Descriptive Pass-vs-Fail separation was ranked with **Cohen's d**
  (standardized mean difference) purely to pick which features to plot -
  **this ranking was not used to select features for any model.** Top 10 by
  |d|: `feature_171` (0.241), `feature_272` (-0.239), `feature_109`
  (-0.234), `feature_262` (-0.233), `feature_460` (0.233), `feature_397`
  (-0.233), `feature_163` (-0.229), `feature_245` (-0.228), `feature_277`
  (-0.228), `feature_85` (0.228).
- **All |d| values are small** (~0.2-0.24 at the very top, decaying toward
  0 for most of the 500 features) - by the usual rule-of-thumb (|d|~0.2
  "small", ~0.5 "medium", ~0.8 "large"), even the single best-separating
  feature only shows a small effect size. No individual feature separates
  Pass from Fail cleanly.
- Full table (mean/std/CV/min/max/outlier rate/Pass-mean/Fail-mean/Cohen's
  d per feature): `outputs/eda_feature_summary.csv`.

## Overlapping Pass/Fail distributions

- Plotted the top 12 features by |Cohen's d| as overlaid Pass/Fail density
  histograms (`outputs/eda_plots/top_features_pass_vs_fail_hist.png`) and
  the top 8 as standardized side-by-side boxplots
  (`outputs/eda_plots/top_features_pass_vs_fail_boxplot.png`).
- **Visually, even the "best" features show heavy overlap** - the Fail
  distribution is shifted slightly relative to Pass (consistent with the
  small Cohen's d values) but the two distributions substantially overlap
  in every plotted feature; there is no feature where Pass and Fail form
  visually separated clusters.
- For contrast, the 4 lowest-|d| features were also plotted
  (`outputs/eda_plots/lowest_signal_features_hist.png`) and show Pass and
  Fail histograms that are nearly indistinguishable.
- Interpretation (descriptive, not causal): no single feature is strongly
  predictive on its own; whatever signal Model A uses (test PR-AUC ~0.53,
  well above the 0.042 no-skill baseline, per Phase 10 above) must come
  from combining many weakly-informative features rather than from one or
  two dominant ones. This is consistent with Model A's logistic-regression
  coefficients needing all ~500 features rather than a small subset.

## Spatial / wafer observations (train.csv)

- Spatial features (`dist_from_center`, `edge_proximity`,
  `neigh_old_fail_density_m3`, etc.) were computed via the existing
  `src/features.add_spatial_features` on the **full per-wafer population**
  (old_label 0 and 1 together), matching the leakage rule already
  documented in the Phase 6 section above, then eligibility-filtered for
  the failure-rate analysis. Excluded (`old_label==1`) dies are kept
  entirely separate from this eligible-only failure-rate analysis, per the
  project's target definition, and are shown only as a distinct gray
  category on the wafer maps for visual context.
- **Edge vs. center**: failure rate rises modestly and roughly
  monotonically from center to edge - by distance-from-center decile,
  fail rate goes from 2.81% (innermost decile) to 5.70% (outermost); by
  edge-proximity decile, 3.00% to 5.22%. Point-biserial correlation of
  `label` with `dist_from_center` is +0.0379 and with `edge_proximity` is
  +0.0355 - both positive (edge dies fail somewhat more) but weak.
- **Neighborhood old-fail density**: dies with a nonzero 3x3-neighborhood
  old-fail density (28,327 dies, density~0.30 on average) show a higher new-fail
  rate (5.00%) than dies with no old-fails nearby (125,710 dies, 4.06%) -
  consistent with the Phase 6 rationale that the generator itself uses
  pre-test neighborhood density when assigning new-fail probability.
- Full decile/bin tables: `outputs/eda_spatial_summary.csv`. Plots:
  `outputs/eda_plots/fail_rate_vs_edge_proximity.png`,
  `outputs/eda_plots/fail_rate_vs_neighbor_density.png`.
- Wafer maps (`outputs/eda_plots/wafer_maps_sample.png`) for the
  lowest-fail-rate (`W_F_0017`, 0.0%), median-fail-rate (`W_F_0035`, 2.8%),
  and highest-fail-rate non-trivial (`W_F_0007`, 50% - but only 2 eligible
  dies, so not representative) wafers show new fails scattered without an
  obvious strong visual clustering pattern on the wafers inspected,
  consistent with the weak (but nonzero) edge/center and neighbor-density
  correlations above - the spatial signal exists but is subtle, not a
  dominant visual pattern.

## Correlation analysis (feature_1..feature_500, train.csv)

- Computed the full 500x500 Pearson correlation matrix (all 124,750
  unique pairs plus diagonal).
- **Max |pairwise correlation| across all 500 features: 0.0131**
  (`feature_401` vs `feature_445`). Median |r| = 0.0021, p90 = 0.0051, p99
  = 0.0079.
- **The 500 features are essentially pairwise uncorrelated** - no pair
  reached even |r|=0.5, so `outputs/eda_feature_correlations.csv` reports
  the top 20 pairs by |r| instead of a threshold-based list (all still
  <0.02).
- This is a notable and useful finding for later interpretability work:
  unlike typical real fab parametric test data (where many parameters are
  physically/electrically correlated), **this feature set shows no
  meaningful redundancy** - consistent with independently-generated
  synthetic parametric measurements. Practically, this means SHAP/feature-
  importance attribution is **not expected to be split across correlated
  siblings** for this dataset - each feature's importance can be read
  fairly directly, which would not be a safe assumption on correlated real
  fab data.
- Heatmap of the top 40 features by |Cohen's d| (a readable subset, not the
  full 500x500 matrix): `outputs/eda_plots/correlation_heatmap_top_features.png` -
  visually confirms an essentially diagonal-only heatmap (no visible
  off-diagonal structure).

## Dimensionality visualization (t-SNE)

- `umap` is not installed in the project venv; used `sklearn.manifold.TSNE`
  instead (available in-repo, no new dependency added).
- Full eligible train set (154,037 rows) is too large for a practical t-SNE
  run, so a **documented, class-proportional stratified sample of 6,000
  rows** (5,746 pass / 254 fail, ~3.9% of the eligible population) was
  used, with `StandardScaler` -> `PCA(50 components)` -> `TSNE` (perplexity
  30, `random_state=42`). This sample and embedding are for visualization
  only and were not used as model input or for any prevalence estimate.
- Plot: `outputs/eda_plots/tsne_pass_fail.png`. Most Fail points are
  interspersed throughout the Pass point cloud with no separation, but a
  distinct small cluster of Fail points appears isolated on one edge of the
  embedding - suggestive of a subgroup of failures that may share
  distinguishing feature characteristics, though t-SNE cluster geometry is
  not a reliable/quantitative claim on its own and this observation is not
  used to draw any modeling conclusion here.

## Limitations / cautions

- All label-bearing analysis is restricted to `train.csv`; findings have
  not been (and per this task's scope, should not be) verified against
  `test.csv`, so they describe the training distribution only.
- Cohen's d, the correlation matrix, and the t-SNE embedding are
  **descriptive tools only** - none of them were used to add, remove, or
  reweight any feature in Model A, and none of them constitute a
  significance test (no multiple-comparison correction was applied across
  500 features; with 500 tests some small effect sizes could arise by
  chance).
- Small-sample wafers (e.g. `W_F_0007` with 2 eligible dies) can show
  extreme failure rates (50%) that are not statistically meaningful -
  visible in `outputs/eda_wafer_failure_rates.csv` and called out
  explicitly above rather than treated as a genuine hotspot.
- The t-SNE sample (6,000 of 154,037 rows) is for visual inspection only;
  its 2D geometry should not be over-interpreted as global structure.

## Confirmation

- **Model A was not modified, retrained, or tuned** in this EDA pass.
- **`test.csv` was not loaded or used** for any analysis or decision here.
- **`company_provided/` was not modified.**
- **`train.csv` and `validation.csv` were not modified** (read-only via the
  existing `src/data_loading.py` loaders).
- No new features were selected for a future model; no thresholds were
  tuned; no new modeling experiments were run; Model B was not started.

---

# MODEL A INTERPRETABILITY (SHAP + COEFFICIENTS)

Date: 2026-09-07

Script: `src/interpretability.py` (run as `python -m src.interpretability`
from repo root).

**This section explains the already-frozen Model A. It does not retrain,
refit, or modify the model in any way.** `models/model_a_final_lr.joblib`
is loaded read-only with `joblib.load` and used only via `.transform()`,
`.decision_function()`, `.predict_proba()`, and direct inspection of
`.coef_`/`.intercept_` - `.fit()` is never called anywhere in
`src/interpretability.py`. The 502-feature set, the
`StandardScaler -> LogisticRegression` structure, and the 0.42 decision
threshold are all treated as fixed inputs.

## Frozen model used

Loaded from `models/model_a_final_lr.joblib` and verified programmatically
(not assumed) before any interpretation:

- Pipeline steps: `scaler` (`StandardScaler`) -> `clf` (`LogisticRegression`)
- `clf` params: `class_weight=None`, `max_iter=2000`, `random_state=42`
  (all asserted equal to the frozen spec; script raises if not)
- `clf.coef_.shape == (1, 502)`, `clf.intercept_ = -4.116370`
- Exact 502 features, in order: `die_row`, `die_col`, `feature_1`
  ... `feature_500` (`pipe.feature_names_in_`, cross-checked byte-for-byte
  against the reconstructed `train.csv` `X` columns before any
  coefficient/SHAP computation - the script asserts equality and aborts
  otherwise, so a silent feature-order mismatch cannot occur).
- Data used to reconstruct `X`: `train.csv` only, via the existing
  `src.data_loading.load_train` + `src.target.build_model_a_dataset`
  (same eligibility filter and feature assembly Model A itself uses).
  `test.csv` is never loaded by `src/interpretability.py`.

## 1. Standardized coefficient interpretation

Because the frozen pipeline is `StandardScaler -> LogisticRegression`,
`clf.coef_` is already expressed per standardized unit (mean 0, std 1 per
feature, per the scaler fit during training) - `|coefficient|` is directly
comparable across all 502 features without further normalization.

Full ranked table: `outputs/model_a_lr_coefficient_importance.csv`
(feature, coefficient, absolute_coefficient, direction, rank). Plot:
`outputs/model_a_lr_global_importance.png` (top 20 by |coefficient|,
orange=positive/higher fail-risk, blue=negative/lower fail-risk).

Top 4 positive (higher predicted fail-probability): `feature_460`
(+0.1117), `feature_85` (+0.1097), `feature_8` (+0.1095), `feature_171`
(+0.1089).

Top 3 negative (lower predicted fail-probability): `feature_231`
(-0.1017), `feature_109` (-0.1000), `feature_204` (-0.0987).

`die_col` ranks 9th of 502 by |coefficient| (+0.0973, positive); `die_row`
ranks 75th (+0.0752, positive) - see the coordinate-feature section below.

All top coefficients are modest in absolute size (~0.07-0.11 on
standardized units) with no single dominant feature - consistent with the
EDA finding that no individual feature separates Pass/Fail cleanly
(largest |Cohen's d| was only ~0.24).

## 2. SHAP method

`shap` was not present in the project venv or `requirements.txt`.
Following the "smallest necessary dependency change" instruction: added
one line (`shap>=0.44.0`) to the root `requirements.txt` (NOT
`company_provided/requirements.txt`, which was not touched) and installed
`shap` (resolved to 0.52.0, prebuilt wheel, no compiler needed) in the
existing project venv. No other packages were added.

**Method: `shap.LinearExplainer`** applied to the fitted `LogisticRegression`
step (`clf`), on `StandardScaler`-transformed features
(`scaler.transform(X)` - transform only, scaler is not refit). This is the
SHAP method built specifically for linear models: it is exact given a
background distribution (not a sampling approximation), and its output
units are the model's log-odds (`decision_function`). Correctness was
verified directly: `decision_function(X) == sum(shap_values, axis=1) +
expected_value` for every row, max absolute discrepancy `2.93e-14`
(floating-point noise) - i.e. the SHAP values exactly reconstruct the
actual frozen model's output, not an approximation of it.

- **Masker**: `shap.maskers.Independent`, background = 200 rows sampled
  from `train.csv` eligible dies (`random_state=42`). Independence is a
  documented, checked assumption here, not a default left unexamined: the
  EDA phase found a max pairwise |correlation| of **0.0131** across all
  500 `feature_*` columns on `train.csv` (see the EDA section above) -
  i.e. the features are empirically almost perfectly uncorrelated, so
  perturbing them independently matches the actual data closely.
- **Global importance**: computed on the **full eligible train.csv
  population** (154,037 rows) - cheap and exact for a linear model with an
  independent masker (0.91s wall-clock for the full population), so no
  subsampling was needed for the ranking itself.
  `outputs/model_a_shap_global_importance.csv` (feature, mean_abs_shap,
  rank). Bar plot: `outputs/model_a_shap_bar.png` (top 20, custom
  matplotlib bar rather than `shap.plots.bar()` directly - the library's
  default "sum of N other features" catch-all bar, summed over the
  remaining 482-483 features, dwarfed the real top-20 bars and made them
  unreadable; the custom bar keeps the same top-20 ranking and mean|SHAP|
  values but drops that misleading aggregate).
- **Beeswarm**: `outputs/model_a_shap_summary.png`, computed on a
  class-proportional stratified sample of 5,000 rows (`random_state=42`,
  same documented-sampling approach as the EDA t-SNE plot) - readability
  only; the ranking/values above already use the full population.
  `group_remaining_features=False` was set for the same reason as the bar
  plot fix.
- **Local/individual explanations**: 5 examples selected by a
  **deterministic, pre-specified rule** (train.csv only, using the frozen
  0.42 threshold) rather than manual cherry-picking:
  1. `most_confident_correct_fail` - argmax(predicted_proba) among true
     Fail & predicted Fail
  2. `most_confident_correct_pass` - argmin(predicted_proba) among true
     Pass & predicted Pass
  3. `borderline_near_threshold` - argmin(|predicted_proba - 0.42|) over
     all eligible rows
  4. `worst_false_negative` - argmin(predicted_proba) among true Fail &
     predicted Pass (biggest missed fail)
  5. `worst_false_positive` - argmax(predicted_proba) among true Pass &
     predicted Fail (biggest false alarm)

  Results: `most_confident_correct_fail` = wafer `W_N_0065`, die (3,12),
  proba=0.9997; `most_confident_correct_pass` = wafer `W_N_0039`, die
  (12,16), proba=0.0001; `borderline_near_threshold` = wafer `W_N_0064`,
  die (28,19), true=Fail, pred=Pass, proba=0.4197 (just below the 0.42
  cutoff); `worst_false_negative` = wafer `W_F_0049`, die (8,32),
  proba=0.0006 (confidently-wrong miss); `worst_false_positive` = wafer
  `W_F_0029`, die (16,31), proba=0.7478 (confidently-wrong false alarm).
  Waterfall plots: `outputs/model_a_shap_local_<rule_name>.png` (one per
  example). Top-10 contributing features per example (raw value, scaled
  value, SHAP value): `outputs/model_a_shap_local_examples.csv`.

## 3. Coordinate-feature (die_row, die_col) interpretation

Reported as **model sensitivity/association only** - the model was given
`die_row`/`die_col` as ordinary standardized features, and this reports
how much the model's output is sensitive to them, which is **not** a
causal claim that die position causes failure (and is a separate question
from the earlier EDA spatial analysis, which measured raw failure-rate
association, not model attribution).

| Feature | Coefficient | Coef rank /502 | Mean \|SHAP\| | SHAP rank /502 |
|---|---:|---:|---:|---:|
| die_row | +0.0752 | 75 | 0.0589 | 82 |
| die_col | +0.0973 | 9 | 0.0744 | 17 |

Both coordinates have positive coefficients (higher raw `die_row`/`die_col`
associated with higher predicted fail-probability in the model). `die_col`
is meaningfully more influential to the model than `die_row` (top-20 by
both methods), while `die_row` sits in the middle of the 502-feature
ranking. This is directionally consistent with the EDA spatial analysis,
which found a weak positive association between edge/center position and
observed failure rate (point-biserial r=+0.0355 to +0.0379) - both analyses
agree the position signal is real but modest, not dominant.

## 4. Coefficient vs. SHAP cross-check

`outputs/model_a_coef_vs_shap_comparison.csv` (feature, coefficient,
absolute_coefficient, coef_rank, mean_abs_shap, shap_rank, rank_diff).

- **Spearman rank correlation** between |coefficient| rank and mean|SHAP|
  rank, across all 502 features: **rho = 0.9999** (p ≈ 0).
- **Top-20 overlap**: **20 of 20** features in the top-20 by |coefficient|
  are also in the top-20 by mean|SHAP| - complete agreement.
- Mean |rank difference| across all 502 features: 0.88 positions.
- **The two methods agree almost perfectly.** This is expected, not
  coincidental: SHAP was computed on the same standardized-feature space
  the coefficients already live in, and given the empirical feature
  independence (below), mean|SHAP| for a linear model reduces to
  approximately `|coefficient| x (mean absolute deviation of the
  standardized feature)` - and every feature already has ~unit standard
  deviation on the training data by construction of `StandardScaler`, so
  there is little room for the two rankings to diverge here.

**Collinearity check**: the EDA phase (see EDA section above) found a
**max pairwise |correlation| of 0.0131** across all 500 `feature_*`
columns on `train.csv` - essentially no collinearity. This directly
supports why coefficient magnitude and SHAP attribution agree so closely
here: collinear features are the standard reason coefficient-based and
SHAP-based importance can disagree (importance gets arbitrarily split
between correlated siblings, and a linear model's coefficients become
unstable/hard to interpret individually under multicollinearity). With
essentially zero pairwise correlation in this feature set, **collinearity
is not a concern for interpreting these importance values individually** -
unlike what would typically be expected on real fab parametric data.

## Limitations

- Interpretability was performed on `train.csv` only, per the task scope;
  it describes what the frozen model learned from training data, not a
  re-validation against `test.csv`.
- SHAP's `Independent` masker assumes feature independence for the
  perturbation distribution; this is well-supported by the EDA
  correlation analysis (max |r|=0.0131) but is still an assumption, not a
  guarantee, for every possible pair among 500 features.
- Local explanations are illustrative examples chosen by documented rules,
  not a systematic error analysis across all misclassifications (that is
  explicitly out of scope for this task, per instructions to stop after
  interpretability).
- Coordinate-feature interpretation is associational (model sensitivity),
  not causal, and should not be read as a physical explanation of why
  edge/position affects yield.

## Confirmation

- **Frozen Model A (`models/model_a_final_lr.joblib`) was not retrained,
  refit, or modified** - only `joblib.load`, `.transform()`,
  `.decision_function()`, `.predict_proba()`, and attribute access were
  used.
- **The 502-feature set, pipeline structure, and 0.42 threshold were not
  changed.**
- **`test.csv` was not loaded or used** for any part of this
  interpretability analysis.
- **`company_provided/` was not modified** (its `requirements.txt` was
  left untouched; only the root `requirements.txt` gained the `shap` line).
- **`train.csv`, `test.csv`, `validation.csv` were not modified** (read via
  the existing loaders only).
- No model selection, threshold tuning, or additional modeling experiments
  were run; Model B was not started.

---

## MODEL A — POST-HOC ERROR ANALYSIS

Date: 2026-09-07

Script: `src/error_analysis.py` (run as `python -m src.error_analysis` from
repo root).

**This section is descriptive post-hoc analysis of the already-frozen
Model A's errors on `test.csv`. It changes nothing about the model.**
`models/model_a_final_lr.joblib` is used read-only (`joblib.load`, then
only `.predict_proba()` - `.fit()` is never called). The 502-feature set
and the 0.42 threshold are unchanged.

### Why test.csv was touched again here

Phase 10 (`src/final_evaluation.py`) evaluated `test.csv` exactly once and
saved only the **aggregate** confusion matrix/metrics to
`outputs/phase10_final_test_evaluation.csv` - it never saved per-die
predicted probabilities or labels anywhere in the repo (`predictions/`
contained only a `.gitkeep`). Sections 2-7 of this error analysis are
inherently per-die questions (probability distributions, which specific
dies are FN/FP, wafer/spatial concentration, feature comparisons) that
cannot be answered from the aggregate CSV alone. Since no per-die artifact
existed, this script performed **one forward pass** of the frozen pipeline
over `test.csv` (`pipe.predict_proba()` only) to generate that missing
artifact - not a re-evaluation, threshold search, or retraining. Before
any analysis proceeded, the script asserted that the resulting confusion
matrix **exactly** matches the already-documented, frozen Phase 10 result
(TP=509, FN=871, FP=24, TN=31,194) - it does, with zero discrepancy - so
Section 1's headline numbers below are read directly from the *existing*
`outputs/phase10_final_test_evaluation.csv`, not recomputed independently.

New per-die artifact produced: `outputs/model_a_test_predictions.csv`
(wafer_id, die_row, die_col, true_label, predicted_proba, predicted_label,
error_type).

### 1. Confusion-matrix breakdown (from existing Phase 10 CSV)

| | Value |
|---|---:|
| TP | 509 |
| FN | 871 |
| FP | 24 |
| TN | 31,194 |
| Precision (Fail) | 0.954972 |
| Recall (Fail) | 0.368841 |
| F1 (Fail) | 0.532148 |
| False-negative rate FN/(FN+TP) | 0.631159 |
| False-positive rate FP/(FP+TN) | 0.000769 |

`outputs/model_a_error_summary.csv`.

### 2. Predicted-probability / threshold findings

`outputs/model_a_error_probability.png`, `outputs/model_a_error_probability_bins.csv`.

- TP probabilities: mean 0.9535, median 0.9837 - the model is very
  confident when it correctly catches a fail.
- FN probabilities: mean 0.0739, median 0.0489, range [0.0010, 0.3935] -
  **every single FN's probability is below the 0.42 threshold by a wide
  margin on average**; none reach anywhere near it.
- FP probabilities: mean 0.5384, median 0.4938, range [0.4264, 0.7668].
- TN probabilities: mean 0.0266, median 0.0137 - very confidently correct.
- **FN bins**: [0.00,0.05)=444, [0.05,0.15)=303, [0.15,0.30)=108,
  [0.30,0.42) near-miss=16. **51.0% of all 871 FN (444 dies) have
  predicted probability under 0.05** - the model is highly confident
  these are Pass, and it is wrong. Only 16 of 871 FN (1.8%) are true
  "near misses" within 0.12 of the threshold.
- **FP bins**: [0.42,0.55) near-miss=15, [0.55,0.70)=6, [0.70,0.85)=3,
  [0.85,1.00]=0. FP skew the opposite way: most (15/24, 62.5%) are
  near-miss, and there are **zero high-confidence FP** (none at or above
  0.85).
- **Key finding: FN and FP are asymmetric in kind, not just count.** The
  majority of FN are confidently-wrong (far below threshold, not
  borderline), while essentially all FP are borderline (close to
  threshold, none confidently-wrong). This means **lowering the threshold
  would catch very few of the actual FN** (only ~16 near-miss cases could
  flip), while raising it would remove most of the already-few FP - i.e.
  the recall ceiling problem is not primarily a threshold-placement issue
  for the majority of misses. (Descriptive observation only - no
  threshold change was made or is being recommended here.)

### 3 & 7. False-negative findings (871 dies)

`outputs/model_a_fn_feature_summary.csv`, `outputs/model_a_fn_vs_tp_top_features.png`.

- **Wafer concentration**: FN dies appear on 39 of 40 test wafers - broadly
  distributed, not concentrated in one or two wafers. Among wafers with
  >=5 true-fail dies (36 wafers, avoiding tiny-sample noise), FN rate
  ranges widely: `W_N_0025` and `W_N_0038` miss 100% of their (small, 5-die)
  true-fail population, `W_N_0068` misses 85.7% (6/7), `W_N_0142` 84.6%
  (11/13), `W_N_0128` 77.8% (14/18) - full ranking in
  `outputs/model_a_error_by_wafer.csv`. 3 additional wafers have 1-4
  true-fail dies and are explicitly excluded from this ranking as
  unreliable (e.g. a single die gives a 0% or 100% "rate").
- **Spatial (edge vs center)**: FN rate by edge-proximity decile ranges
  ~0.57-0.70 with no clear monotonic trend (see
  `outputs/model_a_error_spatial.png`, `outputs/model_a_error_spatial_summary.csv`).
  Point-biserial correlation of edge_proximity with "is FN" (among
  true-Fail dies) is **+0.0194** - negligible. **Misses are not
  meaningfully more likely at the edge or center of the wafer** - unlike
  the (already weak) edge-vs-center failure-rate association found in the
  EDA phase, the *miss* pattern shows essentially no spatial trend at all.
- **Feature-space (TP vs FN)**: top distinguishing features by |Cohen's d|:
  `feature_457` (d=-0.54), `feature_60` (d=-0.51), `feature_269` (d=-0.51),
  `feature_189` (d=+0.49), `feature_123` (d=+0.49), `feature_223` (d=+0.49).
  These effect sizes (~0.49-0.54) are **notably larger than any Pass-vs-Fail
  effect size found in the EDA phase** (max |d|~0.24) - i.e. "easy" fails
  (TP) and "hard" fails (FN) are more separable from each other on these
  features than Pass and Fail are overall. This is a descriptive
  observation about which features characterize missed fails, not a
  causal explanation and not a proposal to change Model A's features.

### 4. False-positive findings (24 dies) - SMALL-SAMPLE CAUTION

`outputs/model_a_fp_feature_summary.csv`, `outputs/model_a_fp_vs_tn_top_features.png`.

**With only 24 observations, all findings in this subsection are
illustrative, not statistically reliable - effect sizes and rates here
have wide uncertainty and should not be treated as established patterns.**

- **Wafer distribution**: the 24 FP spread across 19 of 40 wafers (no
  wafer has more than 4); FP rate per wafer (among true-Pass dies) is tiny
  everywhere it occurs (0.05%-0.46%), consistent with FP being a rare,
  scattered event rather than a wafer-specific failure mode.
- **Probability/confidence**: all 24 FP have probability under 0.77;
  none reach the "high-confidence" band (>=0.85) - see Section 2.
- **Spatial location**: with only 24 points, no reliable decile analysis
  was attempted (see Section 6); the raw scatter in
  `outputs/model_a_error_spatial.png` shows FP spread across the edge-
  proximity range with no visually obvious clustering.
- **Feature characteristics**: top |Cohen's d| features (TN vs FP):
  `feature_116` (d=+0.84), `feature_361` (d=+0.69), `feature_154`
  (d=+0.63), `feature_433` (d=-0.61), `feature_238` (d=-0.60),
  `feature_441` (d=+0.59). These are larger effect sizes than the FN
  comparison, but **with n=24 this is expected sampling variance, not
  necessarily a more real effect** - a handful of outlier values in a
  24-row group can easily produce |d|~0.6-0.8 by chance.

### 5. Wafer-level error concentration

`outputs/model_a_error_by_wafer.csv` (all 40 test wafers).

- FN present on 39/40 wafers; FP present on 19/40 wafers.
- FN: broadly distributed across nearly every wafer (see Section 3 for the
  highest-rate wafers among those with enough true-fail dies to be
  reliable).
- FP: too rare (24 total) to identify a wafer-level "hotspot" - spread
  thinly, at most 4 on any single wafer (`W_F_0010`, out of 1,575
  true-pass dies on that wafer).
- Overall: **errors (especially FN) are broadly distributed across the
  test wafer population rather than concentrated in a small number of
  problem wafers.**

### 6. Spatial error findings

`outputs/model_a_error_spatial.png`, `outputs/model_a_error_spatial_summary.csv`,
`outputs/model_a_error_wafer_maps.png` (sample wafer maps for the two
highest-FN wafers and one FP-containing wafer, plotted on physical
die_row/die_col coordinates, reusing `src.features.add_spatial_features`
- the same spatial-feature machinery and leakage rule as the EDA and
interpretability phases, computed on the full per-wafer population before
the eligibility filter).

- FN rate vs edge-proximity decile: noisy, no clear trend (correlation
  +0.0194, see Section 3).
- FP: too few for a decile breakdown; visually scattered across the
  edge-proximity range in the sample plot.
- Sample wafer maps (`W_N_0015`, `W_F_0010`, `W_N_0018`) show FN dies
  interspersed among TN/TP dies with no obvious visual clustering into
  contiguous regions - consistent with the weak spatial correlation above.

### Train/CV evidence vs. post-hoc test analysis - explicit distinction

- **TRAIN/CV EVIDENCE** (model-selection basis, established in Phases 4-9
  and the EDA/interpretability sections above): wafer-grouped CV Fail F1/
  PR-AUC comparisons, the Phase 5 OOF threshold sweep that selected 0.42,
  the EDA class-imbalance/feature-distribution/correlation findings, and
  the SHAP/coefficient interpretability results. These are what actually
  informed Model A's frozen configuration.
- **POST-HOC TEST ANALYSIS** (this section only): the FN/FP probability
  distributions, wafer-level error rates, spatial error patterns, and
  TP-vs-FN / TN-vs-FP feature comparisons above. **None of this was used
  to select, tune, or validate Model A** - it was computed strictly after
  the model and threshold were already frozen (Phase 10), purely to
  describe how the already-final model behaves on held-out data.
  Nothing in this section fed back into any earlier phase.

### Limitations

- The FP analysis (n=24) is explicitly high-variance and illustrative
  only - see Section 4's caution.
- Wafers with fewer than 5 true-fail (or true-pass) dies were excluded
  from the "reliable" wafer-rate rankings to avoid over-interpreting
  small-sample rates (e.g. 1/1).
- Cohen's d comparisons (TP vs FN, TN vs FP) are descriptive effect sizes,
  not significance-tested, and no multiple-comparison correction was
  applied across 500 features.
- This analysis describes `test.csv` behavior only; it was not repeated
  on `validation.csv` (which has no labels) or used to re-touch `train.csv`.
- Spatial findings use the same `dist_from_center`/`edge_proximity`
  definitions as the EDA phase (normalized per-wafer, not raw
  `die_row`/`die_col`, since wafer bounding boxes vary - see EDA section).

### Confirmation

- **Frozen Model A (`models/model_a_final_lr.joblib`) was not retrained,
  refit, or modified** - `joblib.load` plus `.predict_proba()` only.
- **Threshold 0.42 was not changed** - used exactly as documented, purely
  to classify the already-computed probabilities into TP/FN/FP/TN.
- **`test.csv` was read once via the existing `src.data_loading.load_test`
  loader for a single frozen-pipeline inference pass** (needed because no
  per-die artifact existed yet, per the "why test.csv was touched again"
  note above) - **`test.csv` itself was not modified**, and the resulting
  confusion matrix was verified to exactly reproduce the already-frozen
  Phase 10 result before any error analysis proceeded.
- **`company_provided/` was not modified.**
- **`train.csv` and `validation.csv` were not modified** (not loaded by
  this script at all).
- No model selection, threshold tuning, class-weighting, regularization,
  spatial-feature, or other modeling experiments were run; no existing
  Phase 4-10 outputs were overwritten; Model B was not started.

---

# DOCUMENTATION RECONCILIATION ADDENDUM — PHASE 8, 9, 10 & VALIDATION SUBMISSION (RECONSTRUCTED)

Date: 2026-09-07

**Why this section exists**: an earlier repository audit (see the "MODEL A —
POST-HOC ERROR ANALYSIS" section above, and the standalone repo audit that
preceded this addendum) found that the original detailed prose write-ups
for Phase 8, Phase 9, and Phase 10 - and the documentation of the
validation-set submission - are no longer present in this file, even
though their underlying CSV/script artifacts survive on disk untouched.

**How this section was produced**: every phase below was rebuilt **from
scratch, using only the surviving CSV artifacts and the current source
code** (`outputs/phase8_model_family_experiments.csv`,
`outputs/phase8_per_fold_fail_f1.csv`, `outputs/phase9_spatial_perfold.csv`,
`outputs/phase7_spatial_experiments.csv`, `outputs/phase10_final_test_evaluation.csv`,
`src/experiments_model_family.py`, `src/experiments_spatial.py`,
`src/final_evaluation.py`, `src/generate_validation_submission.py`,
`outputs/validation_submission_model_a.csv`) - **not** from any earlier
in-conversation recollection of the original (now-missing) prose. Every
number below was re-derived directly from a currently-existing file during
this reconciliation pass; where a detail could not be recovered this way,
it is explicitly marked "Not recoverable from surviving artifacts" rather
than guessed or reconstructed from memory. No model was retrained and
`test.csv` was not reloaded to produce this section - all Phase 10 numbers
below are read from the existing `outputs/phase10_final_test_evaluation.csv`.

This addendum does not delete, condense, or reformat anything above it -
it is appended documentation only, and the original phase numbering/
terminology (Phase 6/7/8/9/10) is preserved.

---

## PHASE 8 — MODEL FAMILY COMPARISON (LR vs LightGBM vs ExtraTrees) [RECONSTRUCTED]

Script: `src/experiments_model_family.py`.

### What was tested and why
Per the script's own docstring: this phase isolates whether the model
*family* (not feature set, not imbalance handling, not CV protocol) is the
limiting factor, by comparing Logistic Regression against LightGBM and
ExtraTrees while holding everything else fixed:
- Same 502 non-spatial features (`die_row`, `die_col`, `feature_1..feature_500`)
  - no spatial/neighborhood features (Phase 6/7) included.
- Same shared `GroupKFold(n_splits=5)` split, computed once from a single
  shared `X`/`y`/`wafer_id` so fold membership is **identical by
  construction** across all three models (not re-verified per-model the
  way Phase 7's spatial variants needed to be, since there is only one
  dataset here).
- Unweighted training for every model (`class_weight=None` for LR/
  ExtraTrees). LightGBM additionally sets `is_unbalance=False` and
  `scale_pos_weight=1.0` explicitly (both already that library's defaults)
  so it receives no implicit rebalancing the other two don't also get.
- No hyperparameter search for any model - library defaults only, with a
  fixed `random_state=42` for all three:
  - **Logistic Regression**: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight=None, max_iter=2000, random_state=42))])`.
  - **LightGBM**: `LGBMClassifier(n_estimators=100, num_leaves=31, max_depth=-1, learning_rate=0.1, random_state=42, is_unbalance=False, scale_pos_weight=1.0, n_jobs=-1, verbosity=-1)`.
  - **ExtraTrees**: `ExtraTreesClassifier(n_estimators=100, max_depth=None, class_weight=None, random_state=42, n_jobs=-1)`.
- Each model's classification threshold independently tuned on that
  model's own pooled out-of-fold predictions from `train.csv` only
  (`src.evaluate.select_best_threshold`, maximizing Fail F1), applied only
  after CV is complete.
- The script asserts 0 NaN / 0 inf across all 502 features on the eligible
  population before proceeding (no imputer used).
- `test.csv` is not loaded anywhere in this script.

### Result (verified directly from `outputs/phase8_model_family_experiments.csv`)

| Model | Threshold | Pooled Fail F1 | Pooled Fail Precision | Pooled Fail Recall | Pooled PR-AUC | Pooled ROC-AUC | Pooled Overall Accuracy | Fold Fail F1 (mean ± std) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.4200 | **0.526736** | 0.951503 | 0.364166 | 0.510808 | 0.840485 | 0.972305 | 0.527729 ± 0.011223 |
| LightGBM | 0.14 | 0.435023 | 0.654344 | 0.325817 | 0.402289 | 0.763418 | 0.964184 | 0.436157 ± 0.018851 |
| ExtraTrees | 0.11 | 0.332623 | 0.491892 | 0.251266 | 0.284495 | 0.707901 | 0.957328 | 0.333860 ± 0.015845 |

Per-fold Fail F1 (verified directly from `outputs/phase8_per_fold_fail_f1.csv`, each model's own tuned threshold applied within each fold's OOF slice):

| Fold | Logistic Regression | LightGBM | ExtraTrees |
|---:|---:|---:|---:|
| 0 | 0.524382 | 0.417947 | 0.327596 |
| 1 | 0.530064 | 0.456103 | 0.352819 |
| 2 | 0.513846 | 0.415856 | 0.311367 |
| 3 | 0.544756 | 0.452772 | 0.333677 |
| 4 | 0.525597 | 0.438107 | 0.343840 |

Logistic Regression's per-fold Fail F1 exceeds LightGBM's on every fold by
at least 0.0655 (fold 4, the smallest margin) and up to 0.1064 (fold 3),
and exceeds ExtraTrees's on every fold by at least 0.1707 (fold 4) and up
to 0.2311 (fold 0) - i.e. Logistic Regression wins on every one of the 5
folds with no exceptions, not just on the pooled average.

**Runtime and pickled model-size comparison**: the script (`src/experiments_model_family.py`)
measures and prints per-fold fit/predict time and fold-0 pickled model size
for each model (see `run_model_cv`), but does **not** save these numbers
to any CSV - only the Fail-F1/precision/recall/PR-AUC/ROC-AUC/accuracy
metrics above were persisted to `outputs/phase8_model_family_experiments.csv`.
**Not recoverable from surviving artifacts** without re-running this
script, which was not done for this documentation-repair pass.

### Conclusion (derived directly from the table above)
- **Logistic Regression is retained** as Model A's model family. It beats
  both tree models on every pooled metric (Fail F1, Precision, Recall,
  PR-AUC, ROC-AUC, Accuracy) and on every individual CV fold.
- Per-fold standard deviations (0.0112-0.0189 across the three models) are
  5-10x smaller than the LR-vs-LightGBM and LR-vs-ExtraTrees gaps
  (≥0.0655 and ≥0.1707 respectively, per-fold) - the LR advantage is
  consistent across folds, not attributable to fold-to-fold noise.
- LightGBM and ExtraTrees both required very low tuned thresholds (0.14
  and 0.11 respectively) to reach their best Fail F1, and still lost to LR
  on precision *and* recall simultaneously (not merely a different point
  on a comparable curve) - consistent with the untuned library defaults
  underfitting this task, though this comparison cannot rule out that
  *tuned* tree-model hyperparameters (learning rate, depth/leaves,
  estimator count, imbalance handling) might be more competitive; that
  tuning was not performed in this phase and was deferred as optional
  future work.
- LightGBM and ExtraTrees are **not carried forward** as Model A's model
  family. Logistic Regression (unweighted, StandardScaler, threshold 0.42,
  502 features) is the frozen model family used in Phases 9-10 and in the
  final `models/model_a_final_lr.joblib`.

---

## PHASE 9 — SPATIAL FEATURE FINAL COMPARISON AND VERIFICATION [RECONSTRUCTED]

Underlying artifacts: `outputs/phase9_spatial_perfold.csv` (per-fold Fail F1
for the reference and all three spatial variants, plus per-fold
differences vs. the reference), cross-checked against
`outputs/phase7_spatial_experiments.csv` (pooled CV comparison for the
same four configurations) and `src/experiments_spatial.py` (methodology).
The originating script for `outputs/phase9_spatial_perfold.csv` itself is
**not recoverable from surviving artifacts** (no `src/*.py` file matching
a "Phase 9" per-fold verification script was found in `src/` at the time
of this reconciliation) - the CSV output survives, the script that
produced it does not appear to.

### What was being verified and why
Phase 7 (see the "Phase 7 conclusion" note earlier in this file, and the
pooled table below) had already compared the 502-feature non-spatial
reference against three spatial variants (3x3-only=507 features,
5x5-only=507 features, 3x3+5x5=510 features - all built via
`src.features.add_spatial_features` on the full per-wafer population
before the eligibility filter, per the Phase 6 leakage rule) using the
same wafer-grouped 5-fold CV protocol as every other phase. Phase 9's
surviving artifact (`outputs/phase9_spatial_perfold.csv`) provides the
per-fold breakdown needed to check whether the small pooled differences
seen in Phase 7 are genuine signal or ordinary fold-to-fold noise.

### Pooled CV comparison (verified directly from `outputs/phase7_spatial_experiments.csv`)

| Experiment | Features | Threshold | CV Fail F1 | CV Fail Precision | CV Fail Recall | CV PR-AUC (full precision) | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---|---:|
| Reference (502) | 502 | 0.42 | 0.526736 | 0.951503 | 0.364166 | 0.5108075436667536 | 0.840485 |
| 3x3-only | 507 | 0.46 | 0.526480 | 0.963673 | 0.362172 | 0.5127912218164143 | 0.842471 |
| 5x5-only | 507 | 0.38 | 0.526050 | 0.923018 | 0.367848 | 0.5127882081394063 | 0.842445 |
| 3x3+5x5 | 510 | 0.38 | 0.526477 | 0.922752 | 0.368308 | 0.5127757057780251 | 0.842310 |

Cross-check performed during this reconciliation: the reference row's
`cv_fail_f1` in `outputs/phase7_spatial_experiments.csv`
(0.5267361881517639) is bit-identical to the `pooled_fail_f1` value for
"Logistic Regression" in `outputs/phase8_model_family_experiments.csv`
(0.5267361881517639) - both describe the same 502-feature, unweighted LR,
threshold-0.42 configuration, computed independently in two different
phase scripts, and agree exactly.

### Per-fold verification (verified directly from `outputs/phase9_spatial_perfold.csv`)

| Fold | Reference (502) | 3x3-only (507) | 5x5-only (507) | 3x3+5x5 (510) |
|---:|---:|---:|---:|---:|
| 0 | 0.524382 | 0.521739 | 0.521195 | 0.520472 |
| 1 | 0.530064 | 0.531378 | 0.529885 | 0.530190 |
| 2 | 0.513846 | 0.514836 | 0.513783 | 0.514920 |
| 3 | 0.544756 | 0.542541 | 0.542097 | 0.542888 |
| 4 | 0.525597 | 0.526099 | 0.527946 | 0.527946 |

Per-fold differences (spatial variant minus reference), recomputed
directly from the CSV's own `diff_3x3`/`diff_5x5`/`diff_combined` columns
during this reconciliation:

| Variant | Mean diff vs. reference | Std of diff (ddof=1) |
|---|---:|---:|
| 3x3-only | -0.000410 | 0.001871 |
| 5x5-only | -0.000748 | 0.002235 |
| 3x3+5x5 | -0.000446 | 0.002473 |

For comparison, the reference configuration's own fold-to-fold standard
deviation (from `outputs/phase8_per_fold_fail_f1.csv`, "Logistic
Regression" column, ddof=1) is **0.011223** - roughly **15-27x larger**
than any of the three mean diffs above (0.011223 / 0.000410 ≈ 27.4;
0.011223 / 0.000748 ≈ 15.0; 0.011223 / 0.000446 ≈ 25.2).

### Conclusion (derived directly from the tables above during this reconciliation)
- All three spatial variants show a pooled PR-AUC/ROC-AUC modestly above
  the reference (~0.5128 vs. 0.5108 PR-AUC, ~0.002 gap) but pooled Fail F1
  at or below the reference for all three (diffs of -0.0003, -0.0007,
  -0.0003 respectively per the Phase 7 pooled table).
- The per-fold mean differences (-0.0004 to -0.0007) are an order of
  magnitude smaller than the reference's own fold-to-fold standard
  deviation (0.0112) - i.e. **the apparent spatial-vs-reference gap is not
  distinguishable from ordinary fold-to-fold noise**, for all three
  spatial variants.
- **Spatial features (3x3-only, 5x5-only, and 3x3+5x5 combined) were not
  adopted.** The 502-feature non-spatial Logistic Regression (unweighted,
  StandardScaler, threshold 0.42) remains Model A's frozen configuration.
  The three spatial variants remain documented challengers only, evaluated
  once each on `test.csv` per Phase 7 (3x3-only's one-shot test result:
  Fail F1=0.5286, Precision=0.9618, Recall=0.3645, ROC-AUC=0.8624,
  PR-AUC=0.5304, per the Phase 7 conclusion already documented earlier in
  this file), never adopted as the final model.
- Any additional narrative detail from the original Phase 9 write-up
  beyond what is reconstructed above (e.g. any further verification notes
  or draft decisions that may have existed) is **not recoverable from
  surviving artifacts**.

---

## PHASE 10 — FINAL INDEPENDENT TEST EVALUATION [RECONSTRUCTED]

Script: `src/final_evaluation.py`. Result artifact:
`outputs/phase10_final_test_evaluation.csv` (both read directly for this
reconciliation; `test.csv` was NOT reloaded or re-scored to produce this
section - every number below is read from the existing CSV).

### Frozen configuration used (per `src/final_evaluation.py`, unchanged from Phases 4-9)
- Model: Logistic Regression.
- Features (502): `die_row`, `die_col`, `feature_1..feature_500`.
- Preprocessing: `StandardScaler -> LogisticRegression`, a single fit on
  all of `train.csv`'s eligible rows (no CV at this step - CV was already
  used in Phases 4-9 for model/threshold selection, not here).
- `class_weight=None` (unweighted), `random_state=42`, `max_iter=2000`.
- Threshold = **0.42**, fixed - selected in Phase 5 on `train.csv` pooled
  OOF predictions only, not re-tuned in this phase.
- Spatial features (3x3/5x5/combined): evaluated in Phases 6-9, not
  adopted (see the Phase 9 reconstruction above).
- LightGBM/ExtraTrees: evaluated in Phase 8, not adopted (see the Phase 8
  reconstruction above).

### Procedure (per `src/final_evaluation.py` source code)
1. Fit the frozen pipeline once on the full eligible `train.csv`
   population (154,037 rows, 502 features) - no CV at this step.
2. Load `test.csv`, restrict evaluation to `old_label==0` eligible dies,
   consistent with the Phase 3 target definition (`old_label==1` dies are
   excluded from the eligible metrics because they trivially remain
   failed, per `company_provided/README.md`'s evaluation logic).
3. Apply the frozen threshold (0.42) to `predict_proba` output - no
   threshold tuning performed on `test.csv`.
4. Evaluate `test.csv` exactly once. The script asserts
   `TP+FP+FN+TN == n_eligible == metrics['n_eligible']` before reporting,
   as an explicit sanity check.
5. Save the fitted pipeline to `models/model_a_final_lr.joblib` and the
   metrics to `outputs/phase10_final_test_evaluation.csv`.

**Explicit statement**: `test.csv` was used in this phase **only** for
this single, final, independent evaluation, after the model family
(Phase 8), spatial-feature decision (Phase 7/9), and threshold (Phase 5)
were already selected using `train.csv` CV/OOF evidence alone. `test.csv`
was not used for tuning, feature selection, or model selection at any
point - Phase 10 is a one-shot honesty check on an already-frozen
configuration, not a further selection step.

### Row counts (verified directly from `outputs/phase10_final_test_evaluation.csv`)
| | Count |
|---|---:|
| `n_train_eligible` (train.csv, used to fit the final pipeline) | 154,037 |
| `n_test_total_rows` | 39,351 |
| `n_test_eligible_old_label_0` | 32,598 |
| `n_test_excluded_old_label_1` | 6,753 |
| `confusion_sum_tp_fp_fn_tn` | 32,598 (matches eligible rows exactly) |
| `sanity_check_passed` | True |

### Confusion matrix (eligible dies only, verified directly from the CSV)
| | Pred Fail | Pred Pass |
|---|---:|---:|
| **Actual Fail** | TP = 509 | FN = 871 |
| **Actual Pass** | FP = 24 | TN = 31,194 |

### Final test.csv metrics (verified directly from the CSV)
| Metric | Value |
|---|---:|
| Overall Accuracy | 0.972544 |
| Pass Accuracy (Recall) | 0.999231 |
| Fail Accuracy (Recall) | 0.368841 |
| Pass Precision | 0.972836 |
| Fail Precision | 0.954972 |
| Pass F1 | 0.985857 |
| Fail F1 | 0.532148 |
| PR-AUC | 0.527352 |
| ROC-AUC | 0.859033 |
| Threshold | 0.42 |
| random_state | 42 |

### Final model artifact
`models/model_a_final_lr.joblib` - the pipeline fitted in this phase, on
the full eligible `train.csv` population, at the frozen configuration
above. This is Model A's final, frozen artifact; no further model,
feature, or threshold changes were made after this phase.

### Phase 10 status
Final independent test.csv evaluation complete and unchanged since it was
originally produced. These results are final and were not used for any
further model-selection decisions, then or during this documentation
reconciliation.

---

## VALIDATION SUBMISSION [DOCUMENTED — no prior NOTES.md section existed]

Script: `src/generate_validation_submission.py` (run as
`python -m src.generate_validation_submission`). Applies the already-frozen
`models/model_a_final_lr.joblib` to `validation.csv`, unchanged - does not
train, retune, or modify the model in any way (`joblib.load` only, no
`joblib.dump` back to the frozen model path).

### Output
| | Value |
|---|---|
| Output path | `outputs/validation_submission_model_a.csv` |
| Exact columns | `wafer_id`, `die_row`, `die_col`, `predicted_label` |
| Row count | 39,351 (verified equal to `validation.csv`'s total row count) |

### `old_label` handling
- **`old_label==0` (eligible, 32,598 rows)**: built into the exact frozen
  502-feature matrix via the existing `src.target.build_model_a_dataset`
  (no new feature-construction logic written in this script), scored by
  the frozen pipeline's `predict_proba`, thresholded at the frozen 0.42.
- **`old_label==1` (excluded, 6,753 rows)**: not passed through the model
  at all - assigned `predicted_label=1` directly, per
  `company_provided/README.md`'s rule that pre-failed dies trivially
  remain failed.

### Threshold used
0.42 (frozen, reused unchanged - not re-tuned for validation.csv).

### Predicted-label counts (verified by reading `outputs/validation_submission_model_a.csv` directly)
| | Count |
|---|---:|
| Predicted `0` (Pass) | 32,065 |
| Predicted `1` (Fail) | 7,286 |

Cross-check performed during this reconciliation: 7,286 predicted-1 rows =
6,753 auto-filled `old_label==1` rows + 533 model-predicted-fail rows
among the 32,598 eligible rows (7,286 - 6,753 = 533); 32,065 + 533 =
32,598, matching the eligible row count exactly.

### Sanity checks (all implemented in-script; the script raises and aborts
the write if any fail - the output file's existence confirms all passed)
`row_count_matches_validation_csv`, `no_missing_predictions`,
`no_duplicate_id_rows`, `id_alignment_matches_validation_csv`,
`predicted_label_only_0_or_1`, `all_old_label_1_rows_predicted_1`,
`all_old_label_0_rows_scored_by_model`,
`frozen_threshold_0.42_used_for_old_label_0`,
`no_validation_label_column_present_or_used`, `test_csv_not_loaded`,
`company_provided_not_modified`, `model_file_not_modified` (12 checks
total, verified directly against the script's own `checks` dict).

### Confirmation validation labels were not used
`validation.csv` has no `label` column at all - the script asserts
`val_ds.y is None` before scoring, and `"label" not in val_raw.columns` is
one of the 12 sanity checks above. There is no label column present to
have used.

### Confirmation the frozen model was not modified
The script only ever calls `joblib.load(MODEL_PATH)` on
`models/model_a_final_lr.joblib` - there is no `joblib.dump` call
targeting that path anywhere in `src/generate_validation_submission.py`.

---

## Documentation Integrity Note

- The underlying Phase 5-10 artifacts (`outputs/phase5_imbalance_experiments.csv`,
  `outputs/phase7_spatial_experiments.csv`,
  `outputs/phase8_model_family_experiments.csv`,
  `outputs/phase8_per_fold_fail_f1.csv`,
  `outputs/phase9_spatial_perfold.csv`,
  `outputs/phase10_final_test_evaluation.csv`) **remain intact and were
  not modified** by this reconciliation pass - they were only read.
- The missing Phase 8, Phase 9, and Phase 10 narrative documentation above
  was **reconstructed from those surviving artifacts and the current
  source code only** - not from any earlier in-conversation recollection
  of the original (now-missing) prose. Where a detail was not recoverable
  this way (e.g. Phase 8's runtime/model-size figures, and any Phase 9
  narrative beyond what the surviving per-fold CSV supports), it is
  explicitly marked "Not recoverable from surviving artifacts" above
  rather than guessed.
- **Model A itself was not changed** by this documentation-repair pass -
  no `.fit()` or `joblib.dump()` call was made to
  `models/model_a_final_lr.joblib` or any other model file.
- **No `test.csv` rerun was performed** during this documentation repair -
  all Phase 10 numbers above are read directly from the existing
  `outputs/phase10_final_test_evaluation.csv`, not recomputed.
- **`company_provided/` was not modified** - none of its files were
  written to during this reconciliation.
- The validation-submission documentation above was added based on the
  existing generated artifact (`outputs/validation_submission_model_a.csv`)
  and the existing script (`src/generate_validation_submission.py`) - no
  new submission was generated to produce it.
- **This is documentation reconciliation only.** No model, data, or output
  CSV file was modified; no new modeling experiments were run; Model B was
  not started.