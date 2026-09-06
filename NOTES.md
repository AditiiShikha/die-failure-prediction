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
- After: 510 (when both windows are used together)

Per-experiment feature count (verified directly against src/features.py +
src/target.py output, not assumed - `dist_from_center`/`edge_proximity`
are computed unconditionally by add_spatial_features regardless of which
window sizes are requested, so single-window variants still carry both):
- Base (die_row, die_col, feature_1..500): 502
- 3x3-only: 502 + dist_from_center + edge_proximity + neigh_*_m3 (3 cols) = 507
- 5x5-only: 502 + dist_from_center + edge_proximity + neigh_*_m5 (3 cols) = 507
- 3x3+5x5 combined: 502 + dist_from_center + edge_proximity + neigh_*_m3 (3) + neigh_*_m5 (3) = 510

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

---

## PHASE 7 — SPATIAL MODEL COMPARISON

- Script: `src/experiments_spatial.py`
- All variants: unweighted LR (`class_weight=None`), same `StandardScaler` pipeline, `random_state=42`, 5-fold wafer-grouped CV on `train.csv`.
- Spatial features computed via `add_spatial_features` on the full wafer population (old_label 0 and 1) before eligibility filtering, per the Phase 6 leakage rule.
- Reference keeps its Phase 5 CV-tuned threshold (0.42) fixed rather than re-tuning; each spatial variant's threshold is independently tuned on its own pooled out-of-fold probabilities (maximize Fail F1, train.csv only).
- Row order / wafer_id alignment between the reference and each spatial variant was asserted equal before scoring, to rule out any silent misalignment in the wafer-grouped CV.

### CV results (pooled out-of-fold, train.csv only)

| Experiment | Features | Threshold | CV Fail F1 | CV Fail Precision | CV Fail Recall | CV PR-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference (Phase 5 winner) | 502 | 0.42 | 0.5267 | 0.9515 | 0.3642 | 0.5108 | 0.8405 |
| 3×3-only | 507 | 0.46 | 0.5265 | 0.9637 | 0.3622 | 0.5128 | 0.8425 |
| 5×5-only | 507 | 0.38 | 0.5261 | 0.9230 | 0.3678 | 0.5128 | 0.8424 |
| 3×3 + 5×5 combined | 510 | 0.38 | 0.5265 | 0.9228 | 0.3683 | 0.5128 | 0.8423 |

Raw table: `outputs/phase7_spatial_experiments.csv`.

### Key findings
- All four configurations are within ~0.0007 CV Fail F1 of each other — effectively a statistical tie, not a clear win for spatial features.
- The **non-spatial reference actually has the highest CV Fail F1** of the four; spatial features gave a small PR-AUC/ROC-AUC bump (~0.0018 / ~0.002) but did not improve Fail F1 ranking at the tuned threshold.
- Among the three spatial variants (the only candidates in scope for selection per Phase 7 plan), **3×3-only is marginally best** by CV Fail F1 (0.5265 vs 0.5261 for 5×5, 0.5265 for combined — 3×3 and combined are effectively tied, 3×3 chosen on the reported value).
- Selected variant: **3×3-only, 507 features, threshold 0.46.**

### Test.csv evaluation (evaluated once, selected variant only)

| Metric | Value |
|---|---:|
| Fail Accuracy / Recall | 0.364493 |
| Fail Precision | 0.961759 |
| Fail F1 | 0.528639 |
| Pass Accuracy / Recall | 0.999359 |
| Overall Accuracy | 0.972483 |
| ROC-AUC | 0.862386 |
| PR-AUC | 0.530407 |

Confusion matrix (eligible dies): Actual Fail/Pred Fail 503, Actual Fail/Pred Pass 877, Actual Pass/Pred Fail 20, Actual Pass/Pred Pass 31,198.

Model saved to `models/model_a_spatial_best.joblib`.

### Phase 7 status
- Spatial variants trained, compared via CV, best variant selected, test.csv evaluated exactly once for that variant only. `test.csv` was not used for model or threshold selection at any point.
- Given the marginal/inconclusive CV delta over the non-spatial reference, whether to adopt the spatial variant as the final Model A (vs. the simpler 502-feature reference) is still an open decision - not resolved automatically here.
- Next phase not started.

---

## PHASE 8 — MODEL FAMILY COMPARISON (LR vs LightGBM vs ExtraTrees)

- Script: `src/experiments_model_family.py`
- Objective: isolate whether Logistic Regression itself is the limiting factor, by comparing model families while holding everything else fixed.

### Controlled variables (frozen across all three models)
- Same 502 non-spatial features: `die_row`, `die_col`, `feature_1`...`feature_500`. No spatial/neighborhood features.
- Same wafer-grouped 5-fold `GroupKFold` splits - computed once from a single shared `X`/`y`/`wafer_id`, so fold membership is identical by construction for every model.
- Unweighted training for every model (no `class_weight` rebalancing). LightGBM additionally kept `is_unbalance=False` and `scale_pos_weight=1.0` explicit, so it received no implicit imbalance handling either.
- Each model's classification threshold tuned independently on that model's own pooled out-of-fold predictions from `train.csv` only (same `select_best_threshold` / Fail-F1-maximizing procedure as Phase 5/7), applied only after CV was complete.
- No hyperparameter search for any model - library default settings only, documented explicitly at each constructor, with `random_state=42` fixed for all three.
- Missing-value check: 0 NaN / 0 inf across all 502 features on the eligible population - confirmed directly from the data, no imputer added.
- `test.csv` was not loaded or touched anywhere in this script.

### Model configurations (verified directly from `src/experiments_model_family.py`)
- **Logistic Regression (reference)**: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight=None, max_iter=2000, random_state=42))])`. StandardScaler confirmed present.
- **LightGBM**: `LGBMClassifier(n_estimators=100, num_leaves=31, max_depth=-1, learning_rate=0.1, random_state=42, is_unbalance=False, scale_pos_weight=1.0, n_jobs=-1, verbosity=-1)` - all library defaults except `random_state`/`verbosity`, no scaler (not needed for trees).
- **ExtraTrees**: `ExtraTreesClassifier(n_estimators=100, max_depth=None, class_weight=None, random_state=42, n_jobs=-1)` - all library defaults except `random_state`, no scaler.

### Pooled-OOF comparison (train.csv only, single global threshold per model)

| Model | Threshold | Fail F1 | Fail Precision | Fail Recall | PR-AUC | ROC-AUC | Overall Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (reference) | 0.42 | 0.5267 | 0.9515 | 0.3642 | 0.5108 | 0.8405 | 0.9723 |
| LightGBM | 0.14 | 0.4350 | 0.6543 | 0.3258 | 0.4023 | 0.7634 | 0.9642 |
| ExtraTrees | 0.11 | 0.3326 | 0.4919 | 0.2513 | 0.2845 | 0.7079 | 0.9573 |

Raw table: `outputs/phase8_model_family_experiments.csv`.

### Per-fold Fail F1 (each model's own tuned threshold applied within each fold's OOF slice)

| Fold | LR | LightGBM | ExtraTrees |
|---:|---:|---:|---:|
| 0 | 0.5244 | 0.4179 | 0.3276 |
| 1 | 0.5301 | 0.4561 | 0.3528 |
| 2 | 0.5138 | 0.4159 | 0.3114 |
| 3 | 0.5448 | 0.4528 | 0.3337 |
| 4 | 0.5256 | 0.4381 | 0.3438 |
| **mean ± std** | **0.5277 ± 0.0112** | 0.4362 ± 0.0189 | 0.3339 ± 0.0158 |

Raw table: `outputs/phase8_per_fold_fail_f1.csv`.

### Runtime and model-size comparison

| Model | Total CV time (5 folds) | Avg fold fit | Avg fold predict | Fold-0 model size (pickled) |
|---|---:|---:|---:|---:|
| LR | 12.2s | 2.14s | 0.17s | 0.02 MB |
| LightGBM | 40.4s | 7.79s | 0.13s | 0.38 MB |
| ExtraTrees | 70.0s | 13.53s | 0.31s | 154 MB |

`psutil` is not installed, so process-level memory (RSS) was not measured; wall-clock fit/predict time and pickled fold-0 model size are used as rough complexity/memory proxies instead. ExtraTrees stayed well within the 15-20 minute per-fold abort guard (13-14s/fold), so no early stop was triggered.

### Key findings
- **LR won decisively and consistently across all 5 folds** - it beats both tree models on every fold with no exceptions (margin ≥ 0.07 Fail F1 vs LightGBM, ≥ 0.17 vs ExtraTrees in every single fold), on every pooled metric (Fail F1, Precision, Recall, PR-AUC, ROC-AUC, Accuracy), and on runtime/model size.
- Per-fold std for each model (0.011-0.019) is 5-10x smaller than the gap between LR and the tree models, so this is a real, meaningful model-family effect - not fold-to-fold noise (contrast with Phase 7, where the spatial-vs-reference gap *was* noise).
- Both tree models needed very low thresholds (0.14, 0.11) just to reach their best Fail F1, and lost on precision *and* recall simultaneously versus LR (not a different point on a comparable ROC curve) - consistent with genuine underfitting of the untuned defaults on this task, not a fundamentally worse ceiling for tree methods in general.

### Conclusion
- **LightGBM and ExtraTrees are not carried forward as the primary model family.** Logistic Regression (unweighted, StandardScaler, threshold 0.42, 502 features) remains the primary reference model.
- Tree-model hyperparameter tuning (learning rate, depth/leaves, imbalance handling, estimator count) is **deferred as optional future work**, not required for the main pipeline - Phase 8's untuned comparison cannot rule out tuned tree models being competitive, it only establishes that defaults are not.

### Phase 8 status
- Model-family comparison complete. `test.csv` not evaluated at any point in this phase.
- Next phase not started.

---

## PHASE 9 — SPATIAL FEATURE FINAL COMPARISON AND VERIFICATION

Date: 2026-09-06

- Objective: complete the spatial-feature comparison started in Phase 7 by running the two remaining variants (5×5-only, 3×3+5×5 combined) with the same per-fold breakdown already produced for the reference and 3×3-only, then verify the full-precision PR-AUC values before finalizing the spatial-features decision.
- Reused without rerunning: 502-feature reference (Phase 7/8) and 3×3-only (Phase 7), including their per-fold Fail F1 values from the prior verification pass.
- Newly run: 5×5-only (507 features) and 3×3+5×5 combined (510 features) - pooled OOF generated and per-fold Fail F1 computed for these two only.

### Four-way pooled-OOF comparison

| Experiment | Features | Threshold | Fail F1 | Fail Precision | Fail Recall | PR-AUC (full precision) | ROC-AUC | Overall Accuracy |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 502-feature reference | 502 | 0.42 | 0.5267 | 0.9515 | 0.3642 | 0.5108075436667536 | 0.8405 | 0.9723 |
| 3×3-only | 507 | 0.46 | 0.5265 | 0.9637 | 0.3622 | 0.5127912218164143 | 0.8425 | 0.9724* |
| 5×5-only | 507 | 0.38 | 0.5260 | 0.9230 | 0.3678 | 0.5127882081394063 | 0.8424 | 0.9719 |
| 3×3+5×5 combined | 510 | 0.38 | 0.5265 | 0.9228 | 0.3683 | 0.5127757057780251 | 0.8423 | 0.9720 |

*Reconstructed analytically from recorded precision/recall + fixed eligible-population counts (6,519 fail / 147,518 pass), not directly measured; validated by reproducing the reference's Overall Accuracy exactly against its independently measured value (0.972305) before trusting the reconstruction for 3×3-only.

Full-precision PR-AUC values verified directly from `outputs/phase7_spatial_experiments.csv` and cross-checked as bit-identical against the same reference value independently recorded in `outputs/phase5_imbalance_experiments.csv` and `outputs/phase8_model_family_experiments.csv`.

### Per-fold comparison and noise conclusion
- All three spatial variants show pooled PR-AUC modestly above the reference (~0.5128 vs 0.5108, a ~0.002 gap - the metric least sensitive to threshold choice), but pooled Fail F1 at or below the reference for all three (diffs of -0.0003, -0.0007, -0.0003).
- Paired per-fold Fail F1 differences (spatial variant minus reference) have means an order of magnitude smaller than their own fold-to-fold standard deviation for every variant - the apparent differences are not distinguishable from ordinary fold-to-fold noise.
- Conclusion: spatial features produce a small, consistent ranking-quality improvement (PR-AUC) that does not translate into a meaningful Fail F1 gain at the operating threshold this pipeline optimizes for.

### Verification notes
- A prior draft tie-break clause ("combined must beat 5×5 by at least 0.005 absolute pooled Fail F1") was **not** actually pre-agreed before Phase 9 was implemented. That framing is retracted here. It had no effect on the actual decision, since no spatial variant beat the reference meaningfully in the first place, so the clause was never reached.
- No retraining was performed during the Phase 9 verification pass - full-precision PR-AUC values were recovered from already-saved CSV outputs (`outputs/phase7_spatial_experiments.csv`, `outputs/phase5_imbalance_experiments.csv`, `outputs/phase8_model_family_experiments.csv`) and confirmed via code inspection of `src/experiments_spatial.py`/`src/evaluate.py`, not by rerunning models.
- `test.csv` was not evaluated at any point in Phase 9. `company_provided/` was not modified.

### Final decision
- **The 502-feature non-spatial Logistic Regression (unweighted, StandardScaler, threshold 0.42) remains the primary model.**
- 3×3-only, 5×5-only, and 3×3+5×5 combined all remain documented challengers and were **not adopted**.

### Phase 9 status
- Spatial-feature comparison and verification complete. Checkpoint reached before Phase 10.
- Phase 10 not started.

---

## PHASE 10 — FINAL INDEPENDENT TEST EVALUATION

Date: 2026-09-06

Script: `src/final_evaluation.py`

This is the **one-shot final evaluation** of Model A on `test.csv`. It is
separate in kind from every CV number reported in Phases 4-9: those were
all `train.csv`-only, wafer-grouped, out-of-fold metrics used for model
and threshold *selection*. This section is the single independent
measurement taken after that selection was already frozen, and it was not
used to make any further model, feature, or threshold decision.

### Frozen configuration used (unchanged from Phases 4-9)
- Model: Logistic Regression
- Features (502): `die_row`, `die_col`, `feature_1`...`feature_500`
- Preprocessing: `StandardScaler` -> `LogisticRegression`, scaler fit only on training data at each fitting step (verified leakage-free for the CV procedure earlier in Phase 10 planning; this final fit is a single fit on all of `train.csv`'s eligible rows, no CV at this step)
- `class_weight=None` (unweighted)
- `random_state=42`
- `max_iter=2000`
- Threshold = **0.42**, fixed - selected in Phase 5 on `train.csv` pooled OOF predictions only, not re-tuned here
- Spatial features (3×3 / 5×5 / combined): evaluated in Phases 6-9, not adopted
- LightGBM / ExtraTrees: evaluated in Phase 8, rejected

### Procedure
1. Fit the frozen pipeline once on the full eligible `train.csv` population (154,037 rows, 502 features) - no CV at this step.
2. Loaded `test.csv`, restricted evaluation to `old_label==0` eligible dies, consistent with the Phase 3 target definition.
3. Applied the frozen threshold (0.42) to `predict_proba` output - no threshold tuning performed on `test.csv`.
4. Evaluated `test.csv` exactly once.

### Row counts and confusion-matrix sanity check
- Total `test.csv` rows: 39,351
- Eligible (`old_label==0`) rows: 32,598
- Excluded (`old_label==1`) rows: 6,753
- Confusion matrix below is calculated **only on the 32,598 old_label==0 eligible dies** - excluded old-fail dies are not part of these counts.
- Sanity check: TP + FP + TN + FN = 509 + 24 + 31,194 + 871 = 32,598 = eligible test rows. **PASS.**

### Raw confusion-matrix counts (eligible dies only)

| | Pred Fail | Pred Pass |
|---|---:|---:|
| **Actual Fail** | TP = 509 | FN = 871 |
| **Actual Pass** | FP = 24 | TN = 31,194 |

### Final test.csv metrics (company-required + PR-AUC/ROC-AUC)

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

n_eligible = 32,598; n_actual_fail = 1,380; n_actual_pass = 31,218.

Raw table: `outputs/phase10_final_test_evaluation.csv`. Final fitted pipeline: `models/model_a_final_lr.joblib`.

### Phase 10 status
- Final independent test.csv evaluation complete. These results are final and are not used for any further model-selection decisions.
- Phase 11 not started.


