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


