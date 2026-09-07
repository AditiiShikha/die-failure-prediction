# Model A — Complete Handoff

Document date: 2026-09-07
Source of truth: `NOTES.md`, `config.yaml`, `src/`, and the CSV/PNG files under `outputs/` and `models/` in this repository, as they exist at the time this document was written. Where a claim could not be verified against a currently-existing repository artifact, it is explicitly marked **"Not verified"** rather than guessed or recalled from memory.

This document is descriptive only. Producing it did not retrain, refit, retune, or re-evaluate Model A, and did not touch `test.csv`.

---

## 1. Executive Summary

Model A predicts, for each die on a semiconductor wafer that has **not already failed pre-test** (`old_label == 0`), the probability that it will **newly fail** post-test (`label == 1`). This is the "Model A — Die-Level Only" track of the hackathon problem statement (`company_provided/Hackathon Problem - Die Yield Prediction.docx`): predict per-die pass/fail using die-level parametric test measurements plus positional context, without the ~2,000-value per-die `block_readings` signal (that is reserved for Model B).

The final, frozen model is a **`StandardScaler -> LogisticRegression`** pipeline (`class_weight=None`, `max_iter=2000`, `random_state=42`), trained on 502 features (`die_row`, `die_col`, `feature_1..feature_500`) from the full eligible `train.csv` population (154,037 rows), with a decision threshold of **0.42** selected on train-only out-of-fold (OOF) predictions. It is saved at `models/model_a_final_lr.joblib` and was evaluated exactly once on `test.csv` (Phase 10): **Fail F1 = 0.532, Fail Precision = 0.955, Fail Recall = 0.369, PR-AUC = 0.527, ROC-AUC = 0.859, Overall Accuracy = 0.973**.

The model's main strength is **precision**: when it predicts a die will fail, it is right about 95% of the time on held-out test data, which is far better than the "predict everything Pass" baseline (95.77% accuracy but 0% fail recall) and the no-skill PR-AUC baseline (0.042, i.e. the raw fail prevalence). Its main weakness is **recall**: it only catches about 37% of dies that actually newly fail, and post-hoc error analysis (Section 10) shows the majority of missed fails (51%) are missed with high confidence (predicted probability under 0.05) rather than being borderline near the 0.42 threshold — i.e. the recall ceiling is not primarily a threshold-placement problem. A dedicated exploratory sweep (Section 7) confirmed the maximum achievable Fail Recall across the entire 0.10-0.90 threshold range is only ~0.51, and that gain comes at a steep precision cost.

**Model A is currently frozen.** Nothing about its features, preprocessing, hyperparameters, or threshold should be changed without an explicit, agreed decision to do so. All work described in this document (EDA, interpretability, error analysis, validation submission) was performed strictly *after* the model was frozen and is descriptive/read-only with respect to the model itself.

---

## 2. Dataset

Source: WM-811K wafer map dataset (`company_provided/data/LSWMD.pkl`), with synthetic die-level parametric features and block-level readings generated on top of it by `company_provided/generate_data.py` (not modified by any of this project's own work).

| File | Rows | Wafers | Has `label`? | Role |
|---|---:|---:|---|---|
| `company_provided/input/train.csv` | 173,099 | 160 | Yes | Model training + all CV/OOF model-selection work |
| `company_provided/input/test.csv` | 39,351 | 40 | Yes | Held out; evaluated exactly once (Phase 10) after the model/threshold were already frozen |
| `company_provided/input/validation.csv` | 39,351 | 40 | **No** (label column absent) | Unlabeled; only ever scored, never used for training/selection |

`validation.csv` has the same 40 wafers/rows as `test.csv` with the label column removed, per `generate_data.py` (documented in NOTES.md's "Data Generation" section). Whether the external grader uses this exact file or a separate hidden set was an open question as of the last update to NOTES.md — **Not verified** whether this has since been resolved.

### `old_label` handling and eligible population
- `old_label`: pre-test die status (0=pass, 1=fail, from the WM-811K map). `label`: post-test die status (old fails + new fails).
- Model A's eligible population is `old_label == 0` (dies not already failed pre-test). On that population, `label` is exactly the "newly failed" indicator (verified row-by-row in Phase 2/3 work: `label >= old_label` for every row, zero violations, on both train.csv and test.csv).
- Dies with `old_label == 1` are excluded from model training and from Model A's accuracy/precision/recall calculations (per `company_provided/README.md`'s evaluation logic: they trivially remain failed and would inflate accuracy numbers) but are still included in the final submission format with `predicted_label=1` assigned directly.

### Class distribution (train.csv, eligible dies only — verified via EDA, `outputs/eda_class_distribution.csv`)
| | Count | % |
|---|---:|---:|
| Eligible dies (`old_label==0`) | 154,037 | — |
| Pass (`label=0`) | 147,518 | 95.7679% |
| Fail (`label=1`) | 6,519 | 4.2321% |

Test.csv eligible population: 32,598 dies (excluded old-fails: 6,753), with a newly-failed rate of 4.2334% — consistent with train.csv's rate (per NOTES.md's Phase 2 verification).

### Other dataset characteristics (verified via EDA)
- Wafer bounding boxes vary substantially (48 distinct `(row_max, col_max)` combinations across 160 train wafers, from ~24x26 up to 80x80) — raw `die_row`/`die_col` are **not directly comparable across wafers** without normalization.
- 0 missing values across all id/`old_label`/feature columns in train.csv and validation.csv.
- 0 duplicate rows and 0 duplicate `(wafer_id, die_row, die_col)` coordinate pairs.
- All 500 `feature_*` columns are fully continuous (no truly near-constant features; coefficient-of-variation range 0.0201-0.1502 across all 500).
- `feature_1..feature_500` are essentially pairwise **uncorrelated**: max |Pearson r| across all 500 features on train.csv is **0.0131**.

### Train/CV evidence vs. test-set results vs. post-hoc analysis — explicit distinction
- **Train/CV evidence** (used for model selection): all wafer-grouped 5-fold CV/OOF results in Phases 4-9, the EDA phase, and the exploratory threshold trade-off (Section 7) — all computed from `train.csv` only.
- **Test-set results** (Phase 10, one-shot): the final confusion matrix and metrics in Section 6, computed once on `test.csv` after the model and threshold were already frozen.
- **Post-hoc analysis** (Section 10): FN/FP/wafer/spatial/feature patterns computed from `test.csv` predictions *after* Phase 10, purely descriptive, never fed back into model selection.

---

## 3. Exact Final Model Configuration

Verified directly against the loaded `models/model_a_final_lr.joblib` object (via `joblib.load`, inspecting `.steps`, `.get_params()`, `.coef_.shape`, `.feature_names_in_` — see `src/interpretability.py` Section 0 for the exact assertions run) and `config.yaml`.

| Parameter | Value |
|---|---|
| Model family | Logistic Regression (`sklearn.linear_model.LogisticRegression`) |
| Preprocessing | `sklearn.preprocessing.StandardScaler` (fit on training data only) |
| Pipeline order | `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(...))])` — scaler always applied before the classifier |
| `class_weight` | `None` (unweighted) |
| `max_iter` | `2000` |
| `random_state` | `42` |
| Solver | `lbfgs` (scikit-learn default; not explicitly overridden) |
| Decision threshold | **0.42** |
| Intercept (`clf.intercept_`) | -4.116370 |
| Number of features | **502**, exactly |

**Exact 502 features, in the order the model expects them (`pipe.feature_names_in_`):**
1. `die_row`
2. `die_col`
3. `feature_1` ... `feature_500` (in numeric order)

Features are die-level parametric test measurements (`feature_1..feature_500`) plus raw wafer-grid position (`die_row`, `die_col`). Defined by `config.yaml` (`model_a.feature_prefix="feature_"`, `model_a.num_features=500`, `model_a.spatial_columns=[die_row, die_col]`) and assembled by `src/target.build_model_a_dataset`.

**Spatial (neighborhood) features tested and rejected** (Phase 6/7/9 — see Section 4 below for the comparison): `dist_from_center`, `edge_proximity`, `neigh_old_fail_count_m3`, `neigh_valid_count_m3`, `neigh_old_fail_density_m3`, `neigh_old_fail_count_m5`, `neigh_valid_count_m5`, `neigh_old_fail_density_m5` (8 features total, computed by `src/features.add_spatial_features`). Three variants (3x3-only = 507 features, 5x5-only = 507 features, 3x3+5x5 = 510 features) were compared against the 502-feature reference by CV Fail F1 and **none were adopted** — the frozen 502-feature model does not include any of these.

**Block-level features**: `block_readings` (a ~2,000-value space-separated array per die) is **explicitly excluded** from Model A's scope entirely — it is excluded even at the CSV-reading stage (`src/data_loading.py`, via `config.yaml`'s `model_a.excluded_columns: [block_readings]`), never loaded into memory for Model A. It is reserved for Model B.

**Non-feature columns**: `wafer_id` (grouping/identifier only, never fed to the model), `old_label` (used only to build the eligibility filter — dropped from X since it's constant at 0 across the eligible population), `label` (target only).

**CV protocol**: `sklearn.model_selection.GroupKFold(n_splits=5)`, grouped by `wafer_id` — no wafer's dies ever span both the train and validation side of any fold (asserted explicitly in every CV loop across Phases 4/5/7/8 and the threshold-tradeoff script). 128 train wafers / 32 validation wafers per fold (train.csv, 160 wafers total).

---

## 4. Model Development History

Summarized from `NOTES.md` where its prose is still present, cross-checked against the corresponding CSV outputs. **Audit note**: as of this document, `NOTES.md`'s detailed Phase 8, 9, and 10 write-ups (and part of Phase 7's) have been condensed/removed from the live file by an earlier edit outside this document's scope — the raw CSV outputs still exist and are the source for the numbers below; anything not recoverable from a current artifact is marked "Not verified."

### Phase 4 — Baseline Logistic Regression
- **What was tested**: `StandardScaler -> LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)`, threshold=0.5 (placeholder), no spatial features. 5-fold wafer-grouped CV.
- **Why**: establish whether die-level measurements alone contain predictive signal, before any spatial engineering or threshold tuning.
- **Result**: CV Fail F1 = 0.2400 ± 0.0486, CV PR-AUC = 0.5110 ± 0.0194, CV ROC-AUC = 0.8426 ± 0.0053. Test Fail F1 = 0.2494 (verified — matches `outputs/phase5_imbalance_experiments.csv` row 1 exactly: test_fail_f1=0.249402).
- **Decision**: adopted as the starting baseline, not final — PR-AUC (~0.51) was already well above the no-skill baseline (0.042), showing real signal, but Fail F1 was low due to poor precision at the untuned threshold/balanced weighting.

### Phase 5 — Imbalance / class-weight / threshold experiments
- **What was tested** (all same 502 features, same 5-fold wafer-grouped CV): (A) `class_weight="balanced"`, t=0.5 [Phase 4 baseline]; (B) `class_weight=None`, t=0.5; (C) `class_weight=None`, CV-tuned threshold; (D) `class_weight="balanced"`, CV-tuned threshold.
- **Why**: isolate whether class-weighting or threshold placement (or both) drives the imbalance-handling story.
- **Result** (from `outputs/phase5_imbalance_experiments.csv`):

  | Experiment | Threshold | CV Fail F1 | CV PR-AUC | Test Fail F1 | Test Precision | Test Recall |
  |---|---:|---:|---:|---:|---:|---:|
  | Baseline balanced, t=0.5 | 0.50 | 0.2417 | 0.5100 | 0.2494 | 0.1509 | 0.7174 |
  | A: Unweighted, t=0.5 | 0.50 | 0.5250 | 0.5108 | 0.5301 | 0.9767 | 0.3638 |
  | **B: Unweighted, tuned t** | **0.42** | **0.5267** | **0.5108** | **0.5321** | **0.9550** | **0.3688** |
  | C: Balanced, tuned t | 0.93 | 0.5261 | 0.5100 | 0.5307 | 0.9410 | 0.3696 |

- **Decision**: **B (unweighted + CV-tuned threshold 0.42) selected** as the reference going forward. `class_weight="balanced"` did not improve ranking quality (PR-AUC/ROC-AUC nearly identical to unweighted); threshold placement had a much larger effect on Fail F1 than class weighting (0.24 -> 0.53). This is where the frozen threshold 0.42 originates.

### Phase 6 — Spatial feature engineering
- **What was tested**: implementation of 8 spatial/neighborhood features (`dist_from_center`, `edge_proximity`, and 3x3/5x5 neighbor old-fail count/density), computed on the full per-wafer population (`old_label` 0 and 1) before the eligibility filter, per an explicit leakage rule (neighbor features must see pre-test-only signal).
- **Why**: the company's generator itself uses pre-test neighborhood fail density when computing new-fail probability, so neighborhood signal was hypothesized to help.
- **Result**: features implemented and sanity-checked; no model trained yet in this phase.
- **Decision**: proceed to Phase 7 for an actual CV comparison.

### Phase 7 — Spatial model comparison
- **What was tested**: three spatial variants (3x3-only=507 features, 5x5-only=507 features, 3x3+5x5=510 features) vs. the 502-feature Phase 5 reference, same unweighted LR / wafer-grouped CV protocol; reference kept its threshold fixed at 0.42, each spatial variant independently threshold-tuned on its own OOF.
- **Why**: test whether neighborhood context improves on die-level features alone.
- **Result** (from `outputs/phase7_spatial_experiments.csv`):

  | Experiment | Features | Threshold | CV Fail F1 | CV Fail Precision | CV Fail Recall | CV PR-AUC | CV ROC-AUC |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Reference (502) | 502 | 0.42 | 0.526736 | 0.951503 | 0.364166 | 0.510808 | 0.840485 |
  | 3x3-only | 507 | 0.46 | 0.526480 | 0.963673 | 0.362172 | 0.512791 | 0.842471 |
  | 5x5-only | 507 | 0.38 | 0.526050 | 0.923018 | 0.367848 | 0.512788 | 0.842445 |
  | 3x3+5x5 | 510 | 0.38 | 0.526477 | 0.922752 | 0.368308 | 0.512776 | 0.842310 |

  All four are within ~0.0007 CV Fail F1 of each other. 3x3-only, evaluated once on test.csv as the best spatial variant: Fail F1=0.5286, Precision=0.9618, Recall=0.3645, ROC-AUC=0.8624, PR-AUC=0.5304 (saved as `models/model_a_spatial_best.joblib`).
- **Decision**: **spatial features NOT adopted.** Spatial variants gave a small PR-AUC/ROC-AUC bump (~0.002) but did not improve Fail F1 over the non-spatial reference, which in fact had the (marginally) highest CV Fail F1 of the four.

### Phase 8 — Model-family comparison (LR vs. LightGBM vs. ExtraTrees)
- **What was tested**: same 502 non-spatial features, same shared `GroupKFold(5)` split (identical fold membership by construction), unweighted training for all three, no hyperparameter search (library defaults + fixed `random_state=42`), per-model threshold independently tuned on pooled OOF.
- **Why**: isolate whether Logistic Regression itself is the limiting factor vs. a model-family choice.
- **Result** (from `outputs/phase8_model_family_experiments.csv` and `outputs/phase8_per_fold_fail_f1.csv`):

  | Model | Threshold | Fail F1 | Fail Precision | Fail Recall | PR-AUC | ROC-AUC | Accuracy | Fold Fail F1 (mean ± std) |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | **Logistic Regression** | 0.42 | **0.5267** | 0.9515 | 0.3642 | 0.5108 | 0.8405 | 0.9723 | 0.5277 ± 0.0112 |
  | LightGBM | 0.14 | 0.4350 | 0.6543 | 0.3258 | 0.4023 | 0.7634 | 0.9642 | 0.4362 ± 0.0189 |
  | ExtraTrees | 0.11 | 0.3326 | 0.4919 | 0.2513 | 0.2845 | 0.7079 | 0.9573 | 0.3339 ± 0.0158 |

  Runtime/model-size comparison (LR fastest, smallest pickled size; LightGBM/ExtraTrees slower and larger) was documented in an earlier revision of NOTES.md but is **not present in the current CSV outputs or the current NOTES.md text — marked "Not verified"** for this handoff; it was not re-run to avoid any new modeling execution.
- **Decision**: **Logistic Regression retained.** It beats both tree models on every fold with no exceptions, and on every pooled metric. Tree-model hyperparameter tuning was explicitly deferred as optional future work, not required for the pipeline.

### Phase 9 — Spatial final comparison and verification
- **What was tested**: completed the Phase 7 comparison (ran the two remaining spatial variants' per-fold breakdown) and verified full-precision PR-AUC values.
- **Why**: confirm the Phase 7 spatial-vs-reference gap is genuine signal or noise.
- **Result** (from `outputs/phase9_spatial_perfold.csv`, per-fold Fail F1 differences vs. reference):

  | Fold | Reference | 3x3-only | 5x5-only | 3x3+5x5 |
  |---:|---:|---:|---:|---:|
  | 0 | 0.5244 | 0.5217 | 0.5212 | 0.5205 |
  | 1 | 0.5301 | 0.5314 | 0.5299 | 0.5302 |
  | 2 | 0.5138 | 0.5148 | 0.5138 | 0.5149 |
  | 3 | 0.5448 | 0.5425 | 0.5421 | 0.5429 |
  | 4 | 0.5256 | 0.5261 | 0.5279 | 0.5279 |

  Per-fold differences are an order of magnitude smaller than fold-to-fold standard deviation for every spatial variant.
- **Decision**: **confirmed not adopted** — the apparent spatial-vs-reference gap is not distinguishable from ordinary fold-to-fold noise. The 502-feature non-spatial LR remains the primary model; the three spatial variants remain documented challengers only.

### Phase 10 — Final independent test evaluation
- **What was tested**: the fully frozen configuration (502 features, unweighted LR, threshold=0.42), fit once on all of `train.csv`'s eligible rows (154,037 rows, no CV at this step), evaluated exactly once on `test.csv`.
- **Why**: obtain a single, honest, out-of-sample performance number after all model/feature/threshold selection was already complete.
- **Result** (from `outputs/phase10_final_test_evaluation.csv`, full detail in Section 6 below): Fail F1 = 0.5321, TP=509, FN=871, FP=24, TN=31,194.
- **Decision**: **this is the final, frozen Model A.** Saved to `models/model_a_final_lr.joblib`. No further model/feature/threshold changes were made after this point.

---

## 5. Validation / CV Methodology

- **GroupKFold setup**: `sklearn.model_selection.GroupKFold(n_splits=5)`, applied to `train.csv` only, with `groups=wafer_id`. Used identically (same pattern re-implemented per script) across Phases 4, 5, 7, 8 and the exploratory threshold-tradeoff script.
- **Wafer-level grouping**: every CV loop explicitly asserts `train_wafers.isdisjoint(val_wafers)` per fold — a wafer's dies never appear on both sides of a fold. This is the primary leakage control against within-wafer information leaking between train and validation folds.
- **OOF predictions**: each fold's held-out predictions are collected into one pooled out-of-fold probability array covering all of `train.csv`'s eligible rows exactly once (`oof_proba` arrays in `src/experiments_imbalance.py`, `src/experiments_spatial.py`, `src/experiments_model_family.py`), each asserting `not np.isnan(oof_proba).any()` before use.
- **Threshold-selection methodology**: `src.evaluate.select_best_threshold` sweeps `np.arange(0.01, 1.00, 0.01)`, maximizing Fail F1, applied to **pooled OOF probabilities from `train.csv` only**. This is how the frozen threshold 0.42 was chosen (Phase 5, experiment B).
- **Why `train.csv` for model selection**: it is the only labeled dataset the project treats as available for iterative comparison; every documented phase's CV/OOF numbers come from it exclusively.
- **Why `test.csv` must not be used for tuning**: explicit, repeated project rule (stated in `src/final_evaluation.py`'s docstring and NOTES.md throughout) — `test.csv` is evaluated exactly once (Phase 10), after the model and threshold are already fixed, specifically so it remains an honest, unbiased estimate of generalization performance. This rule was followed: `test.csv` was loaded in Phase 7 and Phase 10 only for one-shot evaluation of an already-selected configuration, never to pick between alternatives.

**Credibility/validation checks actually performed** (verified in source code, not merely recommended):
- Wafer train/val disjointness assertion in every CV fold, every phase (Phases 4/5/7/8, threshold-tradeoff script).
- Row-order / `wafer_id` alignment assertion between the Phase 7 reference and each spatial variant, to rule out silent misalignment.
- Phase 8's shared `GroupKFold` object reused across all three model families, guaranteeing identical fold membership by construction.
- Phase 10's confusion-matrix sanity check: `TP+FP+FN+TN == n_eligible == metrics['n_eligible']`, asserted before reporting.
- Phase 9's full-precision PR-AUC cross-check: values reproduced bit-identically across `outputs/phase7_spatial_experiments.csv`, `outputs/phase5_imbalance_experiments.csv`, and `outputs/phase8_model_family_experiments.csv`.
- Post-hoc error analysis's integrity check (Section 10): re-running the frozen pipeline's inference on `test.csv` reproduced the Phase 10 confusion matrix (TP=509, FN=871, FP=24, TN=31,194) exactly, with zero discrepancy, before any further analysis proceeded.
- SHAP sanity check (Section 9): SHAP values reconstruct the model's actual `decision_function` output to within `2.93e-14` (floating-point noise), confirming the SHAP explanation is of the *actual* fitted model, not an approximation.

---

## 6. Final Performance

| Metric | Train CV/OOF (Phase 5/8, threshold=0.42) | Test (Phase 10, frozen, evaluated once) |
|---|---:|---:|
| PR-AUC | 0.510808 | 0.527352 |
| ROC-AUC | 0.840485 | 0.859033 |
| Fail Precision | 0.951503 | 0.954972 |
| Fail Recall | 0.364166 | 0.368841 |
| Fail F1 | 0.526736 | 0.532148 |
| Pass Precision | Not verified (not saved in Phase 5/8 CSV output) | 0.972836 |
| Pass Recall | Not verified (not saved in Phase 5/8 CSV output) | 0.999231 |
| Pass F1 | Not verified (not saved in Phase 5/8 CSV output) | 0.985857 |
| Overall Accuracy | 0.972305 | 0.972544 |
| TP | — (not a CV-pooled count in saved output) | 509 |
| FN | — | 871 |
| FP | — | 24 |
| TN | — | 31,194 |
| Threshold | 0.42 | 0.42 |
| n_eligible | 154,037 (train) | 32,598 |

Sources: CV/OOF column from `outputs/phase5_imbalance_experiments.csv` (row "B: unweighted, tuned t") and `outputs/phase8_model_family_experiments.csv` (row "Logistic Regression") — both report the identical configuration/numbers, cross-verified. Test column from `outputs/phase10_final_test_evaluation.csv` in full.

---

## 7. Threshold Analysis

Source: `src/explore_threshold_tradeoff.py` (exploratory, not part of the numbered Phase 4-10 pipeline), `outputs/explore_threshold_tradeoff.csv`, `outputs/explore_precision_recall_curve.png`, and NOTES.md's "Exploratory Recall Improvement — Threshold Trade-off" section. Regenerated `train.csv`-only pooled OOF probabilities via the existing, unmodified `wafer_grouped_oof_proba` function (same 502 features, same GroupKFold(5), same unweighted LR) — **`test.csv` and `validation.csv` were never loaded by this script.**

**Frozen threshold = 0.42**: Fail Precision = 0.9515, Fail Recall = 0.3642, Fail F1 = 0.5267 (train OOF).

**Behavior around thresholds 0.30-0.55** (train.csv pooled OOF):

| Threshold | Fail Precision | Fail Recall | Fail F1 |
|---:|---:|---:|---:|
| 0.30 | 0.8527 | 0.3784 | 0.5242 |
| 0.35 | 0.9080 | 0.3708 | 0.5265 |
| 0.40 | 0.9404 | 0.3652 | 0.5261 |
| **0.42 (frozen)** | **0.9515** | **0.3642** | **0.5267** |
| 0.45 | 0.9621 | 0.3617 | 0.5258 |
| 0.50 | 0.9770 | 0.3590 | 0.5250 |
| 0.55 | 0.9873 | 0.3571 | 0.5245 |

Fail F1 is nearly flat (0.524-0.527) across this entire range — 0.42 sits at essentially the empirical Fail-F1 peak, but the peak itself is shallow.

**Recall ceiling**: maximum Fail Recall achieved anywhere in the full 0.10-0.90 threshold sweep is **~0.5064** (at a very low threshold, with correspondingly poor precision). Per NOTES.md's documented recall-floor analysis: reaching Recall≈0.40 requires threshold≈0.22 (Precision≈0.6871, F1≈0.5068); reaching Recall≈0.45 requires threshold≈0.14 (Precision≈0.4533, F1≈0.4543) — F1 *degrades* well before recall improves much further, because precision collapses faster than recall gains.

**Why threshold tuning alone did not solve the recall problem**: the precision/recall trade-off is steep in the useful region (precision falls off a cliff as threshold drops below ~0.30), and even at the most permissive thresholds tested, recall tops out around 0.51 — i.e. roughly half of true fails are never assigned a probability high enough to catch at *any* reasonable threshold. This is consistent with the post-hoc finding (Section 10) that 51% of false negatives on test.csv have predicted probability under 0.05 — these are not borderline cases a threshold shift would fix.

**Explicit statements**:
- **0.42 remains frozen** — this exploratory analysis did not change it.
- **No alternative threshold was selected.**
- This analysis is **exploratory and train-OOF based only** — it does not use `test.csv` or `validation.csv`, and was not used to re-tune or re-select the frozen model.

---

## 8. EDA Findings

Full detail: NOTES.md's "EXPLORATORY DATA ANALYSIS (EDA)" section; outputs under `outputs/eda_*` and `outputs/eda_plots/`. All computed on `train.csv` only (schema audit also covers `validation.csv`; `test.csv` was never loaded by the EDA module).

- **Class imbalance**: 95.77% Pass / 4.23% Fail on the eligible population (154,037 dies). Predict-all-Pass baseline accuracy = 0.9577; no-skill PR-AUC baseline = 0.0423. Per-wafer failure rate is highly uneven (mean 0.0414, std 0.0529, range 0.0-0.50), with several wafers showing 0 new fails despite hundreds of eligible dies.
- **Feature distributions**: 0 missing values across all 500 features; 0 truly near-constant features (all fully continuous, coefficient-of-variation range 0.0201-0.1502 — features differ in raw scale, not in relative variability); 0 features exceed a 1% IQR-outlier rate.
- **Single-feature separation**: ranked by Cohen's d (descriptive only, not used for feature selection) — the single best-separating feature (`feature_171`) has |d|≈0.24, a "small" effect by the usual rule of thumb. No individual feature separates Pass from Fail cleanly; even the best-separating features show substantial histogram overlap.
- **Feature correlations**: features are essentially pairwise **uncorrelated** — max |Pearson r| across all 500 `feature_*` columns is **0.0131**. This is unusual for real fab parametric data and means SHAP/coefficient importance is not expected to be split across correlated siblings (see Section 9).
- **Spatial patterns**: weak positive association between edge position and failure rate (point-biserial r=+0.0355 to +0.0379 for edge_proximity/dist_from_center vs. label) — failure rate rises from ~2.8% (center decile) to ~5.7% (edge decile), a real but modest trend. Dies with nonzero 3x3-neighborhood old-fail density show a higher new-fail rate (5.00% vs. 4.06%).
- **Wafer-level patterns**: failure rate varies widely by wafer (see class-imbalance bullet above); wafer bounding boxes vary substantially, so raw `die_row`/`die_col` require per-wafer normalization (`dist_from_center`/`edge_proximity`) for any cross-wafer spatial comparison.
- **t-SNE**: a documented, class-proportional stratified sample (6,000 of 154,037 rows, visualization only) shows most Fail points interspersed throughout the Pass point cloud with no separation, but one small, visually distinct cluster of Fail points appears isolated — suggestive of a failure subgroup, but not a quantitative or causal claim.
- **Implications for modeling** (descriptive, not causal): no single feature or small feature subset dominates; whatever predictive signal Model A achieves (PR-AUC well above no-skill) comes from combining many weakly-informative, largely-independent features rather than a few strong ones — consistent with why the frozen model uses all 502 features rather than a reduced subset.

---

## 9. Interpretability

Full detail: NOTES.md's "MODEL A INTERPRETABILITY (SHAP + COEFFICIENTS)" section; script `src/interpretability.py`. Explains the **already-frozen** model — `.fit()` is never called; only `.transform()`, `.decision_function()`, `.predict_proba()`, and direct `.coef_`/`.intercept_` inspection.

### Methodology
1. **Standardized coefficient analysis**: since the frozen pipeline is `StandardScaler -> LogisticRegression`, `clf.coef_` is already expressed per standardized unit (mean 0, std 1 per feature) — `|coefficient|` is directly comparable across all 502 features without further normalization.
2. **SHAP**: `shap.LinearExplainer` applied to the fitted `LogisticRegression` step on `StandardScaler`-transformed features. This is the SHAP method built specifically for linear models — exact given a background distribution (not a sampling approximation). Verified correctness: SHAP values reconstruct the model's actual `decision_function` to within `2.93e-14`.
   - **Masker**: `shap.maskers.Independent`, background = 200 rows from `train.csv` eligible dies (`random_state=42`) — independence is justified by the EDA finding of max |r|=0.0131 across features.
   - **Global importance**: computed on the **full 154,037-row eligible train population** (fast/exact for a linear model + independent masker, ~0.9s).
   - Beeswarm plot uses a documented 5,000-row class-proportional stratified sample (readability only).
   - 5 local explanations selected by a **deterministic rule** (not cherry-picked): most-confident correct fail, most-confident correct pass, borderline-near-threshold, worst false negative, worst false positive.

### Top positive/fail-associated features (highest predicted fail-probability)
`feature_460` (+0.1117), `feature_85` (+0.1097), `feature_8` (+0.1095), `feature_171` (+0.1089), `feature_336` (+0.1015) — coefficients on standardized units.

### Top negative/pass-associated features (lowest predicted fail-probability)
`feature_231` (-0.1017), `feature_109` (-0.1000), `feature_204` (-0.0987), `feature_282` (-0.0967), `feature_33` (-0.0947).

### `die_row` / `die_col` importance
Reported as model sensitivity/association only, **not causal**:

| Feature | Coefficient | Coef rank /502 | Mean \|SHAP\| | SHAP rank /502 |
|---|---:|---:|---:|---:|
| `die_row` | +0.0752 | 75 | 0.0589 | 82 |
| `die_col` | +0.0973 | 9 | 0.0744 | 17 |

Both positive; `die_col` is meaningfully more influential than `die_row` — directionally consistent with the EDA's weak positive edge/center association.

### Coefficient-vs-SHAP agreement
Spearman rank correlation across all 502 features: **rho = 0.9999**. Top-20 overlap: **20 of 20**. This near-perfect agreement is expected given the near-zero feature collinearity found in EDA (max |r|=0.0131) — with essentially no collinearity, there is little room for coefficient-based and SHAP-based rankings to diverge.

### Limitations
- Analysis is `train.csv`-only (per task scope).
- The `Independent` masker's feature-independence assumption is well-supported by the correlation analysis but not a guarantee for every possible feature pair.
- Local explanations are illustrative examples, not a systematic error analysis (see Section 10 for that).
- Coordinate-feature interpretation is associational, not a physical/causal explanation.

### Output files (for teammates to inspect directly)
`outputs/model_a_lr_coefficient_importance.csv`, `outputs/model_a_lr_global_importance.png`, `outputs/model_a_shap_global_importance.csv`, `outputs/model_a_shap_bar.png`, `outputs/model_a_shap_summary.png` (beeswarm), `outputs/model_a_shap_local_examples.csv`, `outputs/model_a_shap_local_most_confident_correct_fail.png`, `outputs/model_a_shap_local_most_confident_correct_pass.png`, `outputs/model_a_shap_local_borderline_near_threshold.png`, `outputs/model_a_shap_local_worst_false_negative.png`, `outputs/model_a_shap_local_worst_false_positive.png`, `outputs/model_a_coef_vs_shap_comparison.csv`.

---

## 10. Error Analysis — **POST-HOC TEST ANALYSIS**

**This entire section is post-hoc descriptive analysis of `test.csv` errors, performed strictly after the model and threshold were already frozen (Phase 10). None of these findings were used to select, tune, or validate Model A — they describe how the already-final model behaves, nothing more.**

Full detail: NOTES.md's "MODEL A — POST-HOC ERROR ANALYSIS" section; script `src/error_analysis.py`. Because Phase 10 saved only aggregate metrics (no per-die predictions existed anywhere in the repo), this analysis performed **one** frozen-pipeline inference pass over `test.csv` (`predict_proba` only, no `.fit()`) to generate the missing per-die artifact, then verified the resulting confusion matrix exactly reproduced the already-documented Phase 10 result (zero discrepancy) before any analysis proceeded.

### Confusion matrix (from existing Phase 10 CSV)
TP=509, FN=871, FP=24, TN=31,194. Fail Precision=0.9550, Fail Recall=0.3688, Fail F1=0.5321. False-negative rate FN/(FN+TP)=0.6312. False-positive rate FP/(FP+TN)=0.00077.

### Probability distributions / near-threshold vs. high-confidence errors
- TP mean proba 0.9535; FN mean proba 0.0739 (median 0.0489); FP mean proba 0.5384; TN mean proba 0.0266.
- **51.0% of all 871 FN (444 dies) have predicted probability under 0.05** — the model is confidently wrong, not borderline. Only 1.8% of FN (16 dies) are true near-misses (0.30-0.42).
- FP skew the opposite way: 62.5% are near-miss (0.42-0.55), and **zero FP reach the high-confidence band (≥0.85)**.
- **Key finding**: FN and FP are asymmetric in kind, not just count — this is why threshold tuning alone (Section 7) could not push recall much higher: most misses are far from any reasonable threshold.

### False-negative findings (871 dies)
- Spread across 39 of 40 test wafers — broadly distributed, not concentrated.
- Among wafers with ≥5 true-fail dies (36 wafers, avoiding small-sample noise), FN rate ranges up to 100% (`W_N_0025`, `W_N_0038`, both small 5-die true-fail populations) down through 77-86% for larger wafers (`W_N_0128`, `W_N_0142`, `W_N_0068`).
- Edge-proximity correlation with "is FN" (among true-Fail dies) is **+0.0194** — negligible; misses are not meaningfully more likely at wafer edge or center.
- TP-vs-FN feature comparison: top distinguishing features reach |Cohen's d|≈0.49-0.54 (`feature_457`, `feature_60`, `feature_269`, `feature_189`, `feature_123`, `feature_223`) — notably larger effect sizes than any Pass-vs-Fail separation found in EDA (max ~0.24), i.e. "easy" and "hard" fails are more separable from each other than Pass/Fail overall.

### False-positive findings (24 dies) — **small-sample caution**
**With only 24 observations, all FP findings are illustrative, not statistically reliable.**
- Spread across 19 of 40 wafers, no wafer has more than 4; per-wafer FP rate is tiny everywhere (0.05%-0.46%).
- All 24 FP have probability under 0.77; none reach the high-confidence band.
- TN-vs-FP top |Cohen's d| features reach 0.59-0.84 (`feature_116`, `feature_361`, `feature_154`, `feature_433`, `feature_238`, `feature_441`) — larger than the FN comparison, but expected sampling variance with n=24, not necessarily a more real effect.

### Wafer-level findings
FN present on 39/40 wafers (broadly distributed); FP present on 19/40 wafers, too rare (24 total) to identify a wafer-level hotspot.

### Spatial findings
No meaningful edge/center pattern for FN (correlation +0.0194) or FP (visually scattered, too few points for a reliable decile breakdown). Sample wafer maps show FN interspersed among correct predictions with no obvious spatial clustering.

### Limitations
- FP sample (n=24) is explicitly high-variance/illustrative only.
- Small-sample wafers (<5 true-fail or true-pass dies) excluded from "reliable" wafer-rate rankings.
- Cohen's d comparisons are descriptive effect sizes, not significance-tested; no multiple-comparison correction across 500 features.
- Analysis covers `test.csv` only, not repeated on `validation.csv` (unlabeled) or fed back into `train.csv` work.

### Output files
`outputs/model_a_test_predictions.csv` (per-die artifact), `outputs/model_a_error_summary.csv`, `outputs/model_a_error_probability.png`, `outputs/model_a_error_probability_bins.csv`, `outputs/model_a_error_by_wafer.csv`, `outputs/model_a_error_spatial.png`, `outputs/model_a_error_spatial_summary.csv`, `outputs/model_a_error_wafer_maps.png`, `outputs/model_a_fn_feature_summary.csv`, `outputs/model_a_fn_vs_tp_top_features.png`, `outputs/model_a_fp_feature_summary.csv`, `outputs/model_a_fp_vs_tn_top_features.png`.

---

## 11. Validation Submission

Script: `src/generate_validation_submission.py` (run as `python -m src.generate_validation_submission`). Applies the already-frozen `models/model_a_final_lr.joblib` to `validation.csv`, unchanged — does not train, retune, or modify anything.

| | Value |
|---|---|
| Output path | `outputs/validation_submission_model_a.csv` |
| Exact columns | `wafer_id`, `die_row`, `die_col`, `predicted_label` |
| Row count | 39,351 (verified equal to `validation.csv`'s row count) |
| `old_label==0` (eligible, scored by model) | 32,598 rows |
| `old_label==1` (excluded, auto-filled) | 6,753 rows |
| Predicted `0` count | 32,065 |
| Predicted `1` count | 7,286 |
| Threshold used | 0.42 (frozen) |

Cross-check: 7,286 predicted-1 = 6,753 auto-filled `old_label==1` rows + 533 model-predicted-fail rows among the 32,598 eligible rows (7,286 - 6,753 = 533; 32,065 + 533 = 32,598 ✓, consistent with the eligible row count).

**`old_label==1` handling**: assigned `predicted_label=1` directly, without being passed through the model (matches `company_provided/README.md`'s rule that pre-failed dies trivially remain failed).
**`old_label==0` handling**: scored by the frozen pipeline at the frozen threshold 0.42.

**Sanity checks performed** (all 12 asserted in-script before the file is written; script raises and aborts the write if any fail — the fact that the output file exists confirms all passed): row count matches `validation.csv`; no missing predictions; no duplicate id rows; id alignment matches `validation.csv`'s original row order; `predicted_label` is only 0/1; all `old_label==1` rows predicted 1; all `old_label==0` rows scored by the model and are 0/1; frozen threshold 0.42 used; no `label` column present in `validation.csv`; `test.csv` not loaded (by construction — `load_test` is never imported); `company_provided/` not modified (read-only); model file not modified (`joblib.load` only, no `joblib.dump` to the frozen path).

**Confirmation validation labels were not used**: `validation.csv` has no `label` column at all (verified: `val_ds.y is None` is asserted in-script, consistent with the EDA schema audit's `has_target_label=False` for `validation.csv`) — there is no label to have used.

---

## 12. Artifact Inventory

| File | Contents |
|---|---|
| `models/model_a_final_lr.joblib` | **The frozen final Model A** — `StandardScaler -> LogisticRegression`, fit once on all eligible `train.csv` rows (Phase 10). This is the pipeline all sections above describe. |
| `models/model_a_baseline_logreg.joblib` | Phase 4 baseline (`class_weight="balanced"`, threshold=0.5) — historical, superseded by the frozen model. |
| `models/model_a_spatial_best.joblib` | Phase 7's selected spatial variant (3x3-only, 507 features) — documented challenger, **not adopted**, not the frozen model. |
| `outputs/phase5_imbalance_experiments.csv` | Phase 5 class-weight/threshold experiment comparison table (source of the frozen threshold 0.42). |
| `outputs/phase7_spatial_experiments.csv` | Phase 7 spatial-variant CV comparison table. |
| `outputs/phase8_model_family_experiments.csv` | Phase 8 pooled-OOF LR vs. LightGBM vs. ExtraTrees comparison. |
| `outputs/phase8_per_fold_fail_f1.csv` | Phase 8 per-fold Fail F1 for all three model families. |
| `outputs/phase9_spatial_perfold.csv` | Phase 9 per-fold Fail F1, all four spatial-vs-reference configurations. |
| `outputs/phase10_final_test_evaluation.csv` | **The frozen, one-shot final test.csv evaluation** — source of Section 6's test-side numbers. |
| `outputs/validation_submission_model_a.csv` | Validation-set predictions (Section 11). |
| `outputs/explore_threshold_tradeoff.csv`, `outputs/explore_precision_recall_curve.png` | Exploratory threshold trade-off (Section 7). |
| `outputs/eda_*.csv`, `outputs/eda_plots/*.png` | EDA outputs (Section 8). |
| `outputs/model_a_lr_coefficient_importance.csv`, `model_a_lr_global_importance.png`, `model_a_shap_*` , `model_a_coef_vs_shap_comparison.csv` | Interpretability outputs (Section 9). |
| `outputs/model_a_error_*`, `model_a_fn_*`, `model_a_fp_*`, `model_a_test_predictions.csv` | Post-hoc error analysis outputs (Section 10). |
| `src/data_loading.py` | CSV loading (`block_readings` excluded at read time). |
| `src/target.py` | Eligibility filter + `X`/`y`/`ids` assembly (`build_model_a_dataset`, `feature_columns`). |
| `src/features.py` | Spatial feature engineering (`add_spatial_features`) — not used by the frozen model, only by the rejected spatial variants. |
| `src/evaluate.py` | Metric computation (`evaluate_predictions`), threshold selection (`select_best_threshold`), report formatting. |
| `src/train.py` | Phase 4 baseline training script. |
| `src/experiments_imbalance.py` | Phase 5 imbalance/threshold experiments (also reused by `explore_threshold_tradeoff.py` for OOF regeneration). |
| `src/experiments_spatial.py` | Phase 7 spatial-variant experiments. |
| `src/experiments_model_family.py` | Phase 8 model-family comparison. |
| `src/final_evaluation.py` | **Phase 10 — produces the frozen model and the one-shot test evaluation.** |
| `src/generate_validation_submission.py` | Validation submission generator (Section 11). |
| `src/explore_threshold_tradeoff.py` | Exploratory threshold analysis (Section 7). |
| `src/eda.py` | EDA script (Section 8). |
| `src/interpretability.py` | Interpretability/SHAP script (Section 9). |
| `src/error_analysis.py` | Post-hoc error analysis script (Section 10). |
| `config.yaml` | Model A's feature/target/eligibility contract. |
| `NOTES.md` | Full project log (source of most facts in this document). |

---

## 13. Reproduction Guide

**Safest order to understand Model A without risking any change to it:**

1. **Start with `config.yaml`** — the feature/target/eligibility contract in one place.
2. **Read `src/data_loading.py` and `src/target.py`** — how the 502-feature `X`/`y`/`ids` are assembled from the raw CSVs, and the eligibility filter (`old_label==0`).
3. **Read `src/evaluate.py`** — the metric definitions (`evaluate_predictions`) used everywhere in this project.
4. **Load the frozen model read-only**: `joblib.load("models/model_a_final_lr.joblib")` — inspect `.steps`, `.named_steps["clf"].coef_`, `.feature_names_in_` directly. Do **not** call `.fit()` on it.
5. **Read `outputs/phase10_final_test_evaluation.csv`** — the frozen final test result. This is already computed; there is no need to re-run `test.csv` through the model to see it.
6. **Read `outputs/validation_submission_model_a.csv`** and `src/generate_validation_submission.py` if you need to see how validation-set predictions were produced.
7. **Read the EDA outputs** (`outputs/eda_*`, `outputs/eda_plots/`) for dataset understanding, then the interpretability outputs (`outputs/model_a_shap_*`, `model_a_lr_*`) for feature attribution, then the error-analysis outputs (`outputs/model_a_error_*`, `model_a_fn_*`, `model_a_fp_*`) for known failure patterns — all three are pre-computed and saved; none require re-running the model.
8. **If you need to inspect training/CV mechanics**, read `src/train.py` (Phase 4), `src/experiments_imbalance.py` (Phase 5), `src/experiments_spatial.py` (Phase 7), `src/experiments_model_family.py` (Phase 8) — but running these scripts **retrains models** (on `train.csv` CV folds); do not run them if the goal is only to understand the frozen model, since none of them touch the frozen `model_a_final_lr.joblib` file but they do consume time/compute for no benefit if you just need to read the model.

**Do not** re-run `src/final_evaluation.py` unless you specifically intend to reproduce the one-shot Phase 10 test evaluation from scratch — its output already exists at `outputs/phase10_final_test_evaluation.csv` and its saved model already exists at `models/model_a_final_lr.joblib`; re-running it would re-fit and re-save the pipeline (deterministically, with `random_state=42`, so it should reproduce the same artifact — but there is no need to do this just to review Model A).

---

## 14. Known Limitations / Open Issues

- **Recall limitation**: frozen model catches only ~37% of true fails on test.csv (Fail Recall=0.369); exploratory threshold analysis (Section 7) found the achievable ceiling across the whole threshold range is only ~0.51, at a steep precision cost. This appears to be a genuine modeling-capacity limit rather than a threshold-placement problem (Section 10: 51% of FN are missed with very high confidence).
- **Extreme class imbalance**: 4.23% fail prevalence on the eligible population — inherent to the problem, not something Model A can change.
- **Post-hoc nature of the test error analysis**: Section 10's findings describe the frozen model's test-set behavior but were never used to select or improve it — they are informative for understanding Model A, not evidence that it was validated against them.
- **Small FP sample**: only 24 false positives on test.csv — any FP-specific pattern (wafer distribution, feature differences) should be treated as illustrative, not statistically established.
- **Spatial effects are marginal/inconclusive**: Phase 7/9 found spatial features gave a small PR-AUC bump but no Fail-F1 improvement, within fold-to-fold noise — this doesn't rule out spatial signal existing, only that the tested 3x3/5x5 neighborhood features didn't capture a usable version of it at the frozen operating point.
- **`validation.csv` grading is unresolved**: per NOTES.md's "Important Open Decisions," whether the external grader uses this exact generated validation set or a separate hidden set was never confirmed with the hackathon organizers — **Not verified** whether this has since been resolved.
- **Model B ownership**: per NOTES.md, team assignment for implementing Model B was an open decision as of the last NOTES.md update — **Not verified** current status.
- **NOTES.md documentation gap**: the detailed Phase 8, 9, and 10 write-ups (and part of Phase 7's) have been condensed/removed from the current `NOTES.md` by an edit outside the scope of the work that produced this handoff document. The underlying CSV artifacts survive and are the source for Section 4's numbers, but some narrative detail (e.g. Phase 8's runtime/model-size comparison) is no longer recoverable from any current repository artifact without re-running code — marked "Not verified" where this applies, and **not re-run** to produce this document.
- **Interpretability/error-analysis independence assumption**: SHAP's `Independent` masker and the general "features are near-uncorrelated" framing throughout this document rest on the EDA's correlation measurement (max |r|=0.0131 on train.csv) — well-supported, but not re-verified on `test.csv`.

---

## 15. Model A vs. Future Model B Comparison Checklist

**Model A is the frozen baseline/champion.** Any future Model B (or other challenger) must be compared against it on a **fair, like-for-like basis** — same held-out data, same eligibility filter, same metric definitions — before any claim of improvement is made. Checklist:

- [ ] **PR-AUC** — compare against Model A's frozen test PR-AUC = 0.527352 (and CV/OOF = 0.510808).
- [ ] **Fail F1** — compare against 0.532148 (test) / 0.526736 (CV/OOF).
- [ ] **Fail Precision** — compare against 0.954972 (test) / 0.951503 (CV/OOF).
- [ ] **Fail Recall** — compare against 0.368841 (test) / 0.364166 (CV/OOF) — this is Model A's known weak point; a Model B that meaningfully improves recall without collapsing precision would be the clearest win.
- [ ] **Threshold-selection methodology** — must be selected on train-only OOF predictions (never on test/validation), exactly as Model A's 0.42 was (Phase 5). Report the methodology, not just the resulting number.
- [ ] **CV protocol** — must use the same wafer-grouped `GroupKFold(5)` (or an equally leakage-safe scheme) so results are comparable; a different CV scheme invalidates a direct number-to-number comparison.
- [ ] **Leakage controls** — must demonstrate equivalent controls (wafer disjointness assertions, `test.csv` touched only once for a final evaluation, no test/validation label use in feature engineering or threshold selection).
- [ ] **Feature set** — document exactly which features Model B adds/changes relative to Model A's 502 (e.g. `block_readings` for Model B, per the hackathon problem statement); Model A's spatial-feature experiments (Section 4, Phases 6/7/9) are the existing precedent for how a feature-addition experiment should be run and reported here.
- [ ] **Model complexity** — Model A is a single linear model (~502 coefficients, no hyperparameter search); document Model B's parameter count/training cost for a fair complexity discussion (Model A's interpretability strength, Section 9, is partly a direct consequence of its simplicity).
- [ ] **Interpretability** — Model A has full standardized-coefficient and SHAP-based interpretability with near-perfect method agreement (Section 9); Model B's interpretability approach (especially for `block_readings`) must be documented to the same standard for a fair "interpretability" comparison per the hackathon's evaluation criteria.
- [ ] **Spatial/block information** — Model A explicitly does not use spatial neighborhood features (rejected, Phases 6/7/9) or block-level readings (out of scope). Any Model A→B delta analysis should isolate which of these Model B actually adds, and by how much, rather than attributing an improvement to "more data" generically.
- [ ] **Validation submission behavior** — Model B's validation-set submission should follow the same eligibility handling (old_label==1 auto-filled, old_label==0 model-scored) and the same sanity-check discipline as `src/generate_validation_submission.py` (Section 11).

---

## 16. One-Page Teammate Summary

```
MODEL:            StandardScaler -> LogisticRegression (class_weight=None, max_iter=2000, random_state=42)
FEATURES:         502 — die_row, die_col, feature_1..feature_500 (no spatial, no block_readings)
PREPROCESSING:    StandardScaler, fit on training data only
CV:               GroupKFold(5), grouped by wafer_id, wafer-disjoint train/val per fold
THRESHOLD:        0.42 (frozen; selected via train.csv OOF Fail-F1 maximization, Phase 5)
PR-AUC:           0.527352 (test, Phase 10)   |   0.510808 (train CV/OOF)
FAIL PRECISION:   0.954972 (test)             |   0.951503 (train CV/OOF)
FAIL RECALL:      0.368841 (test)             |   0.364166 (train CV/OOF)
FAIL F1:          0.532148 (test)             |   0.526736 (train CV/OOF)
TP/FN/FP/TN:      509 / 871 / 24 / 31,194  (test.csv, eligible dies, threshold=0.42)
VALIDATION ROWS:  39,351 total (32,598 model-scored + 6,753 auto-filled old-fails)
MAIN STRENGTH:    High precision (~95%) — when it predicts Fail, it's very likely right
MAIN WEAKNESS:    Low recall (~37%) — misses most of the true fails, and most misses are
                   high-confidence (proba<0.05), not threshold-borderline
IMPORTANT ARTIFACT: models/model_a_final_lr.joblib
CURRENT STATUS:   FROZEN — do not retrain, refit, retune, or re-evaluate without an explicit decision
```

**The single most important thing a teammate should know before comparing another model against Model A**: Model A's ~37% recall is not primarily a threshold-tuning artifact — a dedicated exploratory sweep across the entire threshold range (Section 7) found a hard ceiling around ~51% recall, and post-hoc analysis (Section 10) showed the majority of missed fails are missed *confidently* (predicted probability under 0.05), not borderline. Any Model B (or other challenger) that claims to "beat Model A on recall" should be checked against this same pattern — does it actually assign meaningfully higher probability to the specific dies Model A confidently misses, or does it just shift the operating point along a similarly-shaped precision/recall curve? The feature-space comparisons in Section 10 (`outputs/model_a_fn_feature_summary.csv`) are a good starting point for understanding which features characterize Model A's blind spot.
