# MATCARE-AI — Evaluation Report

**Protocol:** `deduplicated_conflict_free_70_15_15` (maternal) and `deduplicated_index_features_70_15_15` (anaemia)
**Random seed:** 42 (fixed for every split, search and bootstrap)
**Held-out test set:** created once, stratified, never used for tuning, threshold selection, weight search or feature choice.

This report supersedes the earlier `model_comparison_report.md`, which described the
original (pre-audit) protocol. Numbers below are taken directly from the saved
artefacts `models/general_risk/evaluation_metrics.json` and `models/anemia/evaluation_metrics.json`.

---

## 1. Module 1 — General Maternal Health Risk

Dataset: UCI Maternal Health Risk. After removing exact duplicates and rows that carry
conflicting labels for the same feature vector, 381 rows remain for modelling.

Partitions: train **265**, validation **58**, held-out test **58** (stratified 70/15/15).

| Approach | Acc % | Macro F1 | Weighted F1 | ROC-AUC | PR-AUC | ECE | High-risk recall % | High-risk FPR % |
| :--- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Random Forest | 72.41 | 68.70 | 72.42 | 0.8475 | 0.7662 | 0.076 | 80.00 | 9.30 |
| XGBoost | 65.52 | 63.90 | 67.05 | 0.8493 | 0.7335 | 0.125 | 66.67 | 9.30 |
| Equal-weight voting (50/50) | 68.97 | 66.23 | 69.63 | 0.8509 | 0.7623 | 0.119 | 73.33 | 11.63 |
| **Adaptive hybrid (RF 96% / XGB 4%)** | **72.41** | **68.70** | **72.42** | 0.8504 | 0.7692 | 0.071 | **80.00** | **9.30** |

Per-class detail (adaptive hybrid):

| Class | Precision | Recall | F1 | FNR | FPR | Support |
| :--- | :--: | :--: | :--: | :--: | :--: | :--: |
| Low Risk | 80.00 | 77.42 | 78.69 | 22.58 | 22.22 | 31 |
| Mid Risk | 50.00 | 50.00 | 50.00 | 50.00 | 13.04 | 12 |
| High Risk | 75.00 | 80.00 | 77.42 | 20.00 | 9.30 | 15 |

Confusion matrix (rows = truth, cols = prediction):

```
[[24  5  2]
 [ 4  6  2]
 [ 2  1 12]]
```

**Reading:** the model separates Low and High risk reasonably well; the Mid-risk class is
the weak point (F1 50%). The Mid-risk band is the most ambiguous label in this dataset and
should not be relied on in isolation.

### Hybrid evidence (Module 1)

- Validation-chosen weights: `w_rf = 0.96`, `w_xgb = 0.04` (exact score plateau 0.96–1.00).
- Weight stability over 40 bootstrapped validation sets: mean `w_rf = 0.963`, sd 0.015,
  95% interval [0.95, 1.00] — the mixture collapses towards Random Forest.
- Macro-F1 gain of the hybrid over the **stronger base model: 0.00 points**.
- Macro-F1 gain over **equal-weight voting: +2.47 points**.
- High-risk recall gain over the stronger base model: 0.00 points.

**Conclusion:** the adaptive hybrid does **not** beat the stronger base model on the
held-out test set. It matches Random Forest, and it does beat naive 50/50 voting on
macro-F1. The hybrid is therefore reported as *equivalent to the better base model*, not superior.

---

## 2. Module 2 — General Haematological Anaemia Classification

Dataset: `anemia_cbc.csv`. 1 421 rows, 887 exact duplicates → 534 unique rows
(no conflicting duplicate groups). The binary target is an **exact** sex-specific
haemoglobin threshold (see the leakage audit). The production model deliberately uses
**only the red-cell indices MCV, MCH, MCHC**.

Partitions: train **373**, validation **80**, held-out test **81**.

| Approach | Acc % | Macro F1 | Weighted F1 | ROC-AUC | PR-AUC | ECE | Anemic recall % | Anemic FPR % |
| :--- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Random Forest | 48.15 | 48.14 | 48.20 | 0.4760 | 0.4630 | 0.114 | 51.35 | 54.55 |
| XGBoost | 41.98 | 41.94 | 42.06 | 0.4840 | 0.4651 | 0.170 | 43.24 | 59.09 |
| Equal-weight voting (50/50) | 46.91 | 46.88 | 46.77 | 0.4773 | 0.4662 | 0.122 | 54.05 | 59.09 |
| **Adaptive hybrid (RF 61% / XGB 39%)** | 46.91 | 46.91 | 46.91 | 0.4797 | 0.4684 | 0.123 | 51.35 | 56.82 |

**Reading:** with haemoglobin removed the task is close to chance (ROC-AUC ≈ 0.48,
accuracy ≈ 42–48%). This is the *expected* result once the label-defining variable is
excluded: MCV/MCH/MCHC alone do not carry enough information to reproduce a
haemoglobin-threshold diagnosis.

### Ablation — why the model is index-only

Identical stratified split, fixed hyper-parameters, deduplicated data:

| Experiment | Features | RF Acc % | XGB Acc % | RF macro-F1 | XGB macro-F1 |
| :--- | :--- | :--: | :--: | :--: | :--: |
| Full features | Gender, Hb, MCH, MCHC, MCV | 100.00 | 100.00 | 100.00 | 100.00 |
| Haemoglobin excluded | Gender, MCH, MCHC, MCV | 66.67 | 59.26 | 66.65 | 59.03 |
| Sex code excluded | Hb, MCH, MCHC, MCV | 88.89 | 87.65 | 88.89 | 87.56 |
| **Hb + sex excluded (production)** | **MCV, MCH, MCHC** | **46.91** | **41.98** | **46.39** | **40.19** |

The 100% on all features is replication of the deterministic label rule, not learning.

### Hybrid evidence (Module 2)

- Validation-chosen weights: `w_rf = 0.61`, `w_xgb = 0.39` (plateau 0.61–0.99).
- Weight stability: mean `w_rf = 0.560`, sd 0.295, 95% interval [0.00, 0.95];
  22.5% of bootstrap runs preferred a near-equal split and 10% collapsed to XGBoost.
  The chosen weight is **not stable** — consistent with both base models being near chance.
- Macro-F1 gain of the hybrid over the stronger base model: **−1.23 points**.

**Conclusion:** the adaptive hybrid does **not** improve on the stronger base model, and
neither base model performs above chance on this index-only task.

---

## 3. Repeated cross-validation (development split only)

Repeated stratified CV on the training partition (15 folds), reported as mean ± sd of macro-F1:

| Module | Random Forest | XGBoost |
| :--- | :--: | :--: |
| Maternal risk | 71.28 ± 5.84 | 67.69 ± 4.07 |
| Anaemia (index-only) | 44.15 ± 3.83 | 46.48 ± 2.29 |

Focused-class recall means: maternal RF 87.60 ± 8.71 / XGB 82.16 ± 6.49;
anaemia RF 39.21 ± 7.19 / XGB 41.11 ± 4.54.

## 4. Bootstrap confidence intervals (held-out test, 1000 resamples)

| Module / model | Metric | Mean | 95% interval |
| :--- | :--- | :--: | :--: |
| Maternal RF | Accuracy | 72.56 | 60.34 – 84.48 |
| Maternal RF | Macro F1 | 68.27 | 55.68 – 80.66 |
| Maternal RF | High-risk recall | 80.27 | 57.14 – 100.0 |
| Maternal XGB | Macro F1 | 63.51 | 50.52 – 75.57 |
| Anaemia RF | Accuracy | 48.00 | 38.24 – 59.26 |
| Anaemia RF | Macro F1 | 47.68 | 37.04 – 59.16 |
| Anaemia hybrid | Accuracy | 46.83 | 37.01 – 58.02 |

The very wide maternal intervals reflect the small held-out test set (n = 58); the point
estimates should be read as indicative, not precise.

## 5. Global explanation (SHAP)

Weight-frequency of absolute attributions on the test partition:

**Maternal risk (hybrid-weighted):** Blood sugar 34.1%, Body temperature 21.7%,
Systolic BP 20.9%, Age 9.4%, Diastolic BP 7.5%, Heart rate 6.4%.

**Anaemia (index-only):** MCHC 37.2%, MCH 32.8%, MCV 30.0% (importances are nearly
uniform, mirroring the near-chance performance).

SHAP values describe how inputs moved a model score. They are **not** evidence of causation.

## 6. Calibration

Expected calibration error on the held-out test set:

| Module | RF | XGB | Equal-weight | Adaptive hybrid |
| :--- | :--: | :--: | :--: | :--: |
| Maternal risk | 0.076 | 0.125 | 0.119 | 0.071 |
| Anaemia | 0.114 | 0.170 | 0.122 | 0.123 |

Maternal hybrid probabilities are reasonably calibrated (ECE 0.071). Anaemia probabilities
are not informative because the underlying model is near chance.

---

## 7. Honest summary

1. The maternal model is a usable **screening** prototype for Low/High risk (high-risk
   recall ≈ 80%, FPR ≈ 9%) but the Mid-risk class is weak and the test set is small.
2. The anaemia module, once the label-defining haemoglobin column is removed, performs at
   chance. This is reported openly rather than hidden behind the 100% figure produced when
   haemoglobin is included.
3. The adaptive hybrid is **not** shown to be superior to the stronger base model in either
   module. It is retained because it matches the better base model while keeping a single
   interface, and because it beats equal-weight voting on maternal macro-F1.
4. No result here constitutes clinical validation.
