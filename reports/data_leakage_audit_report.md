# MATCARE-AI — Data Leakage Audit Report

**Audit scope:** both datasets, both training pipelines, and the shared inference path.
**Method:** inspect the raw files, test label/feature relationships row-by-row, and run
controlled ablations on identical stratified splits. All figures are reproduced from the
saved artefacts and from `scripts/phase2_leakage_audit.py`.

---

## 1. Executive summary

| Finding | Dataset | Severity | Action taken |
| :--- | :--- | :--- | :--- |
| Target label defined exactly by a feature cutoff | `anemia_cbc.csv` | **Critical** | Haemoglobin and sex excluded from the production model |
| 62% exact duplicate rows | `anemia_cbc.csv` | High | Deduplicated before splitting (1 421 → 534) |
| 55% exact duplicate rows | `maternal_health_risk.csv` | Medium | Deduplicated before splitting (1 014 → 452) |
| Same feature vector with conflicting labels | `maternal_health_risk.csv` | Medium | 71 conflicting rows removed (381 rows modelled) |
| Physiologically impossible values | `maternal_health_risk.csv` | Low | `HeartRate < 30` set to NaN and imputed from train medians |

---

## 2. Anaemia module — target-label circularity (critical)

### 2.1 The rule

Cross-tabulating the raw file shows the binary target is a deterministic function of two
columns:

```
Result = 1  if (Gender == 0 and Hemoglobin < 12.0)
           or (Gender == 1 and Hemoglobin < 13.5)
        = 0  otherwise
```

- Rows matched by this rule: **1 421 / 1 421 = 100.0 %** (`rule_is_exact = true`).
- Haemoglobin↔Result correlation: **−0.796**.
- The file contains **no pregnancy indicator, no patient identifier and no RBC column**.

A model that receives `Hemoglobin` and `Gender` is therefore not predicting anaemia — it is
recomputing the label. That is the source of the previously reported **100 % accuracy**.

### 2.2 Ablation on an identical split (deduplicated, fixed hyper-parameters)

| Features | RF accuracy % | XGB accuracy % | RF ROC-AUC | XGB ROC-AUC |
| :--- | :--: | :--: | :--: | :--: |
| Gender, Hb, MCH, MCHC, MCV | 100.00 | 100.00 | 1.0000 | 1.0000 |
| Gender, MCH, MCHC, MCV (Hb removed) | 66.67 | 59.26 | 0.6437 | 0.5885 |
| Hb, MCH, MCHC, MCV (sex removed) | 88.89 | 87.65 | 0.9662 | 0.9656 |
| **MCV, MCH, MCHC (production)** | **46.91** | **41.98** | **0.4896** | **0.4533** |

Removing haemoglobin collapses accuracy to near chance and ROC-AUC to ≈ 0.48. The index-only
model performs no better than a coin flip, which is the honest reflection of the data.

### 2.3 Resolution

Module 2 is designated a **General Haematological Anaemia Classifier** trained on MCV, MCH
and MCHC only. Haemoglobin and the sex code are excluded and are recorded in
`models/anemia/training_config.json` under `excluded_features`.

---

## 3. Maternal module — duplicates and label conflicts

`maternal_health_risk.csv` (1 014 rows):

| Quantity | Value |
| :--- | :--: |
| Exact duplicate rows | 562 |
| Unique rows | 452 |
| Distinct feature groups | 416 |
| Feature groups with conflicting labels | 35 |
| Conflict rows removed | 71 |
| Rows used for modelling | 381 |
| `HeartRate < 30` rows corrected | 2 |

**Why it matters:** with duplicates left in place, a random split places identical rows on
both sides of the train/test boundary and inflates scores. Similarly, feature vectors that
carry two different risk labels can never be predicted consistently; keeping them adds noise
and depresses the achievable ceiling. Both effects are removed before splitting.

---

## 4. Preprocessing isolation (leakage prevention)

- **Split first, fit second.** A stratified 70/15/15 train/validation/test split is created
  with `random_state = 42` before any statistic is computed.
- **Imputation** uses the training-split median only; the medians are stored in
  `training_config.json` (`imputer_medians`).
- **Scaling** (`StandardScaler`) is fitted on the training split only; the fitted
  `Pipeline([imputer, scaler])` is saved as `preprocessor.pkl` and reused verbatim at
  inference time, so the app cannot drift from training.
- **Hyper-parameter search** (`RandomizedSearchCV`) uses `StratifiedKFold` on the training
  partition with the preprocessing inside each fold.
- **Hybrid weights** are chosen on the validation partition only.
- **The held-out test partition is scored once** and is never used for tuning, threshold
  selection, feature choice or calibration. `training_config.json` records
  `test_used_for_tuning: false`.

---

## 5. Residual risks (not fully removable)

1. **Maternal labels are noisy.** 35 feature groups carry more than one label, so some
   irreducible error is baked into the target.
2. **Small held-out sets** (n = 58 maternal, n = 81 anaemia) make the point estimates
   unstable; bootstrap intervals are wide.
3. **No external validation.** Neither module has been tested on an independent patient
   cohort or clinical registry.
4. **Anaemia population is unverified.** The CBC file does not confirm pregnancy status, so
   the module must not be described as pregnancy-specific.

These limitations are carried into the app (Model Intelligence and Dataset tabs) and the
generated patient report rather than being hidden.
