# MATCARE-AI — Clinical Limitations & Research Compliance Notes

This document records what MATCARE-AI is and is not allowed to claim. It is a screening
research prototype, not a medical device, and it has not been clinically validated.

---

## 1. Scope and intended use

MATCARE-AI provides two independent screening outputs for demonstration and research:

1. **General maternal health risk** (Low / Mid / High) from routine vital signs.
2. **General haematological anaemia classification** (Non-Anemic / Anemic) from CBC indices.

Both outputs are statistical model results for the values entered. They are **not diagnoses**
and they must not be used to start, stop or change any treatment.

## 2. Population honesty

The CBC source file (`anemia_cbc.csv`) contains **no obstetrical metadata** (no gestational
age, gravidity, parity or ultrasound data) and records both sexes (740 coded male, 681 coded
female). It therefore **cannot** be described as a maternal or pregnancy-specific anaemia
dataset. It is presented strictly as a **General Haematological Anaemia Classifier**.

## 3. Target-label circularity and its consequence

The anaemia label is an exact, sex-specific haemoglobin threshold
(`Result = 1 if (Gender==0 & Hb<12.0) | (Gender==1 & Hb<13.5)`), matching 100% of raw rows.
Including haemoglobin produced 100% accuracy by simply reproducing the label rule.

To avoid publishing a circular result, the production anaemia model uses **MCV, MCH and MCHC
only**. As a direct consequence, and stated openly:

- Held-out accuracy is **46.9%** (RF) and **42.0%** (XGB) — approximately chance.
- Held-out ROC-AUC is **≈ 0.48** — no discriminative power.
- The module **screens for morphological patterns only** and must not be read as detecting
  anaemia.

This result is the correct, non-fabricated outcome of removing the leaky column.

## 4. Maternal module limitations

- Trained on the UCI Maternal Health Risk dataset; held-out accuracy **72.4%**, high-risk
  recall **80%**, high-risk false-positive rate **9.3%**.
- The **Mid-risk** class is weak (F1 ≈ 50%); its label is intrinsically ambiguous.
- The held-out test set is small (n = 58), so scores carry wide uncertainty.

## 5. Data-quality caveats carried into the tool

- 55% of maternal rows are exact duplicates; 35 feature groups carry conflicting labels.
  Duplicates are removed and conflicting rows dropped before splitting, but residual label
  noise remains.
- The UCI dataset contains impossible values (`HeartRate = 7`); `HeartRate < 30` is treated
  as missing and imputed from training-split medians.
- Reference intervals shown in the UI are common adult/pregnancy screening ranges for
  context only; they are not laboratory reference intervals or diagnostic thresholds.

## 6. What must never be claimed

- No clinical validation, regulatory clearance or deployment readiness.
- No treatment, medication or dosage recommendations.
- No claim that the adaptive hybrid is superior to its base models — the evaluation shows it
  is equivalent to the stronger base model, not better.
- No claim that the anaemia module detects anaemia; it reports morphological screening only.
- No presentation of SHAP values as causal evidence.

## 7. Future work required before any clinical use

1. Independent, prospective validation on a labelled clinical cohort.
2. Recovery of the excluded haemoglobin pathway as a **transparent rule** rather than a
   "model", so the circularity is removed by design.
3. Clear separation of pregnancy-status fields so maternal-specific modelling is possible.
4. Prospective calibration and threshold selection against a chosen operating point.
