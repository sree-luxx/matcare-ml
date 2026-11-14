# MATCARE-AI — Model Comparison Report

**Status: current.** This replaces the earlier version of this file, which described the
pre-audit protocol (raw splits and a 100% anaemia result). Full detail is in
`reports/evaluation_report.md`; leakage analysis is in `reports/data_leakage_audit_report.md`.

Seed 42 · held-out test scored once · hybrid weights chosen on validation only.

---

## Module 1 — General Maternal Health Risk (held-out test n = 58)

| Approach | Acc % | Macro F1 | ROC-AUC | PR-AUC | High-risk recall % | High-risk FPR % |
| :--- | :--: | :--: | :--: | :--: | :--: | :--: |
| Random Forest | 72.41 | 68.70 | 0.8475 | 0.7662 | 80.00 | 9.30 |
| XGBoost | 65.52 | 63.90 | 0.8493 | 0.7335 | 66.67 | 9.30 |
| Equal-weight (50/50) | 68.97 | 66.23 | 0.8509 | 0.7623 | 73.33 | 11.63 |
| **Adaptive hybrid (RF 96% / XGB 4%)** | **72.41** | **68.70** | 0.8504 | 0.7692 | **80.00** | **9.30** |

- Hybrid vs stronger base model: **+0.00** macro-F1, **+0.00** high-risk recall.
- Hybrid vs equal-weight voting: **+2.47** macro-F1 points.

**Verdict:** the hybrid matches the better base model and beats 50/50 voting. It is not
claimed to be superior.

## Module 2 — General Haematological Anaemia (index-only, held-out test n = 81)

| Approach | Acc % | Macro F1 | ROC-AUC | PR-AUC | Anemic recall % |
| :--- | :--: | :--: | :--: | :--: | :--: |
| Random Forest | 48.15 | 48.14 | 0.4760 | 0.4630 | 51.35 |
| XGBoost | 41.98 | 41.94 | 0.4840 | 0.4651 | 43.24 |
| Equal-weight (50/50) | 46.91 | 46.88 | 0.4773 | 0.4662 | 54.05 |
| Adaptive hybrid (RF 61% / XGB 39%) | 46.91 | 46.91 | 0.4797 | 0.4684 | 51.35 |

- Hybrid vs stronger base model: **−1.23** macro-F1 points.

**Verdict:** with haemoglobin removed the task is at chance. The hybrid is not superior.
The earlier 100% figure was label circularity (see the leakage audit) and is not reported as
model performance.

## Global SHAP importances

- **Maternal (hybrid-weighted):** BS 34.1%, BodyTemp 21.7%, SystolicBP 20.9%, Age 9.4%,
  DiastolicBP 7.5%, HeartRate 6.4%.
- **Anaemia (index-only):** MCHC 37.2%, MCH 32.8%, MCV 30.0%.
