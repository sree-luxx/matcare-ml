# MATCARE-AI: Clinical Validation & Research Compliance Notes

## 1. Population Audit: The 473-Sample Cohort Verification
A critical methodological requirement of MATCARE-AI is maintaining strict scientific integrity regarding dataset populations:
- **Finding in CBC Dataset (`anemia_cbc.csv`):**
  The dataset contains 1,421 total records comprising:
  - 740 biological males (`Gender = 1`): 328 non-anemic, 412 anemic.
  - 681 biological females (`Gender = 0`): **473 non-anemic**, 208 anemic.
- **Scientific Conclusion:**
  The **473 non-anemic cases** correspond to the female subgroup of this benchmark. Because the source file lacks obstetrical records (gestational age, gravidity, parity, ultrasound biometry), it is scientifically improper to claim this as a "maternal pregnancy anemia" dataset.
- **Designation:**
  In compliance with clinical research standards, Module 2 is explicitly presented as a **General Hematological Anemia Classifier**, rather than a validated maternal-specific tool.

---

## 2. Physiological Outlier Correction (UCI Maternal Risk)
- **Anomaly Identified:**
  The UCI Maternal Health Risk dataset contained records with `HeartRate = 7` bpm. A resting heart rate of 7 bpm is incompatible with life in ambulatory outpatients and represents human data-entry or transcription error.
- **Mitigation Strategy:**
  Records with `HeartRate < 30` bpm were flagged and replaced with `NaN`. Missing values were imputed strictly using the **median of the training partition** ($X_{\text{train}}$) after a 70/15/15 stratified split, preventing data leakage into validation or test sets.

---

## 3. Data Leakage Prevention Architecture
- **Stratified Partitioning:**
  A 3-way stratified partition was applied:
  - Training Partition (70%)
  - Validation Partition (15%) - utilized exclusively for searching optimal hybrid voting weights $w_{\text{RF}}^*$ and $w_{\text{XGB}}^*$.
  - Held-out Test Partition (15%) - touched exactly once during final benchmark scoring.
- **Scaler Isolation:**
  `StandardScaler` parameters ($\mu, \sigma$) were estimated strictly on the training set and subsequently applied downstream to validation and test folds.

---

## 4. Clinical Decision Thresholds
- **Maternal Health Risk:**
  - *High Risk Criteria:* Marked by blood glucose $\ge 12.0$ mmol/L (indicative of gestational diabetes mellitus) and systolic blood pressure $\ge 140$ mmHg (indicative of gestational hypertension or pre-eclampsia).
- **Anemia Stratification (WHO Guidelines):**
  - *Normal:* Hb $\ge 12.0$ g/dL (non-pregnant women), $\ge 11.0$ g/dL (pregnant women).
  - *Mild Anemia:* Hb 10.0 - 11.9 g/dL.
  - *Moderate Anemia:* Hb 7.0 - 9.9 g/dL.
  - *Severe Anemia:* Hb $< 7.0$ g/dL (requires urgent clinical attention / transfusion workup).
  - *Morphology:*
    - Microcytic ($MCV < 80$ fL): characteristic of iron deficiency.
    - Macrocytic ($MCV > 100$ fL): characteristic of vitamin B12 or folate deficiency.
