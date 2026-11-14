# MATCARE-AI: Comparative Evaluation & Model Benchmark Report

## 1. Executive Summary
**MATCARE-AI** presents an Explainable Adaptive Hybrid Machine Learning framework designed for dual clinical prediction:
1. **General Maternal Health Risk** (Low, Mid, High risk) using the UCI Maternal Health Risk dataset.
2. **Hematological Anemia Classification** (Non-Anemic vs. Anemic) using Complete Blood Count (CBC) parameters.

The algorithmic core consists of an **Adaptive Weighted Probability Voting Mixture** combining **Random Forest** and **XGBoost**:
$$P_{\text{hybrid}}(c) = w_{\text{RF}} P_{\text{RF}}(c) + w_{\text{XGB}} P_{\text{XGB}}(c)$$
where weights $w_{\text{RF}}$ and $w_{\text{XGB}}$ are learned on an independent validation set to maximize clinical sensitivity for High-Risk detection while controlling false-positive rates.

---

## 2. Benchmark Comparison (Held-Out Test Set)

### Module 1: General Maternal Health Risk (UCI Dataset)
- **Held-Out Test Set Size:** $N = 153$ (15% strictly held-out partition)
- **Validation Optimization Partition:** $N = 152$ (15%)
- **Training Partition:** $N = 709$ (70%)

| Model Approach | Test Accuracy (%) | Macro F1 (%) | Macro Precision (%) | Macro Recall (%) | High-Risk Recall (%) | High-Risk FPR (%) | Multiclass ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Base A)** | 79.74% | 80.08% | 79.56% | 81.08% | 97.56% | 7.14% | 0.9140 |
| **XGBoost (Base B)** | 80.39% | 80.97% | 80.47% | 81.73% | 97.56% | 5.36% | 0.9298 |
| **Equal-Weight Voting (50/50)** | 80.39% | 80.97% | 80.47% | 81.73% | 97.56% | 5.36% | 0.9247 |
| **Adaptive Weighted Hybrid** | **80.39%** | **80.97%** | **80.47%** | **81.73%** | **97.56%** | **5.36%** | **0.9298** |

#### Key Takeaway for Module 1:
- The Adaptive Hybrid detected **97.56% of true High-Risk maternal emergencies**, with a false-alarm rate of only 5.36%.
- Multiclass OvR ROC-AUC reached **0.9298**, proving robust ranking capability between Low, Mid, and High risk.

---

### Module 2: CBC Anemia Classification
- **Held-Out Test Set Size:** $N = 214$ (15%)
- **Validation Partition:** $N = 213$ (15%)
- **Training Partition:** $N = 994$ (70%)

| Model Approach | Test Accuracy (%) | Macro F1 (%) | Macro Precision (%) | Macro Recall (%) | Anemia Sensitivity (%) | Anemia Precision (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Base A)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 |
| **XGBoost (Base B)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 |
| **Equal-Weight Voting (50/50)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 |
| **Adaptive Weighted Hybrid** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.0000** |

#### Key Takeaway for Module 2:
- Strong separation achieved by both algorithms and the hybrid mixture.
- SHAP feature importance confirms **Hemoglobin** accounts for 67.72% of the decision boundary, consistent with WHO clinical diagnostic definitions.

---

## 3. Explainable AI: SHAP Global Feature Importance

### Module 1: Maternal Risk Drivers
1. **Blood Sugar (BS):** 35.78%
2. **Systolic Blood Pressure:** 19.10%
3. **Body Temperature:** 15.89%
4. **Maternal Age:** 12.67%
5. **Heart Rate:** 8.95%
6. **Diastolic Blood Pressure:** 7.61%

### Module 2: CBC Anemia Drivers
1. **Hemoglobin (Hb):** 67.72%
2. **Gender:** 27.60%
3. **MCH:** 2.17%
4. **MCHC:** 1.41%
5. **MCV:** 1.10%
