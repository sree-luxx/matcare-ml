"""
MATCARE-AI: Phase 2 - Data Leakage Investigation & Ablation Study
Investigates 100% anemia accuracy, checks Hemoglobin threshold circularity,
runs 5 ablation experiments, and audits maternal dataset duplicates.
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS = {}

# ============================================================
# 1. LOAD DATASETS
# ============================================================
df_m = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets", "maternal_health_risk.csv"))
df_a = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets", "anemia_cbc.csv"))

print("=" * 70)
print("PHASE 2: DATA INTEGRITY & LEAKAGE INVESTIGATION")
print("=" * 70)

# ============================================================
# 2. MATERNAL DATASET DUPLICATE AUDIT
# ============================================================
print("\n--- 2A. UCI Maternal Dataset Duplicates ---")
total_m = len(df_m)
dup_m = df_m.duplicated().sum()
uniq_m = total_m - dup_m
print(f"Total rows: {total_m}")
print(f"Exact duplicates: {dup_m} ({dup_m/total_m*100:.1f}%)")
print(f"Unique rows: {uniq_m}")

# Check if duplicates spread across train/test after split
df_m_clean = df_m.copy()
df_m_clean['RiskLevel_enc'] = df_m_clean['RiskLevel'].str.strip().str.lower().map({'low risk': 0, 'mid risk': 1, 'high risk': 2})
X_m = df_m_clean[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y_m = df_m_clean['RiskLevel_enc']
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_m, y_m, test_size=0.15, random_state=42, stratify=y_m)

# Detect train-test sample overlap (using original dataframe indices)
overlap_m = set(X_tr_m.index).intersection(set(X_te_m.index))
print(f"Train-test index overlap (should be 0): {len(overlap_m)}")

# Check duplicate values across train vs test  
tr_tuples = set(map(tuple, X_tr_m.values.tolist()))
te_tuples = set(map(tuple, X_te_m.values.tolist()))
val_overlap = tr_tuples.intersection(te_tuples)
print(f"Feature-identical rows between train & test: {len(val_overlap)} (data leakage if > 0)")

RESULTS['maternal_duplicates'] = {
    'total': total_m, 'duplicates': int(dup_m), 'unique': uniq_m,
    'feature_overlap_train_test': len(val_overlap)
}

# ============================================================
# 3. ANEMIA DATASET COMPREHENSIVE LEAKAGE TEST
# ============================================================
print("\n--- 2B. CBC Anemia Dataset Duplicates ---")
total_a = len(df_a)
dup_a = df_a.duplicated().sum()
print(f"Total rows: {total_a}")
print(f"Exact duplicates: {dup_a} ({dup_a/total_a*100:.1f}%)")
print(f"Unique rows: {total_a - dup_a}")

print("\n--- 2C. Hemoglobin Threshold Leakage Test ---")
# WHO criteria: Female (Gender=0) Hb < 12; Male (Gender=1) Hb < 13 -> Anemic
def apply_who_rule(row):
    if row['Gender'] == 0:
        return 1 if row['Hemoglobin'] < 12.0 else 0
    else:
        return 1 if row['Hemoglobin'] < 13.0 else 0

who_pred = df_a.apply(apply_who_rule, axis=1)
perfect_match = (who_pred == df_a['Result']).all()
match_rate = (who_pred == df_a['Result']).mean()
print(f"WHO Hb rule (Female<12, Male<13) matches Result: {match_rate*100:.4f}%")
print(f"Perfect rule match: {perfect_match}")

# Try other thresholds
for thr_f, thr_m in [(11.0, 13.0), (12.0, 13.0), (11.5, 13.0), (12.5, 13.5)]:
    pred2 = df_a.apply(lambda r: 1 if r['Hemoglobin'] < (thr_f if r['Gender']==0 else thr_m) else 0, axis=1)
    r2 = (pred2 == df_a['Result']).mean()
    print(f"  Threshold (F<{thr_f}, M<{thr_m}): {r2*100:.3f}% match")

# Single threshold ignoring gender
for thr in [11.0, 11.5, 12.0, 12.5, 13.0]:
    pred3 = (df_a['Hemoglobin'] < thr).astype(int)
    r3 = (pred3 == df_a['Result']).mean()
    print(f"  Single threshold Hb<{thr}: {r3*100:.3f}% match")

# Correlation matrix
print("\n--- 2D. Anemia Feature Correlations with Result ---")
corr = df_a.corr(numeric_only=True)['Result'].drop('Result')
print(corr.sort_values(key=abs, ascending=False))

RESULTS['anemia_leakage'] = {
    'total': total_a,
    'duplicates': int(dup_a),
    'who_rule_match_rate': float(match_rate),
    'perfect_rule_match': bool(perfect_match),
    'correlations': corr.to_dict()
}

# ============================================================
# 4. ABLATION STUDY: 5 EXPERIMENTS (IDENTICAL SPLITS)
# ============================================================
print("\n--- 2E. Ablation Study: 5 Experiments ---")
ALL_FEATURES = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
ablation_configs = [
    ("Full-Feature Baseline", ALL_FEATURES),
    ("Hemoglobin-Excluded", ['Gender', 'MCH', 'MCHC', 'MCV']),
    ("Gender-Excluded", ['Hemoglobin', 'MCH', 'MCHC', 'MCV']),
    ("Hb+Gender-Excluded", ['MCH', 'MCHC', 'MCV']),
    ("MCV+MCH+MCHC-Only", ['MCV', 'MCH', 'MCHC']),
]

X_full = df_a[ALL_FEATURES]
y_a = df_a['Result']

X_tr_base, X_te_base, y_tr_base, y_te_base = train_test_split(
    X_full, y_a, test_size=0.15, random_state=42, stratify=y_a
)

ablation_results = []
for name, feats in ablation_configs:
    X_tr = X_tr_base[feats].values
    X_te = X_te_base[feats].values
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_tr_s, y_tr_base)
    
    pred = rf.predict(X_te_s)
    prob = rf.predict_proba(X_te_s)[:, 1]
    
    acc = accuracy_score(y_te_base, pred) * 100
    f1 = f1_score(y_te_base, pred, average='macro') * 100
    auc = roc_auc_score(y_te_base, prob)
    
    row = {'Experiment': name, 'Features': feats, 'Accuracy': round(acc, 2), 'MacroF1': round(f1, 2), 'AUC': round(auc, 4)}
    ablation_results.append(row)
    print(f"  [{name:30s}] Acc={acc:.2f}%  F1={f1:.2f}%  AUC={auc:.4f}")

RESULTS['ablation_study'] = ablation_results

# Save full results
os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
out_path = os.path.join(PROJECT_ROOT, "reports", "phase2_leakage_audit.json")
with open(out_path, 'w') as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nPhase 2 audit complete. Results saved to: {out_path}")
print("\n=== VERDICT ===")
if match_rate >= 0.995:
    print("CONFIRMED: Hemoglobin threshold directly determines anemia label.")
    print("This is LABEL-DEFINITION CIRCULARITY — not genuine ML learning.")
else:
    print(f"Partial rule match ({match_rate*100:.2f}%). Additional features contribute.")
