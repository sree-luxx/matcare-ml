"""
Script to generate reproducible Jupyter notebooks for MATCARE-AI:
1. notebooks/general_risk_analysis.ipynb
2. notebooks/anemia_analysis.ipynb
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook created at: {filepath}")

# 1. General Maternal Risk Notebook
cells_maternal = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MATCARE-AI: General Maternal Health Risk Analysis & Hybrid Modeling\n",
            "This notebook executes the exploratory data analysis (EDA), anomaly cleaning,\n",
            "stratified train/val/test splitting, base model training (Random Forest & XGBoost),\n",
            "validation-based adaptive probability weight optimization, and SHAP explainability."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Add parent directory to path\n",
            "sys.path.append('..')\n",
            "from preprocessing.general_risk_preprocessor import load_and_inspect_general_risk, get_general_risk_split\n",
            "from training.hybrid_model import AdaptiveWeightedHybridClassifier\n",
            "from evaluation.metrics import compare_four_approaches\n",
            "from explainability.shap_explainers import compute_tree_shap_importances, compute_hybrid_global_importances"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Dataset & Perform Exploratory Data Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = os.path.join('..', 'datasets', 'maternal_health_risk.csv')\n",
            "df = load_and_inspect_general_risk(data_path)\n",
            "print('Dataset Shape:', df.shape)\n",
            "display(df.head())\n",
            "display(df.describe().round(2))\n",
            "print('Risk Level Counts:\\n', df['RiskLevelClean'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Physiological Outlier Inspection & Stratified Splitting"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata = get_general_risk_split(df)\n",
            "print(f'Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}')\n",
            "print('Class distribution preserved across splits.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Train Base Models & Optimize Adaptive Voting Weights on Validation Set"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from xgboost import XGBClassifier\n",
            "\n",
            "rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')\n",
            "rf.fit(X_train, y_train)\n",
            "\n",
            "xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, eval_metric='mlogloss')\n",
            "xgb.fit(X_train, y_train)\n",
            "\n",
            "hybrid = AdaptiveWeightedHybridClassifier(rf_model=rf, xgb_model=xgb, classes=np.array([0, 1, 2]))\n",
            "opt_res = hybrid.optimize_weights(X_val, y_val, metric='clinical_utility', high_risk_idx=2)\n",
            "print(f'Optimal Learned Weights: w_RF={opt_res[\"best_w_rf\"]:.4f}, w_XGB={opt_res[\"best_w_xgb\"]:.4f}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Evaluate Held-Out Test Set: Comparing 4 Approaches"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "comparison = compare_four_approaches(rf, xgb, hybrid, X_test, y_test, metadata['class_names'], high_risk_idx=2)\n",
            "comp_df = pd.DataFrame([{\n",
            "    'Model': c['model_name'],\n",
            "    'Accuracy (%)': c['accuracy'],\n",
            "    'Macro F1 (%)': c['macro_f1'],\n",
            "    'High-Risk Recall (%)': c.get('high_risk_recall', 0),\n",
            "    'High-Risk FPR (%)': c.get('high_risk_fpr', 0),\n",
            "    'ROC-AUC': c['roc_auc']\n",
            "} for c in comparison])\n",
            "display(comp_df)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. SHAP Explainability & Global Feature Importance"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rf_imp = compute_tree_shap_importances(rf, X_val, metadata['feature_names'])\n",
            "xgb_imp = compute_tree_shap_importances(xgb, X_val, metadata['feature_names'])\n",
            "hybrid_imp = compute_hybrid_global_importances(rf_imp, xgb_imp, opt_res['best_w_rf'], opt_res['best_w_xgb'], metadata['feature_names'])\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(x=list(hybrid_imp.values()), y=list(hybrid_imp.keys()), palette='viridis')\n",
            "plt.title('Hybrid SHAP Global Feature Importance (UCI Maternal Risk)')\n",
            "plt.xlabel('Importance Proportion')\n",
            "plt.show()"
        ]
    }
]

# 2. CBC Anemia Notebook
cells_anemia = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MATCARE-AI: CBC Anemia Analysis, Population Audit & Hybrid Modeling\n",
            "This notebook executes Phase 1 population and demographic auditing (including verifying the 473 female non-anemic sample cohort),\n",
            "data leakage analysis, stratified splitting, base models, adaptive hybrid optimization, and SHAP explainability."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "sys.path.append('..')\n",
            "from preprocessing.anemia_preprocessor import load_and_inspect_anemia, audit_cbc_population_and_leakage, get_anemia_split\n",
            "from training.hybrid_model import AdaptiveWeightedHybridClassifier\n",
            "from evaluation.metrics import compare_four_approaches\n",
            "from explainability.shap_explainers import compute_tree_shap_importances, compute_hybrid_global_importances"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Population Audit (Verifying the 473 Sample Finding)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = os.path.join('..', 'datasets', 'anemia_cbc.csv')\n",
            "df_an = load_and_inspect_anemia(data_path)\n",
            "audit = audit_cbc_population_and_leakage(df_an)\n",
            "print('Total Rows:', audit['total_rows'])\n",
            "print('Gender Counts:', audit['gender_distribution'])\n",
            "print('Female Non-Anemic Cohort (473 Check):', audit['female_non_anemic_count'])\n",
            "print('\\nPopulation Research Note:\\n', audit['population_finding'])\n",
            "print('\\nHemoglobin-Result Correlation:', audit['hb_result_correlation'])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Train/Val/Test Split & Hybrid Model Execution"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata = get_anemia_split(df_an)\n",
            "\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from xgboost import XGBClassifier\n",
            "\n",
            "rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced')\n",
            "rf.fit(X_train, y_train)\n",
            "\n",
            "xgb = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='logloss')\n",
            "xgb.fit(X_train, y_train)\n",
            "\n",
            "hybrid = AdaptiveWeightedHybridClassifier(rf_model=rf, xgb_model=xgb, classes=np.array([0, 1]))\n",
            "opt_res = hybrid.optimize_weights(X_val, y_val, metric='macro_f1', high_risk_idx=1)\n",
            "print(f'Optimal Weights: w_RF={opt_res[\"best_w_rf\"]:.4f}, w_XGB={opt_res[\"best_w_xgb\"]:.4f}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Held-Out Evaluation on 4 Approaches"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "comparison = compare_four_approaches(rf, xgb, hybrid, X_test, y_test, metadata['class_names'], high_risk_idx=1)\n",
            "comp_df = pd.DataFrame([{\n",
            "    'Model': c['model_name'],\n",
            "    'Accuracy (%)': c['accuracy'],\n",
            "    'Macro F1 (%)': c['macro_f1'],\n",
            "    'Anemia Sensitivity (%)': c['per_class']['Anemic']['recall'],\n",
            "    'ROC-AUC': c['roc_auc']\n",
            "} for c in comparison])\n",
            "display(comp_df)"
        ]
    }
]

create_notebook(cells_maternal, os.path.join(NOTEBOOKS_DIR, "general_risk_analysis.ipynb"))
create_notebook(cells_anemia, os.path.join(NOTEBOOKS_DIR, "anemia_analysis.ipynb"))
