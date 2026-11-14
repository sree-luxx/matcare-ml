"""
Dataset access and held-out evaluation reconstruction for the dashboard.

The trained artefacts are reused exactly as saved. To draw ROC and
precision-recall curves the deterministic test partition from the training
scripts is recreated with the same cleaning steps and the same seed. No model is
refitted here and none of these rows are used for tuning.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from evaluation.audit import prepare_anemia_modeling_frame, prepare_maternal_modeling_frame
from preprocessing.anemia_preprocessor import INDEX_FEATURE_NAMES, load_and_inspect_anemia
from preprocessing.general_risk_preprocessor import (
    FEATURE_NAMES,
    TARGET_MAPPING,
    load_and_inspect_general_risk,
)

from app.inference import MODULE_CLASSES
from app.model_loader import project_root

SEED = 42

MODULE_LABEL = {"maternal": "Maternal risk", "anemia": "Anaemia"}
FOCUS_INDEX = {"maternal": 2, "anemia": 1}


def dataset_paths():
    root = project_root()
    return {
        "maternal": os.path.join(root, "datasets", "maternal_health_risk.csv"),
        "anemia": os.path.join(root, "datasets", "anemia_cbc.csv"),
    }


@st.cache_data(show_spinner=False)
def load_raw(module):
    paths = dataset_paths()
    if module == "maternal":
        return load_and_inspect_general_risk(paths["maternal"])
    if module == "anemia":
        return load_and_inspect_anemia(paths["anemia"])
    raise ValueError(f"Unknown module: {module}")


def _maternal_test_frame():
    frame, _ = prepare_maternal_modeling_frame(load_raw("maternal"))
    work = frame.copy()
    y = work["RiskLevelClean"].map(TARGET_MAPPING)
    X = work[FEATURE_NAMES]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    return X_test, y_test


def _anemia_test_frame():
    frame, _ = prepare_anemia_modeling_frame(load_raw("anemia"), drop_duplicates=True)
    y = frame["Result"].astype(int)
    X = frame[INDEX_FEATURE_NAMES]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    return X_test, y_test


def _scaled(bundle, frame):
    preprocessor = bundle.get("preprocessor")
    columns = list(frame.columns)
    if preprocessor is not None:
        scaled = preprocessor.transform(frame)
    else:  # legacy fallback: scale only
        filled = frame.fillna(pd.Series(bundle["scaler"].mean_, index=columns))
        scaled = bundle["scaler"].transform(filled)
    return pd.DataFrame(scaled, columns=columns, index=frame.index)


def test_predictions(bundle, module):
    """Recreate the held-out test partition and score every approach on it."""
    X_test, y_test = _maternal_test_frame() if module == "maternal" else _anemia_test_frame()
    scaled = _scaled(bundle, X_test)
    p_rf = np.asarray(bundle["rf"].predict_proba(scaled))
    p_xgb = np.asarray(bundle["xgb"].predict_proba(scaled))
    p_hybrid = np.asarray(bundle["hybrid"].predict_proba(scaled))
    return {
        "module": module,
        "class_names": list(MODULE_CLASSES[module]),
        "y_true": np.asarray(y_test),
        "n_test": int(len(y_test)),
        "probabilities": {
            "Random Forest": p_rf,
            "XGBoost": p_xgb,
            "Equal-weight voting (50/50)": 0.5 * p_rf + 0.5 * p_xgb,
            "Adaptive hybrid": p_hybrid,
        },
    }


def focus_roc_curves(result):
    """One ROC curve per approach for the clinically important (focus) class."""
    focus = FOCUS_INDEX[result["module"]]
    binary = (result["y_true"] == focus).astype(int)
    curves = {}
    for label, probabilities in result["probabilities"].items():
        score = np.asarray(probabilities)[:, focus]
        fpr, tpr, _ = roc_curve(binary, score)
        curves[label] = (fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr)))
    return curves


def focus_pr_curves(result):
    """One precision-recall curve per approach for the focus class."""
    focus = FOCUS_INDEX[result["module"]]
    binary = (result["y_true"] == focus).astype(int)
    curves = {}
    for label, probabilities in result["probabilities"].items():
        score = np.asarray(probabilities)[:, focus]
        precision, recall, _ = precision_recall_curve(binary, score)
        curves[label] = (recall.tolist(), precision.tolist(), float(auc(recall, precision)))
    return curves


def comparison_table(bundle):
    return list((bundle.get("metrics") or {}).get("comparison", []))


def global_importances(bundle):
    return (bundle.get("metrics") or {}).get("shap_importances", {})


def ablation_table(bundle):
    return (bundle.get("metrics") or {}).get("ablation_deduplicated_fixed_hyperparameters", [])


def audit_summary(bundle):
    return (bundle.get("metrics") or {}).get("data_audit", {})


def repeated_cv(bundle):
    return (bundle.get("metrics") or {}).get("repeated_cv_on_development_split", {})


def bootstrap_ci(bundle):
    return (bundle.get("metrics") or {}).get("bootstrap_ci", {})


def hybrid_conclusion(bundle):
    return (bundle.get("metrics") or {}).get("hybrid_conclusion", {})
