"""
Revised maternal-risk training protocol.

The held-out test split is scored once. Hyperparameters are chosen by
stratified cross-validation on the training split. Hybrid weights are chosen
on the validation split. Exact duplicate rows and feature vectors that carry
more than one risk label are removed before the split.
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.audit import prepare_maternal_modeling_frame
from evaluation.metrics import bootstrap_confidence_intervals, compare_four_approaches, json_safe
from explainability.shap_explainers import compute_hybrid_global_importances, compute_tree_shap_importances
from preprocessing.general_risk_preprocessor import (
    CLASS_NAMES,
    FEATURE_NAMES,
    TARGET_MAPPING,
    load_and_inspect_general_risk,
)
from training.hybrid_model import AdaptiveWeightedHybridClassifier
from training.model_search import fit_final_models, repeated_split_scores, search_random_forest, search_xgboost, transform_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("matcare.train.maternal")
SEED = 42


def _split(frame):
    from sklearn.model_selection import train_test_split

    work = frame.copy()
    y = work["RiskLevelClean"].map(TARGET_MAPPING)
    if y.isnull().any():
        raise ValueError(f"Unmapped labels: {work['RiskLevelClean'].unique()}")
    X = work[FEATURE_NAMES]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15 / 0.85, random_state=SEED, stratify=y_dev
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _conclusion(comparison, weights):
    rf, xgb, equal, hybrid = comparison
    best_single = max(rf["macro_f1"], xgb["macro_f1"])
    macro_gain = round(hybrid["macro_f1"] - best_single, 2)
    recall_gain = round(hybrid["high_risk_recall"] - max(rf["high_risk_recall"], xgb["high_risk_recall"]), 2)
    equal_gain = round(hybrid["macro_f1"] - equal["macro_f1"], 2)
    collapsed = weights["best_w_rf"] <= 0.01 or weights["best_w_xgb"] <= 0.01
    improved = macro_gain > 0.5 or (macro_gain >= 0 and recall_gain > 0.5 and equal_gain >= 0)
    if collapsed:
        reason = (
            "On the validation objective the mixture collapsed onto one base model, "
            "so the hybrid decisions match that model."
        )
    elif not improved:
        reason = (
            "The validation-chosen mixture does not improve held-out macro-F1 by more than half a "
            "percentage point over the stronger base model, and it does not improve high-risk recall. "
            "Equal-weight voting is reported beside it. The hybrid is not claimed to be superior."
        )
    else:
        reason = (
            "The validation-chosen mixture is higher than the stronger base model on the held-out "
            "comparison used here. The gain should be read with the confidence interval and the small test count."
        )
    return {
        "improved_on_held_out_test": bool(improved),
        "macro_f1_gain_vs_best_base_points": macro_gain,
        "high_risk_recall_gain_vs_best_base_points": recall_gain,
        "macro_f1_gain_vs_equal_weight_points": equal_gain,
        "collapsed_to_one_model": bool(collapsed),
        "statement": reason,
    }


def run_general_risk_pipeline(n_iter=10):
    LOGGER.info("Maternal risk training started")
    data_path = os.path.join(PROJECT_ROOT, "datasets", "maternal_health_risk.csv")
    raw = load_and_inspect_general_risk(data_path)
    frame, audit = prepare_maternal_modeling_frame(raw)
    X_train, X_val, X_test, y_train, y_val, y_test = _split(frame)
    LOGGER.info("Split sizes train=%s val=%s test=%s", len(X_train), len(X_val), len(X_test))

    rf_params, rf_cv = search_random_forest(X_train, y_train, n_iter=n_iter, seed=SEED)
    xgb_params, xgb_cv = search_xgboost(X_train, y_train, n_iter=n_iter, seed=SEED, binary=False)
    LOGGER.info("Selected RF params %s (cv macro-F1 %.3f)", rf_params, rf_cv)
    LOGGER.info("Selected XGB params %s (cv macro-F1 %.3f)", xgb_params, xgb_cv)

    preprocessor, rf_model, xgb_model = fit_final_models(X_train, y_train, rf_params, xgb_params)
    X_val_s = transform_frame(preprocessor, X_val, FEATURE_NAMES)
    X_test_s = transform_frame(preprocessor, X_test, FEATURE_NAMES)

    hybrid = AdaptiveWeightedHybridClassifier(rf_model=rf_model, xgb_model=xgb_model, classes=np.array([0, 1, 2]))
    weights = hybrid.optimize_weights(X_val_s, y_val, metric="clinical_utility", grid_resolution=101, high_risk_idx=2)
    stability = hybrid.weight_stability(X_val_s, y_val, metric="clinical_utility", high_risk_idx=2, n_boot=40, seed=SEED)

    comparison = compare_four_approaches(rf_model, xgb_model, hybrid, X_test_s, y_test, CLASS_NAMES, high_risk_idx=2)
    intervals = {}
    named_probabilities = {
        "Random Forest": rf_model.predict_proba(X_test_s),
        "XGBoost": xgb_model.predict_proba(X_test_s),
        "Equal-weight voting (50/50)": None,
        comparison[3]["model_name"]: hybrid.predict_proba(X_test_s),
    }
    rf_p = named_probabilities["Random Forest"]
    xgb_p = named_probabilities["XGBoost"]
    named_probabilities["Equal-weight voting (50/50)"] = 0.5 * rf_p + 0.5 * xgb_p
    for item in comparison:
        probabilities = named_probabilities[item["model_name"]]
        predictions = np.argmax(probabilities, axis=1)
        intervals[item["model_name"]] = bootstrap_confidence_intervals(
            y_test, predictions, probabilities, CLASS_NAMES, high_risk_idx=2, n_boot=1000, seed=SEED
        )

    dev = pd.concat([X_train, X_val])
    dev_y = pd.concat([y_train, y_val])
    repeated = repeated_split_scores(
        dev, dev_y, rf_params, xgb_params, CLASS_NAMES, high_risk_idx=2, n_splits=5, n_repeats=3, seed=SEED
    )

    rf_shap = compute_tree_shap_importances(rf_model, X_val_s, FEATURE_NAMES)
    xgb_shap = compute_tree_shap_importances(xgb_model, X_val_s, FEATURE_NAMES)
    hybrid_shap = compute_hybrid_global_importances(rf_shap, xgb_shap, hybrid.w_rf, hybrid.w_xgb, FEATURE_NAMES)
    conclusion = _conclusion(comparison, weights)

    models_dir = os.path.join(PROJECT_ROOT, "models", "general_risk")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl"))
    hybrid.save(os.path.join(models_dir, "hybrid_model.pkl"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    joblib.dump(preprocessor.named_steps["scaler"], os.path.join(models_dir, "scaler.pkl"))

    config = {
        "seed": SEED,
        "protocol": "deduplicated_conflict_free_70_15_15",
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "target_mapping": TARGET_MAPPING,
        "focus_class_index": 2,
        "focus_class_name": "High Risk",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "rf_params": rf_params,
        "xgb_params": xgb_params,
        "search_cv_macro_f1": {"random_forest": rf_cv, "xgboost": xgb_cv},
        "test_used_for_tuning": False,
        "imputer_medians": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, preprocessor.named_steps["imputer"].statistics_)
        },
    }
    payload = {
        "task": "general_maternal_health_risk",
        "data_audit": audit,
        "training_config": config,
        "weights": {key: value for key, value in weights.items() if key != "history"},
        "weight_stability": stability,
        "comparison": comparison,
        "bootstrap_ci": intervals,
        "repeated_cv_on_development_split": repeated,
        "shap_importances": {"rf": rf_shap, "xgb": xgb_shap, "hybrid_weighted_average": hybrid_shap},
        "hybrid_conclusion": conclusion,
        "objective": "0.50*macro-F1 + 0.30*high-risk recall + 0.20*high-risk precision, validation only",
    }
    with open(os.path.join(models_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(config), handle, indent=2)
    with open(os.path.join(models_dir, "weights.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe({"summary": payload["weights"], "stability": stability, "history": weights["history"]}), handle, indent=2)
    with open(os.path.join(models_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    LOGGER.info("Maternal artifacts saved. Hybrid conclusion: %s", conclusion["statement"])
    return payload


if __name__ == "__main__":
    run_general_risk_pipeline()
