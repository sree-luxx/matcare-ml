"""
Anemia experiments and the saved index-only model.

The binary file's label matches a sex-specific hemoglobin cutoff exactly.
The saved clinical model therefore excludes hemoglobin and the sex code.
Full-feature results are stored as a leakage baseline, not as a diagnostic claim.
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.audit import INDEX_FEATURES, prepare_anemia_modeling_frame
from evaluation.metrics import (
    bootstrap_confidence_intervals,
    compare_four_approaches,
    compute_comprehensive_metrics,
    json_safe,
)
from explainability.shap_explainers import compute_hybrid_global_importances, compute_tree_shap_importances
from preprocessing.anemia_preprocessor import (
    ANEMIA_CLASS_NAMES,
    CBC_FEATURE_NAMES,
    DATASET_LABEL_RULE,
    load_and_inspect_anemia,
)
from training.hybrid_model import AdaptiveWeightedHybridClassifier
from training.model_search import (
    fit_final_models,
    fit_preprocessor,
    repeated_split_scores,
    search_random_forest,
    search_xgboost,
    transform_frame,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("matcare.train.anemia")
SEED = 42

ABLATION_SETS = [
    ("Full features", ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]),
    ("Hemoglobin excluded", ["Gender", "MCH", "MCHC", "MCV"]),
    ("Sex code excluded", ["Hemoglobin", "MCH", "MCHC", "MCV"]),
    ("Hemoglobin and sex code excluded", ["MCV", "MCH", "MCHC"]),
]


def _fixed_models():
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
    )
    return rf_model, xgb_model


def _ablation(frame):
    """One stratified split, shared by every feature subset. Fixed hyperparameters, train versus test."""
    y = frame["Result"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        frame, y, test_size=0.15, random_state=SEED, stratify=y
    )
    rows = []
    for name, columns in ABLATION_SETS:
        rf_model, xgb_model = _fixed_models()
        preprocessor = fit_preprocessor(X_train[columns])
        train_s = transform_frame(preprocessor, X_train, columns)
        test_s = transform_frame(preprocessor, X_test, columns)
        rf_model.fit(train_s, y_train)
        xgb_model.fit(train_s, y_train)
        for model_name, model in (("Random Forest", rf_model), ("XGBoost", xgb_model)):
            probabilities = model.predict_proba(test_s)
            predictions = np.argmax(probabilities, axis=1)
            metrics = compute_comprehensive_metrics(
                y_test, predictions, probabilities, ANEMIA_CLASS_NAMES, high_risk_idx=1
            )
            rows.append({
                "experiment": name,
                "features": columns,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "anemic_recall": metrics["high_risk_recall"],
            })
    return rows, (X_train, X_test, y_train, y_test)


def _conclusion(comparison, weights, audit):
    rf, xgb, equal, hybrid = comparison
    best_single = max(rf["macro_f1"], xgb["macro_f1"])
    gain = round(hybrid["macro_f1"] - best_single, 2)
    improved = gain > 0.5
    return {
        "improved_on_held_out_test": bool(improved),
        "macro_f1_gain_vs_best_base_points": gain,
        "collapsed_to_one_model": bool(weights["best_w_rf"] <= 0.01 or weights["best_w_xgb"] <= 0.01),
        "statement": (
            "The saved model predicts the dataset label from MCV, MCH, and MCHC only. "
            "It does not see hemoglobin, which defines that label exactly "
            f"({audit['rule_match_rate']:.0%} of raw rows). "
            + (
                "The mixture improved held-out macro-F1 over the stronger base model."
                if improved
                else "The mixture did not improve held-out macro-F1 over the stronger base model in a meaningful way."
            )
        ),
    }


def run_anemia_pipeline(n_iter=8):
    LOGGER.info("Anemia experiments started")
    raw = load_and_inspect_anemia(os.path.join(PROJECT_ROOT, "datasets", "anemia_cbc.csv"))
    frame, audit = prepare_anemia_modeling_frame(raw, drop_duplicates=True)
    ablation_rows, _ = _ablation(frame)
    LOGGER.info("Ablation complete: %s", ablation_rows)

    y = frame["Result"].astype(int)
    X = frame[INDEX_FEATURES]
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15 / 0.85, random_state=SEED, stratify=y_dev
    )

    rf_params, rf_cv = search_random_forest(X_train, y_train, n_iter=n_iter, seed=SEED)
    xgb_params, xgb_cv = search_xgboost(X_train, y_train, n_iter=n_iter, seed=SEED, binary=True)
    preprocessor, rf_model, xgb_model = fit_final_models(X_train, y_train, rf_params, xgb_params)
    X_val_s = transform_frame(preprocessor, X_val, INDEX_FEATURES)
    X_test_s = transform_frame(preprocessor, X_test, INDEX_FEATURES)

    hybrid = AdaptiveWeightedHybridClassifier(rf_model=rf_model, xgb_model=xgb_model, classes=np.array([0, 1]))
    weights = hybrid.optimize_weights(
        X_val_s, y_val, metric="anemia_detection", grid_resolution=101, high_risk_idx=1
    )
    stability = hybrid.weight_stability(
        X_val_s, y_val, metric="anemia_detection", high_risk_idx=1, n_boot=40, seed=SEED
    )
    comparison = compare_four_approaches(
        rf_model, xgb_model, hybrid, X_test_s, y_test, ANEMIA_CLASS_NAMES, high_risk_idx=1
    )
    intervals = {}
    probability_map = {
        comparison[0]["model_name"]: rf_model.predict_proba(X_test_s),
        comparison[1]["model_name"]: xgb_model.predict_proba(X_test_s),
        comparison[3]["model_name"]: hybrid.predict_proba(X_test_s),
    }
    probability_map[comparison[2]["model_name"]] = 0.5 * probability_map[comparison[0]["model_name"]] + 0.5 * probability_map[comparison[1]["model_name"]]
    for item in comparison:
        probabilities = probability_map[item["model_name"]]
        intervals[item["model_name"]] = bootstrap_confidence_intervals(
            y_test, np.argmax(probabilities, axis=1), probabilities, ANEMIA_CLASS_NAMES, high_risk_idx=1, seed=SEED
        )

    repeated = repeated_split_scores(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
        rf_params,
        xgb_params,
        ANEMIA_CLASS_NAMES,
        high_risk_idx=1,
        n_splits=5,
        n_repeats=3,
        seed=SEED,
    )
    rf_shap = compute_tree_shap_importances(rf_model, X_val_s, INDEX_FEATURES)
    xgb_shap = compute_tree_shap_importances(xgb_model, X_val_s, INDEX_FEATURES)
    hybrid_shap = compute_hybrid_global_importances(rf_shap, xgb_shap, hybrid.w_rf, hybrid.w_xgb, INDEX_FEATURES)
    conclusion = _conclusion(comparison, weights, audit)

    models_dir = os.path.join(PROJECT_ROOT, "models", "anemia")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl"))
    hybrid.save(os.path.join(models_dir, "hybrid_model.pkl"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    joblib.dump(preprocessor.named_steps["scaler"], os.path.join(models_dir, "scaler.pkl"))

    config = {
        "seed": SEED,
        "protocol": "deduplicated_index_features_70_15_15",
        "feature_names": INDEX_FEATURES,
        "excluded_features": ["Gender", "Hemoglobin"],
        "exclusion_reason": DATASET_LABEL_RULE,
        "class_names": ANEMIA_CLASS_NAMES,
        "focus_class_index": 1,
        "focus_class_name": "Anemic",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "rf_params": rf_params,
        "xgb_params": xgb_params,
        "search_cv_macro_f1": {"random_forest": rf_cv, "xgboost": xgb_cv},
        "test_used_for_tuning": False,
        "full_feature_names_in_source_file": CBC_FEATURE_NAMES,
        "rbc_in_source_file": False,
    }
    payload = {
        "task": "general_hematological_index_model",
        "designation": "General Hematological Anemia Classification",
        "data_audit": audit,
        "ablation_deduplicated_fixed_hyperparameters": ablation_rows,
        "training_config": config,
        "weights": {key: value for key, value in weights.items() if key != "history"},
        "weight_stability": stability,
        "comparison": comparison,
        "bootstrap_ci": intervals,
        "repeated_cv_on_development_split": repeated,
        "shap_importances": {"rf": rf_shap, "xgb": xgb_shap, "hybrid_weighted_average": hybrid_shap},
        "hybrid_conclusion": conclusion,
        "objective": "0.50*macro-F1 + 0.30*anemic-class recall + 0.20*anemic-class precision, validation only",
    }
    with open(os.path.join(models_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(config), handle, indent=2)
    with open(os.path.join(models_dir, "weights.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe({"summary": payload["weights"], "stability": stability, "history": weights["history"]}), handle, indent=2)
    with open(os.path.join(models_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    with open(os.path.join(models_dir, "population_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(audit), handle, indent=2)
    LOGGER.info("Anemia artifacts saved. %s", conclusion["statement"])
    return payload


if __name__ == "__main__":
    run_anemia_pipeline()
