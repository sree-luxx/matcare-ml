"""
SHAP helpers for tree models.

Importances and local bars describe how features move a model score.
They do not establish that a measurement caused a clinical outcome.
For a probability mixture, base-model SHAP values are reported separately.
A weight-scaled sum is shown only as a display aid and is labeled as such,
because the two explainers are not guaranteed to share a common unit.
"""

import numpy as np
import pandas as pd
import shap

from explainability.labels import clinical_label


def matrix_for_class(raw_shap, class_idx, n_features):
    if isinstance(raw_shap, list):
        matrix = np.asarray(raw_shap[class_idx])
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix
    array = np.asarray(raw_shap)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[1] == n_features:
            return array[:, :, class_idx]
        if array.shape[2] == n_features:
            return array[:, class_idx, :]
    raise ValueError(f"Unrecognized SHAP shape {getattr(array, 'shape', None)}")


def _class_count(raw_shap, n_features):
    if isinstance(raw_shap, list):
        return len(raw_shap)
    array = np.asarray(raw_shap)
    if array.ndim == 2:
        return 1
    if array.ndim == 3:
        if array.shape[1] == n_features:
            return array.shape[2]
        if array.shape[2] == n_features:
            return array.shape[1]
    return 1


def compute_tree_shap_importances(model, X_sample, feature_names):
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)
    n_features = len(feature_names)
    means = []
    for class_idx in range(_class_count(raw, n_features)):
        matrix = matrix_for_class(raw, class_idx, n_features)
        means.append(np.abs(matrix).mean(axis=0))
    mean_abs = np.mean(means, axis=0)
    total = float(np.sum(mean_abs)) + 1e-12
    ranked = {feature: round(float(value / total), 4) for feature, value in zip(feature_names, mean_abs)}
    return dict(sorted(ranked.items(), key=lambda item: item[1], reverse=True))


def compute_hybrid_global_importances(rf_imp_dict, xgb_imp_dict, w_rf, w_xgb, feature_names):
    """Weight the already-normalized base importances. This is not itself a SHAP value."""
    mixed = {}
    for feature in feature_names:
        mixed[feature] = (w_rf * rf_imp_dict.get(feature, 0.0)) + (w_xgb * xgb_imp_dict.get(feature, 0.0))
    total = sum(mixed.values()) + 1e-12
    mixed = {feature: round(value / total, 4) for feature, value in mixed.items()}
    return dict(sorted(mixed.items(), key=lambda item: item[1], reverse=True))


def explain_single_patient(
    rf_model,
    xgb_model,
    w_rf,
    w_xgb,
    patient_df,
    raw_patient_dict,
    feature_names,
    class_idx=2,
    class_name="predicted class",
    rf_explainer=None,
    xgb_explainer=None,
):
    """
    Local attributions for one row.

    Positive values raise the selected class score of that base model.
    They are not a statement that the measurement increases clinical risk
    when the selected class is a lower-risk class.
    """
    if rf_explainer is None:
        rf_explainer = shap.TreeExplainer(rf_model)
    if xgb_explainer is None:
        xgb_explainer = shap.TreeExplainer(xgb_model)
    rf_raw = rf_explainer.shap_values(patient_df)
    xgb_raw = xgb_explainer.shap_values(patient_df)
    n_features = len(feature_names)
    rf_vec = np.asarray(matrix_for_class(rf_raw, class_idx, n_features)[0], dtype=float)
    xgb_vec = np.asarray(matrix_for_class(xgb_raw, class_idx, n_features)[0], dtype=float)
    mixed = (w_rf * rf_vec) + (w_xgb * xgb_vec)

    rows = []
    for feature, rf_value, xgb_value, mixed_value in zip(feature_names, rf_vec, xgb_vec, mixed):
        direction = (
            f"Raises the {class_name} score"
            if mixed_value > 0
            else f"Lowers the {class_name} score"
        )
        rows.append({
            "Feature": feature,
            "Display name": clinical_label(feature),
            "Clinical Value": raw_patient_dict.get(feature),
            "Hybrid Impact": round(float(mixed_value), 4),
            "RF Impact": round(float(rf_value), 4),
            "XGB Impact": round(float(xgb_value), 4),
            "Direction": direction,
        })
    frame = pd.DataFrame(rows)
    frame["Absolute Impact"] = frame["Hybrid Impact"].abs()
    frame = frame.sort_values("Absolute Impact", ascending=True)
    return frame
