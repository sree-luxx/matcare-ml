"""
Inference for both screening modules.

A prediction always runs through the exact preprocessing pipeline that was
fitted during training (imputation + scaling), so the app and the training
scripts cannot drift apart.
"""

import numpy as np
import pandas as pd

from preprocessing.anemia_preprocessor import ANEMIA_CLASS_NAMES, INDEX_FEATURE_NAMES
from preprocessing.general_risk_preprocessor import CLASS_NAMES as MATERNAL_CLASS_NAMES
from preprocessing.general_risk_preprocessor import FEATURE_NAMES as MATERNAL_FEATURE_NAMES


MODULE_FEATURES = {
    "maternal": list(MATERNAL_FEATURE_NAMES),
    "anemia": list(INDEX_FEATURE_NAMES),
}

MODULE_CLASSES = {
    "maternal": list(MATERNAL_CLASS_NAMES),
    "anemia": list(ANEMIA_CLASS_NAMES),
}

MODEL_LABELS = ["Random Forest", "XGBoost", "Adaptive hybrid"]


def _feature_names(bundle, module):
    config = bundle.get("config") or {}
    names = config.get("feature_names")
    return list(names) if names else list(MODULE_FEATURES[module])


def _class_names(bundle, module):
    config = bundle.get("config") or {}
    names = config.get("class_names")
    return list(names) if names else list(MODULE_CLASSES[module])


def _to_frame(features, feature_names):
    row = {name: features.get(name, np.nan) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _transform(bundle, frame, feature_names):
    preprocessor = bundle.get("preprocessor")
    if preprocessor is not None:
        scaled = preprocessor.transform(frame)
    else:  # legacy fallback: scale only, using the saved scaler
        filled = frame.fillna(pd.Series(bundle["scaler"].mean_, index=feature_names))
        scaled = bundle["scaler"].transform(filled)
    return pd.DataFrame(scaled, columns=feature_names)


def predict_module(bundle, features, module):
    """Predict one patient for one module.

    Returns a dictionary with per-model probabilities, the hybrid decision and
    the list of measurements that had to be imputed.
    """
    feature_names = _feature_names(bundle, module)
    class_names = _class_names(bundle, module)
    frame = _to_frame(features, feature_names)
    missing = [name for name in feature_names if bool(pd.isna(frame.iloc[0][name]))]
    scaled = _transform(bundle, frame, feature_names)

    p_rf = np.asarray(bundle["rf"].predict_proba(scaled))[0]
    p_xgb = np.asarray(bundle["xgb"].predict_proba(scaled))[0]
    p_hybrid = np.asarray(bundle["hybrid"].predict_proba(scaled))[0]
    # keep every probability vector a valid distribution
    p_hybrid = p_hybrid / np.clip(p_hybrid.sum(), 1e-12, None)

    index = int(np.argmax(p_hybrid))
    return {
        "module": module,
        "feature_names": feature_names,
        "class_names": class_names,
        "scaled": scaled,
        "probabilities": {
            "Random Forest": p_rf,
            "XGBoost": p_xgb,
            "Adaptive hybrid": p_hybrid,
        },
        "predicted_index": index,
        "predicted_class": class_names[index],
        "confidence": float(p_hybrid[index] * 100.0),
        "missing_features": missing,
        "w_rf": float(bundle.get("w_rf", 0.5)),
        "w_xgb": float(bundle.get("w_xgb", 0.5)),
    }


def model_agreement(maternal_result, anemia_result):
    """Describe whether the three approaches agree for a module."""
    def _agree(result):
        labels = [np.argmax(result["probabilities"][name]) for name in MODEL_LABELS]
        return len(set(labels)) == 1

    return {
        "maternal": _agree(maternal_result),
        "anemia": _agree(anemia_result),
    }
