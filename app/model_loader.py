"""
Model loading and cached metadata access.

The trained artefacts live under ``models/<module>/``. Loading is expensive, so
it is memoised with ``st.cache_resource``. Metadata such as evaluation metrics
and weights are read once and cached alongside.
"""

import json
import os
import sys

import joblib
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_module(directory):
    """Load one module's models and metadata. Raises a clear error if missing."""
    required = ["rf_model.pkl", "xgb_model.pkl", "hybrid_model.pkl", "scaler.pkl"]
    missing = [name for name in required if not os.path.exists(os.path.join(directory, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing model artefact(s) in {directory}: {', '.join(missing)}. "
            "Run the training pipeline first (python -m training.train_general_risk "
            "and python -m training.train_anemia)."
        )
    hybrid = joblib.load(os.path.join(directory, "hybrid_model.pkl"))
    scale = os.path.join(directory, "scaler.pkl")
    preprocess = os.path.join(directory, "preprocessor.pkl")
    return {
        "rf": joblib.load(os.path.join(directory, "rf_model.pkl")),
        "xgb": joblib.load(os.path.join(directory, "xgb_model.pkl")),
        "hybrid": hybrid,
        "scaler": joblib.load(scale),
        "preprocessor": joblib.load(preprocess) if os.path.exists(preprocess) else None,
        "w_rf": float(getattr(hybrid, "w_rf", 0.5)),
        "w_xgb": float(getattr(hybrid, "w_xgb", 0.5)),
        "metrics": _read_json(os.path.join(directory, "evaluation_metrics.json")),
        "weights": _read_json(os.path.join(directory, "weights.json")),
        "config": _read_json(os.path.join(directory, "training_config.json")),
        "audit": _read_json(os.path.join(directory, "population_audit.json")),
    }


@st.cache_resource(show_spinner=False)
def load_bundle():
    """Return both prediction modules keyed by ``maternal`` and ``anemia``."""
    maternal_dir = os.path.join(PROJECT_ROOT, "models", "general_risk")
    anemia_dir = os.path.join(PROJECT_ROOT, "models", "anemia")
    return {
        "maternal": _load_module(maternal_dir),
        "anemia": _load_module(anemia_dir),
    }


def project_root():
    return PROJECT_ROOT
