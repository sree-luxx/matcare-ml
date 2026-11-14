"""
Adaptive probability mixture of Random Forest and XGBoost.

P(c) = w_rf * P_rf(c) + w_xgb * P_xgb(c), with w_rf + w_xgb = 1.

Weights are chosen on validation predictions only. Tied clinical scores are
broken by validation log-loss, then by proximity to equal weights. The test
set is never an argument of optimize_weights.
"""

import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


class AdaptiveWeightedHybridClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_model=None, xgb_model=None, w_rf=0.5, classes=None, optimization_metric="macro_f1"):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.w_rf = float(w_rf)
        self.w_xgb = float(1.0 - w_rf)
        self.classes_ = np.array(classes) if classes is not None else None
        self.optimization_metric = optimization_metric
        self.optimization_history = []
        self.best_val_score = 0.0
        self.optimization_note = ""

    def fit(self, X, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
        if self.rf_model is not None:
            self.rf_model.fit(X, y)
        if self.xgb_model is not None:
            self.xgb_model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Both base models must be available before prediction.")
        p_rf = self.rf_model.predict_proba(X)
        p_xgb = self.xgb_model.predict_proba(X)
        if p_rf.shape != p_xgb.shape:
            raise ValueError(f"Probability shape mismatch: RF {p_rf.shape}, XGBoost {p_xgb.shape}")
        mixed = (self.w_rf * p_rf) + (self.w_xgb * p_xgb)
        totals = np.sum(mixed, axis=1, keepdims=True)
        return mixed / np.clip(totals, 1e-12, None)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        if self.classes_ is not None:
            return self.classes_[indices]
        return indices

    def _focus_rates(self, y_true, y_pred, focus_idx):
        if focus_idx is None:
            return 0.0, 0.0
        y_bin = (np.asarray(y_true) == focus_idx).astype(int)
        pred_bin = (np.asarray(y_pred) == focus_idx).astype(int)
        recall = recall_score(y_bin, pred_bin, zero_division=0)
        precision = precision_score(y_bin, pred_bin, zero_division=0)
        return float(recall), float(precision)

    def optimize_weights(self, X_val, y_val, metric="macro_f1", grid_resolution=101, high_risk_idx=2):
        """
        Grid-search w_rf on validation data.

        metric:
          macro_f1 — macro-F1 only
          clinical_utility — 0.50 macro-F1 + 0.30 focus recall + 0.20 focus precision
          anemia_detection — same weights, with the focus class set to the anemic label
        """
        self.optimization_metric = metric
        y_val = np.asarray(y_val)
        p_rf = self.rf_model.predict_proba(X_val)
        p_xgb = self.xgb_model.predict_proba(X_val)
        if self.classes_ is None:
            self.classes_ = np.unique(y_val)
        labels = list(self.classes_)

        history = []
        for w in np.linspace(0.0, 1.0, grid_resolution):
            mixed = (w * p_rf) + ((1.0 - w) * p_xgb)
            mixed = mixed / np.clip(mixed.sum(axis=1, keepdims=True), 1e-12, None)
            pred_idx = np.argmax(mixed, axis=1)
            y_pred = self.classes_[pred_idx]
            macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
            focus_recall, focus_precision = self._focus_rates(y_val, y_pred, high_risk_idx)
            if metric in ("clinical_utility", "anemia_detection") and high_risk_idx is not None:
                score = (0.50 * macro_f1) + (0.30 * focus_recall) + (0.20 * focus_precision)
            else:
                score = macro_f1
            clipped = np.clip(mixed, 1e-7, 1.0)
            clipped = clipped / clipped.sum(axis=1, keepdims=True)
            try:
                held_log_loss = float(log_loss(y_val, clipped, labels=labels))
            except Exception:
                held_log_loss = 1e9
            history.append({
                "w_rf": float(w),
                "w_xgb": float(1.0 - w),
                "score": float(score),
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "macro_f1": macro_f1,
                "macro_precision": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
                "high_risk_recall": focus_recall,
                "high_risk_precision": focus_precision,
                "log_loss": held_log_loss,
            })

        best_score = max(item["score"] for item in history)
        plateau = [item for item in history if item["score"] >= best_score - 1e-12]
        best_loss = min(item["log_loss"] for item in plateau)
        loss_tied = [item for item in plateau if item["log_loss"] <= best_loss + 1e-12]
        chosen = min(loss_tied, key=lambda item: (abs(item["w_rf"] - 0.5), item["w_rf"]))

        self.w_rf = float(chosen["w_rf"])
        self.w_xgb = float(chosen["w_xgb"])
        self.best_val_score = float(chosen["score"])
        self.optimization_history = history
        self.optimization_note = (
            "Validation clinical score first, then lower validation log-loss, then the weight closer to 0.50. "
            f"Exact score plateau spans w_rf {min(item['w_rf'] for item in plateau):.2f} to "
            f"{max(item['w_rf'] for item in plateau):.2f}."
        )
        if self.w_rf <= 0.01 or self.w_xgb <= 0.01:
            self.optimization_note += " The selected mixture puts essentially all mass on one base model."

        return {
            "best_w_rf": self.w_rf,
            "best_w_xgb": self.w_xgb,
            "best_val_score": self.best_val_score,
            "optimization_metric": metric,
            "optimization_note": self.optimization_note,
            "plateau_w_rf": [float(min(item["w_rf"] for item in plateau)), float(max(item["w_rf"] for item in plateau))],
            "plateau_size": len(plateau),
            "history": [
                {key: (round(value, 4) if isinstance(value, float) else value) for key, value in item.items()}
                for item in history
            ],
        }

    def weight_stability(self, X_val, y_val, metric="macro_f1", high_risk_idx=2, n_boot=40, seed=42):
        """Refit only the weight grid on bootstrap draws of the validation set. Base models stay fixed."""
        y_val = np.asarray(y_val)
        rng = np.random.default_rng(seed)
        p_rf = self.rf_model.predict_proba(X_val)
        p_xgb = self.xgb_model.predict_proba(X_val)
        chosen = []
        n = len(y_val)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if high_risk_idx is not None and high_risk_idx not in y_val[idx]:
                continue
            if len(np.unique(y_val[idx])) < 2:
                continue
            probe = AdaptiveWeightedHybridClassifier(
                rf_model=_FrozenProba(p_rf[idx]),
                xgb_model=_FrozenProba(p_xgb[idx]),
                classes=self.classes_,
            )
            result = probe.optimize_weights(np.zeros((len(idx), 1)), y_val[idx], metric=metric, high_risk_idx=high_risk_idx)
            chosen.append(result["best_w_rf"])
        arr = np.asarray(chosen, dtype=float)
        if len(arr) == 0:
            return {"n_boot_used": 0}
        return {
            "n_boot_used": int(len(arr)),
            "w_rf_mean": round(float(arr.mean()), 4),
            "w_rf_std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "w_rf_ci95": [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
            "fraction_near_xgb_only": round(float(np.mean(arr <= 0.05)), 4),
            "fraction_near_equal": round(float(np.mean(np.abs(arr - 0.5) <= 0.1)), 4),
        }

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        return joblib.load(file_path)


class _FrozenProba:
    """Validation-bootstrap helper. predict_proba ignores X and returns a fixed matrix."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return self.probabilities
