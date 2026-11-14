"""
Clinical and statistical metrics for MATCARE-AI.

Scores are proportions in the returned raw fields and percentages in the
display fields. Nothing in this module looks at a test set during fitting.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_float(value, digits=4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def expected_calibration_error(y_true, y_prob, n_bins=8):
    """Top-label expected calibration error. This is a model-score check, not a clinical calibration."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2 or len(y_true) == 0:
        return None
    confidence = y_prob.max(axis=1)
    predicted = y_prob.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidence[mask].mean())
        total += (count / len(y_true)) * abs(bin_acc - bin_conf)
    return total


def compute_comprehensive_metrics(y_true, y_pred, y_prob, class_names, high_risk_idx=None):
    """
    Accuracy, precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC,
    false-positive and false-negative rates, and calibration summaries.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    prec_per = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    roc_auc = None
    pr_auc = None
    try:
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            pr_auc = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels))
            pr_parts = []
            for idx in labels:
                pr_parts.append(average_precision_score((y_true == idx).astype(int), y_prob[:, idx]))
            pr_auc = float(np.mean(pr_parts))
    except Exception:
        roc_auc = None
        pr_auc = None

    brier = None
    try:
        if n_classes == 2:
            brier = float(brier_score_loss(y_true, y_prob[:, 1]))
        else:
            one_hot = np.eye(n_classes)[y_true.astype(int)]
            brier = float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))
    except Exception:
        brier = None

    try:
        ece = expected_calibration_error(y_true, y_prob)
    except Exception:
        ece = None
    try:
        ll = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1.0), labels=labels))
    except Exception:
        ll = None

    per_class = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        per_class[name] = {
            "precision": round(float(prec_per[i]) * 100, 2),
            "recall": round(float(rec_per[i]) * 100, 2),
            "f1_score": round(float(f1_per[i]) * 100, 2),
            "false_negative_rate": round((fn / (fn + tp)) * 100, 2) if (fn + tp) else 0.0,
            "false_positive_rate": round((fp / (fp + tn)) * 100, 2) if (fp + tn) else 0.0,
            "support": int(cm[i, :].sum()),
        }

    metrics = {
        "accuracy": round(float(acc) * 100, 2),
        "macro_precision": round(float(prec_macro) * 100, 2),
        "macro_recall": round(float(rec_macro) * 100, 2),
        "macro_f1": round(float(f1_macro) * 100, 2),
        "weighted_f1": round(float(f1_weighted) * 100, 2),
        "roc_auc": _safe_float(roc_auc, 4),
        "pr_auc": _safe_float(pr_auc, 4),
        "brier": _safe_float(brier, 4),
        "ece": _safe_float(ece, 4),
        "log_loss": _safe_float(ll, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    if high_risk_idx is not None and 0 <= high_risk_idx < n_classes:
        name = class_names[high_risk_idx]
        metrics["high_risk_recall"] = per_class[name]["recall"]
        metrics["high_risk_precision"] = per_class[name]["precision"]
        metrics["high_risk_fpr"] = per_class[name]["false_positive_rate"]
        metrics["high_risk_fnr"] = per_class[name]["false_negative_rate"]
    return metrics


def compare_four_approaches(rf_model, xgb_model, adaptive_hybrid, X_test, y_test, class_names, high_risk_idx=None):
    """Random Forest, XGBoost, equal-weight voting, and the validation-chosen hybrid."""
    y_test = np.asarray(y_test)
    prob_rf = rf_model.predict_proba(X_test)
    pred_rf = np.argmax(prob_rf, axis=1)
    metrics_rf = compute_comprehensive_metrics(y_test, pred_rf, prob_rf, class_names, high_risk_idx)
    metrics_rf["model_name"] = "Random Forest"

    prob_xgb = xgb_model.predict_proba(X_test)
    pred_xgb = np.argmax(prob_xgb, axis=1)
    metrics_xgb = compute_comprehensive_metrics(y_test, pred_xgb, prob_xgb, class_names, high_risk_idx)
    metrics_xgb["model_name"] = "XGBoost"

    prob_equal = 0.5 * prob_rf + 0.5 * prob_xgb
    prob_equal = prob_equal / np.sum(prob_equal, axis=1, keepdims=True)
    pred_equal = np.argmax(prob_equal, axis=1)
    metrics_equal = compute_comprehensive_metrics(y_test, pred_equal, prob_equal, class_names, high_risk_idx)
    metrics_equal["model_name"] = "Equal-weight voting (50/50)"

    prob_hybrid = adaptive_hybrid.predict_proba(X_test)
    pred_hybrid = np.asarray(adaptive_hybrid.predict(X_test))
    metrics_hybrid = compute_comprehensive_metrics(y_test, pred_hybrid, prob_hybrid, class_names, high_risk_idx)
    metrics_hybrid["model_name"] = (
        f"Adaptive hybrid (RF {adaptive_hybrid.w_rf * 100:.1f}%, XGB {adaptive_hybrid.w_xgb * 100:.1f}%)"
    )

    return [metrics_rf, metrics_xgb, metrics_equal, metrics_hybrid]


def bootstrap_confidence_intervals(y_true, y_pred, y_prob, class_names, high_risk_idx=None, n_boot=1000, seed=42):
    """Percentile intervals from resampling the already-scored evaluation rows. Not a substitute for external validation."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    if high_risk_idx is not None:
        keys.extend(["high_risk_recall", "high_risk_precision", "high_risk_fpr"])
    collected = {key: [] for key in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics = compute_comprehensive_metrics(
            y_true[idx], y_pred[idx], y_prob[idx], class_names, high_risk_idx
        )
        for key in keys:
            if metrics.get(key) is not None:
                collected[key].append(metrics[key])
    summary = {"n_boot_used": min(len(v) for v in collected.values()) if collected else 0}
    for key, values in collected.items():
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            continue
        summary[key] = {
            "mean": round(float(arr.mean()), 2),
            "ci95_low": round(float(np.percentile(arr, 2.5)), 2),
            "ci95_high": round(float(np.percentile(arr, 97.5)), 2),
        }
    return summary


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
