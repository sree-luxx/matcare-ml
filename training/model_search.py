"""Hyperparameter search and repeated stratified evaluation.

Search scoring uses only the rows passed in. Callers must pass the training
split, not the held-out test split. Preprocessing is inside each fold.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from evaluation.metrics import compute_comprehensive_metrics


def _pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def search_random_forest(X_train, y_train, n_iter=12, seed=42):
    model = RandomForestClassifier(random_state=seed, n_jobs=-1)
    space = {
        "model__n_estimators": [120, 200, 320],
        "model__max_depth": [4, 6, 8, 12, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", 0.5, 0.8],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }
    search = RandomizedSearchCV(
        _pipeline(model),
        space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    params = {key.split("__", 1)[1]: value for key, value in search.best_params_.items()}
    params.update({"random_state": seed, "n_jobs": -1})
    return params, float(search.best_score_)


def search_xgboost(X_train, y_train, n_iter=12, seed=42, binary=False):
    model = XGBClassifier(
        random_state=seed,
        objective="binary:logistic" if binary else "multi:softprob",
        eval_metric="logloss" if binary else "mlogloss",
        tree_method="hist",
        n_jobs=1,
    )
    space = {
        "model__n_estimators": [80, 150, 250],
        "model__max_depth": [2, 3, 4, 6],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1.0, 3.0, 8.0],
        "model__reg_alpha": [0.0, 0.1],
    }
    weights = compute_sample_weight("balanced", y_train)
    search = RandomizedSearchCV(
        _pipeline(model),
        space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, model__sample_weight=weights)
    params = {key.split("__", 1)[1]: value for key, value in search.best_params_.items()}
    params.update({
        "random_state": seed,
        "objective": "binary:logistic" if binary else "multi:softprob",
        "eval_metric": "logloss" if binary else "mlogloss",
        "tree_method": "hist",
        "n_jobs": 1,
    })
    return params, float(search.best_score_)


def fit_preprocessor(X_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipeline.fit(X_train)
    return pipeline


def fit_final_models(X_train, y_train, rf_params, xgb_params):
    preprocessor = fit_preprocessor(X_train)
    scaled = preprocessor.transform(X_train)
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(scaled, y_train)
    xgb_model = XGBClassifier(**xgb_params)
    weights = compute_sample_weight("balanced", y_train)
    xgb_model.fit(scaled, y_train, sample_weight=weights)
    return preprocessor, rf_model, xgb_model


def transform_frame(preprocessor, frame, columns):
    scaled = preprocessor.transform(frame[columns])
    return pd.DataFrame(scaled, columns=columns, index=frame.index)


def repeated_split_scores(
    X,
    y,
    rf_params,
    xgb_params,
    class_names,
    high_risk_idx=None,
    n_splits=5,
    n_repeats=5,
    seed=42,
):
    """Fixed hyperparameters, with imputation and scaling refit inside every fold."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    records = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        preprocessor, rf_model, xgb_model = fit_final_models(
            X.iloc[train_idx], y.iloc[train_idx], rf_params, xgb_params
        )
        scaled = transform_frame(preprocessor, X.iloc[test_idx], list(X.columns))
        y_fold = y.iloc[test_idx].to_numpy()
        for name, model in (("Random Forest", rf_model), ("XGBoost", xgb_model)):
            probabilities = model.predict_proba(scaled)
            predictions = np.argmax(probabilities, axis=1)
            metrics = compute_comprehensive_metrics(y_fold, predictions, probabilities, class_names, high_risk_idx)
            records.append({
                "fold": fold,
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "focus_recall": metrics.get("high_risk_recall"),
            })
    frame = pd.DataFrame(records)
    summary = {}
    for model_name, part in frame.groupby("model"):
        summary[model_name] = {
            "folds": int(len(part)),
            "accuracy_mean": round(float(part["accuracy"].mean()), 2),
            "accuracy_std": round(float(part["accuracy"].std(ddof=1)), 2),
            "macro_f1_mean": round(float(part["macro_f1"].mean()), 2),
            "macro_f1_std": round(float(part["macro_f1"].std(ddof=1)), 2),
            "focus_recall_mean": round(float(part["focus_recall"].mean()), 2),
            "focus_recall_std": round(float(part["focus_recall"].std(ddof=1)), 2),
        }
    return summary
