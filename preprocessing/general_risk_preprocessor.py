"""
MATCARE-AI: General Maternal Health Risk Preprocessing Pipeline
Handles data cleaning, physiological anomaly correction, stratified 3-way split (Train/Val/Test),
and data-leakage-free standardization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
TARGET_NAME = 'RiskLevel'
TARGET_MAPPING = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
REVERSE_MAPPING = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}
CLASS_NAMES = ['Low Risk', 'Mid Risk', 'High Risk']


def load_and_inspect_general_risk(file_path):
    """
    Loads raw UCI dataset, returns basic statistics and class counts.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Normalize string target
    df['RiskLevelClean'] = df[TARGET_NAME].astype(str).str.strip().str.lower()
    return df


def clean_anomalies(df):
    """
    Cleans physiological anomalies in Maternal Health Risk data.
    Known issue in UCI dataset: HeartRate = 7 bpm (impossible in living resting humans).
    Replaces values < 30 bpm with NaN for train-median imputation.
    """
    df_clean = df.copy()
    if 'HeartRate' in df_clean.columns:
        invalid_hr = df_clean['HeartRate'] < 30
        df_clean.loc[invalid_hr, 'HeartRate'] = np.nan
    return df_clean


def get_general_risk_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits into stratified Train, Validation, and Held-out Test sets.
    StandardScaler is fitted strictly on Train to prevent data leakage.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata
    """
    df_clean = clean_anomalies(df)
    
    X = df_clean[FEATURE_NAMES].copy()
    y = df_clean['RiskLevelClean'].map(TARGET_MAPPING)
    
    if y.isnull().any():
        raise ValueError(f"Unmapped target labels: {df_clean['RiskLevelClean'].unique()}")
        
    # First split: Train+Val vs Test
    total_holdout = test_size + val_size
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: Train vs Val
    val_ratio_of_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_of_train_val,
        random_state=random_state, stratify=y_train_val
    )
    
    # Impute missing values using train medians strictly
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    
    # Standardize: fit strictly on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_scaled, columns=FEATURE_NAMES, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES, index=X_test.index)
    
    metadata = {
        'feature_names': FEATURE_NAMES,
        'class_names': CLASS_NAMES,
        'target_mapping': TARGET_MAPPING,
        'reverse_mapping': REVERSE_MAPPING,
        'train_medians': train_medians.to_dict(),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test)
    }
    
    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, scaler, metadata
