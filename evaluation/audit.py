"""Dataset audits used by training and tests. These functions do not fit models."""

import numpy as np
import pandas as pd


MATERNAL_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
ANEMIA_FEATURES = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
INDEX_FEATURES = ["MCV", "MCH", "MCHC"]


def dataset_anemia_label(gender, hemoglobin):
    """
    Reproduce the binary file's label rule.

    Result is 1 when hemoglobin is below 12.0 for Gender 0 and below 13.5 for Gender 1.
    This is a property of anemia_cbc.csv. It is not a pregnancy diagnosis rule.
    """
    gender = np.asarray(gender)
    hemoglobin = np.asarray(hemoglobin, dtype=float)
    threshold = np.where(gender == 0, 12.0, 13.5)
    return (hemoglobin < threshold).astype(int)


def audit_anemia_frame(df):
    labels = dataset_anemia_label(df["Gender"], df["Hemoglobin"])
    match = float(np.mean(labels == df["Result"].to_numpy()))
    feature_groups = df.groupby(ANEMIA_FEATURES, dropna=False)["Result"].nunique()
    return {
        "rows": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_rows": int((~df.duplicated()).sum()),
        "gender_counts": {str(k): int(v) for k, v in df["Gender"].value_counts().to_dict().items()},
        "result_counts": {str(k): int(v) for k, v in df["Result"].value_counts().to_dict().items()},
        "rule": "Result = 1 if (Gender==0 and Hemoglobin<12.0) or (Gender==1 and Hemoglobin<13.5), else 0",
        "rule_match_rate": round(match, 6),
        "rule_is_exact": bool(np.isclose(match, 1.0)),
        "conflicting_feature_groups": int((feature_groups > 1).sum()),
        "hemoglobin_result_correlation": round(float(df["Hemoglobin"].corr(df["Result"])), 4),
        "columns": list(df.columns),
        "rbc_present": "RBC" in df.columns,
        "limitation": (
            "The binary target is an exact sex-specific hemoglobin threshold. "
            "A model that receives hemoglobin is reproducing that rule. "
            "The file has no pregnancy indicator, no patient identifier, and no RBC column. "
            "Duplicate rows are exact copies, not verified repeat visits."
        ),
    }


def prepare_anemia_modeling_frame(df, drop_duplicates=True):
    audit = audit_anemia_frame(df)
    frame = df.copy()
    if drop_duplicates:
        frame = frame.drop_duplicates().reset_index(drop=True)
    group_n = frame.groupby(ANEMIA_FEATURES, dropna=False)["Result"].transform("nunique")
    conflict_rows = int((group_n > 1).sum())
    frame = frame.loc[group_n == 1].reset_index(drop=True)
    audit["rows_after_dedup_and_conflict_removal"] = int(len(frame))
    audit["conflict_rows_removed"] = conflict_rows
    return frame, audit


def audit_maternal_frame(df):
    work = df.copy()
    work["RiskLevelClean"] = work["RiskLevel"].astype(str).str.strip().str.lower()
    groups = work.groupby(MATERNAL_FEATURES, dropna=False)["RiskLevelClean"].nunique()
    return {
        "rows": int(len(work)),
        "duplicate_rows": int(work.duplicated().sum()),
        "unique_rows": int((~work.duplicated()).sum()),
        "class_counts": work["RiskLevelClean"].value_counts().to_dict(),
        "heart_rate_below_30": int((work["HeartRate"] < 30).sum()),
        "feature_groups": int(len(groups)),
        "conflicting_feature_groups": int((groups > 1).sum()),
        "limitation": (
            "Exact duplicate rows are common. Keeping them before a random split places identical "
            "records on both sides of the split. Some distinct feature vectors also carry more than one risk label."
        ),
    }


def prepare_maternal_modeling_frame(df):
    """
    Modeling table for the revised protocol.

    Invalid heart rates stay missing so the training-fold median can impute them.
    Exact duplicate rows are removed. Feature vectors with more than one risk label are removed
    because no classifier can match both labels.
    """
    audit = audit_maternal_frame(df)
    work = df.copy()
    work["RiskLevelClean"] = work["RiskLevel"].astype(str).str.strip().str.lower()
    invalid = work["HeartRate"] < 30
    audit["invalid_heart_rate_rows"] = int(invalid.sum())
    work.loc[invalid, "HeartRate"] = np.nan
    before = len(work)
    work = work.drop_duplicates().copy()
    audit["rows_removed_as_exact_duplicates"] = int(before - len(work))
    group_n = work.groupby(MATERNAL_FEATURES, dropna=False)["RiskLevelClean"].transform("nunique")
    audit["conflict_rows_removed"] = int((group_n > 1).sum())
    work = work.loc[group_n == 1].reset_index(drop=True)
    audit["rows_used_for_modeling"] = int(len(work))
    audit["class_counts_modeling"] = work["RiskLevelClean"].value_counts().to_dict()
    return work, audit
