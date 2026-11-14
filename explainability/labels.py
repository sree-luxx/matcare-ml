"""Display names for model features. These labels do not change the trained columns."""

CLINICAL_FEATURE_LABELS = {
    "Age": "Age",
    "SystolicBP": "Systolic blood pressure",
    "DiastolicBP": "Diastolic blood pressure",
    "BS": "Blood sugar",
    "BodyTemp": "Body temperature",
    "HeartRate": "Heart rate",
    "Gender": "Recorded sex code",
    "Hemoglobin": "Hemoglobin",
    "MCH": "Mean corpuscular hemoglobin",
    "MCHC": "Mean corpuscular hemoglobin concentration",
    "MCV": "Mean corpuscular volume",
}


def clinical_label(feature_name):
    return CLINICAL_FEATURE_LABELS.get(feature_name, feature_name)
