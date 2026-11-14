"""
Clinical reference logic and plain-language explanations.

These helpers turn raw measurements into a screening status and a short
interpretation. The intervals below are common adult pregnancy reference
ranges used for *screening context only*. They are not diagnostic thresholds
and they are not a substitute for laboratory reference intervals.
"""

from app.theme import PALETTE

# ---------------------------------------------------------------------------
# Reference intervals
# ---------------------------------------------------------------------------
# ``low``/``high`` are the inclusive normal band. ``critical_low`` and
# ``critical_high`` optionally mark values that warrant urgent review.
REFERENCES = {
    "Age": {
        "label": "Maternal age",
        "unit": "years",
        "low": 18.0,
        "high": 35.0,
        "note": "Age below 18 or above 35 is associated with higher obstetric risk.",
    },
    "SystolicBP": {
        "label": "Systolic blood pressure",
        "unit": "mmHg",
        "low": 90.0,
        "high": 120.0,
        "critical_high": 140.0,
        "note": "Values at or above 140 mmHg suggest gestational hypertension or pre-eclampsia.",
    },
    "DiastolicBP": {
        "label": "Diastolic blood pressure",
        "unit": "mmHg",
        "low": 60.0,
        "high": 80.0,
        "critical_high": 90.0,
        "note": "Values at or above 90 mmHg suggest gestational hypertension or pre-eclampsia.",
    },
    "BS": {
        "label": "Blood sugar",
        "unit": "mmol/L",
        "low": 4.0,
        "high": 7.8,
        "critical_high": 11.0,
        "note": "Elevated fasting glucose may indicate gestational diabetes.",
    },
    "BodyTemp": {
        "label": "Body temperature",
        "unit": "\u00b0F",
        "low": 97.8,
        "high": 99.0,
        "critical_high": 100.4,
        "note": "Temperature at or above 100.4 \u00b0F indicates fever.",
    },
    "HeartRate": {
        "label": "Heart rate",
        "unit": "bpm",
        "low": 60.0,
        "high": 90.0,
        "note": "Resting heart rate often rises slightly during pregnancy.",
    },
    "Hemoglobin": {
        "label": "Hemoglobin",
        "unit": "g/dL",
        "low": 11.0,
        "high": 16.0,
        "critical_high": None,
        "note": "Pregnancy anaemia screening cut-off is 11.0 g/dL (WHO).",
    },
    "MCV": {
        "label": "Mean corpuscular volume",
        "unit": "fL",
        "low": 80.0,
        "high": 100.0,
        "note": "Below 80 fL suggests microcytosis, above 100 fL macrocytosis.",
    },
    "MCH": {
        "label": "Mean corpuscular hemoglobin",
        "unit": "pg",
        "low": 27.0,
        "high": 33.0,
        "note": "Low values suggest hypochromia.",
    },
    "MCHC": {
        "label": "Mean corpuscular hemoglobin concentration",
        "unit": "g/dL",
        "low": 32.0,
        "high": 36.0,
        "note": "Low values suggest hypochromia.",
    },
    "Gender": {
        "label": "Recorded sex code",
        "unit": "",
        "low": None,
        "high": None,
        "note": "0 = female, 1 = male in the source file.",
    },
}


def evaluate(feature, value):
    """Return ``(status_key, status_text)`` for one measurement.

    ``status_key`` is one of ``normal``, ``warning`` or ``critical``.
    """
    ref = REFERENCES.get(feature)
    if ref is None or value is None or ref.get("low") is None:
        return "info", "No reference interval"

    low, high = ref["low"], ref["high"]
    critical_high = ref.get("critical_high")

    if value < low:
        return "warning", "Below reference interval"
    if critical_high is not None and value >= critical_high:
        return "critical", "Above critical threshold"
    if value > high:
        return "warning", "Above reference interval"
    return "normal", "Within reference interval"


def interpret(feature, value):
    """Short clinical sentence for one measurement."""
    ref = REFERENCES.get(feature, {})
    note = ref.get("note", "")
    status, text = evaluate(feature, value)
    if status == "normal":
        return note
    return f"{text}. {note}".strip()


def anemia_severity(hemoglobin, pregnant=True):
    """WHO-based haemoglobin category used for screening context."""
    if hemoglobin is None:
        return "Unknown", "info", "No hemoglobin value available."
    cutoff = 11.0 if pregnant else 12.0
    if hemoglobin < 7.0:
        return "Severe anemia", "critical", f"Hb < 7.0 g/dL \u2014 below the {cutoff} g/dL cut-off."
    if hemoglobin < 10.0:
        return "Moderate anemia", "critical", f"Hb 7.0\u20139.9 g/dL \u2014 below the {cutoff} g/dL cut-off."
    if hemoglobin < cutoff:
        return "Mild anemia", "warning", f"Hb 10.0\u2013{cutoff - 0.1:.1f} g/dL \u2014 below the {cutoff} g/dL cut-off."
    return "Non-anemic range", "normal", f"Hb at or above the {cutoff} g/dL cut-off."


def morphology(mcv):
    """Red-cell morphology category derived from MCV."""
    if mcv is None:
        return "Not available", "info"
    if mcv < 80:
        return "Microcytic (low MCV)", "warning"
    if mcv > 100:
        return "Macrocytic (high MCV)", "warning"
    return "Normocytic (MCV 80\u2013100 fL)", "normal"


def anemia_screening(hemoglobin, mcv, pregnant=True):
    """Rule-based anaemia screening verdict used as the clinical headline.

    Combines the WHO haemoglobin category with red-cell morphology (MCV) into a
    single clinically readable verdict, e.g. ``"Anaemic — moderate, microcytic"``.
    This is the primary screening result shown to the clinician; the trained
    index-only model is reported separately as a research baseline.

    Returns ``(label, headline, status_key, detail)``.
    """
    severity, status, severity_note = anemia_severity(hemoglobin, pregnant=pregnant)
    morph, _ = morphology(mcv)
    short_morph = morph.split(" (")[0].lower()
    if severity == "Unknown":
        return "Not available", "Not available", "info", severity_note
    if severity == "Non-anemic range":
        return ("Non-anaemic", f"Non-anaemic — {short_morph}", "normal",
                f"{severity_note} Red-cell morphology {short_morph}.")
    level = severity.replace(" anemia", "").lower()
    return ("Anaemic", f"Anaemic — {level}, {short_morph}", status,
            f"{severity_note} Red-cell morphology {short_morph}.")


def risk_plain_language(risk_class, probabilities):
    """Patient-facing sentence for the maternal risk output."""
    p = {name: float(value) for name, value in zip(["Low Risk", "Mid Risk", "High Risk"], probabilities)}
    if risk_class == "High Risk":
        return (
            "The model places this record in the high-risk band. Review blood pressure, "
            "glucose and vital-sign trends promptly with an obstetric clinician."
        )
    if risk_class == "Mid Risk":
        return (
            "The model places this record in the mid-risk band. Closer ante-natal "
            "surveillance is reasonable, especially for blood pressure and glucose."
        )
    return (
        "The model places this record in the low-risk band. Routine ante-natal care "
        "is consistent with the recorded measurements."
    )


def build_vitals_rows(measurements):
    """Build the measurement scorecard rows shared by the UI and the report.

    ``measurements`` maps internal feature names to numeric values.
    """
    rows = []
    for feature, value in measurements.items():
        ref = REFERENCES.get(feature)
        if ref is None or value is None or ref.get("low") is None:
            continue
        status, _ = evaluate(feature, value)
        rows.append({
            "parameter": ref["label"],
            "value": value,
            "unit": ref["unit"],
            "reference": f"{ref['low']:g} \u2013 {ref['high']:g}",
            "status": status,
            "interpretation": interpret(feature, value),
        })
    return rows


RISK_COLORS = {
    "Low Risk": PALETTE["green"],
    "Mid Risk": PALETTE["amber"],
    "High Risk": PALETTE["red"],
}

STATUS_COLORS = {
    "normal": PALETTE["green"],
    "warning": PALETTE["amber"],
    "critical": PALETTE["red"],
    "info": PALETTE["blue"],
}
