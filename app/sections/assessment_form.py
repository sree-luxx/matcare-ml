"""Patient assessment page: structured form, model execution and persistence.

The form collects the patient profile plus both measurement blocks, runs the
two independent screening modules when the clinician submits, and links the
result to a unique patient ID when it is saved.
"""

import datetime

import streamlit as st

from app import storage
from app.clinical import REFERENCES, build_vitals_rows
from app.inference import MODEL_LABELS, model_agreement, predict_module
from app.theme import card, page_head, tag
from explainability.shap_explainers import explain_single_patient


# Presets carry clinical values only. Patient identity is never preset: reusing
# a preset's patient ID would merge different people into one profile.
PRESETS = {
    "Severe risk — pre-eclampsia with microcytic anaemia": {
        "values": {
            "p_age": 36, "p_ga": "32 weeks",
            "p_sys": 155, "p_dia": 98, "p_bs": 15.2, "p_temp": 99.4, "p_hr": 92,
            "p_gender": 0, "p_hb": 8.2, "p_rbc": 3.4, "p_mcv": 72.0, "p_mch": 21.0, "p_mchc": 28.5,
        },
    },
    "Healthy — low risk, normal CBC": {
        "values": {
            "p_age": 24, "p_ga": "20 weeks",
            "p_sys": 110, "p_dia": 72, "p_bs": 6.8, "p_temp": 98.0, "p_hr": 74,
            "p_gender": 0, "p_hb": 13.8, "p_rbc": 4.7, "p_mcv": 89.0, "p_mch": 29.5, "p_mchc": 33.5,
        },
    },
    "Moderate risk — borderline hypertension, mild anaemia": {
        "values": {
            "p_age": 31, "p_ga": "26 weeks",
            "p_sys": 128, "p_dia": 84, "p_bs": 8.4, "p_temp": 98.6, "p_hr": 80,
            "p_gender": 0, "p_hb": 10.6, "p_rbc": 4.2, "p_mcv": 84.5, "p_mch": 27.2, "p_mchc": 32.0,
        },
    },
    "Custom record — enter values below": {
        "values": {
            "p_age": 28, "p_ga": "24 weeks",
            "p_sys": 120, "p_dia": 80, "p_bs": 7.5, "p_temp": 98.0, "p_hr": 76,
            "p_gender": 0, "p_hb": 12.0, "p_rbc": 4.5, "p_mcv": 86.0, "p_mch": 28.0, "p_mchc": 32.5,
        },
    },
}

GESTATIONAL_AGES = [f"{week} weeks" for week in range(4, 43)]

BLANK_PRESET = "Custom record — enter values below"
DEFAULT_PRESET = next(iter(PRESETS))

ID_PREFIX = "PT-"
ID_START = 1001


# ---------------------------------------------------------------------------
# Patient identity
# ---------------------------------------------------------------------------
def next_patient_id():
    """Return the next unused patient ID (PT-1001, PT-1002, ...)."""
    used = set()
    for patient in storage.list_patients():
        raw = str(patient["patient_id"] or "")
        if raw.startswith(ID_PREFIX) and raw[len(ID_PREFIX):].isdigit():
            used.add(int(raw[len(ID_PREFIX):]))
    candidate = ID_START
    while candidate in used:
        candidate += 1
    return f"{ID_PREFIX}{candidate}"


def identity_error(patient_id, patient_name):
    """Return an actionable message when the patient identity is unusable.

    Guards the exact failure that silently merged different people: a new name
    entered against a patient ID that already belongs to somebody else.
    """
    if not patient_name or not patient_id:
        return ("Enter both a patient name and a unique patient ID before submitting "
                "the assessment.")
    existing = storage.get_latest_record(patient_id)
    stored_name = (existing or {}).get("patient_name") or ""
    if stored_name and stored_name.strip().lower() != patient_name.lower():
        return (
            f"Patient ID {patient_id} already belongs to “{stored_name}”. "
            "Choose “New patient” to register a new patient, or correct the ID."
        )
    return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def seed_defaults():
    """Populate the form widgets once per session."""
    for key, value in PRESETS[DEFAULT_PRESET]["values"].items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("preset", DEFAULT_PRESET)
    st.session_state.setdefault("p_name", "")
    if "p_id" not in st.session_state:
        st.session_state["p_id"] = next_patient_id()


def _help(feature, extra=""):
    """Build widget help text from the clinical reference intervals."""
    ref = REFERENCES.get(feature, {})
    parts = []
    if ref.get("low") is not None:
        parts.append(f"Reference interval: {ref['low']:g}–{ref['high']:g} {ref['unit']}".strip())
    if ref.get("note"):
        parts.append(ref["note"])
    if extra:
        parts.append(extra)
    return " ".join(parts)


def apply_preset():
    for key, value in PRESETS[st.session_state["preset"]]["values"].items():
        st.session_state[key] = value


def start_new_patient():
    """Clear the form so a brand-new patient can be registered.

    A fresh unused patient ID is assigned immediately, so a new profile can
    never be registered under an ID that already belongs to somebody else.
    """
    st.session_state["preset"] = BLANK_PRESET
    for key, value in PRESETS[BLANK_PRESET]["values"].items():
        st.session_state[key] = value
    st.session_state["p_name"] = ""
    st.session_state["p_id"] = next_patient_id()
    st.session_state["live_context"] = None


def _goto_profiles():
    st.session_state["page"] = "Patient Profiles"


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
def render(on_submit=None):
    """Draw the structured assessment form.

    ``on_submit`` is the callback wired to the submit button; it runs both
    screening modules and routes to the Results page.
    """
    st.markdown(
        page_head(
            "Patient assessment",
            "Enter the patient profile and both measurement blocks, then submit to run the "
            "general maternal risk and anaemia modules independently.",
        ),
        unsafe_allow_html=True,
    )

    controls = st.columns([3, 2, 2])
    with controls[0]:
        st.selectbox(
            "Profile preset", list(PRESETS), key="preset", on_change=apply_preset,
            help="Load a worked example, or choose “Custom record” to start from a blank template.",
        )
    with controls[1]:
        st.markdown("<div class='mc-spacer' style='height:28px;'></div>", unsafe_allow_html=True)
        st.button("New patient", use_container_width=True, on_click=start_new_patient, key="new_patient")
    with controls[2]:
        st.markdown("<div class='mc-spacer' style='height:28px;'></div>", unsafe_allow_html=True)
        st.button("Patient profiles", use_container_width=True, key="goto_profiles",
                  on_click=_goto_profiles)

    _profile_block()
    _measurement_block()

    st.markdown("<div class='mc-spacer'></div>", unsafe_allow_html=True)

    st.button("Submit assessment", type="primary", use_container_width=True,
              key="submit_assessment", on_click=on_submit)
    if st.session_state.get("live_context"):
        st.caption("An assessment is ready — open **Results** from the sidebar to review it.")


def _profile_block():
    st.markdown("<div class='mc-spacer'></div>", unsafe_allow_html=True)
    st.markdown(tag("Every assessment is grouped under a unique patient ID"), unsafe_allow_html=True)
    st.markdown("<div class='mc-spacer' style='height:8px;'></div>", unsafe_allow_html=True)
    columns = st.columns(4)
    with columns[0]:
        st.text_input("Patient name", key="p_name", help="Patient name or identifier for the saved record.")
    with columns[1]:
        st.text_input("Patient ID", key="p_id",
                      help="Auto-assigned unique profile ID. Keep it to record another "
                           "assessment for the same patient, or choose “New patient” for a "
                           "new profile.")
    with columns[2]:
        st.slider("Maternal age (years)", 12, 60, key="p_age", help=_help("Age"))
    with columns[3]:
        st.selectbox("Gestational age", GESTATIONAL_AGES, key="p_ga",
                     help="Recorded for context only. The trained models do not use gestational age.")


def _measurement_block():
    left, right = st.columns(2)

    with left:
        st.markdown(
            card("1. General maternal health risk",
                 "Six vital-sign inputs — these are the model inputs for the maternal risk module", ""),
            unsafe_allow_html=True,
        )
        st.slider("Systolic blood pressure (mmHg)", 70, 200, key="p_sys", help=_help("SystolicBP"))
        st.slider("Diastolic blood pressure (mmHg)", 40, 130, key="p_dia", help=_help("DiastolicBP"))
        st.number_input("Blood sugar (mmol/L)", 2.0, 30.0, step=0.1, key="p_bs", help=_help("BS"))
        st.number_input("Body temperature (°F)", 95.0, 105.0, step=0.1, key="p_temp", help=_help("BodyTemp"))
        st.slider("Heart rate (bpm)", 40, 140, key="p_hr", help=_help("HeartRate"))

    with right:
        st.markdown(
            card("2. Anaemia assessment",
                 "Index-only model: MCV, MCH and MCHC drive the prediction", ""),
            unsafe_allow_html=True,
        )
        st.number_input("MCV (fL)", 50.0, 125.0, step=0.5, key="p_mcv", help=_help("MCV"))
        st.number_input("MCH (pg)", 12.0, 45.0, step=0.1, key="p_mch", help=_help("MCH"))
        st.number_input("MCHC (g/dL)", 20.0, 45.0, step=0.1, key="p_mchc", help=_help("MCHC"))

        st.markdown(
            f"{tag('Context only — not used by the model', context=True)}",
            unsafe_allow_html=True,
        )
        st.number_input("Haemoglobin (g/dL)", 4.0, 20.0, step=0.1, key="p_hb",
                        help=_help("Hemoglobin", "Used for the screening category, not by the model."))
        st.number_input("RBC count (×10¹²/L)", 1.0, 8.0, step=0.1, key="p_rbc",
                        help=_help("RBC", "The CBC file used for training has no RBC column, so this is "
                                         "recorded for context only."))
        st.selectbox("Recorded sex code (0 = female, 1 = male)", [0, 1], key="p_gender",
                     help="Recorded for context. Excluded from the model because the dataset label is a "
                          "sex-specific haemoglobin threshold.")
        st.caption(
            "The anaemia model deliberately excludes haemoglobin and the sex code: the dataset label "
            "equals a sex-specific haemoglobin cut-off, so including them would only reproduce that rule."
        )


# ---------------------------------------------------------------------------
# Inference and persistence
# ---------------------------------------------------------------------------
def build_context(bundle):
    """Run both modules for the current form values and assemble a page context."""
    maternal_inputs = {
        "Age": st.session_state["p_age"],
        "SystolicBP": st.session_state["p_sys"],
        "DiastolicBP": st.session_state["p_dia"],
        "BS": st.session_state["p_bs"],
        "BodyTemp": st.session_state["p_temp"],
        "HeartRate": st.session_state["p_hr"],
    }
    anemia_inputs = {
        "Gender": st.session_state["p_gender"],
        "Hemoglobin": st.session_state["p_hb"],
        "RBC": st.session_state["p_rbc"],
        "MCV": st.session_state["p_mcv"],
        "MCH": st.session_state["p_mch"],
        "MCHC": st.session_state["p_mchc"],
    }

    maternal_result = predict_module(bundle["maternal"], maternal_inputs, "maternal")
    anemia_result = predict_module(bundle["anemia"], anemia_inputs, "anemia")

    shap = {"maternal": None, "anemia": None}
    try:
        shap["maternal"] = explain_single_patient(
            bundle["maternal"]["rf"], bundle["maternal"]["xgb"],
            maternal_result["w_rf"], maternal_result["w_xgb"], maternal_result["scaled"].to_numpy(),
            maternal_inputs, maternal_result["feature_names"],
            class_idx=maternal_result["predicted_index"], class_name=maternal_result["predicted_class"],
        )
        shap["anemia"] = explain_single_patient(
            bundle["anemia"]["rf"], bundle["anemia"]["xgb"],
            anemia_result["w_rf"], anemia_result["w_xgb"], anemia_result["scaled"].to_numpy(),
            anemia_inputs, anemia_result["feature_names"],
            class_idx=anemia_result["predicted_index"], class_name=anemia_result["predicted_class"],
        )
    except Exception as error:  # pragma: no cover - defensive
        st.session_state["_shap_error"] = str(error)

    return {
        "bundle": bundle,
        "patient": {
            "name": st.session_state["p_name"],
            "id": st.session_state["p_id"],
            "age": st.session_state["p_age"],
            "gestational_age": st.session_state["p_ga"],
        },
        "assessment_date": datetime.datetime.now().strftime("%d %B %Y, %H:%M"),
        "maternal": maternal_result,
        "anemia": anemia_result,
        "maternal_inputs": maternal_inputs,
        "anemia_inputs": anemia_inputs,
        "vitals_rows": build_vitals_rows({**maternal_inputs, **anemia_inputs}),
        "shap": shap,
        "agreement": model_agreement(maternal_result, anemia_result),
        "model_labels": MODEL_LABELS,
    }


def save_current(context):
    """Persist one assessment. Returns ``(record_id, was_duplicate)``."""
    existing = storage.find_duplicate(
        context["patient"]["id"], context["maternal_inputs"], context["anemia_inputs"]
    )
    if existing:
        return existing, True
    record_id = storage.save_assessment(
        patient_id=context["patient"]["id"],
        patient_name=context["patient"]["name"],
        age=context["patient"]["age"],
        gestational_age=context["patient"]["gestational_age"],
        maternal_inputs=context["maternal_inputs"],
        cbc_inputs=context["anemia_inputs"],
        maternal_result=context["maternal"],
        anemia_result=context["anemia"],
        shap=context["shap"],
    )
    return record_id, False


__all__ = [
    "build_context",
    "identity_error",
    "next_patient_id",
    "render",
    "save_current",
    "seed_defaults",
    "start_new_patient",
]
