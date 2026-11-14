"""Patient profiles dashboard: one professional card per saved patient."""

import datetime

import streamlit as st

from app import storage
from app.clinical import anemia_screening
from app.sections.assessment_form import start_new_patient
from app.theme import empty_state, page_head, pill, profile_card


def fmt_date(iso_value):
    """Short, readable date for cards and headers."""
    try:
        return datetime.datetime.fromisoformat(iso_value).strftime("%d %b %Y · %H:%M")
    except Exception:  # pragma: no cover - defensive
        return iso_value or "—"


def risk_pill(risk_class):
    key = {"High Risk": "critical", "Mid Risk": "warning"}.get(risk_class, "normal")
    return pill(key, risk_class or "—")


def anemia_pill(cbc_inputs):
    """Rule-based anaemia screening pill for a saved assessment."""
    cbc = cbc_inputs or {}
    label, _, status, _ = anemia_screening(cbc.get("Hemoglobin"), cbc.get("MCV"))
    key = {"critical": "critical", "warning": "warning"}.get(status, "normal")
    return pill(key, label)


def _new_patient():
    start_new_patient()
    st.session_state["page"] = "Patient Assessment"


def open_patient(patient_id, record_id):
    st.session_state["selected_patient_id"] = patient_id
    st.session_state["dashboard_record_id"] = record_id
    st.session_state["page"] = "Patient Overview"


def _delete_assessment(record_id):
    """Callback: delete one saved assessment, then refresh the profiles list."""
    storage.delete_record(record_id)
    if st.session_state.get("dashboard_record_id") == record_id:
        st.session_state["dashboard_record_id"] = None
    st.session_state["_flash"] = "Assessment deleted."


def _delete_control(patient):
    """Confirm-gated control that removes a single saved assessment.

    One assessment at a time: the other assessments for the same patient are
    left untouched, matching the behaviour of the detailed patient dashboard.
    """
    count = patient["assessment_count"]
    assessments = ([patient["latest"]] if count == 1
                   else list(reversed(storage.get_history(patient["patient_id"]))))

    with st.expander("Delete a saved assessment"):
        if count == 1:
            st.caption("This is the patient's only saved assessment.")
        index = st.selectbox(
            "Assessment",
            options=range(len(assessments)),
            format_func=lambda i: (
                f"{fmt_date(assessments[i]['created_at'])} — "
                f"{assessments[i].get('maternal_prediction') or 'unknown risk'}"
            ),
            key=f"delete_pick_{patient['patient_id']}",
        )
        target = assessments[index]
        confirmed = st.checkbox("Confirm permanent deletion",
                                key=f"delete_confirm_{patient['patient_id']}")
        st.button(
            "Delete assessment",
            key=f"delete_button_{patient['patient_id']}",
            use_container_width=True,
            disabled=not confirmed,
            on_click=_delete_assessment,
            args=(target["id"],),
        )
        if count > 1 and target["id"] == patient["latest"]["id"]:
            st.caption("This is the most recent assessment; deleting it moves the "
                       "profile's latest result to the previous assessment.")


def render():
    st.markdown(
        page_head(
            "Patient profiles",
            "Every patient with at least one saved assessment, most recently assessed first.",
        ),
        unsafe_allow_html=True,
    )

    search_col, action_col = st.columns([3, 1])
    with search_col:
        query = st.text_input("Search patients", key="patient_search",
                              placeholder="Search by patient name or ID")
    with action_col:
        st.markdown("<div class='mc-spacer' style='height:28px;'></div>", unsafe_allow_html=True)
        st.button("New patient", use_container_width=True, key="profiles_new_patient",
                  on_click=_new_patient)

    patients = storage.list_patients(query=query or None)

    if not patients:
        if query:
            st.markdown(
                empty_state(f"No patient profile matches <b>{query}</b>."),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                empty_state(
                    "No patient profiles yet. Complete an assessment and choose "
                    "<b>Save assessment</b> to create the first profile."
                ),
                unsafe_allow_html=True,
            )
        return

    total = storage.count_records()
    st.caption(f"{len(patients)} patient(s) · {total} saved assessment(s)")

    columns = st.columns(3)
    for index, patient in enumerate(patients):
        latest = patient["latest"]
        with columns[index % 3]:
            st.markdown(
                profile_card(
                    latest["patient_name"],
                    patient["patient_id"],
                    [
                        ("Latest assessment", fmt_date(patient["last_assessment"])),
                        ("Maternal risk", risk_pill(latest.get("maternal_prediction"))),
                        ("Anaemia", anemia_pill(latest.get("cbc_inputs"))),
                        ("Assessments on file", str(patient["assessment_count"])),
                    ],
                    title=patient["patient_id"],
                ),
                unsafe_allow_html=True,
            )
            st.button(
                "Open overview",
                key=f"open_patient_{patient['patient_id']}",
                use_container_width=True,
                on_click=open_patient,
                args=(patient["patient_id"], latest["id"]),
            )
            _delete_control(patient)
