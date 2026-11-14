"""Results page: immediate output of both modules for the current assessment.

Shows the risk categories, probabilities, important measurements and plain
language explanations produced by the assessment the clinician just submitted,
together with the save action that adds it to the patient's history.
"""

import streamlit as st

from app import components
from app.sections import assessment
from app.sections.assessment_form import save_current
from app.theme import PALETTE, card, empty_state, page_head


def _save():
    """Callback for the save button: persist the live assessment, then open the
    patient's overview. The callback runs before the script body, so the
    overview renders in the same run."""
    context = st.session_state.get("live_context")
    if not context:
        return
    try:
        record_id, duplicate = save_current(context)
    except Exception as error:  # pragma: no cover - defensive
        st.session_state["_save_error"] = f"Could not save the assessment: {error}"
        return

    st.session_state["selected_patient_id"] = context["patient"]["id"]
    st.session_state["dashboard_record_id"] = record_id
    if duplicate:
        st.session_state["_flash"] = (
            f"This assessment was just saved as record #{record_id}; the duplicate "
            "submission was ignored."
        )
    else:
        st.session_state["_flash"] = (
            f"Assessment saved as record #{record_id}. The patient profile and dashboard "
            "now include it."
        )
    st.session_state["page"] = "Patient Overview"


def render(context):
    if not context:
        st.markdown(
            page_head("Assessment results", "No assessment has been submitted in this session"),
            unsafe_allow_html=True,
        )
        st.markdown(
            empty_state(
                "Complete the form on the <b>Patient assessment</b> page and choose "
                "<b>Submit assessment</b> to run both screening modules here."
            ),
            unsafe_allow_html=True,
        )
        return

    patient = context["patient"]
    st.markdown(
        page_head(
            "Assessment results",
            f"Both independent modules ran on the values submitted for {patient['name']}.",
        ),
        unsafe_allow_html=True,
    )

    components.header(patient, context["assessment_date"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            card(
                "Save this assessment",
                "Stored as a new timestamped record under the patient ID",
                "<p style='font-size:0.86rem;color:" + PALETTE["text_muted"] + ";margin:0;'>Saving appends a record; "
                "earlier assessments for this patient are kept and remain available in the "
                "history and trend charts.</p>",
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='mc-spacer' style='height:12px;'></div>", unsafe_allow_html=True)
        st.button("Save assessment", type="primary", use_container_width=True,
                  key="save_assessment", on_click=_save)

    save_error = st.session_state.pop("_save_error", None)
    if save_error:  # pragma: no cover - defensive
        st.error(save_error)

    if st.session_state.get("_shap_error"):
        st.warning(f"Patient-level explanation unavailable: {st.session_state.pop('_shap_error')}")

    assessment.render(context)
