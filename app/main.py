"""
MATCARE-AI: multi-page explainable adaptive hybrid screening application.

Entry point and page router. The sidebar is the permanent navigation: patient
care pages (assessment, results, profiles, overview, dashboard) and the model
and research pages. Predictions, storage and explainability are unchanged —
this module only decides which page renders.

Run with:  streamlit run app/main.py
"""

import os
import sys

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import storage
from app.model_loader import load_bundle
from app.sections import (
    assessment_form,
    datasets,
    explainability,
    model_intelligence,
    patient_dashboard,
    patient_overview,
    patient_profiles,
    report,
    results,
)
from app.theme import empty_state, inject_theme, page_head

st.set_page_config(
    page_title="MATCARE-AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()
storage.init_db()


PATIENT_PAGES = [
    "Patient Assessment",
    "Results",
    "Patient Profiles",
    "Patient Overview",
    "Patient Dashboard",
]
RESEARCH_PAGES = [
    "Explainable AI",
    "Model Intelligence",
    "Medical Report",
    "Dataset & Research",
]
DEFAULT_PAGE = "Patient Assessment"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _seed_state():
    assessment_form.seed_defaults()
    st.session_state.setdefault("page", DEFAULT_PAGE)
    st.session_state.setdefault("live_context", None)
    st.session_state.setdefault("selected_patient_id", None)
    st.session_state.setdefault("dashboard_record_id", None)


def _navigate(page):
    """Callback: switch pages. Runs before the script body, so the target page
    renders in the same run — no explicit rerun is needed."""
    st.session_state["page"] = page


def _nav_item(label, key):
    active = st.session_state.get("page") == label
    st.button(label, key=key, use_container_width=True,
              type="primary" if active else "secondary",
              on_click=_navigate, args=(label,))


def _sidebar_nav():
    with st.sidebar:
        st.markdown(
            """
            <div class="mc-brand-mini">
                <div class="mc-logo">MC</div>
                <div>
                    <p class="mc-brand-mini-title">MATCARE-AI</p>
                    <p class="mc-brand-mini-sub">Clinical decision support</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="mc-nav-group">Patient care</div>', unsafe_allow_html=True)
        for label in PATIENT_PAGES:
            _nav_item(label, f"nav::{label}")

        st.markdown('<div class="mc-nav-group">Model &amp; research</div>', unsafe_allow_html=True)
        for label in RESEARCH_PAGES:
            _nav_item(label, f"nav::{label}")

        selected = st.session_state.get("selected_patient_id")
        st.markdown(
            f"""<div class="mc-nav-foot">
                {storage.count_patients()} patient profile(s)<br>
                {storage.count_records()} saved assessment(s)<br>
                {"Selected: " + selected if selected else "No patient selected"}
            </div>""",
            unsafe_allow_html=True,
        )

        if st.session_state.get("live_context"):
            st.caption("A submitted assessment is ready on the **Results** page.")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_live_page(renderer, title, subtitle):
    """Render a page that needs the assessment submitted in this session."""
    context = st.session_state.get("live_context")
    if not context:
        st.markdown(page_head(title, subtitle), unsafe_allow_html=True)
        st.markdown(
            empty_state(
                "No assessment has been submitted in this session. Complete the "
                "<b>Patient assessment</b> page and choose <b>Submit assessment</b> first."
            ),
            unsafe_allow_html=True,
        )
        return
    renderer(context)


def _submit_callback(bundle):
    """Build the submit callback. It runs both screening modules for the current
    form values and hands the result to the Results page."""
    def callback():
        patient_id = (st.session_state.get("p_id") or "").strip()
        patient_name = (st.session_state.get("p_name") or "").strip()
        identity_problem = assessment_form.identity_error(patient_id, patient_name)
        if identity_problem:
            st.session_state["_submit_error"] = identity_problem
            return
        try:
            context = assessment_form.build_context(bundle)
        except Exception as error:
            st.session_state["_submit_error"] = (
                f"The screening models could not produce a result for these values: {error}. "
                "Check the measurements, then try again."
            )
            return
        st.session_state.pop("_submit_error", None)
        st.session_state["live_context"] = context
        st.session_state["page"] = "Results"

    return callback


def _assessment_page(bundle):
    error = st.session_state.pop("_submit_error", None)
    if error:
        st.error(error)
    assessment_form.render(on_submit=_submit_callback(bundle))


def _research_pages():
    return {
        "Explainable AI": (
            explainability.render,
            "Explainable AI",
            "Global and patient-level SHAP explanations for the current assessment",
        ),
        "Model Intelligence": (
            model_intelligence.render,
            "Model intelligence",
            "Held-out evaluation, robustness checks and the data-leakage audit",
        ),
        "Medical Report": (
            report.render,
            "Medical report",
            "A structured screening summary generated from the current assessment",
        ),
        "Dataset & Research": (
            datasets.render,
            "Dataset & research",
            "Raw dataset previews, audit findings and research limitations",
        ),
    }


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
_seed_state()

try:
    bundle = load_bundle()
except Exception as error:
    st.error(f"Could not load the saved models. {error}")
    st.info(
        "Run the training pipelines first:\n\n"
        "```\npython -m training.train_general_risk\npython -m training.train_anemia\n```"
    )
    st.stop()

_sidebar_nav()

page = st.session_state["page"]
if page not in PATIENT_PAGES + RESEARCH_PAGES:  # pragma: no cover - defensive
    page = DEFAULT_PAGE
    st.session_state["page"] = page

if st.session_state.get("_flash"):
    st.success(st.session_state.pop("_flash"))

if page == "Patient Assessment":
    _assessment_page(bundle)
elif page == "Results":
    results.render(st.session_state.get("live_context"))
elif page == "Patient Profiles":
    patient_profiles.render()
elif page == "Patient Overview":
    patient_overview.render(st.session_state.get("selected_patient_id"))
elif page == "Patient Dashboard":
    patient_dashboard.render(st.session_state.get("selected_patient_id"))
else:
    renderer, title, subtitle = _research_pages()[page]
    _render_live_page(renderer, title, subtitle)
