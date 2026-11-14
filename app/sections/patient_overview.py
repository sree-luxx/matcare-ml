"""Individual patient overview: a concise summary of one selected patient."""

import streamlit as st

from app import components, storage
from app.clinical import (
    RISK_COLORS,
    STATUS_COLORS,
    anemia_screening,
    build_vitals_rows,
    risk_plain_language,
)
from app.sections import patient_dashboard
from app.sections.patient_profiles import anemia_pill, fmt_date, risk_pill
from app.theme import PALETTE, card, empty_state, page_head, pill


def _abnormal(rows):
    return [row for row in rows if row["status"] in {"warning", "critical"}]


def _back_to_profiles():
    st.session_state["page"] = "Patient Profiles"


def _open_dashboard(record_id):
    st.session_state["dashboard_record_id"] = record_id
    st.session_state["page"] = "Patient Dashboard"


def _needs_selection():
    st.markdown(
        page_head("Patient overview", "No patient has been selected yet"),
        unsafe_allow_html=True,
    )
    st.markdown(
        empty_state(
            "Choose a patient from <b>Patient profiles</b> to open their overview, "
            "or run a new <b>Patient assessment</b>."
        ),
        unsafe_allow_html=True,
    )


def render(patient_id):
    if not patient_id:
        _needs_selection()
        return

    history = storage.get_history(patient_id)  # oldest first
    if not history:
        _needs_selection()
        return

    latest = history[-1]
    maternal = patient_dashboard.record_to_result(latest, "maternal")
    anemia = patient_dashboard.record_to_result(latest, "anemia")
    maternal_inputs = latest.get("maternal_inputs") or {}
    cbc_inputs = latest.get("cbc_inputs") or {}
    vitals_rows = build_vitals_rows({**maternal_inputs, **cbc_inputs})
    abnormal = _abnormal(vitals_rows)
    anemia_label, anemia_headline, anemia_key, anemia_detail = anemia_screening(
        cbc_inputs.get("Hemoglobin"), cbc_inputs.get("MCV")
    )

    top = st.columns([3, 1])
    with top[0]:
        st.markdown(
            page_head(
                f"{latest['patient_name']}",
                f"Patient ID {latest['patient_id']} · "
                f"{len(history)} assessment(s) on file · last assessed {fmt_date(latest['created_at'])}",
            ),
            unsafe_allow_html=True,
        )
    with top[1]:
        st.button("← All patient profiles", use_container_width=True, key="overview_back",
                  on_click=_back_to_profiles)

    components.kpi_row([
        {
            "label": "Maternal risk (latest)",
            "value": maternal["predicted_class"],
            "note": f"Hybrid score {maternal['confidence']:.1f}%",
            "accent": RISK_COLORS.get(maternal["predicted_class"], PALETTE["blue"]),
        },
        {
            "label": "Anaemia (latest)",
            "value": anemia_label,
            "note": f"Clinical screen from haemoglobin and MCV · model baseline {anemia['confidence']:.1f}%",
            "accent": STATUS_COLORS.get(anemia_key, PALETTE["navy"]),
        },
        {
            "label": "Previous assessments",
            "value": str(len(history)),
            "note": "Records stored under this patient ID",
            "accent": PALETTE["cyan"],
        },
        {
            "label": "Measurements to review",
            "value": str(len(abnormal)),
            "note": "Outside the screening reference intervals",
            "accent": PALETTE["amber"] if abnormal else PALETTE["green"],
        },
    ])

    st.markdown("<div class='mc-spacer'></div>", unsafe_allow_html=True)
    action_left, action_right = st.columns([2, 2])
    with action_left:
        st.button("View full dashboard →", type="primary", use_container_width=True,
                  key="open_full_dashboard", on_click=_open_dashboard, args=(latest["id"],))
    with action_right:
        st.caption("The dashboard holds the probability charts, CBC and blood-pressure trends, "
                   "SHAP explanations and the model comparison for this patient.")

    components.section("Patient profile", "Personal information on file")
    info_left, info_right = st.columns([3, 2])
    with info_left:
        st.markdown(
            card(
                "Patient",
                f"Record {latest['patient_id']}",
                f"<p style='font-size:0.9rem;color:{PALETTE['text']};margin:0;line-height:1.8;'>"
                f"<b>Name:</b> {latest['patient_name']}<br>"
                f"<b>Age:</b> {latest.get('age') if latest.get('age') is not None else '—'} years<br>"
                f"<b>Gestational age:</b> {latest.get('gestational_age') or '—'}<br>"
                f"<b>Last assessment:</b> {fmt_date(latest['created_at'])}</p>",
            ),
            unsafe_allow_html=True,
        )
    with info_right:
        st.markdown(
            card(
                "Latest status",
                "Both independent screening modules",
                f"<p style='font-size:0.88rem;color:{PALETTE['text_muted']};margin:0;line-height:1.9;'>"
                f"Maternal risk: {risk_pill(maternal['predicted_class'])}<br>"
                f"Anaemia: {anemia_pill(cbc_inputs)}<br>"
                f"Assessments: <b style='color:{PALETTE['navy']};'>{len(history)}</b><br>"
                f"Last assessment: <b style='color:{PALETTE['navy']};'>{fmt_date(latest['created_at'])}</b></p>",
            ),
            unsafe_allow_html=True,
        )

    components.section("Latest predictions", "Rebuilt from the stored assessment — no model re-run")
    left, right = st.columns(2)
    with left:
        st.markdown(
            card("1. General maternal health risk",
                 "Adaptive weighted mixture of Random Forest and XGBoost", ""),
            unsafe_allow_html=True,
        )
        st.plotly_chart(components.risk_gauge(maternal), use_container_width=True)
        st.caption(risk_plain_language(maternal["predicted_class"], maternal["probabilities"]["Adaptive hybrid"]))
    with right:
        st.markdown(
            card("2. Haematological anaemia screening",
                 "Rule-based verdict from haemoglobin and MCV; the saved model is a research baseline",
                 ""),
            unsafe_allow_html=True,
        )
        st.markdown(f"**{anemia_headline}**")
        st.caption(anemia_detail)
        st.plotly_chart(components.risk_gauge(anemia, "Model anaemia probability"), use_container_width=True)
        st.caption(
            f"Research model baseline: **{anemia['predicted_class']}** "
            f"({anemia['confidence']:.1f}%) — open the full dashboard for the model comparison."
        )

    components.section("Important clinical measurements", "From the latest saved assessment")
    if vitals_rows:
        st.markdown(
            f"<p style='margin:0 0 10px 0;'>{components.status_pills(abnormal)}</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(components.vitals_frame(vitals_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No bounded measurements were recorded for this assessment.")

    components.section("Assessment summary", "A short read of the latest record")
    summary = [
        f"Maternal risk band: **{maternal['predicted_class']}** "
        f"(hybrid score {maternal['confidence']:.1f}%).",
        risk_plain_language(maternal["predicted_class"], maternal["probabilities"]["Adaptive hybrid"]),
        f"Anaemia screening: **{anemia_headline}**. {anemia_detail}",
    ]
    if abnormal:
        summary.append(
            "Measurements to review: "
            + ", ".join(f"{row['parameter']} ({row['value']:g} {row['unit']})" for row in abnormal)
            + "."
        )
    else:
        summary.append("No measurement fell outside the screening reference intervals.")
    summary_items = "".join(f"<li>{item}</li>" for item in summary)
    st.markdown(
        card(
            "Summary",
            f"Based on the assessment of {fmt_date(latest['created_at'])}",
            f"<ul style='margin:0;padding-left:18px;font-size:0.88rem;line-height:1.7;"
            f"color:{PALETTE['text']};'>{summary_items}</ul>",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='mc-disclaimer'>{pill('warning')}&nbsp; "
        "Screening support only. These are statistical model outputs, not confirmed diagnoses.</div>",
        unsafe_allow_html=True,
    )
