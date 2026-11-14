"""Tab 4: Generated medical report (Markdown + PDF)."""

import datetime

import streamlit as st

from app import components
from app.clinical import anemia_severity, morphology, risk_plain_language
from app.reporting import DISCLAIMER, build_report_markdown, build_report_pdf
from app.theme import PALETTE, card


def _observations(vitals_rows, maternal, anemia, agreement):
    notes = []
    for row in vitals_rows:
        if row["status"] in {"warning", "critical"}:
            notes.append(f"{row['parameter']} is {row['status']} ({row['value']:g} {row['unit']}). {row['interpretation']}")
    if not agreement["maternal"]:
        notes.append("The three maternal approaches disagree on the predicted band, so the result should be read with caution.")
    if not agreement["anemia"]:
        notes.append("The three anaemia approaches disagree, which is expected for an index-only model near chance.")
    if maternal["missing_features"]:
        notes.append("Some maternal inputs were missing and were imputed with training medians: "
                     + ", ".join(maternal["missing_features"]))
    if not notes:
        notes.append("No measurement fell outside the screening reference intervals.")
    return notes


def render(context):
    patient = context["patient"]
    maternal = context["maternal"]
    anemia = context["anemia"]
    cbc = context["anemia_inputs"]
    vitals_rows = context["vitals_rows"]

    severity, _, severity_note = anemia_severity(cbc.get("Hemoglobin"))
    morph, _ = morphology(cbc.get("MCV"))
    assessment_date = context["assessment_date"]

    report_context = {
        "patient": patient,
        "assessment_date": assessment_date,
        "maternal_result": maternal,
        "anemia_result": anemia,
        "vitals_rows": vitals_rows,
        "model_labels": context["model_labels"],
        "maternal_plain_language": risk_plain_language(
            maternal["predicted_class"], maternal["probabilities"]["Adaptive hybrid"]
        ),
        "anemia_severity": severity,
        "anemia_severity_note": severity_note,
        "morphology": morph,
        "shap_maternal": context["shap"].get("maternal"),
        "shap_anemia": context["shap"].get("anemia"),
        "observations": _observations(vitals_rows, maternal, anemia, context["agreement"]),
        "maternal_limitation": (
            "The maternal model is a research screening prototype trained on the UCI maternal health "
            "risk dataset. It has not been externally validated and does not replace clinical judgement."
        ),
        "anemia_limitation": (
            "The anaemia module uses only MCV, MCH and MCHC. It deliberately excludes haemoglobin and "
            "the sex code because the dataset label is defined by a sex-specific haemoglobin threshold. "
            "The CBC file does not confirm pregnancy status, so this is a general haematological model."
        ),
    }

    markdown_report = build_report_markdown(report_context)

    components.section("Generated screening summary", "Predictions are clearly separated from diagnoses")
    st.markdown(
        f"<div class='mc-disclaimer'>{DISCLAIMER}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(markdown_report)

    components.section("Download", "Markdown and PDF")
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download report (.md)",
            data=markdown_report,
            file_name=f"MATCARE_report_{patient['id']}_{datetime.date.today().isoformat()}.md",
            mime="text/markdown",
        )
    with right:
        try:
            pdf_bytes = build_report_pdf(markdown_report)
            st.download_button(
                "Download report (.pdf)",
                data=pdf_bytes,
                file_name=f"MATCARE_report_{patient['id']}_{datetime.date.today().isoformat()}.pdf",
                mime="application/pdf",
            )
        except Exception as error:  # pragma: no cover - defensive
            st.warning(f"PDF rendering unavailable: {error}")

    st.markdown(
        card(
            "What this report is not",
            "Scope of the generated summary",
            f"<p style='font-size:0.86rem;color:{PALETTE['text_muted']};margin:0;'>"
            "It is not a diagnosis, it contains no treatment advice, and it has not been reviewed by a "
            "clinician. It is a structured record of the model outputs for the values entered.</p>",
        ),
        unsafe_allow_html=True,
    )
