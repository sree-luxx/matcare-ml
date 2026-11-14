"""Tab 1: Clinical assessment dashboard."""

import streamlit as st

from app import components
from app.clinical import (
    RISK_COLORS,
    STATUS_COLORS,
    anemia_screening,
    anemia_severity,
    risk_plain_language,
)
from app.theme import PALETTE, card


def render(context):
    maternal = context["maternal"]
    anemia = context["anemia"]
    patient = context["patient"]
    inputs = context["maternal_inputs"]
    cbc = context["anemia_inputs"]

    components.risk_banner(maternal["predicted_class"], maternal["confidence"])

    high_probability = float(maternal["probabilities"]["Adaptive hybrid"][-1] * 100.0)
    severity, severity_key, _ = anemia_severity(cbc.get("Hemoglobin"))
    anemia_label, anemia_headline, anemia_key, anemia_detail = anemia_screening(
        cbc.get("Hemoglobin"), cbc.get("MCV")
    )

    components.kpi_row([
        {
            "label": "Maternal risk band",
            "value": maternal["predicted_class"],
            "note": f"Hybrid decision, model score {maternal['confidence']:.1f}%",
            "accent": RISK_COLORS.get(maternal["predicted_class"], PALETTE["blue"]),
        },
        {
            "label": "High-risk probability",
            "value": f"{high_probability:.1f}%",
            "note": "Adaptive hybrid, High Risk class",
            "accent": PALETTE["red"] if high_probability >= 66 else PALETTE["navy"],
        },
        {
            "label": "Anaemia screening",
            "value": anemia_label,
            "note": f"Clinical screen from haemoglobin and MCV · model baseline {anemia['confidence']:.1f}%",
            "accent": STATUS_COLORS.get(anemia_key, PALETTE["navy"]),
        },
        {
            "label": "Haemoglobin",
            "value": f"{cbc.get('Hemoglobin', float('nan')):g} g/dL",
            "note": severity,
            "accent": STATUS_COLORS.get(severity_key, PALETTE["navy"]),
        },
    ])

    # ----- two-column health overview -----
    components.section("Health overview", "Clinical screening results and model views for both modules")
    left, right = st.columns(2)

    with left:
        st.markdown(
            card(
                "1. General maternal health risk",
                "Adaptive weighted mixture of Random Forest and XGBoost",
                "",
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(components.risk_gauge(maternal), use_container_width=True)
        st.plotly_chart(components.probability_bars(maternal), use_container_width=True)
        st.caption(risk_plain_language(maternal["predicted_class"], maternal["probabilities"]["Adaptive hybrid"]))
        if maternal["missing_features"]:
            st.warning("Imputed with training medians: " + ", ".join(maternal["missing_features"]))

    with right:
        st.markdown(
            card(
                "2. Haematological anaemia screening",
                "Rule-based verdict from haemoglobin and MCV — the clinically meaningful result",
                "",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(f"**{anemia_headline}**")
        st.caption(anemia_detail)

        st.markdown(
            "<div class='mc-spacer' style='height:6px;'></div>"
            "<p style='font-size:0.82rem;letter-spacing:0.06em;text-transform:uppercase;"
            "color:#94a3b8;margin:6px 0 2px;'>Research model view — index-only model (MCV, MCH, MCHC)</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(components.risk_gauge(anemia, "Model anaemia probability"), use_container_width=True)
        st.plotly_chart(components.probability_bars(anemia), use_container_width=True)
        st.markdown(
            f"- Model class: **{anemia['predicted_class']}** (score {anemia['confidence']:.1f}%)\n"
            f"- Model inputs: `{'`, `'.join(anemia['feature_names'])}`",
            unsafe_allow_html=True,
        )
        st.caption(
            "The dataset label equals a sex-specific haemoglobin threshold, so a model given "
            "haemoglobin would only reproduce that rule. The saved index-only model is therefore "
            "close to chance and is kept as an honest research baseline — the rule-based screen "
            "above is the result to act on."
        )

    # ----- vitals scorecard -----
    components.section("Measurements and reference intervals", "Units, reference range, status and interpretation")
    vitals = context["vitals_rows"]
    if vitals:
        st.dataframe(components.vitals_frame(vitals), use_container_width=True, hide_index=True)
    else:
        st.info("No measurements available.")

    # ----- model transparency -----
    components.section("Model transparency", "What each approach predicted and why")
    st.dataframe(
        components.model_comparison_frame(maternal, anemia),
        use_container_width=True,
        hide_index=True,
    )

    agree = context["agreement"]
    weight_note = (
        f"Maternal weights — RF {maternal['w_rf'] * 100:.1f}% / XGB {maternal['w_xgb'] * 100:.1f}%. "
        f"Anaemia weights — RF {anemia['w_rf'] * 100:.1f}% / XGB {anemia['w_xgb'] * 100:.1f}%."
    )
    agreement_note = (
        f"Maternal module: three approaches {'agree' if agree['maternal'] else 'disagree'}. "
        f"Anaemia module: three approaches {'agree' if agree['anemia'] else 'disagree'}."
    )
    st.markdown(
        card(
            "How the mixture is formed",
            "Weights are learned on a validation split only, never on the test split",
            f"<p style='font-size:0.88rem;color:{PALETTE['text_muted']};margin:0;'>"
            f"P<sub>hybrid</sub>(c) = w<sub>RF</sub>&middot;P<sub>RF</sub>(c) + w<sub>XGB</sub>&middot;P<sub>XGB</sub>(c). "
            f"{weight_note} {agreement_note}</p>",
        ),
        unsafe_allow_html=True,
    )
