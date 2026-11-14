"""Tab 2: Explainable AI (global and patient-level SHAP)."""

import pandas as pd
import streamlit as st

from app import components
from app.research import global_importances
from app.theme import PALETTE, card

CAUSATION_NOTE = (
    "SHAP values describe how the inputs moved the model score. They are not proof that a "
    "measurement caused a clinical outcome, and they are not a treatment recommendation."
)


def _patient_table(frame):
    display = frame[["Display name", "Clinical Value", "Hybrid Impact", "Direction"]].copy()
    display.columns = ["Measurement", "Value", "Hybrid impact", "Direction"]
    return display.iloc[::-1]


def _global_block(bundle, module_label, module):
    importances = global_importances(bundle)
    if not importances:
        st.info(f"No stored global importances for the {module_label} module.")
        return
    hybrid = importances.get("hybrid_weighted_average")
    left, right = st.columns([3, 2])
    with left:
        if hybrid:
            st.plotly_chart(
                components.importance_chart(hybrid, f"{module_label}: global feature share"),
                use_container_width=True,
            )
    with right:
        rows = []
        for feature, value in sorted(hybrid.items(), key=lambda item: item[1], reverse=True) if hybrid else []:
            rows.append({
                "Feature": feature,
                "Hybrid share": f"{value * 100:.1f}%",
                "RF share": f"{importances.get('rf', {}).get(feature, 0) * 100:.1f}%",
                "XGB share": f"{importances.get('xgb', {}).get(feature, 0) * 100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render(context):
    bundle = context["bundle"]
    shap = context["shap"]
    patient = context["patient"]

    st.markdown(
        card(
            "How to read these explanations",
            "Signed contributions to the model score",
            f"<p style='font-size:0.88rem;color:{PALETTE['text_muted']};margin:0;'>"
            "Red bars raised the score for the class shown; green bars lowered it. "
            "The hybrid bar is a weight-scaled display aid built from the two base models, "
            "which are not guaranteed to share a common unit.</p>",
        ),
        unsafe_allow_html=True,
    )

    components.section("Global feature importance", "Average absolute SHAP value across the validation set")
    _global_block(bundle["maternal"], "Maternal risk", "maternal")
    _global_block(bundle["anemia"], "Anaemia", "anemia")

    components.section(
        "Individual patient explanation",
        f"Why the model produced this result for {patient['name']}",
    )

    maternal_frame = shap.get("maternal")
    anemia_frame = shap.get("anemia")
    left, right = st.columns(2)

    with left:
        st.markdown(f"**Maternal risk — target class: {context['maternal']['predicted_class']}**")
        if maternal_frame is not None and len(maternal_frame):
            st.plotly_chart(
                components.shap_contribution_chart(
                    maternal_frame, f"Contribution to '{context['maternal']['predicted_class']}'"
                ),
                use_container_width=True,
            )
            st.dataframe(_patient_table(maternal_frame), use_container_width=True, hide_index=True)
        else:
            st.info("No maternal explanation available.")

    with right:
        st.markdown(f"**Anaemia — target class: {context['anemia']['predicted_class']}**")
        if anemia_frame is not None and len(anemia_frame):
            st.plotly_chart(
                components.shap_contribution_chart(
                    anemia_frame, f"Contribution to '{context['anemia']['predicted_class']}'"
                ),
                use_container_width=True,
            )
            st.dataframe(_patient_table(anemia_frame), use_container_width=True, hide_index=True)
        else:
            st.info("No anaemia explanation available.")

    st.markdown(
        f"<div class='mc-disclaimer'>{CAUSATION_NOTE}</div>",
        unsafe_allow_html=True,
    )
