"""Tab 5: Dataset and research notes."""

import pandas as pd
import streamlit as st

from app import components
from app.research import ablation_table, audit_summary, dataset_paths, load_raw
from app.theme import PALETTE, card, pill


def _class_balance(frame, column):
    counts = frame[column].value_counts().to_dict()
    return " | ".join(f"{key}: {value}" for key, value in counts.items())


def _maternal_block():
    frame = load_raw("maternal")
    paths = dataset_paths()
    st.markdown(f"**Source file:** `{paths['maternal']}`")
    st.markdown(
        f"- Rows: **{len(frame)}** | Columns: {', '.join(frame.columns)}\n"
        f"- Risk level balance: {_class_balance(frame, 'RiskLevel')}"
    )
    st.dataframe(frame.head(15), use_container_width=True, hide_index=True)
    with open(paths["maternal"], "rb") as handle:
        st.download_button("Download maternal_health_risk.csv", handle, file_name="maternal_health_risk.csv", mime="text/csv")


def _anemia_block():
    frame = load_raw("anemia")
    paths = dataset_paths()
    st.markdown(f"**Source file:** `{paths['anemia']}`")
    st.markdown(
        f"- Rows: **{len(frame)}** | Columns: {', '.join(frame.columns)}\n"
        f"- Label balance: {_class_balance(frame, 'Result')}\n"
        f"- Sex code balance: {_class_balance(frame, 'Gender')}"
    )
    st.dataframe(frame.head(15), use_container_width=True, hide_index=True)
    with open(paths["anemia"], "rb") as handle:
        st.download_button("Download anemia_cbc.csv", handle, file_name="anemia_cbc.csv", mime="text/csv")


def render(context):
    bundle = context["bundle"]

    st.markdown(
        card(
            "Research framing",
            "What these datasets can and cannot support",
            f"<p style='font-size:0.88rem;color:{PALETTE['text_muted']};margin:0;'>"
            "The datasets are used exactly as provided. Neither confirms pregnancy status, patient "
            "identity, or longitudinal follow-up, so the results describe screening behaviour on these "
            "files only. No records were deleted and no values were fabricated.</p>",
        ),
        unsafe_allow_html=True,
    )

    components.section("Dataset explorer", "Raw previews and downloads")
    tab_maternal, tab_anemia = st.tabs(["UCI maternal health risk", "CBC haematology"])
    with tab_maternal:
        _maternal_block()
    with tab_anemia:
        _anemia_block()

    components.section("Audit findings", "Duplicates, label construction and limitations")
    anemia_audit = audit_summary(bundle["anemia"])
    if anemia_audit:
        st.markdown(
            card(
                "CBC dataset — label circularity",
                "The binary target is reproducible from haemoglobin and the sex code",
                f"<p style='font-size:0.86rem;margin:0;color:{PALETTE['text']};'>"
                f"<code>{anemia_audit.get('rule', '')}</code></p>"
                f"<p style='font-size:0.84rem;margin:8px 0 0 0;color:{PALETTE['text_muted']};'>"
                f"Match rate {anemia_audit.get('rule_match_rate', 0) * 100:.0f}% "
                f"(exact: {anemia_audit.get('rule_is_exact')}). "
                f"Duplicates removed: {anemia_audit.get('duplicate_rows')}. "
                f"Rows used for modelling: {anemia_audit.get('rows_after_dedup_and_conflict_removal')}. "
                f"RBC column present: {anemia_audit.get('rbc_present')}.</p>",
            ),
            unsafe_allow_html=True,
        )
        rows = ablation_table(bundle["anemia"])
        if rows:
            st.markdown("**Ablation results**")
            st.dataframe(pd.DataFrame(rows)[["experiment", "model", "accuracy", "macro_f1", "roc_auc", "anemic_recall"]],
                         use_container_width=True, hide_index=True)

    maternal_audit = audit_summary(bundle["maternal"])
    if maternal_audit:
        st.markdown(
            f"- The maternal file contains **{maternal_audit.get('duplicate_rows')}** exact duplicate rows; "
            f"**{maternal_audit.get('rows_used_for_modeling')}** rows remain after removing duplicates and "
            f"feature vectors that carry more than one risk label."
        )
        st.caption(maternal_audit.get("limitation", ""))

    st.markdown(
        f"<div class='mc-disclaimer'>{pill('warning')}&nbsp; "
        "The anaemia module is a <b>general haematological</b> classifier, not a pregnancy-specific "
        "anaemia tool. The CBC file has no pregnancy indicator and no patient identifier.</div>",
        unsafe_allow_html=True,
    )
