"""Patient dashboard: the complete view of one saved assessment.

Reconstructs both module results from the stored record — the raw inputs,
predictions, probability distributions and SHAP explanations — so no model is
re-run when browsing history. Also shows previous assessments and trends.
"""

import datetime

import pandas as pd
import streamlit as st

from app import components, storage
from app.clinical import RISK_COLORS, STATUS_COLORS, anemia_screening, build_vitals_rows, risk_plain_language
from app.inference import MODULE_CLASSES, MODEL_LABELS, model_agreement
from app.theme import PALETTE, card, empty_state, page_head, pill


# ---------------------------------------------------------------------------
# Record → display helpers
# ---------------------------------------------------------------------------
def _fmt_date(iso_value):
    try:
        return datetime.datetime.fromisoformat(iso_value).strftime("%d %B %Y, %H:%M")
    except Exception:  # pragma: no cover - defensive
        return iso_value


def record_to_result(record, module):
    """Rebuild a ``predict_module``-style result dictionary from a stored row."""
    probabilities = record[f"{module}_probabilities"] or {}
    weights = record.get(f"{module}_weights") or {}
    class_names = list(MODULE_CLASSES[module])
    predicted = record.get(f"{module}_prediction") or class_names[0]
    predicted_index = class_names.index(predicted) if predicted in class_names else 0
    return {
        "module": module,
        "class_names": class_names,
        "probabilities": {label: probabilities.get(label, []) for label in MODEL_LABELS},
        "predicted_class": predicted,
        "predicted_index": predicted_index,
        "confidence": float(record.get(f"{module}_confidence") or 0.0),
        "w_rf": float(weights.get("w_rf", 0.5)),
        "w_xgb": float(weights.get("w_xgb", 0.5)),
        "missing_features": [],
    }


def _shap_frame(records):
    if not records:
        return None
    return pd.DataFrame(records)


def _abnormal(rows):
    return [row for row in rows if row["status"] in {"warning", "critical"}]


def _anemia_screen_label(record):
    """Rule-based anaemia screening verdict for a stored record."""
    cbc = record.get("cbc_inputs") or {}
    return anemia_screening(cbc.get("Hemoglobin"), cbc.get("MCV"))[0]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _latest_assessment(record, maternal, anemia, vitals_rows, cbc_inputs):
    components.risk_banner(maternal["predicted_class"], maternal["confidence"])

    high_probability = float(maternal["probabilities"]["Adaptive hybrid"][-1] * 100.0)
    anemia_label, anemia_headline, anemia_key, anemia_detail = anemia_screening(
        cbc_inputs.get("Hemoglobin"), cbc_inputs.get("MCV")
    )
    abnormal = _abnormal(vitals_rows)

    components.kpi_row([
        {
            "label": "Maternal risk band",
            "value": maternal["predicted_class"],
            "note": f"Hybrid score {maternal['confidence']:.1f}%",
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
            "label": "Measurements outside range",
            "value": str(len(abnormal)),
            "note": "Screening reference intervals",
            "accent": PALETTE["amber"] if abnormal else PALETTE["green"],
        },
    ])

    left, right = st.columns(2)
    with left:
        st.markdown(
            card("1. General maternal health risk",
                 "Adaptive weighted mixture of Random Forest and XGBoost", ""),
            unsafe_allow_html=True,
        )
        st.plotly_chart(components.risk_gauge(maternal), use_container_width=True)
        st.plotly_chart(components.probability_bars(maternal), use_container_width=True)
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
        st.plotly_chart(components.probability_bars(anemia), use_container_width=True)
        st.caption(
            f"Research model baseline: **{anemia['predicted_class']}** ({anemia['confidence']:.1f}%)."
        )


def _measurements(vitals_rows):
    components.section("Key clinical measurements", "Units, reference interval, status and interpretation")
    if not vitals_rows:
        st.info("No bounded measurements were recorded for this assessment.")
        return
    abnormal = _abnormal(vitals_rows)
    st.markdown(
        f"<p style='margin:0 0 10px 0;'>{components.status_pills(abnormal)}</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(components.vitals_frame(vitals_rows), use_container_width=True, hide_index=True)


def _explainability(shap):
    components.section("SHAP feature contributions", "Which measurements moved each prediction")
    left, right = st.columns(2)
    maternal_frame = _shap_frame(shap.get("maternal"))
    anemia_frame = _shap_frame(shap.get("anemia"))

    with left:
        st.markdown("**Maternal risk**")
        if maternal_frame is not None and len(maternal_frame):
            st.plotly_chart(
                components.shap_contribution_chart(maternal_frame, "Contribution to maternal risk band"),
                use_container_width=True,
            )
        else:
            st.info("No maternal explanation was stored for this record.")
    with right:
        st.markdown("**Anaemia**")
        if anemia_frame is not None and len(anemia_frame):
            st.plotly_chart(
                components.shap_contribution_chart(anemia_frame, "Contribution to anaemia class"),
                use_container_width=True,
            )
        else:
            st.info("No anaemia explanation was stored for this record.")

    st.markdown(
        f"<div class='mc-disclaimer'>SHAP values describe how inputs moved the model score. "
        f"They are not proof of causation and are not a treatment recommendation.</div>",
        unsafe_allow_html=True,
    )


def _parse_dt(iso_value):
    try:
        return datetime.datetime.fromisoformat(iso_value)
    except Exception:  # pragma: no cover - defensive
        return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _joint_probability(record, module):
    """Hybrid probability of the flagged class (High Risk / Anemic), in percent."""
    probabilities = (record.get(f"{module}_probabilities") or {}).get("Adaptive hybrid")
    if not probabilities:
        return None
    try:
        return float(probabilities[-1]) * 100.0
    except (TypeError, ValueError, IndexError):  # pragma: no cover - defensive
        return None


def _risk_pill(risk_class):
    key = {"High Risk": "critical", "Mid Risk": "warning"}.get(risk_class, "normal")
    return pill(key, risk_class or "—")


def _score_text(value):
    return f"{value:.1f}%" if value is not None else "—"


def _select_record(record_id):
    """Callback: switch the dashboard to another stored assessment."""
    st.session_state["dashboard_record_id"] = record_id


def _back_to_overview():
    st.session_state["page"] = "Patient Overview"


def _back_to_profiles():
    st.session_state["page"] = "Patient Profiles"


def _update_record(record_id):
    """Callback: apply the manage-form edits, then let the body re-read the row."""
    storage.update_record(
        record_id,
        patient_name=st.session_state.get(f"edit_name_{record_id}"),
        notes=st.session_state.get(f"edit_notes_{record_id}"),
    )
    st.session_state["_flash"] = "Record updated."


def _delete_record(record_id):
    """Callback: delete the record and return to the patient list."""
    storage.delete_record(record_id)
    st.session_state["dashboard_record_id"] = None
    st.session_state["page"] = "Patient Profiles"
    st.session_state["_flash"] = "Record deleted."


def _history_table(history, current_id):
    """Assessment history, newest first, with a View details action per row."""
    widths = [2.4, 1.4, 1.0, 1.3, 1.1, 1.2]
    header = st.columns(widths)
    for column, label in zip(header, [
        "Assessment date & time", "Maternal risk", "Risk prob.",
        "Anaemia screen", "Model prob.", "Action",
    ]):
        column.markdown(f"<span class='mc-th'>{label}</span>", unsafe_allow_html=True)

    for item in history:
        cells = st.columns(widths)
        cells[0].markdown(
            f"<span class='mc-td'>{_fmt_date(item['created_at'])}</span>", unsafe_allow_html=True
        )
        cells[1].markdown(_risk_pill(item["maternal_prediction"]), unsafe_allow_html=True)
        cells[2].markdown(
            f"<span class='mc-td'>{_score_text(item['maternal_confidence'])}</span>",
            unsafe_allow_html=True,
        )
        anemia_label, _, anemia_status, _ = anemia_screening(
            (item.get("cbc_inputs") or {}).get("Hemoglobin"),
            (item.get("cbc_inputs") or {}).get("MCV"),
        )
        anemia_key = {"critical": "critical", "warning": "warning"}.get(anemia_status, "normal")
        cells[3].markdown(pill(anemia_key, anemia_label), unsafe_allow_html=True)
        cells[4].markdown(
            f"<span class='mc-td'>{_score_text(item['anemia_confidence'])}</span>",
            unsafe_allow_html=True,
        )
        with cells[5]:
            if item["id"] == current_id:
                st.markdown("<span class='mc-tag'>Viewing</span>", unsafe_allow_html=True)
            else:
                st.button(
                    "View details",
                    key=f"history_view_{item['id']}",
                    use_container_width=True,
                    on_click=_select_record,
                    args=(item["id"],),
                )


# ---------------------------------------------------------------------------
# Historical trends
# ---------------------------------------------------------------------------
def _field_reader(module_key, field):
    def read(item):
        return _as_float((item.get(module_key) or {}).get(field))
    return read


def _probability_reader(module_key):
    def read(item):
        return _joint_probability(item, module_key)
    return read


# Each group keeps measurements with a shared unit together; anything with a
# different unit gets its own chart so no misleading shared scale is drawn.
TREND_GROUPS = [
    {"title": "Blood pressure", "unit": "mmHg", "series": [
        ("Systolic BP", _field_reader("maternal_inputs", "SystolicBP"), PALETTE["amber"]),
        ("Diastolic BP", _field_reader("maternal_inputs", "DiastolicBP"), PALETTE["cyan"]),
    ]},
    {"title": "Haemoglobin", "unit": "g/dL", "series": [
        ("Haemoglobin", _field_reader("cbc_inputs", "Hemoglobin"), PALETTE["red"]),
    ]},
    {"title": "Mean corpuscular volume", "unit": "fL", "series": [
        ("MCV", _field_reader("cbc_inputs", "MCV"), PALETTE["teal"]),
    ]},
    {"title": "Mean corpuscular haemoglobin", "unit": "pg", "series": [
        ("MCH", _field_reader("cbc_inputs", "MCH"), PALETTE["blue"]),
    ]},
    {"title": "MCH concentration", "unit": "g/dL", "series": [
        ("MCHC", _field_reader("cbc_inputs", "MCHC"), PALETTE["green"]),
    ]},
    {"title": "Red blood cell count", "unit": "×10¹²/L", "series": [
        ("RBC", _field_reader("cbc_inputs", "RBC"), PALETTE["cyan"]),
    ]},
    {"title": "Blood sugar", "unit": "mmol/L", "series": [
        ("Blood sugar", _field_reader("maternal_inputs", "BS"), PALETTE["amber"]),
    ]},
    {"title": "Heart rate", "unit": "bpm", "series": [
        ("Heart rate", _field_reader("maternal_inputs", "HeartRate"), PALETTE["red"]),
    ]},
    {"title": "Body temperature", "unit": "°F", "series": [
        ("Body temperature", _field_reader("maternal_inputs", "BodyTemp"), PALETTE["blue"]),
    ]},
    {"title": "Maternal high-risk probability", "unit": "%", "series": [
        ("High-risk probability", _probability_reader("maternal"), PALETTE["red"]),
    ]},
    {"title": "Anaemia probability", "unit": "%", "series": [
        ("Anaemia probability", _probability_reader("anemia"), PALETTE["amber"]),
    ]},
]


def build_trend_series(history):
    """Transform saved assessments into chart-ready series.

    Returns ``(dates, charts)`` where ``dates`` are ``datetime`` objects taken
    from the real assessment timestamps (never row indices) and ``charts`` is a
    list of ``(group, series)`` pairs. Only indicators with at least two recorded
    values are kept, and missing values remain ``None`` so nothing is
    interpolated. A patient with fewer than two dated assessments yields
    ``([], [])``.
    """
    dated = [(dt, item) for item in history if (dt := _parse_dt(item["created_at"])) is not None]
    if len(dated) < 2:
        return [], []

    dates = [dt for dt, _ in dated]
    charts = []
    for group in TREND_GROUPS:
        series = []
        for name, reader, color in group["series"]:
            values = [reader(item) for _, item in dated]
            if sum(value is not None for value in values) >= 2:
                series.append({"name": name, "values": values, "color": color, "unit": group["unit"]})
        if series:
            charts.append((group, series))
    return dates, charts


def _history_trends(history, patient_id):
    components.section(
        "Historical trends",
        "Each indicator is plotted on its own scale against real assessment dates",
    )
    if len(history) < 2:
        st.info(
            "Only one assessment is on file for this patient. Trend charts appear "
            "automatically once a second assessment is saved."
        )
        return

    dates, charts = build_trend_series(history)
    if not dates:
        st.info("Assessment timestamps are unavailable, so trend charts cannot be drawn.")
        return
    if not charts:
        st.info(
            "No measurement has at least two recorded values yet. Trend charts appear "
            "once repeated measurements of the same indicator exist."
        )
        return

    for index in range(0, len(charts), 2):
        left, right = st.columns(2)
        for column, (group, series) in zip((left, right), charts[index:index + 2]):
            with column:
                st.plotly_chart(
                    components.history_trend_chart(dates, series, group["title"], group["unit"]),
                    use_container_width=True,
                    key=f"trend_{patient_id}_{group['title']}",
                )
        st.markdown("<div class='mc-spacer'></div>", unsafe_allow_html=True)


def _change_note(previous, latest):
    if previous is None or latest is None:
        return "Not comparable"
    return "Unchanged" if previous == latest else "Changed"


def _comparison(history):
    """Previous vs latest results for the same indicators."""
    components.section(
        "Previous vs latest assessment",
        "How the two most recent records differ",
    )
    if len(history) < 2:
        st.info("A comparison appears once a second assessment has been saved for this patient.")
        return

    previous, latest = history[-2], history[-1]
    st.caption(f"Comparing {_fmt_date(previous['created_at'])} → {_fmt_date(latest['created_at'])}")

    rows = []
    for group in TREND_GROUPS:
        for name, reader, _ in group["series"]:
            before, after = reader(previous), reader(latest)
            if before is None and after is None:
                continue
            change = (after - before) if (before is not None and after is not None) else None
            rows.append({
                "Indicator": name,
                "Unit": group["unit"],
                "Previous": f"{before:g}" if before is not None else "—",
                "Latest": f"{after:g}" if after is not None else "—",
                "Change": f"{change:+g}" if change is not None else "—",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Screening classification**")
    st.dataframe(
        pd.DataFrame([
            {
                "Module": "Maternal risk",
                "Previous": previous["maternal_prediction"] or "—",
                "Latest": latest["maternal_prediction"] or "—",
                "Change": _change_note(previous["maternal_prediction"], latest["maternal_prediction"]),
            },
            {
                "Module": "Anaemia (clinical screen)",
                "Previous": _anemia_screen_label(previous),
                "Latest": _anemia_screen_label(latest),
                "Change": _change_note(_anemia_screen_label(previous), _anemia_screen_label(latest)),
            },
        ]),
        use_container_width=True,
        hide_index=True,
    )


def _history(history, record):
    components.section(
        "Assessment history",
        f"{len(history)} assessment(s) on file for {record['patient_id']}",
    )
    if not history:
        st.info("No assessments found for this patient.")
        return

    newest_first = sorted(history, key=lambda item: (item["created_at"], item["id"]), reverse=True)
    st.caption("Newest first. Select “View details” to load any stored assessment into this dashboard.")
    _history_table(newest_first, record["id"])


def _manage(record):
    with st.expander("Manage this record"):
        st.caption("Updating edits the saved record's metadata. Deleting removes it permanently.")
        st.text_input("Patient name", value=record["patient_name"], key=f"edit_name_{record['id']}")
        st.text_area("Clinical note", value=record.get("notes") or "", key=f"edit_notes_{record['id']}")

        left, right = st.columns(2)
        with left:
            st.button("Update record", key=f"update_{record['id']}", use_container_width=True,
                      on_click=_update_record, args=(record["id"],))
        with right:
            confirm = st.checkbox("Confirm permanent deletion", key=f"confirm_del_{record['id']}")
            st.button("Delete record", key=f"delete_{record['id']}", use_container_width=True,
                      disabled=not confirm, on_click=_delete_record, args=(record["id"],))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(patient_id):
    """Detailed per-patient dashboard: trends, history, explanations, models."""
    if not patient_id:
        st.markdown(page_head("Patient dashboard", "No patient selected"), unsafe_allow_html=True)
        st.markdown(
            empty_state("Choose a patient from <b>Patient profiles</b> to open their analytics dashboard."),
            unsafe_allow_html=True,
        )
        return

    history = storage.get_history(patient_id)  # oldest first
    if not history:
        st.markdown(page_head("Patient dashboard", "No saved assessments"), unsafe_allow_html=True)
        st.markdown(
            empty_state("This patient has no saved assessments yet."),
            unsafe_allow_html=True,
        )
        return

    newest_first = sorted(history, key=lambda item: (item["created_at"], item["id"]), reverse=True)
    ids = [item["id"] for item in newest_first]
    if st.session_state.get("dashboard_record_id") not in ids:
        st.session_state["dashboard_record_id"] = ids[0]
    record = next(item for item in newest_first if item["id"] == st.session_state["dashboard_record_id"])

    maternal = record_to_result(record, "maternal")
    anemia = record_to_result(record, "anemia")
    maternal_inputs = record.get("maternal_inputs") or {}
    cbc_inputs = record.get("cbc_inputs") or {}
    vitals_rows = build_vitals_rows({**maternal_inputs, **cbc_inputs})

    patient = {
        "name": record["patient_name"],
        "id": record["patient_id"],
        "age": record.get("age") or "-",
        "gestational_age": record.get("gestational_age") or "-",
    }

    top = st.columns([2, 2, 2])
    with top[0]:
        st.button("← Patient overview", use_container_width=True, key="dashboard_back",
                  on_click=_back_to_overview)
    with top[1]:
        st.button("All patient profiles", use_container_width=True, key="dashboard_profiles",
                  on_click=_back_to_profiles)
    with top[2]:
        labels = {item["id"]: f"{_fmt_date(item['created_at'])} · #{item['id']}" for item in newest_first}
        st.selectbox(
            "Assessment to inspect",
            options=ids,
            key="dashboard_record_id",
            format_func=lambda record_id: labels.get(record_id, str(record_id)),
            label_visibility="collapsed",
        )

    st.markdown(
        page_head(
            f"{record['patient_name']} — clinical dashboard",
            f"Patient ID {record['patient_id']} · {len(history)} assessment(s) on file",
        ),
        unsafe_allow_html=True,
    )

    components.header(patient, _fmt_date(record["created_at"]))
    st.markdown(
        f"<div class='mc-disclaimer'>{pill('info', 'Stored record')}&nbsp; "
        f"Viewing the assessment saved on {_fmt_date(record['created_at'])}. Every value and "
        "probability below is the original stored data — no model is re-run for history.</div>",
        unsafe_allow_html=True,
    )

    components.section("Patient profile", "Stored patient details")
    profile, agreement = st.columns([3, 2])
    with profile:
        st.markdown(
            card(
                "Patient",
                f"Record {record['patient_id']}",
                f"<p style='font-size:0.9rem;color:{PALETTE['text']};margin:0;line-height:1.7;'>"
                f"<b>Name:</b> {record['patient_name']}<br>"
                f"<b>Age:</b> {record.get('age') if record.get('age') is not None else '-'} years<br>"
                f"<b>Gestational age:</b> {record.get('gestational_age') or '-'}<br>"
                f"<b>Assessments on file:</b> {len(history)}<br>"
                f"<b>Last assessment:</b> {_fmt_date(newest_first[0]['created_at'])}<br>"
                f"<b>Note:</b> {record.get('notes') or '—'}</p>",
            ),
            unsafe_allow_html=True,
        )
    with agreement:
        agree = model_agreement(maternal, anemia)
        st.markdown(
            card(
                "Model agreement",
                "Do Random Forest, XGBoost and the hybrid agree?",
                f"<p style='font-size:0.88rem;color:{PALETTE['text_muted']};margin:0;'>"
                f"Maternal module: <b style='color:{PALETTE['text']};'>"
                f"{'agree' if agree['maternal'] else 'disagree'}</b><br>"
                f"Anaemia module: <b style='color:{PALETTE['text']};'>"
                f"{'agree' if agree['anemia'] else 'disagree'}</b><br>"
                f"Weights — RF {maternal['w_rf'] * 100:.0f}% / XGB {maternal['w_xgb'] * 100:.0f}% (maternal), "
                f"RF {anemia['w_rf'] * 100:.0f}% / XGB {anemia['w_xgb'] * 100:.0f}% (anaemia).</p>",
            ),
            unsafe_allow_html=True,
        )

    components.section("Latest predictions", "Risk categories and probabilities for the selected assessment")
    _latest_assessment(record, maternal, anemia, vitals_rows, cbc_inputs)

    _measurements(vitals_rows)

    _history_trends(history, record["patient_id"])

    _history(history, record)

    _comparison(history)

    _explainability({"maternal": record.get("shap_maternal"), "anemia": record.get("shap_anemia")})

    components.section("Model comparison", "Random Forest vs XGBoost vs Adaptive hybrid")
    st.dataframe(components.model_comparison_frame(maternal, anemia), use_container_width=True, hide_index=True)

    _manage(record)
