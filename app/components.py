"""
Reusable Streamlit + Plotly building blocks for the dashboard.

Everything here is presentation only. Clinical rules live in ``app.clinical``
and model calls live in ``app.inference``. All charts are styled for the dark
theme through the tokens in ``app.theme``.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.clinical import RISK_COLORS
from app.inference import MODEL_LABELS
from app.theme import PALETTE, card, kpi_card, pill


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _style(figure, title=None, height=None, legend=None, margin=None):
    """Apply the shared dark chart styling."""
    layout = {
        "paper_bgcolor": PALETTE["chart_bg"],
        "plot_bgcolor": PALETTE["chart_bg"],
        "font": {"color": PALETTE["chart_font"], "family": "Inter, sans-serif", "size": 12},
        "margin": margin or dict(l=10, r=16, t=46, b=10),
        "showlegend": legend is not None,
    }
    if title:
        layout["title"] = {"text": title, "font": {"size": 13, "color": PALETTE["navy"]}}
    if height:
        layout["height"] = height
    if legend is not None:
        layout["legend"] = legend
    figure.update_layout(**layout)
    return figure


def section(title, subtitle=""):
    sub = f"<small>{subtitle}</small>" if subtitle else ""
    st.markdown(f'<div class="mc-section">{title}{sub}</div>', unsafe_allow_html=True)


def header(patient, assessment_date):
    st.markdown(
        f"""
        <div class="mc-header">
            <div class="mc-brand">
                <div class="mc-logo">MC</div>
                <div>
                    <p class="mc-title">MATCARE-AI Clinical Decision Support</p>
                    <p class="mc-subtitle">Explainable adaptive hybrid screening for maternal health risk and haematological anaemia</p>
                </div>
            </div>
            <div class="mc-meta">
                <div class="mc-meta-item">Patient <b>{patient['name']}</b> ({patient['id']})</div>
                <div class="mc-meta-item">Age <b>{patient['age']} yrs</b></div>
                <div class="mc-meta-item">Gestational age <b>{patient['gestational_age']}</b></div>
                <div class="mc-meta-item">Assessment date <b>{assessment_date}</b></div>
            </div>
            <div class="mc-disclaimer">
                Screening support only. Outputs are statistical model predictions, not confirmed clinical
                diagnoses, and no treatment is recommended by this tool. Always confirm with a qualified clinician.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_banner(risk_class, confidence):
    key = {"High Risk": "critical", "Mid Risk": "warning"}.get(risk_class, "normal")
    headline = {
        "High Risk": "High maternal risk band",
        "Mid Risk": "Mid maternal risk band",
    }.get(risk_class, "Low maternal risk band")
    message = {
        "High Risk": "Vital signs and blood chemistry are consistent with the high-risk band. Prompt obstetric review is advisable.",
        "Mid Risk": "Some parameters fall outside the screening reference interval. Closer ante-natal follow-up is reasonable.",
    }.get(risk_class, "Recorded measurements are consistent with the low-risk band. Routine ante-natal care applies.")
    st.markdown(
        f'<div class="mc-banner mc-banner-{key}">'
        f"<div><h3>{headline}</h3><p>{message}</p></div>"
        f'<div class="mc-banner-score">{confidence:.1f}%</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.markdown(
                kpi_card(item["label"], item["value"], item.get("note", ""), item.get("accent")),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def risk_gauge(result, title="High-risk probability"):
    probabilities = result["probabilities"]["Adaptive hybrid"]
    value = float(probabilities[-1] * 100.0)
    bar = RISK_COLORS.get(result["predicted_class"], PALETTE["blue"])
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 30, "color": PALETTE["navy"]}},
        title={"text": title, "font": {"size": 13, "color": PALETTE["text_muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PALETTE["text_subtle"],
                     "tickfont": {"color": PALETTE["chart_font"]}},
            "bar": {"color": bar, "thickness": 0.28},
            "bgcolor": "rgba(148, 163, 184, 0.08)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": PALETTE["green_soft"]},
                {"range": [33, 66], "color": PALETTE["amber_soft"]},
                {"range": [66, 100], "color": PALETTE["red_soft"]},
            ],
        },
    ))
    return _style(figure, height=210, margin=dict(l=16, r=16, t=44, b=8))


def probability_bars(result):
    probabilities = result["probabilities"]["Adaptive hybrid"]
    labels = result["class_names"]
    colors = [RISK_COLORS.get(name, PALETTE["blue"]) for name in labels]
    if len(labels) == 2:
        colors = [PALETTE["green"], PALETTE["red"]]
    figure = go.Figure(go.Bar(
        x=[float(p) * 100 for p in probabilities],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{float(p) * 100:.1f}%" for p in probabilities],
        textposition="auto",
        textfont={"color": PALETTE["navy"]},
    ))
    figure.update_layout(
        xaxis=dict(range=[0, 100], title="Model probability (%)", gridcolor=PALETTE["chart_grid"],
                   zerolinecolor=PALETTE["chart_zeroline"]),
        yaxis=dict(title=""),
    )
    return _style(figure, height=210, margin=dict(l=10, r=16, t=10, b=10))


def shap_contribution_chart(frame, title, positive_label="raises the score"):
    ordered = frame.sort_values("Hybrid Impact")
    colors = [PALETTE["red"] if value > 0 else PALETTE["green"] for value in ordered["Hybrid Impact"]]
    figure = go.Figure(go.Bar(
        x=ordered["Hybrid Impact"],
        y=ordered["Display name"],
        orientation="h",
        marker_color=colors,
        customdata=ordered["Clinical Value"],
        hovertemplate="%{y}<br>contribution %{x:+.3f}<br>value %{customdata}<extra></extra>",
    ))
    figure.update_layout(
        xaxis=dict(title=f"SHAP contribution (positive {positive_label})", zeroline=True,
                   zerolinecolor=PALETTE["chart_zeroline"], gridcolor=PALETTE["chart_grid"]),
    )
    return _style(figure, title=title, height=max(260, 42 * len(ordered)),
                  margin=dict(l=10, r=20, t=44, b=10))


def confusion_matrix_figure(matrix, class_names, title):
    matrix = np.asarray(matrix)
    figure = go.Figure(go.Heatmap(
        z=matrix,
        x=class_names,
        y=class_names,
        colorscale=[[0, "rgba(15, 26, 46, 1)"], [1, PALETTE["teal"]]],
        text=matrix,
        texttemplate="%{text}",
        showscale=False,
    ))
    figure.update_layout(
        xaxis=dict(title="Predicted", color=PALETTE["chart_font"]),
        yaxis=dict(title="Actual", autorange="reversed", color=PALETTE["chart_font"]),
    )
    return _style(figure, title=title, height=320, margin=dict(l=10, r=10, t=44, b=10))


def importance_chart(importance_map, title):
    items = sorted(importance_map.items(), key=lambda pair: pair[1])
    figure = go.Figure(go.Bar(
        x=[value * 100 for _, value in items],
        y=[key for key, _ in items],
        orientation="h",
        marker_color=PALETTE["teal"],
        text=[f"{value * 100:.1f}%" for _, value in items],
        textposition="auto",
        textfont={"color": PALETTE["navy"]},
    ))
    figure.update_layout(xaxis=dict(title="Share of total attribution (%)", gridcolor=PALETTE["chart_grid"]))
    return _style(figure, title=title, height=max(240, 40 * len(items)))


def model_comparison_chart(metrics_rows, title):
    labels = [row["model_name"] for row in metrics_rows]
    accuracies = [row["accuracy"] for row in metrics_rows]
    macro_f1 = [row["macro_f1"] for row in metrics_rows]
    figure = go.Figure()
    figure.add_bar(name="Accuracy", x=labels, y=accuracies, marker_color=PALETTE["blue"],
                   text=[f"{v:.1f}%" for v in accuracies], textposition="auto",
                   textfont={"color": PALETTE["navy"]})
    figure.add_bar(name="Macro-F1", x=labels, y=macro_f1, marker_color=PALETTE["teal"],
                   text=[f"{v:.1f}%" for v in macro_f1], textposition="auto",
                   textfont={"color": PALETTE["navy"]})
    figure.update_layout(
        barmode="group",
        yaxis=dict(title="%", range=[0, 105], gridcolor=PALETTE["chart_grid"]),
    )
    return _style(figure, title=title, height=340,
                  legend=dict(orientation="h", y=1.14, x=0, font={"color": PALETTE["chart_font"]}))


def roc_curve_figure(curves, title):
    figure = go.Figure()
    for label, (fpr, tpr, auc) in curves.items():
        figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC {auc:.3f})"))
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                line=dict(dash="dash", color=PALETTE["text_subtle"]),
                                name="Chance"))
    figure.update_layout(
        xaxis=dict(title="False positive rate", gridcolor=PALETTE["chart_grid"]),
        yaxis=dict(title="True positive rate", gridcolor=PALETTE["chart_grid"]),
    )
    return _style(figure, title=title, height=340,
                  legend=dict(orientation="h", y=-0.22, font={"color": PALETTE["chart_font"]}))


def history_trend_chart(dates, series, title, y_title):
    """Dated line chart for one measurement across saved assessments.

    ``dates`` is a list of ``datetime`` objects used directly as the x-axis, so
    the axis reflects real assessment timestamps rather than row indices.
    ``series`` is a list of dicts ``{"name", "values", "color", "unit"}`` whose
    ``values`` align to ``dates``; ``None`` entries are kept as gaps and never
    interpolated.
    """
    figure = go.Figure()
    for item in series:
        figure.add_trace(go.Scatter(
            x=dates,
            y=item["values"],
            mode="lines+markers",
            name=item["name"],
            connectgaps=False,
            line=dict(width=2.4, color=item["color"]),
            marker=dict(size=9, color=item["color"],
                        line=dict(width=1.4, color=PALETTE["page_bg"])),
            hovertemplate=(
                "%{x|%d %b %Y · %H:%M}<br>" + item["name"] + ": %{y:g} "
                + item["unit"] + "<extra></extra>"
            ),
        ))
    figure.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title="Assessment date", type="date", tickformat="%d %b\n%Y",
            showgrid=False, zeroline=False, linecolor=PALETTE["border"],
            tickfont={"color": PALETTE["chart_font"]},
        ),
        yaxis=dict(
            title=y_title, gridcolor=PALETTE["chart_grid"], zeroline=False,
            tickfont={"color": PALETTE["chart_font"]},
        ),
    )
    return _style(figure, title=title, height=300,
                  margin=dict(l=10, r=18, t=76, b=10),
                  legend=dict(orientation="h", y=1.14, x=0, font={"color": PALETTE["chart_font"]}))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def vitals_frame(rows):
    import pandas as pd

    records = []
    for row in rows:
        records.append({
            "Parameter": row["parameter"],
            "Value": f"{row['value']:g} {row['unit']}".strip(),
            "Reference interval": row["reference"],
            "Status": {"normal": "Normal", "warning": "Review", "critical": "Critical"}.get(row["status"], "Info"),
            "Interpretation": row["interpretation"],
        })
    return pd.DataFrame(records)


def model_comparison_frame(maternal_result, anemia_result):
    import pandas as pd

    records = []
    for module_name, result in (("Maternal risk", maternal_result), ("Anaemia", anemia_result)):
        for label in MODEL_LABELS:
            probabilities = result["probabilities"][label]
            index = int(np.argmax(probabilities))
            records.append({
                "Module": module_name,
                "Approach": label,
                "Predicted class": result["class_names"][index],
                "Probability": f"{float(probabilities[index]) * 100:.1f}%",
                "Full distribution": ", ".join(
                    f"{name} {float(p) * 100:.1f}%" for name, p in zip(result["class_names"], probabilities)
                ),
            })
    return pd.DataFrame(records)


def status_pills(rows):
    """Inline status chips for the abnormal-measurement summary."""
    if not rows:
        return pill("normal", "No measurements")
    return " ".join(
        f'<span class="mc-tag">{row["parameter"]}: {row["status"]}</span>' for row in rows
    )
