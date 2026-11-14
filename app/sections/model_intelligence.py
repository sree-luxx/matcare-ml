"""Tab 3: Model intelligence (evaluation, leakage audit, hybrid evidence)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components
from app.research import (
    ablation_table,
    audit_summary,
    bootstrap_ci,
    comparison_table,
    focus_pr_curves,
    focus_roc_curves,
    hybrid_conclusion,
    repeated_cv,
    test_predictions,
)
from app.theme import PALETTE, STATUS, card, pill

MODULES = [("maternal", "Maternal risk"), ("anemia", "Anaemia")]


def _comparison_frame(rows):
    records = []
    for row in rows:
        records.append({
            "Approach": row["model_name"],
            "Accuracy (%)": row["accuracy"],
            "Macro-F1 (%)": row["macro_f1"],
            "Macro-precision (%)": row["macro_precision"],
            "Macro-recall (%)": row["macro_recall"],
            "ROC-AUC": row["roc_auc"],
            "PR-AUC": row["pr_auc"],
            "Focus recall (%)": row["high_risk_recall"],
            "Focus FPR (%)": row["high_risk_fpr"],
            "Brier": row["brier"],
            "ECE": row["ece"],
        })
    return pd.DataFrame(records)


def _bootstrap_frame(bundle, module_label):
    intervals = bootstrap_ci(bundle)
    records = []
    for model_name, payload in intervals.items():
        row = {"Module": module_label, "Approach": model_name}
        for metric in ("accuracy", "macro_f1", "high_risk_recall"):
            stats = payload.get(metric)
            if isinstance(stats, dict) and "mean" in stats:
                row[metric] = f"{stats['mean']:.2f}% [{stats.get('ci95_low', float('nan')):.1f}, {stats.get('ci95_high', float('nan')):.1f}]"
        records.append(row)
    return pd.DataFrame(records)


def _cv_frame(bundle, module_label):
    summary = repeated_cv(bundle)
    records = []
    for model_name, payload in summary.items():
        records.append({
            "Module": module_label,
            "Approach": model_name,
            "Folds": payload.get("folds"),
            "Accuracy mean": f"{payload.get('accuracy_mean')} ± {payload.get('accuracy_std')}",
            "Macro-F1 mean": f"{payload.get('macro_f1_mean')} ± {payload.get('macro_f1_std')}",
            "Focus recall mean": f"{payload.get('focus_recall_mean')} ± {payload.get('focus_recall_std')}",
        })
    return pd.DataFrame(records)


def _ablation_frame(rows):
    records = []
    for row in rows:
        records.append({
            "Experiment": row["experiment"],
            "Features": ", ".join(row["features"]),
            "Model": row["model"],
            "Accuracy (%)": row["accuracy"],
            "Macro-F1 (%)": row["macro_f1"],
            "ROC-AUC": row["roc_auc"],
            "Anemic recall (%)": row["anemic_recall"],
        })
    return pd.DataFrame(records)


def _hybrid_evidence(bundle, module_label):
    conclusion = hybrid_conclusion(bundle)
    weights = (bundle.get("weights") or {}).get("summary", {})
    if not conclusion:
        return
    improved = conclusion.get("improved_on_held_out_test")
    status_key = "critical" if conclusion.get("collapsed_to_one_model") else ("normal" if improved else "warning")
    st.markdown(
        card(
            f"{module_label} — hybrid evidence",
            "Is the adaptive mixture better than its base models?",
            f"<p style='font-size:0.88rem;color:{PALETTE['text']};margin:0 0 8px 0;'>"
            f"{pill(status_key)}&nbsp; {conclusion.get('statement', '')}</p>"
            f"<p style='font-size:0.82rem;color:{PALETTE['text_muted']};margin:0;'>"
            f"Selected weights: RF {weights.get('best_w_rf', 0) * 100:.1f}% / XGB {weights.get('best_w_xgb', 0) * 100:.1f}%. "
            f"Plateau on the validation objective: {weights.get('plateau_w_rf', ['-'])}.</p>",
        ),
        unsafe_allow_html=True,
    )


def render(context):
    bundle = context["bundle"]

    st.markdown(
        card(
            "Evaluation protocol",
            "Held-out test split, untouched during optimisation",
            f"<p style='font-size:0.88rem;color:{PALETTE['text_muted']};margin:0;'>"
            "Hyperparameters are chosen by stratified cross-validation on the training split. "
            "Hybrid weights are chosen on the validation split. The test split is scored once. "
            "Splits, weights and metrics are read from the saved artefacts.</p>",
        ),
        unsafe_allow_html=True,
    )

    components.section("Model comparison", "Four approaches on the same held-out test split")
    for module, label in MODULES:
        rows = comparison_table(bundle[module])
        if rows:
            st.markdown(f"**{label}**")
            st.dataframe(_comparison_frame(rows), use_container_width=True, hide_index=True)
            st.plotly_chart(
                components.model_comparison_chart(rows, f"{label}: accuracy vs macro-F1"),
                use_container_width=True,
            )

    components.section("Confusion matrices and curves", "Focus class is High Risk (maternal) and Anaemic (anaemia)")
    for module, label in MODULES:
        rows = comparison_table(bundle[module])
        if not rows:
            continue
        result = test_predictions(bundle[module], module)
        class_names = result["class_names"]
        columns = st.columns(len(rows))
        for column, row in zip(columns, rows):
            with column:
                st.plotly_chart(
                    components.confusion_matrix_figure(
                        row["confusion_matrix"], class_names, row["model_name"].split(" (")[0]
                    ),
                    use_container_width=True,
                )
        roc_left, pr_right = st.columns(2)
        with roc_left:
            st.plotly_chart(
                components.roc_curve_figure(focus_roc_curves(result), f"{label}: ROC (focus class)"),
                use_container_width=True,
            )
        with pr_right:
            pr = focus_pr_curves(result)
            figure = go.Figure()
            for name, (recall, precision, area) in pr.items():
                figure.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AUC {area:.3f})"))
            figure.update_layout(
                title={"text": f"{label}: precision-recall (focus class)", "font": {"size": 13, "color": PALETTE["navy"]}},
                height=340,
                margin=dict(l=10, r=10, t=44, b=10),
                xaxis=dict(title="Recall", gridcolor=PALETTE["border"]),
                yaxis=dict(title="Precision", gridcolor=PALETTE["border"]),
                plot_bgcolor="white",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(figure, use_container_width=True)

    components.section("Robustness", "Repeated cross-validation and bootstrap confidence intervals")
    for module, label in MODULES:
        cv = _cv_frame(bundle[module], label)
        if len(cv):
            st.markdown(f"**{label} — repeated stratified CV on the development split**")
            st.dataframe(cv, use_container_width=True, hide_index=True)
        ci = _bootstrap_frame(bundle[module], label)
        if len(ci):
            st.markdown(f"**{label} — bootstrap 95% intervals on the test split**")
            st.dataframe(ci, use_container_width=True, hide_index=True)

    components.section("Hybrid evidence", "Whether the adaptive mixture is supported by the held-out results")
    for module, label in MODULES:
        _hybrid_evidence(bundle[module], label)

    components.section("Data leakage audit", "Why the anaemia model excludes haemoglobin and sex code")
    anemia_audit = audit_summary(bundle["anemia"])
    if anemia_audit:
        st.markdown(
            card(
                "Anaemia label circularity",
                "The target is reproducible from haemoglobin and the sex code",
                f"<p style='font-size:0.86rem;color:{PALETTE['text']};margin:0 0 6px 0;'>"
                f"<code>{anemia_audit.get('rule', '')}</code></p>"
                f"<p style='font-size:0.84rem;color:{PALETTE['text_muted']};margin:0;'>"
                f"Rule reproduces the stored label in {anemia_audit.get('rule_match_rate', 0) * 100:.0f}% of raw rows "
                f"(exact: {anemia_audit.get('rule_is_exact')}). Duplicate rows: {anemia_audit.get('duplicate_rows')}. "
                f"Rows kept for modelling: {anemia_audit.get('rows_after_dedup_and_conflict_removal')}. "
                f"{anemia_audit.get('limitation', '')}</p>",
            ),
            unsafe_allow_html=True,
        )
        rows = ablation_table(bundle["anemia"])
        if rows:
            st.markdown("**Ablation study (deduplicated data, fixed hyperparameters)**")
            st.dataframe(_ablation_frame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Full-feature performance is a leakage baseline. Removing haemoglobin, then the sex code, "
                "drops the index-only model to chance — the CBC indices alone cannot recover the "
                "haemoglobin-threshold label."
            )

    maternal_audit = audit_summary(bundle["maternal"])
    if maternal_audit:
        st.markdown("**Maternal dataset audit**")
        st.markdown(
            f"- Rows: {maternal_audit.get('rows')} | exact duplicates: {maternal_audit.get('duplicate_rows')}\n"
            f"- Rows kept for modelling: {maternal_audit.get('rows_used_for_modeling')} "
            f"(removed duplicates: {maternal_audit.get('rows_removed_as_exact_duplicates')}, "
            f"conflicting feature vectors: {maternal_audit.get('conflict_rows_removed')})\n"
            f"- Invalid heart-rate rows set to missing: {maternal_audit.get('invalid_heart_rate_rows')}"
        )
        st.caption(maternal_audit.get("limitation", ""))
