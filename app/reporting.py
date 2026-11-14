"""
Patient report generation.

Produces a structured screening summary in two formats:

* Markdown (``build_report_markdown``) for on-screen display and download.
* PDF (``build_report_pdf``) rendered with matplotlib so no extra runtime
  dependency is required.

The report always keeps machine-learning predictions clearly separated from
clinical diagnoses and never recommends treatment.
"""

import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

DISCLAIMER = (
    "This summary was produced by the MATCARE-AI screening prototype. Values are "
    "statistical model outputs on the given inputs, not confirmed clinical diagnoses, "
    "and they have not been externally validated. No treatment or medication is "
    "recommended by this tool. Confirm all findings with a qualified clinician."
)


def _probability_line(class_names, probabilities):
    return ", ".join(f"{name} {float(p) * 100:.1f}%" for name, p in zip(class_names, probabilities))


def _top_contributors(frame, positive_only=True, count=4):
    if frame is None or len(frame) == 0:
        return []
    ordered = frame.sort_values("Absolute Impact", ascending=False)
    if positive_only:
        ordered = ordered[ordered["Hybrid Impact"] > 0]
    rows = []
    for _, row in ordered.head(count).iterrows():
        rows.append(f"{row['Display name']} = {row['Clinical Value']} (contribution {row['Hybrid Impact']:+.3f})")
    return rows


def build_report_markdown(context):
    """Return the full patient report as a Markdown string."""
    patient = context["patient"]
    maternal = context["maternal_result"]
    anemia = context["anemia_result"]
    vitals_rows = context["vitals_rows"]
    assessment_date = context["assessment_date"]

    lines = []
    lines.append("# MATCARE-AI Screening Summary")
    lines.append(f"**Assessment date:** {assessment_date}  ")
    lines.append(f"**Patient:** {patient['name']} ({patient['id']})  ")
    lines.append(f"**Maternal age:** {patient['age']} years | **Gestational age:** {patient['gestational_age']}")
    lines.append("")

    lines.append("## 1. Maternal health measurements")
    lines.append("| Measurement | Value | Reference interval | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for row in vitals_rows:
        lines.append(
            f"| {row['parameter']} | {row['value']:g} {row['unit']} | {row['reference']} | "
            f"{row['status'].capitalize()} |"
        )
    lines.append("")

    lines.append("## 2. Maternal risk prediction")
    lines.append(f"**Predicted band:** {maternal['predicted_class']} (model score {maternal['confidence']:.1f}%)  ")
    lines.append(f"**Probability distribution:** {_probability_line(maternal['class_names'], maternal['probabilities']['Adaptive hybrid'])}")
    lines.append("")
    lines.append(f"_{context['maternal_plain_language']}_")
    lines.append("")

    lines.append("## 3. Anaemia screening")
    lines.append(f"**Clinical screen:** {context['anemia_severity']} ({context['anemia_severity_note']})  ")
    lines.append(f"**Red-cell morphology (from MCV):** {context['morphology']}  ")
    lines.append("")
    lines.append(f"**Research model baseline:** {anemia['predicted_class']} (model score {anemia['confidence']:.1f}%)  ")
    lines.append(f"**Probability distribution:** {_probability_line(anemia['class_names'], anemia['probabilities']['Adaptive hybrid'])}  ")
    lines.append(f"**Model inputs:** {', '.join(anemia['feature_names'])} "
                 "(haemoglobin and sex code are deliberately excluded \u2014 see the leakage audit).")
    lines.append("")

    lines.append("## 4. Model comparison")
    lines.append("| Module | Approach | Predicted class | Probability |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for module_name, result in (("Maternal risk", maternal), ("Anaemia", anemia)):
        for label in context["model_labels"]:
            probs = result["probabilities"][label]
            index = int(max(range(len(probs)), key=lambda i: probs[i]))
            lines.append(f"| {module_name} | {label} | {result['class_names'][index]} | {float(probs[index]) * 100:.1f}% |")
    lines.append("")

    lines.append("## 5. Explanation of the prediction")
    for module_label, frame in (("Maternal risk", context.get("shap_maternal")),
                                ("Anaemia", context.get("shap_anemia"))):
        drivers = _top_contributors(frame)
        lines.append(f"**{module_label} \u2014 measurements that raised the predicted band:**")
        if drivers:
            for item in drivers:
                lines.append(f"- {item}")
        else:
            lines.append("- No single measurement increased the predicted score.")
        lines.append("")
    lines.append("SHAP values describe how inputs moved the model score. They do not prove causation.")
    lines.append("")

    lines.append("## 6. Important observations")
    for observation in context.get("observations", []):
        lines.append(f"- {observation}")
    lines.append("")

    lines.append("## 7. Limitations and disclaimer")
    lines.append(f"- {context['maternal_limitation']}")
    lines.append(f"- {context['anemia_limitation']}")
    lines.append(f"- {DISCLAIMER}")
    lines.append("")
    return "\n".join(lines)


def _paginate(text, width=104, lines_per_page=52):
    wrapped = []
    for raw in text.split("\n"):
        if not raw.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(raw, width=width) or [""])
    return [wrapped[i:i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)]


def build_report_pdf(markdown_text):
    """Render the Markdown report into a simple paginated PDF (bytes)."""
    plain = markdown_text.replace("**", "").replace("_", "").replace("# ", "").replace("## ", "")
    pages = _paginate(plain)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for number, page_lines in enumerate(pages, start=1):
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.text(0.08, 0.96, "MATCARE-AI Screening Summary", fontsize=13, fontweight="bold")
            figure.text(0.08, 0.945, f"page {number} of {len(pages)}", fontsize=8, color="#64748b")
            figure.text(
                0.08, 0.925, "\n".join(page_lines),
                fontsize=8.2, family="monospace", va="top", linespacing=1.45,
            )
            figure.text(0.08, 0.035, "Screening output only \u2014 not a clinical diagnosis.",
                        fontsize=7.5, color="#b45309")
            pdf.savefig(figure)
            plt.close(figure)
    buffer.seek(0)
    return buffer.getvalue()
