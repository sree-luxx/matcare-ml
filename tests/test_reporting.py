"""Report generation tests: Markdown structure, disclaimer, PDF bytes."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.clinical import anemia_severity, build_vitals_rows, morphology, risk_plain_language
from app.inference import MODEL_LABELS, predict_module
from app.model_loader import load_bundle
from app.reporting import DISCLAIMER, build_report_markdown, build_report_pdf
from explainability.shap_explainers import explain_single_patient

MATERNAL_INPUTS = {
    "Age": 36, "SystolicBP": 155, "DiastolicBP": 98, "BS": 15.2, "BodyTemp": 99.4, "HeartRate": 92,
}
ANEMIA_INPUTS = {"Gender": 0, "Hemoglobin": 8.2, "MCV": 72.0, "MCH": 21.0, "MCHC": 28.5}


def _report_context(bundle):
    maternal = predict_module(bundle["maternal"], MATERNAL_INPUTS, "maternal")
    anemia = predict_module(bundle["anemia"], ANEMIA_INPUTS, "anemia")
    shap_maternal = explain_single_patient(
        bundle["maternal"]["rf"], bundle["maternal"]["xgb"],
        maternal["w_rf"], maternal["w_xgb"], maternal["scaled"].to_numpy(),
        MATERNAL_INPUTS, maternal["feature_names"],
        class_idx=maternal["predicted_index"], class_name=maternal["predicted_class"],
    )
    shap_anemia = explain_single_patient(
        bundle["anemia"]["rf"], bundle["anemia"]["xgb"],
        anemia["w_rf"], anemia["w_xgb"], anemia["scaled"].to_numpy(),
        ANEMIA_INPUTS, anemia["feature_names"],
        class_idx=anemia["predicted_index"], class_name=anemia["predicted_class"],
    )
    severity, _, severity_note = anemia_severity(ANEMIA_INPUTS["Hemoglobin"])
    morph, _ = morphology(ANEMIA_INPUTS["MCV"])
    return {
        "patient": {"name": "Test Patient", "id": "PT-TEST", "age": 36, "gestational_age": "32 weeks"},
        "assessment_date": "01 January 2026, 09:00",
        "maternal_result": maternal,
        "anemia_result": anemia,
        "vitals_rows": build_vitals_rows({**MATERNAL_INPUTS, **ANEMIA_INPUTS}),
        "model_labels": MODEL_LABELS,
        "maternal_plain_language": risk_plain_language(
            maternal["predicted_class"], maternal["probabilities"]["Adaptive hybrid"]
        ),
        "anemia_severity": severity,
        "anemia_severity_note": severity_note,
        "morphology": morph,
        "shap_maternal": shap_maternal,
        "shap_anemia": shap_anemia,
        "observations": ["Systolic blood pressure is critical (155 mmHg)."],
        "maternal_limitation": "Research prototype, not externally validated.",
        "anemia_limitation": "Index-only module; haemoglobin and sex excluded.",
    }


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.markdown = build_report_markdown(_report_context(cls.bundle))

    def test_required_sections_present(self):
        for heading in (
            "# MATCARE-AI Screening Summary",
            "## 1. Maternal health measurements",
            "## 2. Maternal risk prediction",
            "## 3. Anaemia screening",
            "## 4. Model comparison",
            "## 5. Explanation of the prediction",
            "## 6. Important observations",
            "## 7. Limitations and disclaimer",
        ):
            self.assertIn(heading, self.markdown)

    def test_disclaimer_and_noncausal_note(self):
        self.assertIn(DISCLAIMER, self.markdown)
        self.assertIn("do not prove causation", self.markdown)
        self.assertIn("haemoglobin and sex code are deliberately excluded", self.markdown)

    def test_patient_details_are_rendered(self):
        self.assertIn("Test Patient", self.markdown)
        self.assertIn("PT-TEST", self.markdown)

    def test_pdf_is_produced(self):
        pdf_bytes = build_report_pdf(self.markdown)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)


if __name__ == "__main__":
    unittest.main()
