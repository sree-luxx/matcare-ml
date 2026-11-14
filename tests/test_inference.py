"""Inference tests: probability normalisation, class output, imputation."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.inference import MODEL_LABELS, predict_module
from app.model_loader import load_bundle

MATERNAL_INPUTS = {
    "Age": 36, "SystolicBP": 155, "DiastolicBP": 98, "BS": 15.2, "BodyTemp": 99.4, "HeartRate": 92,
}
ANEMIA_INPUTS = {"Gender": 0, "Hemoglobin": 8.2, "MCV": 72.0, "MCH": 21.0, "MCHC": 28.5}


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_probabilities_are_normalised(self):
        for module, inputs in (("maternal", MATERNAL_INPUTS), ("anemia", ANEMIA_INPUTS)):
            result = predict_module(self.bundle[module], inputs, module)
            for label in MODEL_LABELS:
                probabilities = result["probabilities"][label]
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=3, msg=f"{module} {label}")

    def test_predicted_class_is_known(self):
        result = predict_module(self.bundle["maternal"], MATERNAL_INPUTS, "maternal")
        self.assertIn(result["predicted_class"], result["class_names"])
        anemia = predict_module(self.bundle["anemia"], ANEMIA_INPUTS, "anemia")
        self.assertIn(anemia["predicted_class"], anemia["class_names"])

    def test_full_input_needs_no_imputation(self):
        result = predict_module(self.bundle["maternal"], MATERNAL_INPUTS, "maternal")
        self.assertEqual(result["missing_features"], [])

    def test_missing_input_is_imputed_and_still_valid(self):
        result = predict_module(self.bundle["maternal"], {}, "maternal")
        self.assertEqual(set(result["missing_features"]), set(result["feature_names"]))
        for label in MODEL_LABELS:
            self.assertAlmostEqual(float(result["probabilities"][label].sum()), 1.0, places=3)

    def test_hybrid_matches_weighted_base_models(self):
        result = predict_module(self.bundle["maternal"], MATERNAL_INPUTS, "maternal")
        expected = (
            result["w_rf"] * result["probabilities"]["Random Forest"]
            + result["w_xgb"] * result["probabilities"]["XGBoost"]
        )
        self.assertTrue(
            all(abs(a - b) < 1e-3 for a, b in zip(result["probabilities"]["Adaptive hybrid"], expected))
        )

    def test_anemia_model_ignores_hemoglobin_and_sex(self):
        """Changing the excluded columns must not change the anaemia prediction."""
        base = predict_module(self.bundle["anemia"], ANEMIA_INPUTS, "anemia")
        altered = predict_module(
            self.bundle["anemia"], {**ANEMIA_INPUTS, "Hemoglobin": 4.0, "Gender": 1}, "anemia"
        )
        self.assertEqual(base["predicted_class"], altered["predicted_class"])
        self.assertTrue(
            all(abs(a - b) < 1e-9 for a, b in zip(
                base["probabilities"]["Adaptive hybrid"], altered["probabilities"]["Adaptive hybrid"]
            ))
        )


if __name__ == "__main__":
    unittest.main()
